use std::{any::Any, collections::HashMap};

use alloy::primitives::U256;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use tycho_common::{
    dto::ProtocolStateDelta,
    models::token::Token,
    simulation::{
        errors::{SimulationError, TransitionError},
        protocol_sim::{Balances, GetAmountOutResult, PoolSwap, ProtocolSim, QueryPoolSwapParams},
    },
    Bytes,
};

use super::math::{self, FEE_DENOMINATOR};
use crate::evm::protocol::{
    safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256},
    u256_num::{biguint_to_u256, u256_to_biguint, u256_to_f64},
    utils::add_fee_markup,
};

/// Curve StableSwap 2-token pool state.
///
/// Reserves are stored in raw (native) token decimals as received from the indexer.
/// Normalization to 18 decimals happens internally before invariant math, matching
/// Curve's on-chain RATES mechanism.
///
/// Reference: <https://github.com/curvefi/curve-contract/blob/master/contracts/pool-templates/base/SwapTemplateBase.vy>
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurveStableSwapState {
    reserve0: U256,
    reserve1: U256,
    /// Amplification coefficient in internal precision: `A * A_PRECISION` (A_PRECISION=100).
    /// Note: on-chain `A()` returns `A / A_PRECISION`. Use `_A()` or multiply `A()` by 100.
    amplification: U256,
    /// Fee in Curve's FEE_DENOMINATOR (10^10). Example: 4_000_000 = 0.04%.
    fee: U256,
    /// Normalization rate for token0: 10^(18 - token0_decimals).
    rate0: U256,
    /// Normalization rate for token1: 10^(18 - token1_decimals).
    rate1: U256,
}

impl CurveStableSwapState {
    pub fn new(
        reserve0: U256,
        reserve1: U256,
        amplification: U256,
        fee: U256,
        rate0: U256,
        rate1: U256,
    ) -> Self {
        Self { reserve0, reserve1, amplification, fee, rate0, rate1 }
    }

    /// Normalize reserves to 18-decimal space (Curve RATES).
    fn normalized_reserves(&self) -> Result<(U256, U256), SimulationError> {
        Ok((safe_mul_u256(self.reserve0, self.rate0)?, safe_mul_u256(self.reserve1, self.rate1)?))
    }
}

#[typetag::serde]
impl ProtocolSim for CurveStableSwapState {
    fn fee(&self) -> f64 {
        u256_to_f64(self.fee).expect("Fee value is safe to convert") / 1e10
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        if self.reserve0.is_zero() || self.reserve1.is_zero() {
            return Err(SimulationError::RecoverableError("No liquidity".to_string()));
        }
        let zero_to_one = quote.address < base.address;
        let (norm_reserve0, norm_reserve1) = self.normalized_reserves()?;

        let d = math::get_d(norm_reserve0, norm_reserve1, self.amplification)?;

        // Analytical marginal price: dy/dx from the StableSwap invariant derivative.
        // quote is the input token (what you pay), base is what you buy.
        let (norm_in, norm_out) = if zero_to_one {
            (norm_reserve0, norm_reserve1)
        } else {
            (norm_reserve1, norm_reserve0)
        };
        let (num, den) = math::spot_price_raw(norm_in, norm_out, d, self.amplification)?;

        // base_per_quote = num / den (how much base you get per 1 quote)
        let base_per_quote = u256_to_f64(num)? / u256_to_f64(den)?;

        // Invert: price = quote per base
        let price = 1.0 / base_per_quote;

        Ok(add_fee_markup(price, self.fee()))
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        let amount_in_u256 = biguint_to_u256(&amount_in);
        if amount_in_u256.is_zero() {
            return Err(SimulationError::InvalidInput("Amount in cannot be zero".to_string(), None));
        }
        if self.reserve0.is_zero() || self.reserve1.is_zero() {
            return Err(SimulationError::RecoverableError("No liquidity".to_string()));
        }
        let zero_to_one = token_in.address < token_out.address;

        let (norm_reserve0, norm_reserve1) = self.normalized_reserves()?;
        let (rate_in, rate_out) =
            if zero_to_one { (self.rate0, self.rate1) } else { (self.rate1, self.rate0) };
        let (norm_in, norm_out) = if zero_to_one {
            (norm_reserve0, norm_reserve1)
        } else {
            (norm_reserve1, norm_reserve0)
        };

        // Scale input to 18 decimals
        let amount_in_normalized = safe_mul_u256(amount_in_u256, rate_in)?;

        let d = math::get_d(norm_reserve0, norm_reserve1, self.amplification)?;
        let x_new = safe_add_u256(norm_in, amount_in_normalized)?;
        let y_new = math::get_y(x_new, d, self.amplification)?;

        // Curve subtracts 1 wei as safety margin: dy = xp[j] - y - 1
        let dy_raw = safe_sub_u256(safe_sub_u256(norm_out, y_new)?, U256::from(1))?;

        // Apply fee (in normalized space)
        let fee_amount = safe_div_u256(safe_mul_u256(dy_raw, self.fee)?, FEE_DENOMINATOR)?;
        let dy_after_fee = safe_sub_u256(dy_raw, fee_amount)?;

        // Scale output back to token's native decimals
        let amount_out = safe_div_u256(dy_after_fee, rate_out)?;

        let mut new_state = self.clone();
        if zero_to_one {
            new_state.reserve0 = safe_add_u256(self.reserve0, amount_in_u256)?;
            new_state.reserve1 = safe_sub_u256(self.reserve1, amount_out)?;
        } else {
            new_state.reserve0 = safe_sub_u256(self.reserve0, amount_out)?;
            new_state.reserve1 = safe_add_u256(self.reserve1, amount_in_u256)?;
        }

        Ok(GetAmountOutResult::new(
            u256_to_biguint(amount_out),
            BigUint::from(150_000u32),
            Box::new(new_state),
        ))
    }

    fn get_limits(
        &self,
        sell_token: Bytes,
        buy_token: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        let zero_to_one = sell_token < buy_token;
        let (reserve_in, reserve_out) = if zero_to_one {
            (self.reserve0, self.reserve1)
        } else {
            (self.reserve1, self.reserve0)
        };

        if reserve_in.is_zero() || reserve_out.is_zero() {
            return Ok((BigUint::ZERO, BigUint::ZERO));
        }

        // Compute the input amount that would drain ~90% of the output reserve.
        // This is the actual protocol limit — Curve reverts when output reserve hits 0.
        let (norm_reserve0, norm_reserve1) = self.normalized_reserves()?;
        let (rate_in, rate_out) =
            if zero_to_one { (self.rate0, self.rate1) } else { (self.rate1, self.rate0) };
        let (norm_in, norm_out) = if zero_to_one {
            (norm_reserve0, norm_reserve1)
        } else {
            (norm_reserve1, norm_reserve0)
        };

        // Target: output reserve drops to 10% of current (90% drained)
        let y_target = safe_div_u256(norm_out, U256::from(10))?;

        let d = math::get_d(norm_reserve0, norm_reserve1, self.amplification)?;
        // get_y is symmetric: given target y, find required x
        let x_required = math::get_y(y_target, d, self.amplification)?;
        let max_in_normalized = safe_sub_u256(x_required, norm_in)?;

        // Scale back to native decimals
        let max_in = safe_div_u256(max_in_normalized, rate_in)?;

        // Compute max_out for this max_in, including fee
        let dy_raw = safe_sub_u256(norm_out, y_target)?;
        let fee_amount = safe_div_u256(safe_mul_u256(dy_raw, self.fee)?, FEE_DENOMINATOR)?;
        let dy_after_fee = safe_sub_u256(dy_raw, fee_amount)?;
        let max_out = safe_div_u256(dy_after_fee, rate_out)?;

        Ok((u256_to_biguint(max_in), u256_to_biguint(max_out)))
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        _tokens: &HashMap<Bytes, Token>,
        _balances: &Balances,
    ) -> Result<(), TransitionError> {
        self.reserve0 = U256::from_be_slice(
            delta
                .updated_attributes
                .get("reserve0")
                .ok_or(TransitionError::MissingAttribute("reserve0".to_string()))?,
        );

        self.reserve1 = U256::from_be_slice(
            delta
                .updated_attributes
                .get("reserve1")
                .ok_or(TransitionError::MissingAttribute("reserve1".to_string()))?,
        );

        if let Some(amp_bytes) = delta.updated_attributes.get("A") {
            self.amplification = U256::from_be_slice(amp_bytes) * math::A_PRECISION;
        }

        if let Some(fee_bytes) = delta.updated_attributes.get("fee") {
            self.fee = U256::from_be_slice(fee_bytes);
        }

        Ok(())
    }

    fn query_pool_swap(&self, _params: &QueryPoolSwapParams) -> Result<PoolSwap, SimulationError> {
        Err(SimulationError::InvalidInput(
            "CurveStableSwapState does not support query_pool_swap yet".to_string(),
            None,
        ))
    }

    fn clone_box(&self) -> Box<dyn ProtocolSim> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn eq(&self, other: &dyn ProtocolSim) -> bool {
        if let Some(other_state) = other.as_any().downcast_ref::<Self>() {
            self.reserve0 == other_state.reserve0 &&
                self.reserve1 == other_state.reserve1 &&
                self.amplification == other_state.amplification &&
                self.fee == other_state.fee &&
                self.rate0 == other_state.rate0 &&
                self.rate1 == other_state.rate1
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        str::FromStr,
    };

    use num_bigint::BigUint;
    use num_traits::Zero;
    use tycho_common::{
        dto::ProtocolStateDelta,
        hex_bytes::Bytes,
        models::{token::Token, Chain},
        simulation::{
            errors::TransitionError,
            protocol_sim::{Balances, ProtocolSim},
        },
    };

    use super::*;

    fn token_dai() -> Token {
        Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn token_usdc() -> Token {
        Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap(),
            "USDC",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    /// Balanced pool: 1M DAI (18 dec) + 1M USDC (6 dec)
    fn balanced_pool() -> CurveStableSwapState {
        CurveStableSwapState::new(
            U256::from(1_000_000_000_000_000_000_000_000u128), // 1M DAI in wei
            U256::from(1_000_000_000_000u128),                 // 1M USDC in 6-dec
            U256::from(10_000u64),                             // A = 100
            U256::from(4_000_000u64),                          // 0.04% fee
            math::rate_from_decimals(18).unwrap(),
            math::rate_from_decimals(6).unwrap(),
        )
    }

    /// Same-decimal pool for simpler math verification
    fn balanced_pool_same_decimals() -> CurveStableSwapState {
        CurveStableSwapState::new(
            U256::from(1_000_000_000_000_000_000_000_000u128),
            U256::from(1_000_000_000_000_000_000_000_000u128),
            U256::from(10_000u64),
            U256::from(4_000_000u64),
            math::rate_from_decimals(18).unwrap(),
            math::rate_from_decimals(18).unwrap(),
        )
    }

    #[test]
    fn test_get_amount_out_same_decimals() {
        let state = balanced_pool_same_decimals();
        let amount_in = BigUint::from(1_000_000_000_000_000_000_000u128); // 1000 tokens

        let result = state
            .get_amount_out(amount_in, &token_dai(), &token_usdc())
            .unwrap();

        let expected_min = BigUint::from(999_000_000_000_000_000_000u128);
        let expected_max = BigUint::from(1_000_000_000_000_000_000_000u128);
        assert!(result.amount > expected_min && result.amount < expected_max);
    }

    #[test]
    fn test_get_amount_out_different_decimals() {
        let state = balanced_pool();

        // Swap 1000 DAI (18 dec) for USDC (6 dec)
        let amount_in = BigUint::from(1_000_000_000_000_000_000_000u128); // 1000 DAI
        let result = state
            .get_amount_out(amount_in, &token_dai(), &token_usdc())
            .unwrap();

        // Should get ~999 USDC (6 decimals)
        let expected_min = BigUint::from(999_000_000u128); // 999 USDC
        let expected_max = BigUint::from(1_000_000_000u128); // 1000 USDC
        assert!(
            result.amount > expected_min && result.amount < expected_max,
            "Expected ~999 USDC, got {}",
            result.amount
        );
    }

    #[test]
    fn test_get_amount_out_usdc_to_dai() {
        let state = balanced_pool();

        // Swap 1000 USDC (6 dec) for DAI (18 dec)
        let amount_in = BigUint::from(1_000_000_000u128); // 1000 USDC
        let result = state
            .get_amount_out(amount_in, &token_usdc(), &token_dai())
            .unwrap();

        // Should get ~999 DAI (18 decimals)
        let expected_min = BigUint::from(999_000_000_000_000_000_000u128);
        let expected_max = BigUint::from(1_000_000_000_000_000_000_000u128);
        assert!(
            result.amount > expected_min && result.amount < expected_max,
            "Expected ~999 DAI, got {}",
            result.amount
        );
    }

    #[test]
    fn test_get_amount_out_imbalanced() {
        let state = CurveStableSwapState::new(
            U256::from(900_000_000_000_000_000_000_000u128), // 900k DAI
            U256::from(100_000_000_000u128),                 // 100k USDC
            U256::from(10_000u64),
            U256::from(4_000_000u64),
            math::rate_from_decimals(18).unwrap(),
            math::rate_from_decimals(6).unwrap(),
        );

        let amount_in = BigUint::from(10_000_000_000_000_000_000_000u128); // 10k DAI
        let result = state
            .get_amount_out(amount_in, &token_dai(), &token_usdc())
            .unwrap();

        // Swapping abundant token for scarce → slippage, output < 10k USDC
        assert!(result.amount < BigUint::from(10_000_000_000u128)); // < 10k USDC
    }

    #[test]
    fn test_fee() {
        let state = balanced_pool();
        let fee = state.fee();
        assert!((fee - 0.0004).abs() < 1e-10);
    }

    #[test]
    fn test_spot_price_balanced() {
        let state = balanced_pool();
        let price = state
            .spot_price(&token_dai(), &token_usdc())
            .unwrap();
        // Balanced stableswap → price ≈ 1.0
        assert!((price - 1.0).abs() < 0.01, "Expected price ~1.0, got {}", price);
    }

    #[test]
    fn test_roundtrip() {
        let state = balanced_pool_same_decimals();
        let amount_in = BigUint::from(5_000_000_000_000_000_000_000u128); // 5000

        let result1 = state
            .get_amount_out(amount_in.clone(), &token_dai(), &token_usdc())
            .unwrap();
        let new_state = result1
            .new_state
            .as_any()
            .downcast_ref::<CurveStableSwapState>()
            .unwrap();

        let result2 = new_state
            .get_amount_out(result1.amount.clone(), &token_usdc(), &token_dai())
            .unwrap();

        assert!(result2.amount < amount_in);
        let min_expected = &amount_in * BigUint::from(99u32) / BigUint::from(100u32);
        assert!(
            result2.amount > min_expected,
            "roundtrip loss too high: got {} back from {}",
            result2.amount,
            amount_in
        );
    }

    #[test]
    fn test_symmetry_same_decimals() {
        let state = balanced_pool_same_decimals();
        let amount = BigUint::from(1_000_000_000_000_000_000_000u128);

        let out_0_to_1 = state
            .get_amount_out(amount.clone(), &token_dai(), &token_usdc())
            .unwrap();
        let out_1_to_0 = state
            .get_amount_out(amount.clone(), &token_usdc(), &token_dai())
            .unwrap();

        assert_eq!(out_0_to_1.amount, out_1_to_0.amount);
    }

    #[test]
    fn test_delta_transition() {
        let mut state = balanced_pool();
        let attributes: HashMap<String, Bytes> = vec![
            ("reserve0".to_string(), Bytes::from(U256::from(2_000_000u64).to_be_bytes_vec())),
            ("reserve1".to_string(), Bytes::from(U256::from(3_000_000u64).to_be_bytes_vec())),
        ]
        .into_iter()
        .collect();

        let delta = ProtocolStateDelta {
            component_id: "State1".to_owned(),
            updated_attributes: attributes,
            deleted_attributes: HashSet::new(),
        };

        let result = state.delta_transition(delta, &HashMap::new(), &Balances::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_delta_transition_missing_attribute() {
        let mut state = balanced_pool();
        let attributes: HashMap<String, Bytes> =
            vec![("reserve0".to_string(), Bytes::from(U256::from(1500u64).to_be_bytes_vec()))]
                .into_iter()
                .collect();

        let delta = ProtocolStateDelta {
            component_id: "State1".to_owned(),
            updated_attributes: attributes,
            deleted_attributes: HashSet::new(),
        };

        let result = state.delta_transition(delta, &HashMap::new(), &Balances::default());
        assert!(result.is_err());
        match result {
            Err(TransitionError::MissingAttribute(attr)) => assert_eq!(attr, "reserve1"),
            _ => panic!("Expected MissingAttribute error"),
        }
    }

    #[test]
    fn test_delta_transition_updates_amplification() {
        let mut state = balanced_pool();
        let attributes: HashMap<String, Bytes> = vec![
            ("reserve0".to_string(), Bytes::from(state.reserve0.to_be_bytes_vec())),
            ("reserve1".to_string(), Bytes::from(state.reserve1.to_be_bytes_vec())),
            ("A".to_string(), Bytes::from(U256::from(200u64).to_be_bytes_vec())),
        ]
        .into_iter()
        .collect();

        let delta = ProtocolStateDelta {
            component_id: "State1".to_owned(),
            updated_attributes: attributes,
            deleted_attributes: HashSet::new(),
        };

        state
            .delta_transition(delta, &HashMap::new(), &Balances::default())
            .unwrap();
        assert_eq!(state.amplification, U256::from(200u64) * super::math::A_PRECISION);
    }

    #[test]
    fn test_get_limits_dai_to_usdc() {
        // DAI (18 dec) → USDC (6 dec), balanced 1M each
        let state = balanced_pool();
        let dai = Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap();
        let usdc = Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap();

        let (max_in, max_out) = state
            .get_limits(dai.clone(), usdc.clone())
            .unwrap();

        // max_in should be in DAI decimals (18), significantly more than reserve
        assert!(max_in > BigUint::ZERO);
        // max_out should be in USDC decimals (6), close to 90% of 1M USDC
        let usdc_900k = BigUint::from(900_000_000_000u128); // 900k USDC
        assert!(
            max_out > usdc_900k / BigUint::from(2u32),
            "max_out should be substantial, got {}",
            max_out
        );

        // Verify get_amount_out works at max_in
        let result = state.get_amount_out(max_in, &token_dai(), &token_usdc());
        assert!(result.is_ok(), "get_amount_out should succeed at max_in");
    }

    #[test]
    fn test_get_limits_usdc_to_dai() {
        // USDC (6 dec) → DAI (18 dec)
        let state = balanced_pool();
        let dai = Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap();
        let usdc = Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap();

        let (max_in, max_out) = state.get_limits(usdc, dai).unwrap();

        // max_in should be in USDC decimals (6)
        assert!(max_in > BigUint::ZERO);
        // max_out should be in DAI decimals (18)
        let dai_900k = BigUint::from(900_000_000_000_000_000_000_000u128); // 900k DAI
        assert!(
            max_out > dai_900k / BigUint::from(2u32),
            "max_out should be substantial, got {}",
            max_out
        );

        // Verify get_amount_out works at max_in
        let result = state.get_amount_out(max_in, &token_usdc(), &token_dai());
        assert!(result.is_ok(), "get_amount_out should succeed at max_in");
    }

    #[test]
    fn test_get_limits_empty_pool() {
        let state = CurveStableSwapState::new(
            U256::ZERO,
            U256::ZERO,
            U256::from(10_000u64),
            U256::from(4_000_000u64),
            math::rate_from_decimals(18).unwrap(),
            math::rate_from_decimals(6).unwrap(),
        );
        let t0 = Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap();
        let t1 = Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap();

        let (max_in, max_out) = state.get_limits(t0, t1).unwrap();
        assert!(max_in.is_zero());
        assert!(max_out.is_zero());
    }

    #[test]
    fn test_clone_box_and_eq() {
        let state = balanced_pool();
        let cloned = state.clone_box();
        assert!(ProtocolSim::eq(&state, cloned.as_ref()));
    }

    #[test]
    fn test_new_state_is_immutable() {
        let state = balanced_pool();
        let original_r0 = state.reserve0;
        let original_r1 = state.reserve1;

        let _ = state.get_amount_out(
            BigUint::from(1_000_000_000_000_000_000_000u128),
            &token_dai(),
            &token_usdc(),
        );

        assert_eq!(state.reserve0, original_r0);
        assert_eq!(state.reserve1, original_r1);
    }

    #[test]
    fn test_zero_amount_returns_err() {
        let state = balanced_pool();
        assert!(state
            .get_amount_out(BigUint::ZERO, &token_dai(), &token_usdc())
            .is_err());
    }
}
