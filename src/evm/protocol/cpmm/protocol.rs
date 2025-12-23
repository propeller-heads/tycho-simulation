use alloy::primitives::{U256, U512};
use num_bigint::BigUint;
use num_traits::Zero;
use tycho_client::feed::synchronizer::ComponentWithState;
use tycho_common::{
    dto::ProtocolStateDelta,
    models::token::Token,
    simulation::{
        errors::{SimulationError, TransitionError},
        protocol_sim::Price,
    },
    Bytes,
};

use super::reserve_price::spot_price_from_reserves;
use crate::{
    evm::protocol::{
        safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256, sqrt_u512},
        u256_num::{biguint_to_u256, u256_to_biguint},
        utils::solidity_math::mul_div,
    },
    protocol::errors::InvalidSnapshotError,
};

pub fn cpmm_try_from_with_header(
    snapshot: ComponentWithState,
) -> Result<(U256, U256), InvalidSnapshotError> {
    let reserve0 = U256::from_be_slice(
        snapshot
            .state
            .attributes
            .get("reserve0")
            .ok_or(InvalidSnapshotError::MissingAttribute("reserve0".to_string()))?,
    );

    let reserve1 = U256::from_be_slice(
        snapshot
            .state
            .attributes
            .get("reserve1")
            .ok_or(InvalidSnapshotError::MissingAttribute("reserve1".to_string()))?,
    );
    Ok((reserve0, reserve1))
}

pub fn cpmm_fee(fee_bps: u32) -> f64 {
    fee_bps as f64 / 10000.0
}

pub fn cpmm_spot_price(
    base: &Token,
    quote: &Token,
    reserve0: U256,
    reserve1: U256,
) -> Result<f64, SimulationError> {
    if base < quote {
        spot_price_from_reserves(reserve0, reserve1, base.decimals, quote.decimals)
    } else {
        spot_price_from_reserves(reserve1, reserve0, base.decimals, quote.decimals)
    }
}

pub fn cpmm_get_amount_out(
    amount_in: U256,
    reserve_in: U256,
    reserve_out: U256,
    fee: ProtocolFee,
) -> Result<U256, SimulationError> {
    if amount_in == U256::from(0u64) {
        return Err(SimulationError::InvalidInput("Amount in cannot be zero".to_string(), None));
    }

    if reserve_in == U256::from(0u64) || reserve_out == U256::from(0u64) {
        return Err(SimulationError::RecoverableError("No liquidity".to_string()));
    }

    let amount_in_with_fee = safe_mul_u256(amount_in, fee.numerator)?;
    let numerator = safe_mul_u256(amount_in_with_fee, reserve_out)?;
    let denominator = safe_add_u256(safe_mul_u256(reserve_in, fee.precision)?, amount_in_with_fee)?;

    safe_div_u256(numerator, denominator)
}

/// Returns the soft limit for amount_in to achieve approximately 90% price impact.
///
/// This calculation assumes a fee-less constant product formula:
/// - 90% price impact: `(reserve_out - y) / (reserve_in + x) = 0.1 × (reserve_out / reserve_in)`
/// - Constant product: `(reserve_in + x) × (reserve_out - y) = reserve_in × reserve_out`
///
/// However, in practice, swaps apply fees to the input amount, with the fee portion then added to
/// the pool's liquidity. So the constant formula becomes:
/// - `(reserve_in + x × (1 - fee)) × (reserve_out - y) = reserve_in × reserve_out`
///
/// This asymmetry (full `x` added to reserves, but `y` based on fee-adjusted input) causes
/// the actual price impact to be slightly less than 90%. For typical fees (0.25%-0.3%), the
/// actual price impact is approximately 89.9% instead of exactly 90%.
pub fn cpmm_get_limits(
    sell_token: Bytes,
    buy_token: Bytes,
    reserve0: U256,
    reserve1: U256,
    fee_bps: u32,
) -> Result<(BigUint, BigUint), SimulationError> {
    if reserve0 == U256::from(0u64) || reserve1 == U256::from(0u64) {
        return Ok((BigUint::zero(), BigUint::zero()));
    }

    let zero_for_one = sell_token < buy_token;
    let (reserve_in, reserve_out) =
        if zero_for_one { (reserve0, reserve1) } else { (reserve1, reserve0) };

    // Soft limit for amount in is the amount to get approximately 90% price impact.
    // Solving the fee-less equations yields: x = (√10 - 1) × reserve_in ≈ 2.162 × reserve_in
    let amount_in = mul_div(reserve_in, U256::from(2162u128), U256::from(1e3))?;

    // Calculate amount_out using the actual swap formula that accounts for fees
    // amount_in_with_fee = amount_in * (10000 - fee_bps) / 10000
    // amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    let fee_multiplier = U256::from(10000 - fee_bps);
    let amount_in_with_fee = safe_mul_u256(amount_in, fee_multiplier)?;

    let amount_out =
        mul_div(reserve_out, amount_in_with_fee, safe_add_u256(reserve_in, amount_in)?)?;

    Ok((u256_to_biguint(amount_in), u256_to_biguint(amount_out)))
}

pub fn cpmm_delta_transition(
    delta: ProtocolStateDelta,
    reserve0_mut: &mut U256,
    reserve1_mut: &mut U256,
) -> Result<(), TransitionError<String>> {
    // reserve0 and reserve1 are considered required attributes and are expected in every delta
    // we process
    let reserve0 = U256::from_be_slice(
        delta
            .updated_attributes
            .get("reserve0")
            .ok_or(TransitionError::MissingAttribute("reserve0".to_string()))?,
    );
    let reserve1 = U256::from_be_slice(
        delta
            .updated_attributes
            .get("reserve1")
            .ok_or(TransitionError::MissingAttribute("reserve1".to_string()))?,
    );
    *reserve0_mut = reserve0;
    *reserve1_mut = reserve1;
    Ok(())
}

/// Represents a protocol fee as a numerator and precision.
pub struct ProtocolFee {
    pub numerator: U256,
    pub precision: U256,
}

impl ProtocolFee {
    pub fn new(numerator: U256, precision: U256) -> Self {
        ProtocolFee { numerator, precision }
    }
}

/// Calculates the exact amount of token_in required to move the pool's marginal price down to
/// a target price.
///
/// # Algorithm
///
/// Derives how much to swap to reach a target price using the constant product formula.
/// **Note**: This method assumes k remains constant, but in reality fees accrue to the pool,
/// causing k to increase slightly. This simplification leads to a conservative
/// underestimation of the pool's supply capacity.
///
/// ## Base equations
/// 1. Constant product: `x * y = k` where x = reserve_in, y = reserve_out
/// 2. Swap with 0.3% fee: Only 99.7% of input affects price
/// 3. Marginal price after swap: `price = (x' * 10000) / (y' * 9970)`
///
/// ## Derivation
/// We want the pool to reach target price: `price = sell_price / buy_price`
///
/// From marginal price formula:
/// ```text,no_run
/// x' / y' = (sell_price * 9970) / (buy_price * 10000)  [call this target_price_w_fee]
/// ```
///
/// From constant product:
/// ```text,no_run
/// x' * y' = k
/// ```
///
///
/// Substituting the first into the second:
/// ```text,no_run
/// x' = target_price_w_fee * y'
/// (target_price_w_fee * y') * y' = k
/// y'^2 = k / target_price_w_fee
/// y' = sqrt(k / target_price_w_fee)
/// ```
///
/// Therefore:
/// ```text,no_run
/// x' = target_price_w_fee * y'
///    = target_price_w_fee * sqrt(k / target_price_w_fee)
///    = sqrt(k * target_price_w_fee)
/// ```
///
/// Amount to swap in:
/// ```text,no_run
/// amount_in = x' - x = sqrt(k * target_price_w_fee) - reserve_in
/// ```
///
/// where `target_price_w_fee = (sell_price * 9970) / (buy_price * 10000)`
/// Then swap to get amount_out.
///
/// # Returns
/// * `Ok((amount_in, implied_amount_out))` - The implied amount out is computed analytically and
///   will be smaller than actually swapping against the pool.
///
/// # Errors
/// * `FatalError` - If reserves are zero or calculation overflows
/// * `InvalidInput` - If target price is unreachable (already below current spot price)
pub fn cpmm_swap_to_price(
    reserve_in: U256,
    reserve_out: U256,
    target_price: &Price,
    fee: ProtocolFee,
) -> Result<(BigUint, BigUint), SimulationError> {
    if reserve_in == U256::ZERO || reserve_out == U256::ZERO {
        return Err(SimulationError::FatalError("Reserves cannot be zero".to_string()));
    }

    // Flip target pool price to swap price
    let swap_price_num = biguint_to_u256(&target_price.denominator);
    let swap_price_den = biguint_to_u256(&target_price.numerator);

    // Check reachability: target price must be above the spot price (with fees)
    // swap_price_num/swap_price_den >= (reserve_in * FEE_PRECISION) / (reserve_out *
    // FEE_NUMERATOR)
    // Cross-multiply to avoid division: swap_price_num * reserve_out * FEE_NUMERATOR >=
    // swap_price_den * reserve_in * FEE_PRECISION
    // Use U512 precision to match the calculation of new reserves
    let target_price_cross_mult = U512::from(swap_price_num)
        .checked_mul(U512::from(reserve_out))
        .and_then(|x| x.checked_mul(U512::from(fee.numerator)))
        .ok_or_else(|| SimulationError::FatalError("Overflow in price check".to_string()))?;
    let current_price_cross_mult = U512::from(swap_price_den)
        .checked_mul(U512::from(reserve_in))
        .and_then(|x| x.checked_mul(U512::from(fee.precision)))
        .ok_or_else(|| SimulationError::FatalError("Overflow in price check".to_string()))?;

    if target_price_cross_mult < current_price_cross_mult {
        return Err(SimulationError::InvalidInput(
            "Target price is unreachable (already below current spot price)".to_string(),
            None,
        ));
    }

    // Calculate new reserve_in: x' = sqrt(k * price_num * FEE_NUMERATOR / (price_den *
    // FEE_PRECISION))
    let k = U512::from(reserve_in) * U512::from(reserve_out);
    let k_times_price = k * U512::from(swap_price_num) * U512::from(fee.numerator) /
        (U512::from(swap_price_den) * U512::from(fee.precision));
    let x_prime_u512 = sqrt_u512(k_times_price);

    // Convert back to U256 and calculate amount_in
    let limbs = x_prime_u512.as_limbs();
    let x_prime = U256::from_limbs([limbs[0], limbs[1], limbs[2], limbs[3]]);

    if x_prime <= reserve_in {
        return Ok((BigUint::ZERO, BigUint::ZERO));
    }

    let amount_in = safe_sub_u256(x_prime, reserve_in)?;
    if amount_in == U256::ZERO {
        return Ok((BigUint::ZERO, BigUint::ZERO));
    }

    let implied_amount_out = mul_div(amount_in, swap_price_den, swap_price_num)?;

    Ok((u256_to_biguint(amount_in), u256_to_biguint(implied_amount_out)))
}

/// Calculates the exact amount of token_in required such that the trade's execution price
/// equals the target price.
///
/// # Algorithm
///
/// Unlike `cpmm_swap_to_price` which targets the *marginal* (spot) price after the swap,
/// this function targets the *trade* (average execution) price of the swap itself.
///
/// ## Base equations
/// 1. Trade price: `limit_price = amount_in / amount_out`
/// 2. Amount out: `amount_out = (reserve_out * amount_in * fee_numerator) / (reserve_in *
///    fee_precision + amount_in * fee_numerator)`
///
/// ## Derivation
///
/// We want `amount_in / amount_out = limit_price`.
///
/// Let `γ = fee_numerator / fee_precision` (e.g., 9970/10000 for 0.3% fee).
///
/// Substituting the CPMM formula for `amount_out`:
/// ```text,no_run
/// limit_price = amount_in / [ (reserve_out * amount_in * γ) / (reserve_in + amount_in * γ) ]
/// limit_price = (reserve_in + amount_in * γ) / (reserve_out * γ)
/// limit_price * reserve_out * γ = reserve_in + amount_in * γ
/// amount_in * γ = limit_price * reserve_out * γ - reserve_in
/// amount_in = limit_price * reserve_out - reserve_in / γ
/// ```
///
/// Translated to integer math with precision:
/// ```text,no_run
/// amount_in = (P_num * reserve_out * fee_num - reserve_in * P_den * fee_prec) / (P_den * fee_num)
/// ```
///
/// where `limit_price = P_num / P_den` after flipping the input `Price` (which is in
/// amount_out/amount_in convention).
///
/// # Returns
/// * `Ok((amount_in, implied_amount_out))` - The implied amount out satisfies `amount_in /
///   implied_amount_out ≈ limit_price` (after flipping, with integer rounding).
///
/// # Errors
/// * `FatalError` - If reserves are zero or calculation overflows
/// * `InvalidInput` - If limit price is unreachable (at or better than current spot price)
pub fn cpmm_swap_to_trade_price(
    reserve_in: U256,
    reserve_out: U256,
    limit_price: &Price,
    fee: ProtocolFee,
) -> Result<(BigUint, BigUint), SimulationError> {
    if reserve_in == U256::ZERO || reserve_out == U256::ZERO {
        return Err(SimulationError::FatalError("Reserves cannot be zero".to_string()));
    }

    // Flip limit pool price to swap price (P = amount_in / amount_out)
    // Input Price is in amount_out/amount_in convention, so we invert it
    let swap_price_num = biguint_to_u256(&limit_price.denominator);
    let swap_price_den = biguint_to_u256(&limit_price.numerator);

    // Check reachability: limit trade price P must be strictly greater than current spot price
    // (with fees). As you trade, P increases (you pay more per unit received), so the limit
    // must be above the starting spot price.
    //
    // Current spot price (in P convention) = reserve_in / (reserve_out * gamma)
    //                                      = (reserve_in * fee_precision) / (reserve_out *
    // fee_numerator)
    //
    // Condition: P > spot_price
    // swap_price_num / swap_price_den > (reserve_in * fee_precision) / (reserve_out *
    // fee_numerator) Cross-multiply: swap_price_num * reserve_out * fee_numerator >
    // swap_price_den * reserve_in * fee_precision
    let limit_price_cross_mult = U512::from(swap_price_num)
        .checked_mul(U512::from(reserve_out))
        .and_then(|x| x.checked_mul(U512::from(fee.numerator)))
        .ok_or_else(|| SimulationError::FatalError("Overflow in price check".to_string()))?;
    let current_price_cross_mult = U512::from(swap_price_den)
        .checked_mul(U512::from(reserve_in))
        .and_then(|x| x.checked_mul(U512::from(fee.precision)))
        .ok_or_else(|| SimulationError::FatalError("Overflow in price check".to_string()))?;

    if limit_price_cross_mult <= current_price_cross_mult {
        return Err(SimulationError::InvalidInput(
            "Limit trade price is unreachable (at or better than current spot price)".to_string(),
            None,
        ));
    }

    // Calculate amount_in using the derived formula:
    // amount_in = (P_num * reserve_out * fee_num - P_den * reserve_in * fee_prec) / (P_den *
    // fee_num) We already have limit_price_cross_mult = P_num * reserve_out * fee_num
    //            and current_price_cross_mult = P_den * reserve_in * fee_prec
    let numerator = limit_price_cross_mult - current_price_cross_mult;

    let denominator = U512::from(swap_price_den)
        .checked_mul(U512::from(fee.numerator))
        .ok_or_else(|| SimulationError::FatalError("Overflow in denominator".to_string()))?;

    let amount_in_u512 = numerator / denominator;

    // Check for U256 overflow
    let limbs = amount_in_u512.as_limbs();
    if limbs[4] != 0 || limbs[5] != 0 || limbs[6] != 0 || limbs[7] != 0 {
        return Err(SimulationError::FatalError("Calculated amount_in overflows U256".to_string()));
    }
    let amount_in = U256::from_limbs([limbs[0], limbs[1], limbs[2], limbs[3]]);

    if amount_in == U256::ZERO {
        return Ok((BigUint::ZERO, BigUint::ZERO));
    }

    // Calculate implied_amount_out such that amount_in / amount_out = P
    // amount_out = amount_in / P = amount_in * P_den / P_num
    let implied_amount_out = mul_div(amount_in, swap_price_den, swap_price_num)?;

    Ok((u256_to_biguint(amount_in), u256_to_biguint(implied_amount_out)))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use num_bigint::BigUint;
    use tycho_common::{
        hex_bytes::Bytes,
        models::{token::Token, Chain},
        simulation::protocol_sim::Price,
    };

    use super::*;

    fn token_0() -> Token {
        Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap(),
            "T0",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn token_1() -> Token {
        Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap(),
            "T1",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    #[test]
    fn test_swap_to_price_verifies_spot_price() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        let target_price = Price::new(BigUint::from(2u32), BigUint::from(5u32));

        let (amount_in, _implied_amount_out) =
            cpmm_swap_to_price(reserve0, reserve1, &target_price, fee).unwrap();

        let amount_in_u256 = biguint_to_u256(&amount_in);

        let fee_for_get_amount_out = ProtocolFee::new(fee_numerator, fee_precision);
        let actual_amount_out =
            cpmm_get_amount_out(amount_in_u256, reserve0, reserve1, fee_for_get_amount_out)
                .unwrap();

        let new_reserve0 = safe_add_u256(reserve0, amount_in_u256).unwrap();
        let new_reserve1 = safe_sub_u256(reserve1, actual_amount_out).unwrap();

        let base = token_0();
        let quote = token_1();
        let new_spot_price = cpmm_spot_price(&base, &quote, new_reserve0, new_reserve1).unwrap();

        let target_price_f64 = target_price
            .numerator
            .to_string()
            .parse::<f64>()
            .unwrap() /
            target_price
                .denominator
                .to_string()
                .parse::<f64>()
                .unwrap();

        let relative_diff = (new_spot_price - target_price_f64).abs() / target_price_f64;
        assert!(
            relative_diff < 0.01,
            "New spot price {new_spot_price} should be close to target price {target_price_f64}, \
             relative diff: {relative_diff}"
        );
    }

    #[test]
    fn test_swap_to_price_unreachable() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        // Target price of 3.0 means swap_price = 1/3 ≈ 0.33
        // Current spot price (with fees) ≈ 2.006 (reserve_in * fee_precision / (reserve_out *
        // fee_numerator)) Since swap_price < current spot price, this target is unreachable
        let target_price = Price::new(BigUint::from(3u32), BigUint::from(1u32));

        let result = cpmm_swap_to_price(reserve0, reserve1, &target_price, fee);

        assert!(
            matches!(result, Err(SimulationError::InvalidInput(ref msg, _)) if msg.contains("unreachable")),
            "Expected InvalidInput error about unreachable price, got: {:?}",
            result
        );
    }

    #[test]
    fn test_swap_to_price_at_spot() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        // Current spot price (with fees) = reserve_in * fee_precision / (reserve_out *
        // fee_numerator)                 = 2_000_000 * 10000 / (1_000_000 * 9970)
        //                                = 2000/997 ≈ 2.006
        // Setting target_price = 997/2000 means swap_price = 2000/997 = current spot
        let target_price = Price::new(BigUint::from(997u32), BigUint::from(2000u32));

        let (amount_in, implied_amount_out) =
            cpmm_swap_to_price(reserve0, reserve1, &target_price, fee).unwrap();

        // When target price equals current spot price, no swap is needed
        assert!(
            amount_in.is_zero(),
            "amount_in should be zero when target is at spot price, got: {}",
            amount_in
        );
        assert!(
            implied_amount_out.is_zero(),
            "implied_amount_out should be zero when target is at spot price, got: {}",
            implied_amount_out
        );
    }

    #[test]
    fn test_swap_to_trade_price_verifies_trade_price() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        // Target trade price: 2.5 amount_in per amount_out (in Price convention: 1/2.5 = 0.4)
        // This is worse than current spot price (≈0.5), so it's achievable
        let target_price = Price::new(BigUint::from(2u32), BigUint::from(5u32));

        let (amount_in, implied_amount_out) =
            cpmm_swap_to_trade_price(reserve0, reserve1, &target_price, fee).unwrap();

        let amount_in_u256 = biguint_to_u256(&amount_in);

        // Execute actual swap to get real amount_out
        let fee_for_get_amount_out = ProtocolFee::new(fee_numerator, fee_precision);
        let actual_amount_out =
            cpmm_get_amount_out(amount_in_u256, reserve0, reserve1, fee_for_get_amount_out)
                .unwrap();

        // Compute actual trade price: amount_in / amount_out (in P convention)
        // In Q convention (amount_out / amount_in), this is what we compare to target_price
        let amount_in_f64 = amount_in
            .to_string()
            .parse::<f64>()
            .unwrap();
        let actual_amount_out_f64 = actual_amount_out
            .to_string()
            .parse::<f64>()
            .unwrap();
        let actual_trade_price_q = actual_amount_out_f64 / amount_in_f64;

        let target_price_f64 = target_price
            .numerator
            .to_string()
            .parse::<f64>()
            .unwrap() /
            target_price
                .denominator
                .to_string()
                .parse::<f64>()
                .unwrap();

        let relative_diff = (actual_trade_price_q - target_price_f64).abs() / target_price_f64;
        assert!(
            relative_diff < 0.001,
            "Actual trade price {actual_trade_price_q} should be close to target price \
             {target_price_f64}, relative diff: {relative_diff}"
        );

        // Verify implied_amount_out is close to actual
        let implied_amount_out_f64 = implied_amount_out
            .to_string()
            .parse::<f64>()
            .unwrap();
        let amount_out_diff =
            (implied_amount_out_f64 - actual_amount_out_f64).abs() / actual_amount_out_f64;
        assert!(
            amount_out_diff < 0.001,
            "Implied amount out {implied_amount_out_f64} should be close to actual \
             {actual_amount_out_f64}, relative diff: {amount_out_diff}"
        );
    }

    #[test]
    fn test_swap_to_trade_price_unreachable() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        // Target trade price: 0.6 amount_out per amount_in (better than spot ≈0.5)
        // This is unreachable - you can't get a better price than spot
        let target_price = Price::new(BigUint::from(3u32), BigUint::from(5u32));

        let result = cpmm_swap_to_trade_price(reserve0, reserve1, &target_price, fee);

        assert!(result.is_err(), "Should return error when target trade price is better than spot");
    }

    #[test]
    fn test_swap_to_trade_price_at_spot() {
        let reserve0 = U256::from(2_000_000u64);
        let reserve1 = U256::from(1_000_000u64);

        let fee_bps: u32 = 30;
        let fee_numerator = U256::from(10000 - fee_bps);
        let fee_precision = U256::from(10000u32);
        let fee = ProtocolFee::new(fee_numerator, fee_precision);

        // Target exactly at spot price (with fees):
        // Spot price in Q convention = reserve_out * fee_num / (reserve_in * fee_prec)
        //                            = 1_000_000 * 9970 / (2_000_000 * 10000)
        let spot_price_num = U256::from(1_000_000u64) * fee_numerator;
        let spot_price_den = U256::from(2_000_000u64) * fee_precision;

        let target_price =
            Price::new(u256_to_biguint(spot_price_num), u256_to_biguint(spot_price_den));

        let result = cpmm_swap_to_trade_price(reserve0, reserve1, &target_price, fee);

        // At exactly spot price, we can't achieve that trade price (need to be strictly worse)
        assert!(result.is_err(), "Should return error when target trade price equals spot price");
    }
}
