use std::{any::Any, collections::HashMap};

use alloy::primitives::{Sign, I256, U256};
use num_bigint::BigUint;
use num_traits::Zero;
use tracing::trace;
use tycho_common::{
    dto::ProtocolStateDelta,
    models::token::Token,
    simulation::{
        errors::{SimulationError, TransitionError},
        protocol_sim::{Balances, GetAmountOutResult, ProtocolSim},
    },
    Bytes,
};

use super::enums::FeeAmount;
use crate::evm::protocol::{
    safe_math::{safe_add_u256, safe_sub_u256},
    u256_num::u256_to_biguint,
    utils::uniswap::{
        i24_be_bytes_to_i32, liquidity_math,
        sqrt_price_math::{get_amount0_delta, get_amount1_delta, sqrt_price_q96_to_f64},
        swap_math,
        tick_list::{TickInfo, TickList, TickListErrorKind},
        tick_math::{
            get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio, MAX_SQRT_RATIO, MAX_TICK,
            MIN_SQRT_RATIO, MIN_TICK,
        },
        StepComputation, SwapResults, SwapState,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniswapV3State {
    liquidity: u128,
    sqrt_price: U256,
    fee: FeeAmount,
    tick: i32,
    ticks: TickList,
}

impl UniswapV3State {
    /// Creates a new instance of `UniswapV3State`.
    ///
    /// # Arguments
    /// - `liquidity`: The initial liquidity of the pool.
    /// - `sqrt_price`: The square root of the current price.
    /// - `fee`: The fee tier for the pool.
    /// - `tick`: The current tick of the pool.
    /// - `ticks`: A vector of `TickInfo` representing the tick information for the pool.
    pub fn new(
        liquidity: u128,
        sqrt_price: U256,
        fee: FeeAmount,
        tick: i32,
        ticks: Vec<TickInfo>,
    ) -> Self {
        let spacing = UniswapV3State::get_spacing(fee);
        let tick_list = TickList::from(spacing, ticks);
        UniswapV3State { liquidity, sqrt_price, fee, tick, ticks: tick_list }
    }

    fn get_spacing(fee: FeeAmount) -> u16 {
        match fee {
            FeeAmount::Lowest => 1,
            FeeAmount::Lowest2 => 2,
            FeeAmount::Lowest3 => 3,
            FeeAmount::Lowest4 => 4,
            FeeAmount::Low => 10,
            FeeAmount::MediumLow => 50,
            FeeAmount::Medium => 60,
            FeeAmount::MediumHigh => 100,
            FeeAmount::High => 200,
        }
    }

    fn swap(
        &self,
        zero_for_one: bool,
        amount_specified: I256,
        sqrt_price_limit: Option<U256>,
    ) -> Result<SwapResults, SimulationError> {
        if self.liquidity == 0 {
            return Err(SimulationError::RecoverableError("No liquidity".to_string()));
        }
        let price_limit = if let Some(limit) = sqrt_price_limit {
            limit
        } else if zero_for_one {
            safe_add_u256(MIN_SQRT_RATIO, U256::from(1u64))?
        } else {
            safe_sub_u256(MAX_SQRT_RATIO, U256::from(1u64))?
        };

        if zero_for_one {
            assert!(price_limit > MIN_SQRT_RATIO);
            assert!(price_limit < self.sqrt_price);
        } else {
            assert!(price_limit < MAX_SQRT_RATIO);
            assert!(price_limit > self.sqrt_price);
        }

        let exact_input = amount_specified > I256::from_raw(U256::from(0u64));

        let mut state = SwapState {
            amount_remaining: amount_specified,
            amount_calculated: I256::from_raw(U256::from(0u64)),
            sqrt_price: self.sqrt_price,
            tick: self.tick,
            liquidity: self.liquidity,
        };
        let mut gas_used = U256::from(130_000);

        while state.amount_remaining != I256::from_raw(U256::from(0u64)) &&
            state.sqrt_price != price_limit
        {
            let (mut next_tick, initialized) = match self
                .ticks
                .next_initialized_tick_within_one_word(state.tick, zero_for_one)
            {
                Ok((tick, init)) => (tick, init),
                Err(tick_err) => match tick_err.kind {
                    TickListErrorKind::TicksExeeded => {
                        let mut new_state = self.clone();
                        new_state.liquidity = state.liquidity;
                        new_state.tick = state.tick;
                        new_state.sqrt_price = state.sqrt_price;
                        return Err(SimulationError::InvalidInput(
                            "Ticks exceeded".into(),
                            Some(GetAmountOutResult::new(
                                u256_to_biguint(state.amount_calculated.abs().into_raw()),
                                u256_to_biguint(gas_used),
                                Box::new(new_state),
                            )),
                        ));
                    }
                    _ => return Err(SimulationError::FatalError("Unknown error".to_string())),
                },
            };

            next_tick = next_tick.clamp(MIN_TICK, MAX_TICK);

            let sqrt_price_next = get_sqrt_ratio_at_tick(next_tick)?;
            let (sqrt_price, amount_in, amount_out, fee_amount) = swap_math::compute_swap_step(
                state.sqrt_price,
                UniswapV3State::get_sqrt_ratio_target(sqrt_price_next, price_limit, zero_for_one),
                state.liquidity,
                state.amount_remaining,
                self.fee as u32,
            )?;
            state.sqrt_price = sqrt_price;

            let step = StepComputation {
                sqrt_price_start: state.sqrt_price,
                tick_next: next_tick,
                initialized,
                sqrt_price_next,
                amount_in,
                amount_out,
                fee_amount,
            };
            if exact_input {
                state.amount_remaining -= I256::checked_from_sign_and_abs(
                    Sign::Positive,
                    safe_add_u256(step.amount_in, step.fee_amount)?,
                )
                .unwrap();
                state.amount_calculated -=
                    I256::checked_from_sign_and_abs(Sign::Positive, step.amount_out).unwrap();
            } else {
                state.amount_remaining +=
                    I256::checked_from_sign_and_abs(Sign::Positive, step.amount_out).unwrap();
                state.amount_calculated += I256::checked_from_sign_and_abs(
                    Sign::Positive,
                    safe_add_u256(step.amount_in, step.fee_amount)?,
                )
                .unwrap();
            }
            if state.sqrt_price == step.sqrt_price_next {
                if step.initialized {
                    let liquidity_raw = self
                        .ticks
                        .get_tick(step.tick_next)
                        .unwrap()
                        .net_liquidity;
                    let liquidity_net = if zero_for_one { -liquidity_raw } else { liquidity_raw };
                    state.liquidity =
                        liquidity_math::add_liquidity_delta(state.liquidity, liquidity_net)?;
                }
                state.tick = if zero_for_one { step.tick_next - 1 } else { step.tick_next };
            } else if state.sqrt_price != step.sqrt_price_start {
                state.tick = get_tick_at_sqrt_ratio(state.sqrt_price)?;
            }
            gas_used = safe_add_u256(gas_used, U256::from(2000))?;
        }
        Ok(SwapResults {
            amount_calculated: state.amount_calculated,
            sqrt_price: state.sqrt_price,
            liquidity: state.liquidity,
            tick: state.tick,
            gas_used,
        })
    }

    fn get_sqrt_ratio_target(
        sqrt_price_next: U256,
        sqrt_price_limit: U256,
        zero_for_one: bool,
    ) -> U256 {
        let cond1 = if zero_for_one {
            sqrt_price_next < sqrt_price_limit
        } else {
            sqrt_price_next > sqrt_price_limit
        };

        if cond1 {
            sqrt_price_limit
        } else {
            sqrt_price_next
        }
    }
}

impl ProtocolSim for UniswapV3State {
    fn fee(&self) -> f64 {
        (self.fee as u32) as f64 / 1_000_000.0
    }

    fn spot_price(&self, a: &Token, b: &Token) -> Result<f64, SimulationError> {
        if a < b {
            Ok(sqrt_price_q96_to_f64(self.sqrt_price, a.decimals, b.decimals))
        } else {
            Ok(1.0f64 / sqrt_price_q96_to_f64(self.sqrt_price, b.decimals, a.decimals))
        }
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_a: &Token,
        token_b: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        let zero_for_one = token_a < token_b;
        let amount_specified = I256::checked_from_sign_and_abs(
            Sign::Positive,
            U256::from_be_slice(&amount_in.to_bytes_be()),
        )
        .ok_or_else(|| {
            SimulationError::InvalidInput("I256 overflow: amount_in".to_string(), None)
        })?;

        let result = self.swap(zero_for_one, amount_specified, None)?;

        trace!(?amount_in, ?token_a, ?token_b, ?zero_for_one, ?result, "V3 SWAP");
        let mut new_state = self.clone();
        new_state.liquidity = result.liquidity;
        new_state.tick = result.tick;
        new_state.sqrt_price = result.sqrt_price;

        Ok(GetAmountOutResult::new(
            u256_to_biguint(
                result
                    .amount_calculated
                    .abs()
                    .into_raw(),
            ),
            u256_to_biguint(result.gas_used),
            Box::new(new_state),
        ))
    }

    fn get_limits(
        &self,
        token_in: Bytes,
        token_out: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        // If the pool has no liquidity, return zeros for both limits
        if self.liquidity == 0 {
            return Ok((BigUint::zero(), BigUint::zero()));
        }

        let zero_for_one = token_in < token_out;
        let mut current_tick = self.tick;
        let mut current_sqrt_price = self.sqrt_price;
        let mut current_liquidity = self.liquidity;
        let mut total_amount_in = U256::from(0u64);
        let mut total_amount_out = U256::from(0u64);

        // Iterate through all ticks in the direction of the swap
        // Continues until there is no more liquidity in the pool or no more ticks to process
        while let Ok((tick, initialized)) = self
            .ticks
            .next_initialized_tick_within_one_word(current_tick, zero_for_one)
        {
            // Clamp the tick value to ensure it's within valid range
            let next_tick = tick.clamp(MIN_TICK, MAX_TICK);

            // Calculate the sqrt price at the next tick boundary
            let sqrt_price_next = get_sqrt_ratio_at_tick(next_tick)?;

            // Calculate the amount of tokens swapped when moving from current_sqrt_price to
            // sqrt_price_next. Direction determines which token is being swapped in vs out
            let (amount_in, amount_out) = if zero_for_one {
                let amount0 = get_amount0_delta(
                    sqrt_price_next,
                    current_sqrt_price,
                    current_liquidity,
                    true,
                )?;
                let amount1 = get_amount1_delta(
                    sqrt_price_next,
                    current_sqrt_price,
                    current_liquidity,
                    false,
                )?;
                (amount0, amount1)
            } else {
                let amount0 = get_amount0_delta(
                    sqrt_price_next,
                    current_sqrt_price,
                    current_liquidity,
                    false,
                )?;
                let amount1 = get_amount1_delta(
                    sqrt_price_next,
                    current_sqrt_price,
                    current_liquidity,
                    true,
                )?;
                (amount1, amount0)
            };

            // Accumulate total amounts for this tick range
            total_amount_in = safe_add_u256(total_amount_in, amount_in)?;
            total_amount_out = safe_add_u256(total_amount_out, amount_out)?;

            // If this tick is "initialized" (meaning its someone's position boundary), update the
            // liquidity when crossing it
            // For zero_for_one, liquidity is removed when crossing a tick
            // For one_for_zero, liquidity is added when crossing a tick
            if initialized {
                let liquidity_raw = self
                    .ticks
                    .get_tick(next_tick)
                    .unwrap()
                    .net_liquidity;
                let liquidity_delta = if zero_for_one { -liquidity_raw } else { liquidity_raw };
                current_liquidity =
                    liquidity_math::add_liquidity_delta(current_liquidity, liquidity_delta)?;
            }

            // Move to the next tick position
            current_tick = if zero_for_one { next_tick - 1 } else { next_tick };
            current_sqrt_price = sqrt_price_next;
        }

        Ok((u256_to_biguint(total_amount_in), u256_to_biguint(total_amount_out)))
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        _tokens: &HashMap<Bytes, Token>,
        _balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        // apply attribute changes
        if let Some(liquidity) = delta
            .updated_attributes
            .get("liquidity")
        {
            // This is a hotfix because if the liquidity has never been updated after creation, it's
            // currently encoded as H256::zero(), therefore, we can't decode this as u128.
            // We can remove this once it has been fixed on the tycho side.
            let liq_16_bytes = if liquidity.len() == 32 {
                // Make sure it only happens for 0 values, otherwise error.
                if liquidity == &Bytes::zero(32) {
                    Bytes::from([0; 16])
                } else {
                    return Err(TransitionError::DecodeError(format!(
                        "Liquidity bytes too long for {liquidity}, expected 16",
                    )));
                }
            } else {
                liquidity.clone()
            };

            self.liquidity = u128::from(liq_16_bytes);
        }
        if let Some(sqrt_price) = delta
            .updated_attributes
            .get("sqrt_price_x96")
        {
            self.sqrt_price = U256::from_be_slice(sqrt_price);
        }
        if let Some(tick) = delta.updated_attributes.get("tick") {
            // This is a hotfix because if the tick has never been updated after creation, it's
            // currently encoded as H256::zero(), therefore, we can't decode this as i32.
            // We can remove this once it has been fixed on the tycho side.
            let ticks_4_bytes = if tick.len() == 32 {
                // Make sure it only happens for 0 values, otherwise error.
                if tick == &Bytes::zero(32) {
                    Bytes::from([0; 4])
                } else {
                    return Err(TransitionError::DecodeError(format!(
                        "Tick bytes too long for {tick}, expected 4"
                    )));
                }
            } else {
                tick.clone()
            };
            self.tick = i24_be_bytes_to_i32(&ticks_4_bytes);
        }

        // apply tick changes
        for (key, value) in delta.updated_attributes.iter() {
            // tick liquidity keys are in the format "tick/{tick_index}/net_liquidity"
            if key.starts_with("ticks/") {
                let parts: Vec<&str> = key.split('/').collect();
                self.ticks.set_tick_liquidity(
                    parts[1]
                        .parse::<i32>()
                        .map_err(|err| TransitionError::DecodeError(err.to_string()))?,
                    i128::from(value.clone()),
                )
            }
        }
        // delete ticks - ignores deletes for attributes other than tick liquidity
        for key in delta.deleted_attributes.iter() {
            // tick liquidity keys are in the format "tick/{tick_index}/net_liquidity"
            if key.starts_with("tick/") {
                let parts: Vec<&str> = key.split('/').collect();
                self.ticks.set_tick_liquidity(
                    parts[1]
                        .parse::<i32>()
                        .map_err(|err| TransitionError::DecodeError(err.to_string()))?,
                    0,
                )
            }
        }
        Ok(())
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
        if let Some(other_state) = other
            .as_any()
            .downcast_ref::<UniswapV3State>()
        {
            self.liquidity == other_state.liquidity &&
                self.sqrt_price == other_state.sqrt_price &&
                self.fee == other_state.fee &&
                self.tick == other_state.tick &&
                self.ticks == other_state.ticks
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        fs,
        path::Path,
        str::FromStr,
    };

    use num_bigint::ToBigUint;
    use num_traits::FromPrimitive;
    use serde_json::Value;
    use tycho_client::feed::synchronizer::ComponentWithState;
    use tycho_common::{hex_bytes::Bytes, models::Chain};

    use super::*;
    use crate::protocol::models::{DecoderContext, TryFromWithBlock};

    #[test]
    fn test_get_amount_out_full_range_liquidity() {
        let token_x = Token::new(
            &Bytes::from_str("0x6b175474e89094c44da98b954eedeac495271d0f").unwrap(),
            "X",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let token_y = Token::new(
            &Bytes::from_str("0xf1ca9cb74685755965c7458528a36934df52a3ef").unwrap(),
            "Y",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );

        let pool = UniswapV3State::new(
            8330443394424070888454257,
            U256::from_str("188562464004052255423565206602").unwrap(),
            FeeAmount::Medium,
            17342,
            vec![TickInfo::new(0, 0), TickInfo::new(46080, 0)],
        );
        let sell_amount = BigUint::from_str("11_000_000000000000000000").unwrap();
        let expected = BigUint::from_str("61927070842678722935941").unwrap();

        let res = pool
            .get_amount_out(sell_amount, &token_x, &token_y)
            .unwrap();

        assert_eq!(res.amount, expected);
    }

    struct SwapTestCase {
        symbol: &'static str,
        sell: BigUint,
        exp: BigUint,
    }

    #[test]
    fn test_get_amount_out() {
        let wbtc = Token::new(
            &Bytes::from_str("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599").unwrap(),
            "WBTC",
            8,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let weth = Token::new(
            &Bytes::from_str("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2").unwrap(),
            "WETH",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let pool = UniswapV3State::new(
            377952820878029838,
            U256::from_str("28437325270877025820973479874632004").unwrap(),
            FeeAmount::Low,
            255830,
            vec![
                TickInfo::new(255760, 1759015528199933i128),
                TickInfo::new(255770, 6393138051835308i128),
                TickInfo::new(255780, 228206673808681i128),
                TickInfo::new(255820, 1319490609195820i128),
                TickInfo::new(255830, 678916926147901i128),
                TickInfo::new(255840, 12208947683433103i128),
                TickInfo::new(255850, 1177970713095301i128),
                TickInfo::new(255860, 8752304680520407i128),
                TickInfo::new(255880, 1486478248067104i128),
                TickInfo::new(255890, 1878744276123248i128),
                TickInfo::new(255900, 77340284046725227i128),
            ],
        );
        let cases = vec![
            SwapTestCase {
                symbol: "WBTC",
                sell: 500000000.to_biguint().unwrap(),
                exp: BigUint::from_str("64352395915550406461").unwrap(),
            },
            SwapTestCase {
                symbol: "WBTC",
                sell: 550000000.to_biguint().unwrap(),
                exp: BigUint::from_str("70784271504035662865").unwrap(),
            },
            SwapTestCase {
                symbol: "WBTC",
                sell: 600000000.to_biguint().unwrap(),
                exp: BigUint::from_str("77215534856185613494").unwrap(),
            },
            SwapTestCase {
                symbol: "WBTC",
                sell: BigUint::from_str("1000000000").unwrap(),
                exp: BigUint::from_str("128643569649663616249").unwrap(),
            },
            SwapTestCase {
                symbol: "WBTC",
                sell: BigUint::from_str("3000000000").unwrap(),
                exp: BigUint::from_str("385196519076234662939").unwrap(),
            },
            SwapTestCase {
                symbol: "WETH",
                sell: BigUint::from_str("64000000000000000000").unwrap(),
                exp: BigUint::from_str("496294784").unwrap(),
            },
            SwapTestCase {
                symbol: "WETH",
                sell: BigUint::from_str("70000000000000000000").unwrap(),
                exp: BigUint::from_str("542798479").unwrap(),
            },
            SwapTestCase {
                symbol: "WETH",
                sell: BigUint::from_str("77000000000000000000").unwrap(),
                exp: BigUint::from_str("597047757").unwrap(),
            },
            SwapTestCase {
                symbol: "WETH",
                sell: BigUint::from_str("128000000000000000000").unwrap(),
                exp: BigUint::from_str("992129037").unwrap(),
            },
            SwapTestCase {
                symbol: "WETH",
                sell: BigUint::from_str("385000000000000000000").unwrap(),
                exp: BigUint::from_str("2978713582").unwrap(),
            },
        ];

        for case in cases {
            let (token_a, token_b) =
                if case.symbol == "WBTC" { (&wbtc, &weth) } else { (&weth, &wbtc) };
            let res = pool
                .get_amount_out(case.sell, token_a, token_b)
                .unwrap();

            assert_eq!(res.amount, case.exp);
        }
    }

    #[test]
    fn test_err_with_partial_trade() {
        let dai = Token::new(
            &Bytes::from_str("0x6b175474e89094c44da98b954eedeac495271d0f").unwrap(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let usdc = Token::new(
            &Bytes::from_str("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap(),
            "USDC",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let pool = UniswapV3State::new(
            73015811375239994,
            U256::from_str("148273042406850898575413").unwrap(),
            FeeAmount::High,
            -263789,
            vec![
                TickInfo::new(-269600, 3612326326695492i128),
                TickInfo::new(-268800, 1487613939516867i128),
                TickInfo::new(-267800, 1557587121322546i128),
                TickInfo::new(-267400, 424592076717375i128),
                TickInfo::new(-267200, 11691597431643916i128),
                TickInfo::new(-266800, -218742815100986i128),
                TickInfo::new(-266600, 1118947532495477i128),
                TickInfo::new(-266200, 1233064286622365i128),
                TickInfo::new(-265000, 4252603063356107i128),
                TickInfo::new(-263200, -351282010325232i128),
                TickInfo::new(-262800, -2352011819117842i128),
                TickInfo::new(-262600, -424592076717375i128),
                TickInfo::new(-262200, -11923662433672566i128),
                TickInfo::new(-261600, -2432911749667741i128),
                TickInfo::new(-260200, -4032727022572273i128),
                TickInfo::new(-260000, -22889492064625028i128),
                TickInfo::new(-259400, -1557587121322546i128),
                TickInfo::new(-259200, -1487613939516867i128),
                TickInfo::new(-258400, -400137022888262i128),
            ],
        );
        let amount_in = BigUint::from_str("50000000000").unwrap();
        let exp = BigUint::from_str("6820591625999718100883").unwrap();

        let err = pool
            .get_amount_out(amount_in, &usdc, &dai)
            .unwrap_err();

        match err {
            SimulationError::InvalidInput(ref _err, ref amount_out_result) => {
                match amount_out_result {
                    Some(amount_out_result) => {
                        assert_eq!(amount_out_result.amount, exp);
                        let new_state = amount_out_result
                            .new_state
                            .as_any()
                            .downcast_ref::<UniswapV3State>()
                            .unwrap();
                        assert_ne!(new_state.tick, pool.tick);
                        assert_ne!(new_state.liquidity, pool.liquidity);
                    }
                    _ => panic!("Partial amount out result is None. Expected partial result."),
                }
            }
            _ => panic!("Test failed: was expecting a SimulationError::InsufficientData"),
        }
    }

    #[test]
    fn test_delta_transition() {
        let mut pool = UniswapV3State::new(
            1000,
            U256::from_str("1000").unwrap(),
            FeeAmount::Low,
            100,
            vec![TickInfo::new(255760, 10000), TickInfo::new(255900, -10000)],
        );
        let attributes: HashMap<String, Bytes> = [
            ("liquidity".to_string(), Bytes::from(2000_u64.to_be_bytes().to_vec())),
            ("sqrt_price_x96".to_string(), Bytes::from(1001_u64.to_be_bytes().to_vec())),
            ("tick".to_string(), Bytes::from(120_i32.to_be_bytes().to_vec())),
            (
                "ticks/-255760/net_liquidity".to_string(),
                Bytes::from(10200_u64.to_be_bytes().to_vec()),
            ),
            (
                "ticks/255900/net_liquidity".to_string(),
                Bytes::from(9800_u64.to_be_bytes().to_vec()),
            ),
        ]
        .into_iter()
        .collect();
        let delta = ProtocolStateDelta {
            component_id: "State1".to_owned(),
            updated_attributes: attributes,
            deleted_attributes: HashSet::new(),
        };

        pool.delta_transition(delta, &HashMap::new(), &Balances::default())
            .unwrap();

        assert_eq!(pool.liquidity, 2000);
        assert_eq!(pool.sqrt_price, U256::from(1001));
        assert_eq!(pool.tick, 120);
        assert_eq!(
            pool.ticks
                .get_tick(-255760)
                .unwrap()
                .net_liquidity,
            10200
        );
        assert_eq!(
            pool.ticks
                .get_tick(255900)
                .unwrap()
                .net_liquidity,
            9800
        );
    }

    #[test]
    fn test_get_limits_usdc_weth_high_values() {
        // Reproduce the issue where get_limits returns astronomically high values
        // for the USDC-WETH pool 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
        // Observed values: max input 425778233505282382766106886059816494 USDC (6 decimals)
        //                  max output 3530436799538040685700 WETH (18 decimals)
        // Expected: Reasonable values based on actual pool liquidity

        // Real on-chain data from the USDC-WETH pool
        let liquidity = 2003687413977649104u128;
        let current_sqrt_price = U256::from_str("1230166506324753154737293410227554").unwrap();
        let current_tick = 193016;

        // Simplified tick list - using key ticks that span a wide range
        // FeeAmount::Low has tick spacing of 10
        let ticks = vec![
            TickInfo::new(-887270, 23083107119390899),
            TickInfo::new(-92110, 398290794261),
            TickInfo::new(-76280, 29651014881301328537),
            TickInfo::new(-75980, -29651014881301328537),
            TickInfo::new(0, 13570724854034),
            TickInfo::new(100, 100),
            TickInfo::new(110, -100),
            TickInfo::new(100000, -13570724854034),
            TickInfo::new(108340, 2266789968),
            TickInfo::new(108360, 2273693713),
            TickInfo::new(108390, 44739669244),
            TickInfo::new(115120, 144810683332502),
            TickInfo::new(115140, 24308826080),
            TickInfo::new(138160, 2344264358099),
            TickInfo::new(149150, 9403707956281),
            TickInfo::new(161190, 574355480),
            TickInfo::new(166300, 607207624019674),
            TickInfo::new(175170, 494806614022118),
            TickInfo::new(177250, 1142415145764782),
            TickInfo::new(177280, 2588778177052193),
            TickInfo::new(178340, 68579783371570),
            TickInfo::new(179480, 229060865165778),
            TickInfo::new(180160, 542209191207296),
            TickInfo::new(180180, 419968624358816),
            TickInfo::new(181260, 88680311208),
            TickInfo::new(181590, 105220981852142),
            TickInfo::new(182390, 3451150785812214),
            TickInfo::new(182560, 4740323423268196),
            TickInfo::new(182770, 1084560033827482),
            TickInfo::new(183080, -4740323423268196),
            TickInfo::new(183160, 1104057347094),
            TickInfo::new(183260, 14920025509875),
            TickInfo::new(183310, 159498928900521),
            TickInfo::new(184200, 17409290878),
            TickInfo::new(184210, 559777428971808),
            TickInfo::new(184220, 9491189899070617),
            TickInfo::new(184230, -3397635186717387),
            TickInfo::new(184940, 1389538197360745),
            TickInfo::new(185050, 477197831652989),
            TickInfo::new(185210, 1049687190863),
            TickInfo::new(185240, 148422519006153),
            TickInfo::new(185260, 2942212203299448),
            TickInfo::new(185270, 404583613408945),
            TickInfo::new(185540, 4283793404312660),
            TickInfo::new(185840, 297728093),
            TickInfo::new(186060, 16737492612081),
            TickInfo::new(186070, 86119889539),
            TickInfo::new(186320, 8039254639562),
            TickInfo::new(186420, 228699723058089),
            TickInfo::new(186430, 261844689822052),
            TickInfo::new(186440, 4182138301843280),
            TickInfo::new(186450, 5750887401156881),
            TickInfo::new(186460, 416983925278171),
            TickInfo::new(186940, 238266340863267),
            TickInfo::new(187090, 476499315256829),
            TickInfo::new(187220, 16895160683208),
            TickInfo::new(187350, 17643911357026),
            TickInfo::new(187360, 2533050010148058),
            TickInfo::new(187440, 11661475258274),
            TickInfo::new(187780, 32374077410530097),
            TickInfo::new(187940, 54743213803487),
            TickInfo::new(188070, 926478815741581),
            TickInfo::new(188200, 701176288517482),
            TickInfo::new(188380, 1270657089796),
            TickInfo::new(188420, 278014747494314),
            TickInfo::new(188450, 10013802424018),
            TickInfo::new(188460, 12890223829892),
            TickInfo::new(188520, 2711846872564618),
            TickInfo::new(188530, 167770740693836),
            TickInfo::new(188630, 3582410944595),
            TickInfo::new(188780, 224653000969518),
            TickInfo::new(188820, 250039264055),
            TickInfo::new(188920, 28509577839199252),
            TickInfo::new(188950, 200611939639690),
            TickInfo::new(188990, 6391022318941913),
            TickInfo::new(189020, 431254538102),
            TickInfo::new(189040, 54558453571517),
            TickInfo::new(189130, 1491132445622),
            TickInfo::new(189170, 31371938231827),
            TickInfo::new(189200, 1230671402600057),
            TickInfo::new(189210, 107388554969),
            TickInfo::new(189230, 116526800732),
            TickInfo::new(189320, 167487115137896176),
            TickInfo::new(189330, 50730409600022),
            TickInfo::new(189340, 30347006557822),
            TickInfo::new(189360, 1838984005587729),
            TickInfo::new(189370, 139708241399466),
            TickInfo::new(189440, 19400085413161),
            TickInfo::new(189460, 2659554920775),
            TickInfo::new(189480, 79481654850350),
            TickInfo::new(189490, 877667016856307),
            TickInfo::new(189520, 3791294749733727),
            TickInfo::new(189530, 25705399149457),
            TickInfo::new(189600, 251688016150017),
            TickInfo::new(189610, 11621068399680073),
            TickInfo::new(189650, 39184646932425),
            TickInfo::new(189660, 650044387022091),
            TickInfo::new(189700, 487607661387),
            TickInfo::new(189810, 7523717528979396),
            TickInfo::new(189830, 287293254907818),
            TickInfo::new(189840, 745828656031025),
            TickInfo::new(189870, 4582054987788346),
            TickInfo::new(189880, 1034480184182597),
            TickInfo::new(189920, 1212341758108364),
            TickInfo::new(189960, 34750727469721),
            TickInfo::new(189980, 603093375742265),
            TickInfo::new(189990, 941735738993078),
            TickInfo::new(190000, 5365882529126),
            TickInfo::new(190010, 14297995036012488),
            TickInfo::new(190020, 70041547412365),
            TickInfo::new(190050, 11018365830831284),
            TickInfo::new(190100, 42835525318364),
            TickInfo::new(190110, 1930510700526076),
            TickInfo::new(190130, 217639345420681),
            TickInfo::new(190140, 2082544348785232),
            TickInfo::new(190180, 388710541704459),
            TickInfo::new(190190, 21787614493017904),
            TickInfo::new(190200, 2296866628846815),
            TickInfo::new(190220, 1375109716821800),
            TickInfo::new(190280, 94616825643640),
            TickInfo::new(190300, 137729364911365),
            TickInfo::new(190310, 30292588971765),
            TickInfo::new(190320, 1340502249746583),
            TickInfo::new(190350, 82555302645888),
            TickInfo::new(190370, 186968191595275),
            TickInfo::new(190380, 76826013196651613),
            TickInfo::new(190410, 24290767711963731),
            TickInfo::new(190420, 152846646454223),
            TickInfo::new(190470, 3795478697228972),
            TickInfo::new(190480, 549134932915826),
            TickInfo::new(190490, 24364329331668),
            TickInfo::new(190500, 14327762966156),
            TickInfo::new(190540, 696734104851138),
            TickInfo::new(190560, 129185481225073),
            TickInfo::new(190570, 530229278844440),
            TickInfo::new(190580, 30520418856337),
            TickInfo::new(190610, 78877582031074),
            TickInfo::new(190630, 476398527221144),
            TickInfo::new(190650, 13141201568201),
            TickInfo::new(190690, 15660410301265),
            TickInfo::new(190700, 992005100373726),
            TickInfo::new(190710, 1512508835869810),
            TickInfo::new(190720, 71510532495962),
            TickInfo::new(190730, 2360767931033961),
            TickInfo::new(190740, 15999796812446947),
            TickInfo::new(190750, 876153399866934),
            TickInfo::new(190760, 131959895110779155),
            TickInfo::new(190770, 264980598312960),
            TickInfo::new(190780, 1176552317157296),
            TickInfo::new(190800, 224082011199),
            TickInfo::new(190830, 1472809195659503),
            TickInfo::new(190860, 784302512953798),
            TickInfo::new(190880, 446220170233695),
            TickInfo::new(190890, 4174066817331886),
            TickInfo::new(190900, 2233595926396318),
            TickInfo::new(190920, 1561390393945330),
            TickInfo::new(190940, 531413766147870958),
            TickInfo::new(190950, 20565461227504270),
            TickInfo::new(191010, 394322066720857),
            TickInfo::new(191020, 2782053929017767),
            TickInfo::new(191030, 780642218292288),
            TickInfo::new(191040, 702163789719809),
            TickInfo::new(191050, -1240536852594445),
            TickInfo::new(191070, 7463128176735693),
            TickInfo::new(191080, 2593851273572619),
            TickInfo::new(191090, 514747739316929),
            TickInfo::new(191110, 8326101247310152),
            TickInfo::new(191120, 208037332753207),
            TickInfo::new(191130, 27563266076594),
            TickInfo::new(191140, 537327730325874810),
            TickInfo::new(191150, 46163699741856823),
            TickInfo::new(191160, 902982190948710),
            TickInfo::new(191170, 15936898545036553),
            TickInfo::new(191180, 249267944321718),
            TickInfo::new(191200, 1084475207711498),
            TickInfo::new(191210, 15592776725384973),
            TickInfo::new(191220, 955591808173212),
            TickInfo::new(191230, 860643410822165),
            TickInfo::new(191240, 2940861103894532),
            TickInfo::new(191250, 26315259228347568),
            TickInfo::new(191270, 2653562060045981),
            TickInfo::new(191290, 4401650642802915),
            TickInfo::new(191300, 87563876036283),
            TickInfo::new(191310, 203217098680788),
            TickInfo::new(191320, 69247239417235),
            TickInfo::new(191340, -480642253889760270),
            TickInfo::new(191350, 12756913509452478),
            TickInfo::new(191360, 6976395916092448),
            TickInfo::new(191370, 933931176924050),
            TickInfo::new(191380, 1662440493406421),
            TickInfo::new(191390, 6783329557647),
            TickInfo::new(191400, 5039433626820),
            TickInfo::new(191410, 135831735509197163),
            TickInfo::new(191420, 13958512179887),
            TickInfo::new(191440, 243396021313348),
            TickInfo::new(191450, 87229677460617719),
            TickInfo::new(191470, 22269132569362425),
            TickInfo::new(191480, 65890530025971227),
            TickInfo::new(191490, 2181308783461349),
            TickInfo::new(191500, 26696366051068531),
            TickInfo::new(191510, 435685738452201),
            TickInfo::new(191530, -26465443208261779),
            TickInfo::new(191540, 639110252349782884),
            TickInfo::new(191550, -382682096992183696),
            TickInfo::new(191560, 21965819658046884),
            TickInfo::new(191570, 19001530236645),
            TickInfo::new(191580, 111195856952706),
            TickInfo::new(191600, 22994211383098),
            TickInfo::new(191610, 2028300382284212),
            TickInfo::new(191620, 733880314320360),
            TickInfo::new(191630, 278138066838164168),
            TickInfo::new(191640, -637232871567988026),
            TickInfo::new(191650, 47412791913653139),
            TickInfo::new(191660, -7078188227985942),
            TickInfo::new(191680, 28864571466385),
            TickInfo::new(191690, -61828390008018719),
            TickInfo::new(191710, 10260828334177035),
            TickInfo::new(191720, 42682476834069),
            TickInfo::new(191730, 3349621485010294),
            TickInfo::new(191750, 1008881321756356),
            TickInfo::new(191760, -70793889442030581),
            TickInfo::new(191770, 56103632110126489),
            TickInfo::new(191790, 464150240663224),
            TickInfo::new(191820, 101032505423413),
            TickInfo::new(191830, 418733228634139),
            TickInfo::new(191840, -98020715517607015),
            TickInfo::new(191870, -74834676130644879),
            TickInfo::new(191910, 3764156203262613),
            TickInfo::new(191930, 10497484848368857),
            TickInfo::new(191940, -249267944321718),
            TickInfo::new(191950, 3239926654449),
            TickInfo::new(191960, 23761922827311069697),
            TickInfo::new(191970, -23759163828571689215),
            TickInfo::new(191980, 156864026852047121),
            TickInfo::new(192010, -1291672839955340),
            TickInfo::new(192020, 2162651440570489),
            TickInfo::new(192040, 2650121881977623),
            TickInfo::new(192060, 19295032710262794),
            TickInfo::new(192070, 140326580976),
            TickInfo::new(192090, 16133267644139219),
            TickInfo::new(192130, 32619216540607338),
            TickInfo::new(192170, 16645749706690585),
            TickInfo::new(192180, 3186017983195186),
            TickInfo::new(192190, -13921264950966619),
            TickInfo::new(192200, -242515715818309553),
            TickInfo::new(192210, 24722532194444),
            TickInfo::new(192220, 614237638901623),
            TickInfo::new(192230, 20805746830537171),
            TickInfo::new(192240, -601097675364598),
            TickInfo::new(192250, 5985637188346111),
            TickInfo::new(192260, 351706710167148),
            TickInfo::new(192270, 635967191524438),
            TickInfo::new(192280, -617733939550643),
            TickInfo::new(192300, -2142510777022216),
            TickInfo::new(192310, 4414111338631631),
            TickInfo::new(192320, 39505639142374413),
            TickInfo::new(192330, -6924602134902535),
            TickInfo::new(192340, -76015894219710),
            TickInfo::new(192350, -2782188499094454),
            TickInfo::new(192360, -175258803896687),
            TickInfo::new(192370, 360996256936080),
            TickInfo::new(192380, -184066445460675),
            TickInfo::new(192390, -360996256936080),
            TickInfo::new(192400, 128403486033504),
            TickInfo::new(192410, 783407267359097),
            TickInfo::new(192420, -138490174011457390),
            TickInfo::new(192430, 7844068696166480),
            TickInfo::new(192450, 3407473150558),
            TickInfo::new(192470, 1212516163900957),
            TickInfo::new(192490, -962879662115954),
            TickInfo::new(192510, 27553790403817195),
            TickInfo::new(192520, 5908614246791188),
            TickInfo::new(192540, -4999248988051935),
            TickInfo::new(192550, -1671288430145311),
            TickInfo::new(192560, -13323151164218433),
            TickInfo::new(192570, 1101283002618),
            TickInfo::new(192580, 5109098565236964),
            TickInfo::new(192590, 783567992901259),
            TickInfo::new(192600, -3435101371606126),
            TickInfo::new(192630, 3024377869308990),
            TickInfo::new(192650, 7870491187005346),
            TickInfo::new(192660, -10963330829885589),
            TickInfo::new(192670, 3580990712879278),
            TickInfo::new(192680, 35996658956886),
            TickInfo::new(192700, 70820938255846229),
            TickInfo::new(192710, 18760148772244291),
            TickInfo::new(192720, 2918091978167594),
            TickInfo::new(192730, 480768229066840),
            TickInfo::new(192740, -4417449992782603),
            TickInfo::new(192760, -22994211383098),
            TickInfo::new(192770, -2921776836758084),
            TickInfo::new(192790, -151011264004795),
            TickInfo::new(192800, -19345599385182994),
            TickInfo::new(192820, 124851511767452780),
            TickInfo::new(192830, -333586932958048),
            TickInfo::new(192840, -28864571466385),
            TickInfo::new(192850, 8057235718143160),
            TickInfo::new(192860, -715940857113992),
            TickInfo::new(192870, 2810314206998071),
            TickInfo::new(192880, -10508299915250201),
            TickInfo::new(192890, -74527839516035530),
            TickInfo::new(192900, -1374702806752939),
            TickInfo::new(192910, -1340502249746583),
            TickInfo::new(192920, 1187098685043851),
            TickInfo::new(192930, -3249739640038760),
            TickInfo::new(192950, -121836700984848),
            TickInfo::new(192960, 331899325460838697),
            TickInfo::new(192970, -7647208189691461),
            TickInfo::new(193000, -458977744793482),
            TickInfo::new(193010, 123221345665484645),
            TickInfo::new(193020, -124861565069576681),
            TickInfo::new(193030, -955493109212457),
            TickInfo::new(193040, -531353309491851),
            TickInfo::new(193050, -173347964091375),
            TickInfo::new(193070, -2810314206998071),
            TickInfo::new(193080, -334357456632342732),
            TickInfo::new(193090, -561662227789932),
            TickInfo::new(193100, -1148954928033),
            TickInfo::new(193110, -684018547417528),
            TickInfo::new(193120, 28586655333517),
            TickInfo::new(193130, -28486626337519748),
            TickInfo::new(193140, -2940861103894532),
            TickInfo::new(193150, 192538878168260),
            TickInfo::new(193160, -28430039721947526),
            TickInfo::new(193180, -3851687179254859),
            TickInfo::new(193190, -43483961820601228),
            TickInfo::new(193200, -45308297154560),
            TickInfo::new(193210, 4444951682919187),
            TickInfo::new(193220, -1575116142100671),
            TickInfo::new(193230, -7534738344815665),
            TickInfo::new(193250, -2106667057951642),
            TickInfo::new(193260, -7509905445773146),
            TickInfo::new(193280, 16386661756838),
            TickInfo::new(193290, 134408482979845),
            TickInfo::new(193300, -1040649823609683),
            TickInfo::new(193310, -1838982684409764),
            TickInfo::new(193320, -2349468138820706),
            TickInfo::new(193330, -15854259816213781),
            TickInfo::new(193340, 22072894619172356),
            TickInfo::new(193350, 52585024153162),
            TickInfo::new(193360, -8662857362106179),
            TickInfo::new(193370, -486556554948863),
            TickInfo::new(193380, -303749321770597936),
            TickInfo::new(193390, -112267682655545),
            TickInfo::new(193400, -3708819013729436),
            TickInfo::new(193410, 32148749576131557),
            TickInfo::new(193420, -808601172337223),
            TickInfo::new(193440, 3639563600270095),
            TickInfo::new(193450, -203538227431601),
            TickInfo::new(193460, -22032398519496101),
            TickInfo::new(193480, -415235853425725),
            TickInfo::new(193490, -135918389147150),
            TickInfo::new(193510, -6209496757876825),
            TickInfo::new(193520, -562804943193258),
            TickInfo::new(193530, -157444189101167),
            TickInfo::new(193540, -2724577384520569),
            TickInfo::new(193550, -12494970087148875),
            TickInfo::new(193560, -51191301283478),
            TickInfo::new(193570, 317944704921646),
            TickInfo::new(193580, -2247347377981878),
            TickInfo::new(193590, -355561901348697),
            TickInfo::new(193600, -417345483866658),
            TickInfo::new(193610, -94327137207819),
            TickInfo::new(193620, -301081063951601),
            TickInfo::new(193630, -23625998598228848),
            TickInfo::new(193640, -152846646454223),
            TickInfo::new(193650, -25572153063432720),
            TickInfo::new(193660, -369648040727497),
            TickInfo::new(193670, -2380574243163748),
            TickInfo::new(193680, -1347626660410452),
            TickInfo::new(193700, 132349475865655),
            TickInfo::new(193710, -1151680918791608),
            TickInfo::new(193720, -5340871701858954),
            TickInfo::new(193730, -30596114141732),
            TickInfo::new(193740, -119777424499387),
            TickInfo::new(193760, -70968293525310033),
            TickInfo::new(193770, 3343964387309574),
            TickInfo::new(193780, -2662695390803818),
            TickInfo::new(193790, 28858669997289002),
            TickInfo::new(193800, 49809698499372773),
            TickInfo::new(193810, -75677653807476697),
            TickInfo::new(193820, 1042789587533),
            TickInfo::new(193830, -1246250556448807),
            TickInfo::new(193840, -44458864621),
            TickInfo::new(193850, -7099002566557534),
            TickInfo::new(193860, 40480786716227),
            TickInfo::new(193870, -852052109874268),
            TickInfo::new(193880, -783407267359097),
            TickInfo::new(193890, -268782510911571619),
            TickInfo::new(193900, -7810126672257),
            TickInfo::new(193910, -11018365830831284),
            TickInfo::new(193920, 19106053362174780),
            TickInfo::new(193930, 2881358531908961),
            TickInfo::new(193950, -713520553076294),
            TickInfo::new(193960, 12764552050362243),
            TickInfo::new(193970, -4484817067780013),
            TickInfo::new(193980, -7475466172060282),
            TickInfo::new(193990, -250040995586732),
            TickInfo::new(194000, -958157376635214),
            TickInfo::new(194010, 1773397621154237),
            TickInfo::new(194020, -5917977251158594),
            TickInfo::new(194040, -2146307409545697),
            TickInfo::new(194050, 143718443099437871),
            TickInfo::new(194060, 98448842312524),
            TickInfo::new(194070, -460699832995314),
            TickInfo::new(194080, 2263075909514466),
            TickInfo::new(194090, -1092378745891843),
            TickInfo::new(194100, -826069224497083),
            TickInfo::new(194110, -304199831859031),
            TickInfo::new(194120, -207880225603710),
            TickInfo::new(194130, -15260644175364),
            TickInfo::new(194140, -1514198386829832),
            TickInfo::new(194150, -4122835607346620),
            TickInfo::new(194160, -39803836641501622),
            TickInfo::new(194170, 7753345031338662),
            TickInfo::new(194180, 127333777645379),
            TickInfo::new(194190, 1372383597911125),
            TickInfo::new(194200, -4720453657968947),
            TickInfo::new(194210, -35730929942559),
            TickInfo::new(194220, -48772735110768260),
            TickInfo::new(194230, -110146420452018),
            TickInfo::new(194250, 22385491048117),
            TickInfo::new(194260, -34664303109401),
            TickInfo::new(194270, 14692573273309),
            TickInfo::new(194290, 37434213071258),
            TickInfo::new(194300, -2617410482815967),
            TickInfo::new(194330, -2169859963379613),
            TickInfo::new(194340, 2101382342032497),
            TickInfo::new(194350, 3531029817741),
            TickInfo::new(194360, -2369207749645922),
            TickInfo::new(194370, -2004302405748620),
            TickInfo::new(194380, -2209802648926),
            TickInfo::new(194390, -701176288517482),
            TickInfo::new(194410, -4059854252894),
            TickInfo::new(194420, -68336395674612),
            TickInfo::new(194430, -2112488455410514),
            TickInfo::new(194440, -5752531529662246),
            TickInfo::new(194460, 7479063997816858),
            TickInfo::new(194470, -6881699369420418),
            TickInfo::new(194480, -109307854712499),
            TickInfo::new(194500, -351706710167148),
            TickInfo::new(194510, -10037989948886290),
            TickInfo::new(194520, 13700782543363),
            TickInfo::new(194550, -145190552984840610),
            TickInfo::new(194560, -209457311084167),
            TickInfo::new(194570, -3987334919769389),
            TickInfo::new(194580, 613127307514207),
            TickInfo::new(194590, 379925300279897),
            TickInfo::new(194600, 377986830376374),
            TickInfo::new(194610, 5967819685),
            TickInfo::new(194620, 731786621768846),
            TickInfo::new(194630, -1389192788527898),
            TickInfo::new(194640, 928704455057256),
            TickInfo::new(194650, 148075143147539109),
            TickInfo::new(194660, -602117276185157),
            TickInfo::new(194670, 106775541336400),
            TickInfo::new(194680, -144705560496685514),
            TickInfo::new(194690, -2753164146855951),
            TickInfo::new(194700, 1179108338928795),
            TickInfo::new(194710, 5117872918144375),
            TickInfo::new(194720, -547764420492692),
            TickInfo::new(194730, 12299707948666929),
            TickInfo::new(194740, 1139674486633280),
            TickInfo::new(194750, -17006924401402593),
            TickInfo::new(194760, 3454477469766185),
            TickInfo::new(194770, -801211904258285),
            TickInfo::new(194780, -3362617199463505),
            TickInfo::new(194790, 17615278098816611),
            TickInfo::new(194800, -1615963031826208),
            TickInfo::new(194810, 1198884537287),
            TickInfo::new(194820, -5468120091862623),
            TickInfo::new(194830, 195665307069921),
            TickInfo::new(194840, -82504467490575),
            TickInfo::new(194850, -5016460502904761),
            TickInfo::new(194860, 13193550602438429),
            TickInfo::new(194880, 136325190564383),
            TickInfo::new(194890, 777118296410956),
            TickInfo::new(194900, 352955509937721),
            TickInfo::new(194910, -656764981419647),
            TickInfo::new(194920, -483414982336380),
            TickInfo::new(194930, -326355274269041),
            TickInfo::new(194940, 2654537836611793),
            TickInfo::new(194950, 1159451428480215),
            TickInfo::new(194960, -1328439979490704),
            TickInfo::new(194970, 5838435324041791),
            TickInfo::new(194980, 7717531886704583),
            TickInfo::new(194990, -6329348890558669),
            TickInfo::new(195000, -8012877159783295),
            TickInfo::new(195010, 79076664559610),
            TickInfo::new(195020, 55706007896799),
            TickInfo::new(195030, 325694837437904),
            TickInfo::new(195040, -1088761183603421),
            TickInfo::new(195050, -141422004777885),
            TickInfo::new(195060, 538639824090),
            TickInfo::new(195070, -189656584407909),
            TickInfo::new(195080, 2087802447024457),
            TickInfo::new(195090, -1769634197397228),
            TickInfo::new(195100, 405209805857335),
            TickInfo::new(195110, -135154106951644),
            TickInfo::new(195120, -124270538921018),
            TickInfo::new(195130, 37955507766696),
            TickInfo::new(195140, -33216871201408),
            TickInfo::new(195150, -268920263453773),
            TickInfo::new(195170, 58579662710063),
            TickInfo::new(195180, -294790772633901),
            TickInfo::new(195190, -141787631251309),
            TickInfo::new(195200, 164227612092352),
            TickInfo::new(195210, -219252698),
            TickInfo::new(195220, -16791310816433502),
            TickInfo::new(195230, -511582788812774),
            TickInfo::new(195240, 12141064759829),
            TickInfo::new(195250, 350181855246871),
            TickInfo::new(195260, 15982241300548),
            TickInfo::new(195270, -345436228075577),
            TickInfo::new(195300, -32645304218886922),
            TickInfo::new(195310, 2760786175125),
            TickInfo::new(195320, -84076121299767),
            TickInfo::new(195330, 50869624045273),
            TickInfo::new(195340, -7369965552654201),
            TickInfo::new(195350, -22419547258713),
            TickInfo::new(195360, 3501014825541),
            TickInfo::new(195370, -278012037285390),
            TickInfo::new(195380, 104),
            TickInfo::new(195390, -2445316477494430),
            TickInfo::new(195400, 39442333693954),
            TickInfo::new(195420, 34686419853248),
            TickInfo::new(195430, -28404473623409),
            TickInfo::new(195440, -27817040790419),
            TickInfo::new(195450, 98203969114105),
            TickInfo::new(195460, 217393173717114),
            TickInfo::new(195470, -408556704607616),
            TickInfo::new(195480, 5020870184962128),
            TickInfo::new(195490, 218469140122260),
            TickInfo::new(195520, -5264960535011316),
            TickInfo::new(195530, -1037151075004626),
            TickInfo::new(195540, -550442191879715),
            TickInfo::new(195550, 650765631333407),
            TickInfo::new(195560, -17191670679762),
            TickInfo::new(195570, 13421706473467912),
            TickInfo::new(195580, 9672322328918),
            TickInfo::new(195590, -16154833820723031),
            TickInfo::new(195600, -384687958841511),
            TickInfo::new(195610, -3228286235833285),
            TickInfo::new(195620, 42679098236154),
            TickInfo::new(195630, 18613271753294),
            TickInfo::new(195640, 90919265604651),
            TickInfo::new(195650, -69247239417235),
            TickInfo::new(195660, 567060156698400),
            TickInfo::new(195670, 194885666253823),
            TickInfo::new(195680, -1968345557432667),
            TickInfo::new(195690, -724660478811199),
            TickInfo::new(195700, -201739572244771),
            TickInfo::new(195710, -42837753572237),
            TickInfo::new(195720, -532955572982589),
            TickInfo::new(195730, 4677017405753453),
            TickInfo::new(195740, 1090254404594628),
            TickInfo::new(195750, -3772234203760020),
            TickInfo::new(195760, -1601473723243607),
            TickInfo::new(195770, -2601717756104414),
            TickInfo::new(195780, 1607765910693575),
            TickInfo::new(195790, 5269229259151),
            TickInfo::new(195800, -1678279868021937),
            TickInfo::new(195810, -704453670888),
            TickInfo::new(195820, -3502437613585755),
            TickInfo::new(195830, -11220150993306),
            TickInfo::new(195840, -603227802714980),
            TickInfo::new(195850, 604423560711429),
            TickInfo::new(195860, -14272882892337706),
            TickInfo::new(195870, -829679390217598),
            TickInfo::new(195880, 254168326085822),
            TickInfo::new(195890, 14067997442399905),
            TickInfo::new(195900, -246502283733188),
            TickInfo::new(195910, -14055305205711167),
            TickInfo::new(195920, -2860221146779857),
            TickInfo::new(195930, 674720956872992),
            TickInfo::new(195940, -160766814340139),
            TickInfo::new(195960, -656814229710),
            TickInfo::new(195970, -1833125383713494),
            TickInfo::new(195990, -5491570229589),
            TickInfo::new(196000, 462813201860095),
            TickInfo::new(196010, -420244135655232),
            TickInfo::new(196020, -15038253251540),
            TickInfo::new(196030, 11689098596230964),
            TickInfo::new(196040, 2920714493200402),
            TickInfo::new(196050, -48725826558184),
            TickInfo::new(196060, -514782249692785),
            TickInfo::new(196070, -3012574567294640),
            TickInfo::new(196080, 233864248362155),
            TickInfo::new(196090, 27682279651365),
            TickInfo::new(196100, -233864248362155),
            TickInfo::new(196110, -1565522914726323),
            TickInfo::new(196120, -186496977983265),
            TickInfo::new(196140, -860500607508),
            TickInfo::new(196150, 566032523879),
            TickInfo::new(196160, 683156711966037),
            TickInfo::new(196170, -455296452060595),
            TickInfo::new(196180, -4373379595919090),
            TickInfo::new(196190, 1443176916637037),
            TickInfo::new(196210, -4888017823148),
            TickInfo::new(196220, 161367614277373),
            TickInfo::new(196230, -85651143943809),
            TickInfo::new(196240, 832673790145),
            TickInfo::new(196250, -7364051505611460),
            TickInfo::new(196260, -33905079727641898),
            TickInfo::new(196270, 257949016690013),
            TickInfo::new(196280, 103444812730821),
            TickInfo::new(196290, -8171423444647946),
            TickInfo::new(196300, 1681147127179411),
            TickInfo::new(196310, -33957210118740621),
            TickInfo::new(196320, 1572450960698),
            TickInfo::new(196330, -7078403028662),
            TickInfo::new(196340, -189662625501685),
            TickInfo::new(196360, -44279667104524),
            TickInfo::new(196380, 68967570782517),
            TickInfo::new(196390, 291722824867927),
            TickInfo::new(196400, 260077680036104),
            TickInfo::new(196410, -291722824867927),
            TickInfo::new(196420, -450175903873477),
            TickInfo::new(196440, -174491192869730),
            TickInfo::new(196450, -1142974422066087),
            TickInfo::new(196460, 23716808157624),
            TickInfo::new(196470, 16813896239854),
            TickInfo::new(196480, -6245861528179271),
            TickInfo::new(196490, 76325096396232),
            TickInfo::new(196500, 184913570239120),
            TickInfo::new(196510, 319528687331476),
            TickInfo::new(196520, -184913570239120),
            TickInfo::new(196530, -517581405166413),
            TickInfo::new(196540, -4699291558494),
            TickInfo::new(196560, 687061659425998),
            TickInfo::new(196570, 275211829108371),
            TickInfo::new(196580, -696421060987894),
            TickInfo::new(196590, -275211829108371),
            TickInfo::new(196600, 491866254570155),
            TickInfo::new(196610, -33109507204443),
            TickInfo::new(196620, -13601083638),
            TickInfo::new(196630, -606668021377386),
            TickInfo::new(196640, 2822561197168283),
            TickInfo::new(196650, -87029620885755),
            TickInfo::new(196660, -277416577183292),
            TickInfo::new(196680, -228739591168735),
            TickInfo::new(196690, -1990233101382540),
            TickInfo::new(196710, 477264930847132),
            TickInfo::new(196720, 351126201990803),
            TickInfo::new(196730, 68095834915471),
            TickInfo::new(196740, -691498167384),
            TickInfo::new(196750, -6041575136322),
            TickInfo::new(196760, -255147773903835),
            TickInfo::new(196770, 14691343731744),
            TickInfo::new(196780, -2459033625423),
            TickInfo::new(196790, 1063903322650331),
            TickInfo::new(196800, -1320579380532456),
            TickInfo::new(196810, -50099074879937),
            TickInfo::new(196820, 6620733776444749),
            TickInfo::new(196840, -14334110),
            TickInfo::new(196850, 56882),
            TickInfo::new(196870, 10135906124416),
            TickInfo::new(196880, -15042028409493),
            TickInfo::new(196900, 2660309908620727),
            TickInfo::new(196910, -4811971477690),
            TickInfo::new(196920, 58484208444917),
            TickInfo::new(196930, -1460551175454691),
            TickInfo::new(196940, -16302416809366581),
            TickInfo::new(196950, -3990811678765292),
            TickInfo::new(196960, -43249317349795),
            TickInfo::new(196970, 619434952584772),
            TickInfo::new(196980, 3959478091923),
            TickInfo::new(196990, -1021236380313973),
            TickInfo::new(197000, 4494901289223705),
            TickInfo::new(197010, -6390358767843717),
            TickInfo::new(197020, 47760155),
            TickInfo::new(197030, 1607497067360447),
            TickInfo::new(197040, -47760155),
            TickInfo::new(197050, -1064336541331),
            TickInfo::new(197060, 30422423310469),
            TickInfo::new(197070, -130633561795810),
            TickInfo::new(197080, -108838902558678),
            TickInfo::new(197090, 9394042929301352),
            TickInfo::new(197100, -5880440729039),
            TickInfo::new(197110, -15686405820345),
            TickInfo::new(197130, -22481743678107320),
            TickInfo::new(197140, 1404382673471161),
            TickInfo::new(197150, 59283650865347),
            TickInfo::new(197160, 1045073826424212),
            TickInfo::new(197180, 1379369439272641),
            TickInfo::new(197190, -74240964321383),
            TickInfo::new(197200, -1285928851405615),
            TickInfo::new(197220, 32576198885432),
            TickInfo::new(197230, 107746496109199),
            TickInfo::new(197240, 212828189966921),
            TickInfo::new(197250, 26627335411421),
            TickInfo::new(197260, -268242300047918),
            TickInfo::new(197270, 299117231476348),
            TickInfo::new(197290, -321189761116248),
            TickInfo::new(197310, -17731575051374491),
            TickInfo::new(197330, 134640636153769),
            TickInfo::new(197340, -80489425535468),
            TickInfo::new(197350, -316164739913418),
            TickInfo::new(197360, -847182171257),
            TickInfo::new(197370, 111580463696280),
            TickInfo::new(197380, -752056005096855),
            TickInfo::new(197390, 5651252445553),
            TickInfo::new(197400, 5248503423456263),
            TickInfo::new(197410, 557949097433222),
            TickInfo::new(197420, -427386792905201),
            TickInfo::new(197430, 2694300761945565),
            TickInfo::new(197440, -4736302076672598),
            TickInfo::new(197450, -23868680164168),
            TickInfo::new(197460, -1398454345514152),
            TickInfo::new(197470, 1634428937928623),
            TickInfo::new(197480, -3073114744199753),
            TickInfo::new(197490, -12210402708863),
            TickInfo::new(197500, -4873384021599149),
            TickInfo::new(197510, -251054225367),
            TickInfo::new(197520, 361346765013654),
            TickInfo::new(197530, 201021515384362),
            TickInfo::new(197540, -268815166292558),
            TickInfo::new(197550, -195608658405383),
            TickInfo::new(197560, 10976623986803697),
            TickInfo::new(197570, 1409179795951465),
            TickInfo::new(197580, -11037147452304866),
            TickInfo::new(197590, -892169573277407),
            TickInfo::new(197600, 1754105197363844),
            TickInfo::new(197610, 167583061487984),
            TickInfo::new(197620, -1847429021911883),
            TickInfo::new(197630, 193739327053775),
            TickInfo::new(197640, 207591694004054),
            TickInfo::new(197650, 4126567458044491),
            TickInfo::new(197660, -197248894708313),
            TickInfo::new(197670, -4376644206972401),
            TickInfo::new(197680, 857382257338122),
            TickInfo::new(197690, -29281341163917790),
            TickInfo::new(197700, 66596947231437),
            TickInfo::new(197710, 56696890135),
            TickInfo::new(197720, -1073109110639785),
            TickInfo::new(197730, 2229793195379915470),
            TickInfo::new(197740, -2227345450299561462),
            TickInfo::new(197750, -1721422109838698),
            TickInfo::new(197760, -6071716122228),
            TickInfo::new(197780, -383366919250281),
            TickInfo::new(197790, 56015721937652),
            TickInfo::new(197800, 1316020909718482),
            TickInfo::new(197810, -31362008543507),
            TickInfo::new(197820, 3165477434124),
            TickInfo::new(197830, -1293418572576637),
            TickInfo::new(197840, -10463644269478),
            TickInfo::new(197850, -12141064759829),
            TickInfo::new(197860, -6581266493858149),
            TickInfo::new(197870, 637607665423487),
            TickInfo::new(197880, 43645986313600),
            TickInfo::new(197890, -527321090162260),
            TickInfo::new(197900, 139641090876619),
            TickInfo::new(197910, 3598490209837783),
            TickInfo::new(197920, -155011488206174),
            TickInfo::new(197930, -13554370420632),
            TickInfo::new(197940, -3373078624420383),
            TickInfo::new(197950, 1113055660529935),
            TickInfo::new(197960, -2001016307140),
            TickInfo::new(197970, -935721011585279),
            TickInfo::new(197980, -2801848355158361),
            TickInfo::new(197990, 272092801187),
            TickInfo::new(198000, 208621214404126),
            TickInfo::new(198010, -2984372891292173),
            TickInfo::new(198020, 7252688040227),
            TickInfo::new(198030, 338079598778340),
            TickInfo::new(198040, 784111021185317),
            TickInfo::new(198050, -1086347379250441),
            TickInfo::new(198070, -4305434898959029),
            TickInfo::new(198080, -40782934764873814),
            TickInfo::new(198090, -39383549767089602),
            TickInfo::new(198100, 8743237610319699),
            TickInfo::new(198120, -15497856512577588),
            TickInfo::new(198130, -24069662215151207),
            TickInfo::new(198140, -768755543026544),
            TickInfo::new(198160, 479779145875449),
            TickInfo::new(198170, -16507237651455),
            TickInfo::new(198180, -20815245879999),
            TickInfo::new(198190, 398808670381539),
            TickInfo::new(198200, 628650267437247),
            TickInfo::new(198210, -1559597444604115),
            TickInfo::new(198230, -121560313876077),
            TickInfo::new(198240, 2519036035195),
            TickInfo::new(198250, -301623200895122),
            TickInfo::new(198260, 26270459940136),
            TickInfo::new(198270, -10135906124416),
            TickInfo::new(198280, -73392587069442),
            TickInfo::new(198290, 47739347251),
            TickInfo::new(198300, -31838653988823),
            TickInfo::new(198310, 18157443316294683),
            TickInfo::new(198320, 1131825612344787),
            TickInfo::new(198330, -17573910184836852),
            TickInfo::new(198340, -1147770206691903),
            TickInfo::new(198360, 245592076582059),
            TickInfo::new(198380, -826601339),
            TickInfo::new(198390, 719385737917328),
            TickInfo::new(198400, 15580378389654),
            TickInfo::new(198410, -2124157394070),
            TickInfo::new(198420, 451500832683482),
            TickInfo::new(198430, 283463112072528),
            TickInfo::new(198440, 583919962818375),
            TickInfo::new(198450, -290277545196536),
            TickInfo::new(198460, -770472220213029),
            TickInfo::new(198470, 99862653496),
            TickInfo::new(198480, 900839097196264),
            TickInfo::new(198490, -5259826088156424),
            TickInfo::new(198510, -11496411404286),
            TickInfo::new(198540, 173732600262594),
            TickInfo::new(198570, 3783916266906828),
            TickInfo::new(198580, -5549435394448),
            TickInfo::new(198590, -3968797884756267),
            TickInfo::new(198610, -8474131399635),
            TickInfo::new(198620, 282817264728283),
            TickInfo::new(198630, 231257853830576),
            TickInfo::new(198640, -13283408140229),
            TickInfo::new(198650, -230133082661011),
            TickInfo::new(198660, 60889973703997),
            TickInfo::new(198670, -93068856583417),
            TickInfo::new(198680, -10568367650),
            TickInfo::new(198690, 121987180817589),
            TickInfo::new(198700, -60939093687807),
            TickInfo::new(198710, 308480905945336),
            TickInfo::new(198720, -1329378141046676),
            TickInfo::new(198730, -34695508046861),
            TickInfo::new(198740, -32014372783357),
            TickInfo::new(198750, 6293937469386956),
            TickInfo::new(198760, -6474856383277037),
            TickInfo::new(198770, -45918026204723),
            TickInfo::new(198780, 37834206153297),
            TickInfo::new(198790, -67299727759981),
            TickInfo::new(198810, -8372045892911),
            TickInfo::new(198820, -42729429656536),
            TickInfo::new(198830, -2797835679058747),
            TickInfo::new(198840, -4029643996556),
            TickInfo::new(198850, -34485672668104),
            TickInfo::new(198860, -598528128940),
            TickInfo::new(198870, -63354520726),
            TickInfo::new(198880, 318751421923936),
            TickInfo::new(198890, 464608994416258),
            TickInfo::new(198900, 959622567567724),
            TickInfo::new(198910, -1781399117959140),
            TickInfo::new(198920, -2693121882085043),
            TickInfo::new(198930, 9432688312744859),
            TickInfo::new(198940, -61455478067194),
            TickInfo::new(198950, -9523392769960157),
            TickInfo::new(198960, -613611879278974),
            TickInfo::new(198970, -11600521422629),
            TickInfo::new(198980, -6057272388149),
            TickInfo::new(198990, 116038509072488),
            TickInfo::new(199000, 2941147221545),
            TickInfo::new(199010, -132041755579250),
            TickInfo::new(199020, 81881594487534),
            TickInfo::new(199030, 38047204275619),
            TickInfo::new(199040, -1608314815216),
            TickInfo::new(199050, -559759276453364),
            TickInfo::new(199060, -82545145586010),
            TickInfo::new(199070, 73788088044),
            TickInfo::new(199080, 4386037171978715),
            TickInfo::new(199090, 155380486075820),
            TickInfo::new(199100, -19048869057949),
            TickInfo::new(199110, 26941645942926228),
            TickInfo::new(199130, 2214819435643762),
            TickInfo::new(199140, 6752318500854474),
            TickInfo::new(199150, 47373281153672),
            TickInfo::new(199160, -6469904903001750),
            TickInfo::new(199170, -25679251608679),
            TickInfo::new(199180, -282432206519017),
            TickInfo::new(199190, 232595088633026),
            TickInfo::new(199200, 34893328671790489),
            TickInfo::new(199210, -256161256197068),
            TickInfo::new(199220, 49285086809113),
            TickInfo::new(199230, 147674185025024),
            TickInfo::new(199240, 14164582632099),
            TickInfo::new(199250, 1927115828601846),
            TickInfo::new(199260, 25427533995542624),
            TickInfo::new(199270, -1874510080240612),
            TickInfo::new(199280, -1992932662768916),
            TickInfo::new(199290, 148016922752040),
            TickInfo::new(199300, -373605264873176),
            TickInfo::new(199310, 1352885195903299456),
            TickInfo::new(199320, -33776187944116123),
            TickInfo::new(199330, 14346432094203),
            TickInfo::new(199340, -817554709350),
            TickInfo::new(199350, -2129948289393),
            TickInfo::new(199360, -5972537845246932),
            TickInfo::new(199370, -4642147956327),
            TickInfo::new(199380, 7598705807111945502),
            TickInfo::new(199390, -6908484258181),
            TickInfo::new(199400, 233),
            TickInfo::new(199420, -1),
            TickInfo::new(199430, 16427088716168126),
            TickInfo::new(199440, 1165566570179008),
            TickInfo::new(199450, -600001754019951),
            TickInfo::new(199460, -16404462357409696),
            TickInfo::new(199470, 126146815564161),
            TickInfo::new(199480, -33324672753),
            TickInfo::new(199490, 109441723946229),
            TickInfo::new(199500, -2816471387297651),
            TickInfo::new(199520, 371),
            TickInfo::new(199540, -7049110270089969),
            TickInfo::new(199550, 74496991922150),
            TickInfo::new(199560, 1616020322756620),
            TickInfo::new(199570, 6114792526958),
            TickInfo::new(199580, -1616020322746218),
            TickInfo::new(199590, -184506482331579),
            TickInfo::new(199600, -10779293),
            TickInfo::new(199610, -787418080409433),
            TickInfo::new(199620, -67582646792077),
            TickInfo::new(199630, 513032659291062),
            TickInfo::new(199640, -231),
            TickInfo::new(199650, 71560570730),
            TickInfo::new(199660, -231),
            TickInfo::new(199680, 26101660793654),
            TickInfo::new(199690, 191701986657523),
            TickInfo::new(199700, 425879693738647),
            TickInfo::new(199710, -1394522275932262),
            TickInfo::new(199720, 783773814508095),
            TickInfo::new(199730, 14403264449810),
            TickInfo::new(199740, 215),
            TickInfo::new(199750, 338386054535499),
            TickInfo::new(199760, 14403),
            TickInfo::new(199770, 224088807239040),
            TickInfo::new(199780, -306411751523488),
            TickInfo::new(199800, 9774472201415),
            TickInfo::new(199810, 19841324105038),
            TickInfo::new(199820, -181988844886207),
            TickInfo::new(199830, 24648158712216),
            TickInfo::new(199840, -10909651),
            TickInfo::new(199860, -8754785440878),
            TickInfo::new(199870, -31794472274990),
            TickInfo::new(199880, 1),
            TickInfo::new(199890, -5524425706845),
            TickInfo::new(199900, 21906),
            TickInfo::new(199910, 123704697866143),
            TickInfo::new(199920, 433690381460236),
            TickInfo::new(199930, -23826526836849),
            TickInfo::new(199940, -86119867589),
            TickInfo::new(199950, 1788833624868295),
            TickInfo::new(199960, 646985384531357),
            TickInfo::new(199970, 1110724847423183),
            TickInfo::new(199980, 21993),
            TickInfo::new(199990, -1110724847423184),
            TickInfo::new(200000, -21584269815665),
            TickInfo::new(200020, -227365776986897),
            TickInfo::new(200030, -90734592066),
            TickInfo::new(200040, 22060),
            TickInfo::new(200050, -104340034120545),
            TickInfo::new(200060, 22081),
            TickInfo::new(200080, 22104),
            TickInfo::new(200100, 22127),
            TickInfo::new(200110, -7231564215441),
            TickInfo::new(200120, 374023533237),
            TickInfo::new(200140, 22170),
            TickInfo::new(200150, 2794350631607160463),
            TickInfo::new(200160, -68558788127722),
            TickInfo::new(200180, 22214),
            TickInfo::new(200190, 490373193234813),
            TickInfo::new(200200, 3692609760423),
            TickInfo::new(200210, 2023873390418693),
            TickInfo::new(200220, 993179348993),
            TickInfo::new(200230, -48494646361),
            TickInfo::new(200240, -1138921463695256),
            TickInfo::new(200250, -1),
            TickInfo::new(200260, 1897811301938688),
            TickInfo::new(200280, 22325),
            TickInfo::new(200290, 1529687658581818),
            TickInfo::new(200300, -20756378733009373),
            TickInfo::new(200310, -20173889312110874),
            TickInfo::new(200320, 96449144525111),
            TickInfo::new(200330, -27372571432279),
            TickInfo::new(200340, 2742414754506642),
            TickInfo::new(200350, -15129925699259),
            TickInfo::new(200360, -374023511091),
            TickInfo::new(200370, -3930566173970),
            TickInfo::new(200380, 93855006829908),
            TickInfo::new(200390, 996569352538131),
            TickInfo::new(200400, -510838932108302),
            TickInfo::new(200410, -178609528144522),
            TickInfo::new(200420, 122792561080834992),
            TickInfo::new(200430, 6957598209700),
            TickInfo::new(200440, -122792561080834992),
            TickInfo::new(200450, 14566093142379),
            TickInfo::new(200460, -8511086224541862),
            TickInfo::new(200470, 351337821359505),
            TickInfo::new(200480, -16051775222728),
            TickInfo::new(200490, -351337821359505),
            TickInfo::new(200510, 36513897146351),
            TickInfo::new(200520, 45319483288907),
            TickInfo::new(200530, 21173915535375),
            TickInfo::new(200540, -1),
            TickInfo::new(200550, -8284070484708),
            TickInfo::new(200560, -473812700333203),
            TickInfo::new(200570, -1982310279884812),
            TickInfo::new(200580, 30330952715726),
            TickInfo::new(200590, 71570634871),
            TickInfo::new(200600, 68095114764129),
            TickInfo::new(200610, -1836850799640),
            TickInfo::new(200620, -68095114764130),
            TickInfo::new(200630, -420589345654229),
            TickInfo::new(200640, 815702302406377),
            TickInfo::new(200660, -812983526946829),
            TickInfo::new(200680, 106429529364921),
            TickInfo::new(200690, 88575924976180),
            TickInfo::new(200700, 117537856462031),
            TickInfo::new(200710, -96612801794301),
            TickInfo::new(200720, 328276238126211),
            TickInfo::new(200730, 2525400803780),
            TickInfo::new(200740, 756037507768727),
            TickInfo::new(200750, -437640355755909),
            TickInfo::new(200760, -411429389961272),
            TickInfo::new(200770, 2635830155326988),
            TickInfo::new(200780, -350592129264),
            TickInfo::new(200790, -1965897650637536),
            TickInfo::new(200800, -4817758115102),
            TickInfo::new(200810, 406809738308173),
            TickInfo::new(200820, -226615127015775),
            TickInfo::new(200830, -1857694841713043),
            TickInfo::new(200840, 5935929477015562),
            TickInfo::new(200850, 844988115259702),
            TickInfo::new(200860, -7590074359994158930),
            TickInfo::new(200870, -1353585769302967830),
            TickInfo::new(200880, -5807344029001399),
            TickInfo::new(200890, 160869487974152025),
            TickInfo::new(200900, 372),
            TickInfo::new(200910, -160753138694254855),
            TickInfo::new(200920, -122613609468868),
            TickInfo::new(200930, -5491879378840760),
            TickInfo::new(200940, -1034480172633653),
            TickInfo::new(200950, -126458672292395),
            TickInfo::new(200960, 23098),
            TickInfo::new(200970, 1357652577477990),
            TickInfo::new(200980, 1812508146521759),
            TickInfo::new(200990, -1367414389390946),
            TickInfo::new(201000, 702617002410770),
            TickInfo::new(201010, 4315586199005097),
            TickInfo::new(201020, 582822049176087),
            TickInfo::new(201030, -6048431382098680),
            TickInfo::new(201040, -3557979882117344),
            TickInfo::new(201050, 80880024376906),
            TickInfo::new(201060, 504391598442),
            TickInfo::new(201070, -298187954299110),
            TickInfo::new(201080, 68956702828135),
            TickInfo::new(201090, 2668953208653137),
            TickInfo::new(201100, -3055376945953063),
            TickInfo::new(201110, -34814551861353),
            TickInfo::new(201120, 370998123320649),
            TickInfo::new(201130, -77307541856785),
            TickInfo::new(201140, -355906641703505),
            TickInfo::new(201150, 3199296154466082),
            TickInfo::new(201160, 60677488234077),
            TickInfo::new(201170, -1501373152030),
            TickInfo::new(201180, -59302831729477),
            TickInfo::new(201190, 1483277048793441),
            TickInfo::new(201200, 51360526510960),
            TickInfo::new(201210, 1797141477859690),
            TickInfo::new(201220, 23400),
            TickInfo::new(201230, -1766635752917669),
            TickInfo::new(201240, 818883235855889),
            TickInfo::new(201250, -8157871729957143),
            TickInfo::new(201260, 1771057903642426),
            TickInfo::new(201270, 9752955799890292),
            TickInfo::new(201280, -10249529180754647),
            TickInfo::new(201290, -1402489897593015),
            TickInfo::new(201300, -160525415309438),
            TickInfo::new(201310, 477712266859626),
            TickInfo::new(201320, 66974483738175),
            TickInfo::new(201330, -84128988190204),
            TickInfo::new(201340, -67777587699958),
            TickInfo::new(201350, 4011809697059212),
            TickInfo::new(201360, -1961889913047020),
            TickInfo::new(201370, 354680333401936),
            TickInfo::new(201380, -1539253825653133),
            TickInfo::new(201390, -4990386824878),
            TickInfo::new(201400, 1886785011478798),
            TickInfo::new(201410, 3028722380488513),
            TickInfo::new(201420, -1234960806301203),
            TickInfo::new(201430, -3729169504894378),
            TickInfo::new(201440, -733314569528550),
            TickInfo::new(201450, 384794861559651),
            TickInfo::new(201460, -650907398108510),
            TickInfo::new(201470, -16827750181546),
            TickInfo::new(201480, -1833688456947804),
            TickInfo::new(201490, 90499272574400),
            TickInfo::new(201500, -3855572938769733),
            TickInfo::new(201510, -727076815925031),
            TickInfo::new(201520, -113477480720098),
            TickInfo::new(201530, 6824239110568007),
            TickInfo::new(201550, 32563202638530),
            TickInfo::new(201560, -1114705087551475),
            TickInfo::new(201570, -2164250889876547),
            TickInfo::new(201580, 903071009663854),
            TickInfo::new(201590, -1029158315195265),
            TickInfo::new(201600, -206779111675761),
            TickInfo::new(201610, -101525303618758),
            TickInfo::new(201620, 4792164193),
            TickInfo::new(201630, -3872515054515),
            TickInfo::new(201640, -26982940616495631),
            TickInfo::new(201650, 765361740583450),
            TickInfo::new(201660, -33390114239104),
            TickInfo::new(201670, -775342162142592),
            TickInfo::new(201680, -198242218931975),
            TickInfo::new(201690, -99253126273823),
            TickInfo::new(201700, -23144534925279),
            TickInfo::new(201710, 98614049897619),
            TickInfo::new(201720, -78809282445782),
            TickInfo::new(201730, 128194750968894),
            TickInfo::new(201740, -140590409734278),
            TickInfo::new(201750, -31477761011204),
            TickInfo::new(201760, 55594628576024),
            TickInfo::new(201770, -8544138),
            TickInfo::new(201780, -15486538387791),
            TickInfo::new(201790, -74496991922150),
            TickInfo::new(201800, -9319971869154),
            TickInfo::new(201810, 122846685345363),
            TickInfo::new(201820, -32789672023349),
            TickInfo::new(201830, -122846322537263),
            TickInfo::new(201840, 3293926376305),
            TickInfo::new(201850, -75303493831926),
            TickInfo::new(201860, 98747458896728),
            TickInfo::new(201880, 27576606725510),
            TickInfo::new(201890, 304140118800551),
            TickInfo::new(201900, 26681127819988),
            TickInfo::new(201910, -415471330504993),
            TickInfo::new(201920, 1490752855615372),
            TickInfo::new(201930, -1328000075072396),
            TickInfo::new(201940, -2795584118052655586),
            TickInfo::new(201950, 8772087391684),
            TickInfo::new(201960, -2022334482141697),
            TickInfo::new(201980, -10066320121934),
            TickInfo::new(201990, -918253170159524),
            TickInfo::new(202000, 1293604400121640),
            TickInfo::new(202020, 371214130284398),
            TickInfo::new(202030, -52164127452446),
            TickInfo::new(202040, -1165747826076894),
            TickInfo::new(202050, 36208171340227),
            TickInfo::new(202060, 385341402660873),
            TickInfo::new(202070, 1256094250164338),
            TickInfo::new(202080, -1289117518890826),
            TickInfo::new(202090, 899518794073598),
            TickInfo::new(202100, 34556401736423),
            TickInfo::new(202110, -207614746038809),
            TickInfo::new(202120, -405770532020821),
            TickInfo::new(202130, -873719823639174),
            TickInfo::new(202150, 1203784374257720),
            TickInfo::new(202160, 1460098731480152),
            TickInfo::new(202170, -1205520087204318),
            TickInfo::new(202180, -666672863230672),
            TickInfo::new(202190, -721127241915418),
            TickInfo::new(202200, 2841271059224749),
            TickInfo::new(202210, 20804432915779),
            TickInfo::new(202220, -2841271059224749),
            TickInfo::new(202230, 3890419816627696),
            TickInfo::new(202240, 4846086921315790),
            TickInfo::new(202250, -1845831689201499),
            TickInfo::new(202260, -98747458896728),
            TickInfo::new(202270, 948515720825901),
            TickInfo::new(202280, -606932490709277),
            TickInfo::new(202290, -948515720825901),
            TickInfo::new(202300, -84431771944744),
            TickInfo::new(202310, 1016658362362261),
            TickInfo::new(202320, 1485425173462),
            TickInfo::new(202330, -530380071325622),
            TickInfo::new(202340, 17826169961373778),
            TickInfo::new(202350, -2110649311887226),
            TickInfo::new(202360, -528464800249724),
            TickInfo::new(202370, -5181401192827),
            TickInfo::new(202380, 275850618898897),
            TickInfo::new(202390, -202239912360275),
            TickInfo::new(202400, 11907509894361),
            TickInfo::new(202410, -104380890878967),
            TickInfo::new(202420, -478379313437571),
            TickInfo::new(202430, -20652721952631),
            TickInfo::new(202440, 49749484),
            TickInfo::new(202450, -21045839514248),
            TickInfo::new(202470, -6586339534015),
            TickInfo::new(202480, -730144639967080),
            TickInfo::new(202490, 294207921674184),
            TickInfo::new(202510, -17957417719219246),
            TickInfo::new(202520, -4385939184862505),
            TickInfo::new(202540, 584567585465525),
            TickInfo::new(202550, -162102198680786),
            TickInfo::new(202560, 4400484934565610),
            TickInfo::new(202570, -4172116416249325),
            TickInfo::new(202580, 13305292578630),
            TickInfo::new(202590, 3102843738936),
            TickInfo::new(202600, -13305292578630),
            TickInfo::new(202610, -425305726470974),
            TickInfo::new(202620, 17868651158522),
            TickInfo::new(202630, -29245318865145),
            TickInfo::new(202640, -240392410784920),
            TickInfo::new(202650, -1693579555728),
            TickInfo::new(202660, 586789738158),
            TickInfo::new(202670, -4375844532703458),
            TickInfo::new(202680, -4234912725548),
            TickInfo::new(202690, -209975311487342),
            TickInfo::new(202700, 3912691503080106),
            TickInfo::new(202710, -3045642408061188),
            TickInfo::new(202720, 405576329100360),
            TickInfo::new(202730, 92075674360476319),
            TickInfo::new(202750, -93398308354173048),
            TickInfo::new(202760, -405576329100360),
            TickInfo::new(202770, -635875721240268),
            TickInfo::new(202780, 39367994112418),
            TickInfo::new(202790, 86543163008874),
            TickInfo::new(202800, 534741969435138),
            TickInfo::new(202810, 49766248128352),
            TickInfo::new(202820, -853206539302727),
            TickInfo::new(202830, 314569659716951),
            TickInfo::new(202840, -19032019196998),
            TickInfo::new(202850, -298478209754927),
            TickInfo::new(202860, -2170513430308334),
            TickInfo::new(202870, -1790722478715099),
            TickInfo::new(202880, 86481145519812),
            TickInfo::new(202890, -159707892442444),
            TickInfo::new(202900, -124779696182196),
            TickInfo::new(202920, 1559500721976),
            TickInfo::new(202960, -44770824606692),
            TickInfo::new(202970, -87133314853926),
            TickInfo::new(202980, -17566865042867),
            TickInfo::new(202990, -53120179115163),
            TickInfo::new(203000, -7181254743610),
            TickInfo::new(203020, -8407139530500),
            TickInfo::new(203040, -101785000042774),
            TickInfo::new(203050, 598080051491788),
            TickInfo::new(203060, -6088612861499),
            TickInfo::new(203100, -28277919904324),
            TickInfo::new(203120, -5231689491892),
            TickInfo::new(203130, -32865497418105),
            TickInfo::new(203150, -41481054197019),
            TickInfo::new(203160, -15336904627316),
            TickInfo::new(203170, -24388615498285),
            TickInfo::new(203180, -41827190104880),
            TickInfo::new(203190, -52315735489592798),
            TickInfo::new(203200, -584738576398555),
            TickInfo::new(203220, -7619473815173),
            TickInfo::new(203260, -636124044933302),
            TickInfo::new(203270, -354433757869),
            TickInfo::new(203320, -17732822804684),
            TickInfo::new(203330, -42823465704219),
            TickInfo::new(203360, -1590441756031),
            TickInfo::new(203390, -22560478618918),
            TickInfo::new(203400, -507090483264063),
            TickInfo::new(203460, -4401764424180),
            TickInfo::new(203470, -9287401875839),
            TickInfo::new(203490, -66885192117),
            TickInfo::new(203520, -6560098309762),
            TickInfo::new(203530, -786980171399450),
            TickInfo::new(203540, -73502870437),
            TickInfo::new(203550, -146194548038870),
            TickInfo::new(203600, 41148245238),
            TickInfo::new(203640, -598080051491778),
            TickInfo::new(203700, -16732059572749),
            TickInfo::new(203740, -25808506038260),
            TickInfo::new(203780, 37593079364642),
            TickInfo::new(203790, -37593079364642),
            TickInfo::new(203820, -4459378691580),
            TickInfo::new(203840, -190075317894),
            TickInfo::new(203860, -2563707068368625),
            TickInfo::new(203870, -4039297443936569),
            TickInfo::new(203880, -22698763804104424),
            TickInfo::new(203890, 6489986004223),
            TickInfo::new(203950, 33906611355),
            TickInfo::new(203960, 1184074883211),
            TickInfo::new(203990, 15213000014794),
            TickInfo::new(204020, -2721055168095),
            TickInfo::new(204050, -224653000969518),
            TickInfo::new(204060, -2594583748343629),
            TickInfo::new(204070, -23405696260743),
            TickInfo::new(204080, 44749132440304),
            TickInfo::new(204090, 366010862813302),
            TickInfo::new(204100, -15580378389654),
            TickInfo::new(204130, -21173915535375),
            TickInfo::new(204160, 527279115156),
            TickInfo::new(204170, 81134504679928),
            TickInfo::new(204180, 1004655231506),
            TickInfo::new(204200, 696765182566037),
            TickInfo::new(204240, -954979947124),
            TickInfo::new(204260, 2298787003010616),
            TickInfo::new(204280, -1004655231506),
            TickInfo::new(204340, -205167652606593),
            TickInfo::new(204360, 46922130405323),
            TickInfo::new(204380, -497297810940347),
            TickInfo::new(204390, 86249643007776),
            TickInfo::new(204420, -3990344019813),
            TickInfo::new(204470, -93413305517238),
            TickInfo::new(204490, 9024164118422),
            TickInfo::new(204500, -10),
            TickInfo::new(204530, 21821905),
            TickInfo::new(204550, -33615509923089),
            TickInfo::new(204570, 17618420282610),
            TickInfo::new(204600, -23695521257064),
            TickInfo::new(204610, 1341183578617033),
            TickInfo::new(204620, 83217762574215),
            TickInfo::new(204630, 39197183406877),
            TickInfo::new(204640, 394387707481),
            TickInfo::new(204650, 84644554722165),
            TickInfo::new(204660, -394387707481),
            TickInfo::new(204670, -1966158769478),
            TickInfo::new(204710, -721318221347292),
            TickInfo::new(204720, -52991559159211),
            TickInfo::new(204740, -1290203203245374),
            TickInfo::new(204750, 1087800237554443),
            TickInfo::new(204790, -58556667865517),
            TickInfo::new(204800, 8811768495241),
            TickInfo::new(204820, -8811768495241),
            TickInfo::new(204860, 26579569),
            TickInfo::new(204890, -3777730624536),
            TickInfo::new(204900, -26579569),
            TickInfo::new(204910, -910271637044215),
            TickInfo::new(204930, -6510470788869),
            TickInfo::new(204950, -35112050975070),
            TickInfo::new(204960, 5708588153368),
            TickInfo::new(204970, -37231024637399),
            TickInfo::new(204990, -939053049297),
            TickInfo::new(205000, -15213000014794),
            TickInfo::new(205010, 2229579203960),
            TickInfo::new(205070, 3258972779416),
            TickInfo::new(205090, -3258972779416),
            TickInfo::new(205140, 3815506156622),
            TickInfo::new(205180, 72247538655926),
            TickInfo::new(205190, -5708588153368),
            TickInfo::new(205200, -2154791887441158),
            TickInfo::new(205210, 12887701135087),
            TickInfo::new(205220, -2413943077381),
            TickInfo::new(205240, -12887701135087),
            TickInfo::new(205250, 2118391249583235),
            TickInfo::new(205260, 92409582075501),
            TickInfo::new(205270, -2177218599761408),
            TickInfo::new(205280, -75931054461905),
            TickInfo::new(205290, -41769168482150),
            TickInfo::new(205300, -163892614366023),
            TickInfo::new(205330, 35601822000),
            TickInfo::new(205340, -7721290634729),
            TickInfo::new(205350, 35413176029881),
            TickInfo::new(205370, -35601822000),
            TickInfo::new(205380, 56032323483674),
            TickInfo::new(205390, -35413176029881),
            TickInfo::new(205420, -2223869642786316),
            TickInfo::new(205430, 221591903313724),
            TickInfo::new(205460, -2337388845635494),
            TickInfo::new(205490, 167222976955551),
            TickInfo::new(205500, -5141291429292),
            TickInfo::new(205510, -35547267802725),
            TickInfo::new(205530, -167222976955551),
            TickInfo::new(205560, 22741687028344),
            TickInfo::new(205590, -10644388542016),
            TickInfo::new(205610, -22741687028344),
            TickInfo::new(205630, -22525689953564),
            TickInfo::new(205800, -46922130405323),
            TickInfo::new(205850, -483224541245025),
            TickInfo::new(205870, -1653347163134),
            TickInfo::new(205920, 5921513161873),
            TickInfo::new(205930, -5921513161873),
            TickInfo::new(205990, -14566093142378),
            TickInfo::new(206010, 4132589508320855),
            TickInfo::new(206030, -821491371774),
            TickInfo::new(206060, -36825342664461),
            TickInfo::new(206130, 382765298101521),
            TickInfo::new(206190, -3815527978527),
            TickInfo::new(206250, -30410086366488),
            TickInfo::new(206260, -477197831652989),
            TickInfo::new(206290, 3526014420553790),
            TickInfo::new(206310, -744366934478),
            TickInfo::new(206320, -4132589508320855),
            TickInfo::new(206370, 122973486683145),
            TickInfo::new(206380, -128273656740017),
            TickInfo::new(206410, 5517325292675),
            TickInfo::new(206450, -5517325292675),
            TickInfo::new(206530, -46445625148087),
            TickInfo::new(206560, -10456351239484),
            TickInfo::new(206690, 39249186082639),
            TickInfo::new(206710, -39249186082639),
            TickInfo::new(206850, -71560570730),
            TickInfo::new(207040, -1060355204688),
            TickInfo::new(207090, -21599131813462),
            TickInfo::new(207190, -116526800732),
            TickInfo::new(207200, 23690981880736),
            TickInfo::new(207220, -229060865165778),
            TickInfo::new(207240, -14937241110954516),
            TickInfo::new(207250, -108381237000),
            TickInfo::new(207280, -37137476019528),
            TickInfo::new(207330, -1457334165987),
            TickInfo::new(207350, -1777620945252),
            TickInfo::new(207380, -1049687190863),
            TickInfo::new(207540, -6783329557647),
            TickInfo::new(207670, -17618420282610),
            TickInfo::new(207760, -1024516309504589),
            TickInfo::new(207900, -33663120308853),
            TickInfo::new(207910, -8),
            TickInfo::new(208290, -3047476124369),
            TickInfo::new(208300, -483896091878231),
            TickInfo::new(208430, -15816971942115),
            TickInfo::new(208520, -54743213803487),
            TickInfo::new(208650, -1238910149582598),
            TickInfo::new(208760, -1759243635560),
            TickInfo::new(208870, -116383157570658),
            TickInfo::new(209120, -14039449939),
            TickInfo::new(209140, -34367459497789),
            TickInfo::new(209310, -4225060974401),
            TickInfo::new(209470, -9059350051728),
            TickInfo::new(209730, -1084560033827482),
            TickInfo::new(209900, -191701986657523),
            TickInfo::new(210120, -7222183383424229),
            TickInfo::new(210380, -228699723058089),
            TickInfo::new(210810, -69273863244383),
            TickInfo::new(210830, -13348687223838),
            TickInfo::new(211310, -2649113876628352),
            TickInfo::new(211440, -2659554920775),
            TickInfo::new(211470, -4107565474162),
            TickInfo::new(211550, -1366052033351),
            TickInfo::new(211560, -706192325),
            TickInfo::new(212080, -4860235671811),
            TickInfo::new(212160, -4554637977519),
            TickInfo::new(212350, -1367905074718435),
            TickInfo::new(212510, -187004715148085),
            TickInfo::new(213060, -504391575228),
            TickInfo::new(214170, -1000926761713858),
            TickInfo::new(214210, -419968624358816),
            TickInfo::new(216410, -102499926684047),
            TickInfo::new(217600, -470800979737),
            TickInfo::new(218060, -123988887729631),
            TickInfo::new(219600, -224082011199),
            TickInfo::new(221220, -494806614022261),
            TickInfo::new(222430, -63150),
            TickInfo::new(223340, -43059560597867),
            TickInfo::new(223370, -1142415145764782),
            TickInfo::new(225630, -35531670596),
            TickInfo::new(225700, -34337229047),
            TickInfo::new(230270, -21381469789743),
            TickInfo::new(238030, 1356881278244106196),
            TickInfo::new(238040, -1356881278244106196),
            TickInfo::new(242650, -426144530607),
            TickInfo::new(249240, -280102524730),
            TickInfo::new(260820, 158140808402),
            TickInfo::new(269390, -24250664520),
            TickInfo::new(276300, 2998029234933125),
            TickInfo::new(276320, -2987584474068099),
            TickInfo::new(292530, -2266789968),
            TickInfo::new(292560, -2273693713),
            TickInfo::new(292600, -44739669244),
            TickInfo::new(310340, -28436567813),
            TickInfo::new(344450, 3216578076775223),
            TickInfo::new(345410, -3216578076775223),
            TickInfo::new(349460, 146118690838789),
            TickInfo::new(349620, -2525400803780),
            TickInfo::new(351240, 127004648638536),
            TickInfo::new(352340, -122036674458985),
            TickInfo::new(352770, 500858985503915),
            TickInfo::new(354570, 70487808690953408),
            TickInfo::new(354900, 1021617313228699),
            TickInfo::new(356050, 13379584971254501),
            TickInfo::new(356800, -500858985503915),
            TickInfo::new(356900, 511481352845),
            TickInfo::new(357040, -13379584971254501),
            TickInfo::new(357790, -1017111148),
            TickInfo::new(357930, 1941059489935650987),
            TickInfo::new(358040, 297698757815726047),
            TickInfo::new(358080, -297698757815726047),
            TickInfo::new(359080, 971834702212488),
            TickInfo::new(359140, 4956000407814055),
            TickInfo::new(359260, -47845114297170429),
            TickInfo::new(359510, 4486512288248773),
            TickInfo::new(359660, -1021617313228699),
            TickInfo::new(359760, -22644412210070448),
            TickInfo::new(360220, 14773138965994406),
            TickInfo::new(360310, -14773138965994406),
            TickInfo::new(360450, -4956000407814055),
            TickInfo::new(360460, 8127959324508536),
            TickInfo::new(360880, -1941059489935650987),
            TickInfo::new(361090, -971834702212488),
            TickInfo::new(361300, -4486512288248773),
            TickInfo::new(361500, -103724168037440),
            TickInfo::new(361990, -8127959324508536),
            TickInfo::new(391460, 11099171960),
            TickInfo::new(407550, 1205661623580111),
            TickInfo::new(414490, -1350483406084573),
            TickInfo::new(759890, -398290794261),
            TickInfo::new(887270, -23120020447085342),
        ];

        let pool = UniswapV3State::new(
            liquidity,
            current_sqrt_price,
            FeeAmount::Low, // 0.05% fee, tick spacing = 10
            current_tick,
            ticks,
        );

        // USDC address < WETH address
        let usdc_addr = Bytes::from_str("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap();
        let weth_addr = Bytes::from_str("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2").unwrap();

        let (max_input, max_output) = pool
            .get_limits(usdc_addr.clone(), weth_addr.clone())
            .expect("get_limits should work");

        println!("Max input (USDC): {}", max_input);
        println!("Max output (WETH): {}", max_output);

        // The pool has ~2e15 liquidity at price ~0.0005 WETH per USDC (or ~2000 USDC per WETH)
        // With real pool data, reasonable limits would be:
        // - Max USDC input: on the order of 1e12 to 1e18 (millions to trillions of USDC base units)
        // - Max WETH output: on the order of 1e15 to 1e21 wei (thousands to millions of ETH)

        // The observed bug manifests as values around 1e38, which is 20+ orders of magnitude too
        // high A reasonable threshold is 1e30 - anything above this suggests the algorithm
        // is broken
        assert!(
            max_input < BigUint::from(10u128).pow(30),
            "Max input is unrealistically high: {} (should be < 1e30)",
            max_input
        );
        assert!(
            max_output < BigUint::from(10u128).pow(30),
            "Max output is unrealistically high: {} (should be < 1e30)",
            max_output
        );
    }

    #[tokio::test]
    async fn test_get_limits() {
        let project_root = env!("CARGO_MANIFEST_DIR");
        let asset_path =
            Path::new(project_root).join("tests/assets/decoder/uniswap_v3_snapshot.json");
        let json_data = fs::read_to_string(asset_path).expect("Failed to read test asset");
        let data: Value = serde_json::from_str(&json_data).expect("Failed to parse JSON");

        let state: ComponentWithState = serde_json::from_value(data)
            .expect("Expected json to match ComponentWithState structure");

        let usv3_state = UniswapV3State::try_from_with_header(
            state,
            Default::default(),
            &Default::default(),
            &Default::default(),
            &DecoderContext::new(),
        )
        .await
        .unwrap();

        let t0 = Token::new(
            &Bytes::from_str("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599").unwrap(),
            "WBTC",
            8,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let t1 = Token::new(
            &Bytes::from_str("0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf").unwrap(),
            "cbBTC",
            8,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );

        let res = usv3_state
            .get_limits(t0.address.clone(), t1.address.clone())
            .unwrap();

        assert_eq!(&res.0, &BigUint::from_u128(20358481906554983980330155).unwrap()); // Crazy amount because of this tick: "ticks/-887272/net-liquidity": "0x10d73d"

        let out = usv3_state
            .get_amount_out(res.0, &t0, &t1)
            .expect("swap for limit in didn't work");

        assert_eq!(&res.1, &out.amount);
    }
}

#[cfg(test)]
mod tests_forks {
    use std::{fs, path::Path, str::FromStr};

    use serde_json::Value;
    use tycho_client::feed::synchronizer::ComponentWithState;
    use tycho_common::models::Chain;

    use super::*;
    use crate::protocol::models::{DecoderContext, TryFromWithBlock};

    #[tokio::test]
    async fn test_pancakeswap_get_amount_out() {
        let project_root = env!("CARGO_MANIFEST_DIR");
        let asset_path =
            Path::new(project_root).join("tests/assets/decoder/pancakeswap_v3_snapshot.json");
        let json_data = fs::read_to_string(asset_path).expect("Failed to read test asset");
        let data: Value = serde_json::from_str(&json_data).expect("Failed to parse JSON");

        let state: ComponentWithState = serde_json::from_value(data)
            .expect("Expected json to match ComponentWithState structure");

        let pool_state = UniswapV3State::try_from_with_header(
            state,
            Default::default(),
            &Default::default(),
            &Default::default(),
            &DecoderContext::new(),
        )
        .await
        .unwrap();

        let usdc = Token::new(
            &Bytes::from_str("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap(),
            "USDC",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let usdt = Token::new(
            &Bytes::from_str("0xdac17f958d2ee523a2206206994597c13d831ec7").unwrap(),
            "USDT",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );

        // Swap from https://etherscan.io/tx/0x641b1e98990ae49fd00157a29e1530ff6403706b2864aa52b1c30849ce020b2c#eventlog
        let res = pool_state
            .get_amount_out(BigUint::from_str("5976361609").unwrap(), &usdt, &usdc)
            .unwrap();

        assert_eq!(res.amount, BigUint::from_str("5975901673").unwrap());
    }
}
