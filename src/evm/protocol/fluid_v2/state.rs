use std::{
    any::Any,
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use alloy::primitives::{Sign, I256, U256};
use num_bigint::{BigUint, ToBigUint};
use num_traits::Euclid;
use serde::{Deserialize, Serialize};
use thiserror::Error;
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

use crate::evm::{
    engine_db::{create_engine, SHARED_TYCHO_DB},
    protocol::{
        safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256},
        u256_num::{biguint_to_u256, u256_to_biguint, u256_to_f64},
        utils::{
            add_fee_markup,
            uniswap::{
                liquidity_math, swap_math,
                tick_list::TickList,
                tick_math::{
                    get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio, MAX_SQRT_RATIO, MAX_TICK,
                    MIN_SQRT_RATIO, MIN_TICK,
                },
                StepComputation, SwapResults, SwapState,
            },
        },
    },
};

const TOKENS_DECIMALS_PRECISION: u8 = 9;
const GAS_BASE: u64 = 155_000;
const GAS_CROSS_INIT_TICK: u64 = 21_492;
const MAX_SQRT_PRICE_CHANGE_PERCENTAGE: u64 = 2_000_000_000;
const MIN_SQRT_PRICE_CHANGE_PERCENTAGE: u64 = 5;
const DEFAULT_EXPONENT_SIZE: u32 = 8;
const DEFAULT_EXPONENT_MASK: u64 = 0xFF;
const ROUNDING_FACTOR: u64 = 1_000_000_000;
const ROUNDING_FACTOR_MINUS_ONE: u64 = ROUNDING_FACTOR - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DexType {
    D3,
    D4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FluidV2State {
    dex_id: Bytes,
    token0: Token,
    token1: Token,
    dex_type: DexType,
    fee: u32,
    dynamic_fee: bool,
    tick_spacing: i32,
    controller: Bytes,
    dex_variables: DexVariables,
    dex_variables2: DexVariables2,
    token0_reserve: U256,
    token1_reserve: U256,
    token0_borrow_exchange_price: U256,
    token0_supply_exchange_price: U256,
    token1_borrow_exchange_price: U256,
    token1_supply_exchange_price: U256,
    ticks: Vec<crate::evm::protocol::utils::uniswap::tick_list::TickInfo>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DexVariables {
    pub current_tick: i32,
    pub current_sqrt_price_x96: U256,
    pub fee_growth_global0_x102: U256,
    pub fee_growth_global1_x102: U256,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DexVariables2 {
    pub protocol_fee_0_to_1: U256,
    pub protocol_fee_1_to_0: U256,
    pub protocol_cut_fee: U256,
    pub token0_decimals: u8,
    pub token1_decimals: u8,
    pub active_liquidity: U256,
    pub pool_accounting_flag: bool,
    pub fetch_dynamic_fee_flag: bool,
    pub fee_version: U256,
    pub lp_fee: U256,
    pub max_decay_time: U256,
    pub price_impact_to_fee_division_factor: U256,
    pub min_fee: U256,
    pub max_fee: U256,
    pub net_price_impact: i64,
    pub last_update_timestamp: U256,
    pub decay_time_remaining: U256,
}

impl FluidV2State {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dex_id: Bytes,
        token0: Token,
        token1: Token,
        dex_type: DexType,
        fee: u32,
        dynamic_fee: bool,
        tick_spacing: i32,
        controller: Bytes,
        dex_variables: DexVariables,
        dex_variables2: DexVariables2,
        token0_reserve: U256,
        token1_reserve: U256,
        token0_borrow_exchange_price: U256,
        token0_supply_exchange_price: U256,
        token1_borrow_exchange_price: U256,
        token1_supply_exchange_price: U256,
        ticks: Vec<crate::evm::protocol::utils::uniswap::tick_list::TickInfo>,
    ) -> Self {
        FluidV2State {
            dex_id,
            token0,
            token1,
            dex_type,
            fee,
            dynamic_fee,
            tick_spacing,
            controller,
            dex_variables,
            dex_variables2,
            token0_reserve,
            token1_reserve,
            token0_borrow_exchange_price,
            token0_supply_exchange_price,
            token1_borrow_exchange_price,
            token1_supply_exchange_price,
            ticks,
        }
    }

    fn swap_v3(
        &self,
        ticks: &TickList,
        zero_for_one: bool,
        amount_specified: I256,
        sqrt_price_limit: U256,
        fee_pips: u32,
        mut liquidity: u128,
        mut tick: i32,
        mut sqrt_price: U256,
    ) -> Result<SwapResults, SimulationError> {
        if liquidity == 0 {
            return Err(SimulationError::RecoverableError("No liquidity".to_string()));
        }

        if zero_for_one {
            if sqrt_price_limit <= MIN_SQRT_RATIO || sqrt_price_limit >= sqrt_price {
                return Err(SimulationError::InvalidInput(
                    "Invalid sqrt price limit".to_string(),
                    None,
                ));
            }
        } else if sqrt_price_limit >= MAX_SQRT_RATIO || sqrt_price_limit <= sqrt_price {
            return Err(SimulationError::InvalidInput("Invalid sqrt price limit".to_string(), None));
        }

        let exact_input = amount_specified > I256::from_raw(U256::from(0u64));
        let mut state = SwapState {
            amount_remaining: amount_specified,
            amount_calculated: I256::from_raw(U256::from(0u64)),
            sqrt_price,
            tick,
            liquidity,
        };
        let mut gas_used = U256::from(130_000);

        while state.amount_remaining != I256::from_raw(U256::from(0u64)) &&
            state.sqrt_price != sqrt_price_limit
        {
            let (mut next_tick, initialized) = match ticks
                .next_initialized_tick_within_one_word(state.tick, zero_for_one)
            {
                Ok((tick, init)) => (tick, init),
                Err(tick_err) => match tick_err.kind {
                    crate::evm::protocol::utils::uniswap::tick_list::TickListErrorKind::TicksExeeded => {
                        let mut new_state = self.clone();
                        new_state.dex_variables2.active_liquidity =
                            U256::from(state.liquidity);
                        new_state.dex_variables.current_tick = state.tick;
                        new_state.dex_variables.current_sqrt_price_x96 = state.sqrt_price;
                        return Err(SimulationError::InvalidInput(
                            "Ticks exceeded".into(),
                            Some(GetAmountOutResult::new(
                                u256_to_biguint(state.amount_calculated.abs().into_raw()),
                                u256_to_biguint(gas_used),
                                Box::new(new_state),
                            )),
                        ));
                    }
                    _ => {
                        return Err(SimulationError::FatalError(
                            "Unknown tick list error".to_string(),
                        ));
                    }
                },
            };

            next_tick = next_tick.clamp(MIN_TICK, MAX_TICK);
            let sqrt_price_next = get_sqrt_ratio_at_tick(next_tick)?;
            let (sqrt_price_next, amount_in, amount_out, fee_amount) =
                swap_math::compute_swap_step(
                    state.sqrt_price,
                    FluidV2State::get_sqrt_ratio_target(
                        sqrt_price_next,
                        sqrt_price_limit,
                        zero_for_one,
                    ),
                    state.liquidity,
                    state.amount_remaining,
                    fee_pips,
                )?;

            let step = StepComputation {
                sqrt_price_start: state.sqrt_price,
                tick_next: next_tick,
                initialized,
                sqrt_price_next,
                amount_in,
                amount_out,
                fee_amount,
            };

            state.sqrt_price = sqrt_price_next;
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
                    let liquidity_raw = ticks
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
            amount_specified,
            amount_remaining: state.amount_remaining,
            sqrt_price: state.sqrt_price,
            liquidity: state.liquidity,
            tick: state.tick,
            gas_used,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn swap_d3(
        &self,
        ticks: &TickList,
        zero_for_one: bool,
        amount_specified: I256,
        sqrt_price_limit: U256,
        protocol_fee_pips: u32,
        lp_fee_pips: u32,
        mut liquidity: u128,
        mut tick: i32,
        mut sqrt_price: U256,
    ) -> Result<SwapResults, SimulationError> {
        if liquidity == 0 {
            return Err(SimulationError::RecoverableError("No liquidity".to_string()));
        }

        if zero_for_one {
            if sqrt_price_limit <= MIN_SQRT_RATIO || sqrt_price_limit >= sqrt_price {
                return Err(SimulationError::InvalidInput(
                    "Invalid sqrt price limit".to_string(),
                    None,
                ));
            }
        } else if sqrt_price_limit >= MAX_SQRT_RATIO || sqrt_price_limit <= sqrt_price {
            return Err(SimulationError::InvalidInput("Invalid sqrt price limit".to_string(), None));
        }

        let exact_input = amount_specified > I256::from_raw(U256::from(0u64));
        let mut state = SwapState {
            amount_remaining: amount_specified,
            amount_calculated: I256::from_raw(U256::from(0u64)),
            sqrt_price,
            tick,
            liquidity,
        };
        let mut cross_init_tick_loops: u64 = 0;

        let protocol_cut_fee = self
            .dex_variables2
            .protocol_cut_fee
            .to::<u32>();
        let fee_version = self
            .dex_variables2
            .fee_version
            .to::<u32>();
        let mut constant_lp_fee_pips = lp_fee_pips;

        // TODO: fee_version = 1 dynamic fee variables + controller override fetch.
        if fee_version == 1 {
            if self
                .dex_variables2
                .price_impact_to_fee_division_factor ==
                U256::from(0u64)
            {
                constant_lp_fee_pips = self.dex_variables2.min_fee.to::<u32>();
            } else {
                // TODO: dynamic fee path. For now, fall back to minFee to keep simulation running.
                constant_lp_fee_pips = self.dex_variables2.min_fee.to::<u32>();
            }
        }
        // TODO: if fetchDynamicFeeFlag is on and controller is not swapper,
        // fetchDynamicFeeForSwapIn should override constant_lp_fee_pips.

        let active_liquidity_start = state.liquidity;
        let mut sqrt_price_start = state.sqrt_price;
        let mut sqrt_price_start_changed = false;

        while state.amount_remaining != I256::from_raw(U256::from(0u64)) &&
            state.sqrt_price != sqrt_price_limit
        {
            let (mut next_tick, initialized) = ticks
                .next_initialized_tick_within_one_word(state.tick, zero_for_one)
                .map_err(|err| match err.kind {
                    crate::evm::protocol::utils::uniswap::tick_list::TickListErrorKind::TicksExeeded => {
                        SimulationError::InvalidInput("Ticks exceeded".to_string(), None)
                    }
                    _ => SimulationError::FatalError("Unknown tick list error".to_string()),
                })?;

            next_tick = next_tick.clamp(MIN_TICK, MAX_TICK);
            let sqrt_price_next = get_sqrt_ratio_at_tick(next_tick)?;
            if (zero_for_one && sqrt_price_next > state.sqrt_price) ||
                (!zero_for_one && sqrt_price_next < state.sqrt_price)
            {
                let diff = if sqrt_price_next > state.sqrt_price {
                    safe_sub_u256(sqrt_price_next, state.sqrt_price)?
                } else {
                    safe_sub_u256(state.sqrt_price, sqrt_price_next)?
                };
                if safe_mul_u256(diff, four_decimals())? > state.sqrt_price {
                    return Err(SimulationError::InvalidInput(
                        "Sqrt price deviation too high".to_string(),
                        None,
                    ));
                }
                state.sqrt_price = sqrt_price_next;
            }
            let (sqrt_price_next, amount_in, amount_out, _fee_amount) =
                swap_math::compute_swap_step(
                    state.sqrt_price,
                    FluidV2State::get_sqrt_ratio_target(
                        sqrt_price_next,
                        sqrt_price_limit,
                        zero_for_one,
                    ),
                    state.liquidity,
                    state.amount_remaining,
                    0,
                )?;

            let step = StepComputation {
                sqrt_price_start: state.sqrt_price,
                tick_next: next_tick,
                initialized,
                sqrt_price_next,
                amount_in,
                amount_out,
                fee_amount: U256::from(0u64),
            };

            state.sqrt_price = sqrt_price_next;
            if exact_input {
                let mut step_amount_out = step.amount_out;
                let mut step_protocol_fee = U256::from(0u64);
                let mut step_lp_fee = U256::from(0u64);

                if step_amount_out > U256::from(0u64) && protocol_fee_pips > 0 {
                    let numerator = safe_add_u256(
                        safe_mul_u256(step_amount_out, U256::from(protocol_fee_pips))?,
                        U256::from(1u64),
                    )?;
                    step_protocol_fee = safe_add_u256(
                        safe_div_u256(numerator, U256::from(1_000_000u64))?,
                        U256::from(1u64),
                    )?;
                }
                if step_amount_out > step_protocol_fee {
                    step_amount_out = safe_sub_u256(step_amount_out, step_protocol_fee)?;
                } else {
                    step_amount_out = U256::from(0u64);
                }

                if step_amount_out > U256::from(0u64) && constant_lp_fee_pips > 0 {
                    let numerator = safe_add_u256(
                        safe_mul_u256(step_amount_out, U256::from(constant_lp_fee_pips))?,
                        U256::from(1u64),
                    )?;
                    step_lp_fee = safe_add_u256(
                        safe_div_u256(numerator, U256::from(1_000_000u64))?,
                        U256::from(1u64),
                    )?;
                }
                if step_amount_out > step_lp_fee {
                    step_amount_out = safe_sub_u256(step_amount_out, step_lp_fee)?;
                } else {
                    step_amount_out = U256::from(0u64);
                }

                if step_lp_fee > U256::from(0u64) && protocol_cut_fee > 0 {
                    let numerator = safe_add_u256(
                        safe_mul_u256(step_lp_fee, U256::from(protocol_cut_fee))?,
                        U256::from(1u64),
                    )?;
                    let step_protocol_cut = safe_add_u256(
                        safe_div_u256(numerator, U256::from(100u64))?,
                        U256::from(1u64),
                    )?;
                    if step_lp_fee > step_protocol_cut {
                        step_lp_fee = safe_sub_u256(step_lp_fee, step_protocol_cut)?;
                        step_protocol_fee = safe_add_u256(step_protocol_fee, step_protocol_cut)?;
                    } else {
                        step_protocol_fee = safe_add_u256(step_protocol_fee, step_lp_fee)?;
                        step_lp_fee = U256::from(0u64);
                    }
                }

                state.amount_remaining -=
                    I256::checked_from_sign_and_abs(Sign::Positive, step.amount_in).unwrap();
                state.amount_calculated -=
                    I256::checked_from_sign_and_abs(Sign::Positive, step_amount_out).unwrap();
            } else {
                let amount_in_with_lp = safe_div_u256(
                    safe_mul_u256(step.amount_in, U256::from(1_000_000u64))?,
                    safe_sub_u256(U256::from(1_000_000u64), U256::from(constant_lp_fee_pips))?,
                )?;
                let amount_in_with_fees = safe_div_u256(
                    safe_mul_u256(amount_in_with_lp, U256::from(1_000_000u64))?,
                    safe_sub_u256(U256::from(1_000_000u64), U256::from(protocol_fee_pips))?,
                )?;

                state.amount_remaining +=
                    I256::checked_from_sign_and_abs(Sign::Positive, step.amount_out).unwrap();
                state.amount_calculated +=
                    I256::checked_from_sign_and_abs(Sign::Positive, amount_in_with_fees).unwrap();
            }

            // TODO: update feeGrowthGlobal and feeGrowthOutside on ticks.

            if state.sqrt_price == step.sqrt_price_next {
                if step.initialized {
                    let liquidity_raw = ticks
                        .get_tick(step.tick_next)
                        .unwrap()
                        .net_liquidity;
                    let liquidity_net = if zero_for_one { -liquidity_raw } else { liquidity_raw };
                    state.liquidity =
                        liquidity_math::add_liquidity_delta(state.liquidity, liquidity_net)?;
                    cross_init_tick_loops += 1;

                    if active_liquidity_start == 0 &&
                        !sqrt_price_start_changed &&
                        state.liquidity > 0
                    {
                        sqrt_price_start = state.sqrt_price;
                        sqrt_price_start_changed = true;
                    }
                }
                state.tick = if zero_for_one { step.tick_next - 1 } else { step.tick_next };
            } else if state.sqrt_price != step.sqrt_price_start {
                state.tick = get_tick_at_sqrt_ratio(state.sqrt_price)?;
            }
        }

        verify_sqrt_price_change_limits(sqrt_price_start, state.sqrt_price)?;
        // TODO: if fee_version == 1, update dynamic fee variables based on final price impact.

        let gas_used = U256::from(GAS_BASE) +
            U256::from(GAS_CROSS_INIT_TICK.saturating_mul(cross_init_tick_loops));

        Ok(SwapResults {
            amount_calculated: state.amount_calculated,
            amount_specified,
            amount_remaining: state.amount_remaining,
            sqrt_price: state.sqrt_price,
            liquidity: state.liquidity,
            tick: state.tick,
            gas_used,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn swap_d4(
        &self,
        _ticks: &TickList,
        _zero_for_one: bool,
        _amount_specified: I256,
        _sqrt_price_limit: U256,
        _fee_pips: u32,
        _liquidity: u128,
        _tick: i32,
        _sqrt_price: U256,
    ) -> Result<SwapResults, SimulationError> {
        Err(SimulationError::InvalidInput("D4 swap algo not implemented".to_string(), None))
    }

    fn get_sqrt_ratio_target(
        sqrt_price_next: U256,
        sqrt_price_limit: U256,
        zero_for_one: bool,
    ) -> U256 {
        let cond = if zero_for_one {
            sqrt_price_next < sqrt_price_limit
        } else {
            sqrt_price_next > sqrt_price_limit
        };
        if cond {
            sqrt_price_limit
        } else {
            sqrt_price_next
        }
    }
}

impl DexVariables {
    pub fn from_packed(value: U256) -> Self {
        let sign_is_positive = (value & U256::from(1u64)) == U256::from(1u64);
        let abs_tick = ((value >> 1u32) & u256_mask(19)).to::<u32>();
        let current_tick = if abs_tick == 0 {
            0
        } else if sign_is_positive {
            abs_tick as i32
        } else {
            -(abs_tick as i32)
        };

        let current_sqrt_price_x96 = (value >> 20u32) & u256_mask(72);
        let fee_growth_global0_x102 = (value >> 92u32) & u256_mask(82);
        let fee_growth_global1_x102 = (value >> 174u32) & u256_mask(82);

        Self {
            current_tick,
            current_sqrt_price_x96,
            fee_growth_global0_x102,
            fee_growth_global1_x102,
        }
    }
}

impl DexVariables2 {
    pub fn from_packed(value: U256) -> Self {
        let protocol_fee_0_to_1 = value & u256_mask(12);
        let protocol_fee_1_to_0 = (value >> 12u32) & u256_mask(12);
        let protocol_cut_fee = (value >> 24u32) & u256_mask(6);

        let token0_decimals_raw = ((value >> 30u32) & u256_mask(4)).to::<u8>();
        let token1_decimals_raw = ((value >> 34u32) & u256_mask(4)).to::<u8>();
        let token0_decimals = if token0_decimals_raw == 15 { 18 } else { token0_decimals_raw };
        let token1_decimals = if token1_decimals_raw == 15 { 18 } else { token1_decimals_raw };

        let active_liquidity = (value >> 38u32) & u256_mask(102);
        let pool_accounting_flag = ((value >> 140u32) & U256::from(1u64)) == U256::from(1u64);
        let fetch_dynamic_fee_flag = ((value >> 141u32) & U256::from(1u64)) == U256::from(1u64);
        let fee_version = (value >> 152u32) & u256_mask(4);

        let lp_fee = (value >> 156u32) & u256_mask(16);
        let max_decay_time = (value >> 156u32) & u256_mask(12);
        let price_impact_to_fee_division_factor = (value >> 168u32) & u256_mask(8);
        let min_fee = (value >> 176u32) & u256_mask(16);
        let max_fee = (value >> 192u32) & u256_mask(16);

        let net_price_impact_sign = ((value >> 208u32) & U256::from(1u64)) == U256::from(1u64);
        let net_price_impact_abs = ((value >> 209u32) & u256_mask(20)).to::<u32>() as i64;
        let net_price_impact = if net_price_impact_abs == 0 {
            0
        } else if net_price_impact_sign {
            net_price_impact_abs
        } else {
            -net_price_impact_abs
        };

        let last_update_timestamp = (value >> 229u32) & u256_mask(15);
        let decay_time_remaining = (value >> 244u32) & u256_mask(12);

        Self {
            protocol_fee_0_to_1,
            protocol_fee_1_to_0,
            protocol_cut_fee,
            token0_decimals,
            token1_decimals,
            active_liquidity,
            pool_accounting_flag,
            fetch_dynamic_fee_flag,
            fee_version,
            lp_fee,
            max_decay_time,
            price_impact_to_fee_division_factor,
            min_fee,
            max_fee,
            net_price_impact,
            last_update_timestamp,
            decay_time_remaining,
        }
    }
}

fn u256_mask(bits: u32) -> U256 {
    if bits == 0 {
        return U256::from(0u64);
    }
    (U256::from(1u64) << bits) - U256::from(1u64)
}

fn pow10_u256(exp: u8) -> U256 {
    U256::from(10u64).pow(U256::from(exp as u64))
}

fn exchange_price_precision() -> U256 {
    pow10_u256(12)
}

fn calculate_precisions(decimals: u8) -> (U256, U256) {
    if decimals > TOKENS_DECIMALS_PRECISION {
        (U256::from(1u64), pow10_u256(decimals - TOKENS_DECIMALS_PRECISION))
    } else {
        (pow10_u256(TOKENS_DECIMALS_PRECISION - decimals), U256::from(1u64))
    }
}

fn amount_to_adjusted(
    amount: U256,
    token_numerator_precision: U256,
    token_denominator_precision: U256,
    exchange_price: U256,
) -> Result<U256, SimulationError> {
    let scaled = safe_mul_u256(amount, exchange_price_precision())?;
    let scaled = safe_mul_u256(scaled, token_numerator_precision)?;
    let denominator = safe_mul_u256(exchange_price, token_denominator_precision)?;
    safe_div_u256(scaled, denominator)
}

fn adjusted_to_amount(
    adjusted: U256,
    token_numerator_precision: U256,
    token_denominator_precision: U256,
    exchange_price: U256,
) -> Result<U256, SimulationError> {
    let scaled = safe_mul_u256(adjusted, token_denominator_precision)?;
    let scaled = safe_mul_u256(scaled, exchange_price)?;
    let denominator = safe_mul_u256(token_numerator_precision, exchange_price_precision())?;
    safe_div_u256(scaled, denominator)
}

fn four_decimals() -> U256 {
    pow10_u256(4)
}

fn ten_decimals() -> U256 {
    pow10_u256(10)
}

fn rounding_factor() -> U256 {
    U256::from(ROUNDING_FACTOR)
}

fn rounding_factor_minus_one() -> U256 {
    U256::from(ROUNDING_FACTOR_MINUS_ONE)
}

fn round_down_raw(amount: U256) -> U256 {
    let scaled = amount.saturating_mul(rounding_factor_minus_one());
    let mut out = if ROUNDING_FACTOR == 0 { amount } else { scaled / rounding_factor() };
    if out > U256::from(0u64) {
        out = out.saturating_sub(U256::from(1u64));
    }
    out
}

fn decode_sqrt_price(raw: U256, swap0_to_1: bool) -> U256 {
    let mut coeff = raw >> DEFAULT_EXPONENT_SIZE;
    let exp = (raw & U256::from(DEFAULT_EXPONENT_MASK)).to::<u32>();
    if exp > 0 && !swap0_to_1 {
        coeff = coeff + U256::from(1u64);
    }
    coeff << exp
}

fn encode_big_number(value: U256, coeff_bits: u32) -> U256 {
    if value.is_zero() {
        return U256::from(0u64);
    }
    let max_coeff = (U256::from(1u64) << coeff_bits) - U256::from(1u64);
    let mut coeff = value;
    let mut exp: u32 = 0;
    while coeff > max_coeff && exp < 255 {
        coeff >>= 1u32;
        exp += 1;
    }
    (coeff << DEFAULT_EXPONENT_SIZE) | U256::from(exp as u64)
}

fn max_x(bits: u32) -> U256 {
    (U256::from(1u64) << bits) - U256::from(1u64)
}

fn verify_amount_limits(amount: U256) -> Result<(), SimulationError> {
    if amount < four_decimals() || amount > max_x(128) {
        return Err(SimulationError::InvalidInput("Amount out of limits".to_string(), None));
    }
    Ok(())
}

fn verify_adjusted_amount_limits(amount: U256) -> Result<(), SimulationError> {
    if amount < four_decimals() || amount > max_x(86) {
        return Err(SimulationError::InvalidInput(
            "Adjusted amount out of limits".to_string(),
            None,
        ));
    }
    Ok(())
}

fn verify_sqrt_price_change_limits(
    sqrt_price_start: U256,
    sqrt_price_end: U256,
) -> Result<(), SimulationError> {
    let diff = if sqrt_price_end > sqrt_price_start {
        safe_sub_u256(sqrt_price_end, sqrt_price_start)?
    } else {
        safe_sub_u256(sqrt_price_start, sqrt_price_end)?
    };
    let percentage_change = safe_div_u256(safe_mul_u256(diff, ten_decimals())?, sqrt_price_start)?;
    if percentage_change > U256::from(MAX_SQRT_PRICE_CHANGE_PERCENTAGE) ||
        percentage_change < U256::from(MIN_SQRT_PRICE_CHANGE_PERCENTAGE)
    {
        return Err(SimulationError::InvalidInput(
            "Sqrt price change out of bounds".to_string(),
            None,
        ));
    }
    Ok(())
}

fn is_nonzero_bytes(bytes: &Bytes) -> bool {
    bytes.as_ref().iter().any(|b| *b != 0)
}

#[typetag::serde]
impl ProtocolSim for FluidV2State {
    fn fee(&self) -> f64 {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        todo!()
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        if amount_in == BigUint::from(0u32) {
            return Ok(GetAmountOutResult::new(
                BigUint::from(0u32),
                BigUint::from(0u32),
                Box::new(self.clone()),
            ));
        }

        if is_nonzero_bytes(&self.controller) {
            return Err(SimulationError::InvalidInput(
                "Controller pools are not supported".to_string(),
                None,
            ));
        }

        let zero_for_one = if token_in.address == self.token0.address {
            true
        } else if token_in.address == self.token1.address {
            false
        } else {
            return Err(SimulationError::InvalidInput(
                "Token in not part of pool".to_string(),
                None,
            ));
        };

        if !(token_out.address == self.token0.address || token_out.address == self.token1.address) {
            return Err(SimulationError::InvalidInput(
                "Token out not part of pool".to_string(),
                None,
            ));
        }

        let amount_in_u256 = biguint_to_u256(&amount_in);
        verify_amount_limits(amount_in_u256)?;

        let (token_in_decimals, token_out_decimals) = if zero_for_one {
            (self.dex_variables2.token0_decimals, self.dex_variables2.token1_decimals)
        } else {
            (self.dex_variables2.token1_decimals, self.dex_variables2.token0_decimals)
        };

        let (token_in_num_precision, token_in_den_precision) =
            calculate_precisions(token_in_decimals);
        let (token_out_num_precision, token_out_den_precision) =
            calculate_precisions(token_out_decimals);

        let (exchange_price_in, exchange_price_out) = if zero_for_one {
            match self.dex_type {
                DexType::D3 => {
                    (self.token0_supply_exchange_price, self.token1_supply_exchange_price)
                }
                DexType::D4 => {
                    (self.token0_borrow_exchange_price, self.token1_borrow_exchange_price)
                }
            }
        } else {
            match self.dex_type {
                DexType::D3 => {
                    (self.token1_supply_exchange_price, self.token0_supply_exchange_price)
                }
                DexType::D4 => {
                    (self.token1_borrow_exchange_price, self.token0_borrow_exchange_price)
                }
            }
        };

        let amount_in_adjusted = amount_to_adjusted(
            amount_in_u256,
            token_in_num_precision,
            token_in_den_precision,
            exchange_price_in,
        )?;
        let amount_in_adjusted = round_down_raw(amount_in_adjusted);
        verify_adjusted_amount_limits(amount_in_adjusted)?;

        if self.tick_spacing <= 0 {
            return Err(SimulationError::FatalError("Tick spacing must be positive".to_string()));
        }
        let liquidity = self
            .dex_variables2
            .active_liquidity
            .to::<u128>();
        let tick_list = TickList::from(self.tick_spacing as u16, self.ticks.clone())?;

        let (tick_min, tick_max) = match (self.ticks.first(), self.ticks.last()) {
            (Some(min), Some(max)) => (min.index, max.index),
            _ => {
                return Err(SimulationError::FatalError("No ticks available".to_string()));
            }
        };
        let tick_limit = if zero_for_one { tick_min } else { tick_max };
        let mut sqrt_price_limit = get_sqrt_ratio_at_tick(tick_limit)?;
        sqrt_price_limit = if zero_for_one {
            safe_add_u256(sqrt_price_limit, U256::from(1u64))?
        } else {
            safe_sub_u256(sqrt_price_limit, U256::from(1u64))?
        };

        let amount_specified = I256::checked_from_sign_and_abs(Sign::Positive, amount_in_adjusted)
            .ok_or_else(|| {
                SimulationError::InvalidInput("I256 overflow: amount_in".to_string(), None)
            })?;

        let (protocol_fee_pips, lp_fee_pips) = if zero_for_one {
            (
                self.dex_variables2
                    .protocol_fee_0_to_1
                    .to::<u32>(),
                self.dex_variables2.lp_fee.to::<u32>(),
            )
        } else {
            (
                self.dex_variables2
                    .protocol_fee_1_to_0
                    .to::<u32>(),
                self.dex_variables2.lp_fee.to::<u32>(),
            )
        };

        let result = match self.dex_type {
            DexType::D3 => self.swap_d3(
                &tick_list,
                zero_for_one,
                amount_specified,
                sqrt_price_limit,
                protocol_fee_pips,
                lp_fee_pips,
                liquidity,
                self.dex_variables.current_tick,
                decode_sqrt_price(
                    self.dex_variables
                        .current_sqrt_price_x96,
                    zero_for_one,
                ),
            )?,
            DexType::D4 => self.swap_d4(
                &tick_list,
                zero_for_one,
                amount_specified,
                sqrt_price_limit,
                lp_fee_pips,
                liquidity,
                self.dex_variables.current_tick,
                decode_sqrt_price(
                    self.dex_variables
                        .current_sqrt_price_x96,
                    zero_for_one,
                ),
            )?,
        };

        if result.amount_remaining > I256::from_raw(U256::from(0u64)) {
            return Err(SimulationError::InvalidInput("Next tick out of bounds".to_string(), None));
        }

        if matches!(self.dex_type, DexType::D4) {
            verify_sqrt_price_change_limits(
                decode_sqrt_price(
                    self.dex_variables
                        .current_sqrt_price_x96,
                    zero_for_one,
                ),
                result.sqrt_price,
            )?;
        }

        let amount_out_raw_adjusted = result
            .amount_calculated
            .abs()
            .into_raw();
        let amount_out_raw_adjusted = round_down_raw(amount_out_raw_adjusted);
        verify_adjusted_amount_limits(amount_out_raw_adjusted)?;
        let mut amount_out = adjusted_to_amount(
            amount_out_raw_adjusted,
            token_out_num_precision,
            token_out_den_precision,
            exchange_price_out,
        )?;
        if amount_out > U256::from(0u64) {
            amount_out = safe_sub_u256(amount_out, U256::from(1u64))?;
        }
        verify_amount_limits(amount_out)?;

        let pool_accounting_on = !self.dex_variables2.pool_accounting_flag;
        let mut new_state = self.clone();
        new_state.dex_variables.current_tick = result.tick;
        new_state
            .dex_variables
            .current_sqrt_price_x96 = encode_big_number(result.sqrt_price, 64);
        new_state
            .dex_variables2
            .active_liquidity = U256::from(result.liquidity);

        if pool_accounting_on {
            let reserve_limit = max_x(128);
            if zero_for_one {
                let reserve_in = safe_add_u256(self.token0_reserve, amount_in_adjusted)?;
                if reserve_in > reserve_limit {
                    return Err(SimulationError::InvalidInput(
                        "Token reserves overflow".to_string(),
                        None,
                    ));
                }
                if self.token1_reserve < amount_out_raw_adjusted {
                    return Err(SimulationError::InvalidInput(
                        "Token reserves underflow".to_string(),
                        None,
                    ));
                }
                new_state.token0_reserve = reserve_in;
                new_state.token1_reserve =
                    safe_sub_u256(self.token1_reserve, amount_out_raw_adjusted)?;
            } else {
                let reserve_in = safe_add_u256(self.token1_reserve, amount_in_adjusted)?;
                if reserve_in > reserve_limit {
                    return Err(SimulationError::InvalidInput(
                        "Token reserves overflow".to_string(),
                        None,
                    ));
                }
                if self.token0_reserve < amount_out_raw_adjusted {
                    return Err(SimulationError::InvalidInput(
                        "Token reserves underflow".to_string(),
                        None,
                    ));
                }
                new_state.token1_reserve = reserve_in;
                new_state.token0_reserve =
                    safe_sub_u256(self.token0_reserve, amount_out_raw_adjusted)?;
            }
        }

        Ok(GetAmountOutResult::new(
            u256_to_biguint(amount_out),
            u256_to_biguint(result.gas_used),
            Box::new(new_state),
        ))
    }

    fn get_limits(
        &self,
        sell_token: Bytes,
        buy_token: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        todo!()
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        tokens: &HashMap<Bytes, Token>,
        balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn ProtocolSim> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        todo!()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        todo!()
    }

    fn eq(&self, other: &dyn ProtocolSim) -> bool {
        todo!()
    }
}
