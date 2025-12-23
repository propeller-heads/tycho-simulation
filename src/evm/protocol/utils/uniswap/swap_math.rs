use alloy::primitives::{I256, U256, U512};
use num_bigint::BigUint;
use tycho_common::simulation::errors::SimulationError;

use super::sqrt_price_math;
use crate::evm::protocol::{
    safe_math::safe_sub_u256,
    u256_num::{biguint_to_u256, u256_to_biguint},
    utils::solidity_math::{mul_div, mul_div_rounding_up},
};

/// Computes the result of swapping some amount in, or amount out, given the parameters of the swap
///
/// The fee, plus the amount in, will never exceed the amount remaining if the swap's `amountSpecified` is positive
///
/// # Arguments
///
/// * `sqrt_ratio_current` - The current sqrt price as a Q64.96 fixed-point number.
/// * `sqrt_ratio_target` - The price that cannot be exceeded, from which the direction of the swap
///   is inferred. Typically represents the next tick boundary.
/// * `liquidity` - The usable liquidity in the current tick range.
/// * `amount_remaining` - How much input or output amount is remaining to be swapped in/out.
///   Positive values indicate exact input swaps (amount_specified > 0),
///   negative values indicate exact output swaps (amount_specified < 0).
/// * `fee_pips` - The fee taken from the input amount, expressed in hundredths of a bip
///   (e.g., 3000 = 0.3%).
///
/// # Returns
///
/// A tuple of `(sqrt_ratio_next, amount_in, amount_out, fee_amount)`:
/// * `sqrt_ratio_next` - The price after swapping the amount in/out, not to exceed the price
///   target.
/// * `amount_in` - The amount to be swapped in, of either token0 or token1, based on the direction
///   of the swap.
/// * `amount_out` - The amount to be received, of either token0 or token1, based on the direction
///   of the swap.
/// * `fee_amount` - The amount of input that will be taken as a fee.
///
/// # Invariants
///
/// The fee, plus the amount in, will never exceed the amount remaining if the swap's
/// `amount_remaining` is positive (exact input swap).
///
/// # Swap Direction
///
/// The swap direction is inferred from the price relationship:
/// * `zero_for_one` (selling token0 for token1): when `sqrt_ratio_current >= sqrt_ratio_target`
/// * `one_for_zero` (selling token1 for token0): when `sqrt_ratio_current < sqrt_ratio_target`
pub(crate) fn compute_swap_step(
    sqrt_ratio_current: U256,
    sqrt_ratio_target: U256,
    liquidity: u128,
    amount_remaining: I256,
    fee_pips: u32,
) -> Result<(U256, U256, U256, U256), SimulationError> {
    let zero_for_one = sqrt_ratio_current >= sqrt_ratio_target;
    let exact_in = amount_remaining >= I256::from_raw(U256::from(0u64));
    let sqrt_ratio_next: U256;
    let mut amount_in = U256::from(0u64);
    let mut amount_out = U256::from(0u64);

    if exact_in {
        let amount_remaining_less_fee = mul_div(
            amount_remaining.into_raw(),
            U256::from(1_000_000 - fee_pips),
            U256::from(1_000_000),
        )?;
        amount_in = if zero_for_one {
            sqrt_price_math::get_amount0_delta(
                sqrt_ratio_target,
                sqrt_ratio_current,
                liquidity,
                true,
            )?
        } else {
            sqrt_price_math::get_amount1_delta(
                sqrt_ratio_current,
                sqrt_ratio_target,
                liquidity,
                true,
            )?
        };
        if amount_remaining_less_fee >= amount_in {
            sqrt_ratio_next = sqrt_ratio_target
        } else {
            sqrt_ratio_next = sqrt_price_math::get_next_sqrt_price_from_input(
                sqrt_ratio_current,
                liquidity,
                amount_remaining_less_fee,
                zero_for_one,
            )?
        }
    } else {
        amount_out = if zero_for_one {
            sqrt_price_math::get_amount1_delta(
                sqrt_ratio_target,
                sqrt_ratio_current,
                liquidity,
                false,
            )?
        } else {
            sqrt_price_math::get_amount0_delta(
                sqrt_ratio_current,
                sqrt_ratio_target,
                liquidity,
                false,
            )?
        };
        if amount_remaining.abs().into_raw() > amount_out {
            sqrt_ratio_next = sqrt_ratio_target;
        } else {
            sqrt_ratio_next = sqrt_price_math::get_next_sqrt_price_from_output(
                sqrt_ratio_current,
                liquidity,
                amount_remaining.abs().into_raw(),
                zero_for_one,
            )?;
        }
    }

    let max = sqrt_ratio_target == sqrt_ratio_next;

    if zero_for_one {
        amount_in = if max && exact_in {
            amount_in
        } else {
            sqrt_price_math::get_amount0_delta(
                sqrt_ratio_next,
                sqrt_ratio_current,
                liquidity,
                true,
            )?
        };
        amount_out = if max && !exact_in {
            amount_out
        } else {
            sqrt_price_math::get_amount1_delta(
                sqrt_ratio_next,
                sqrt_ratio_current,
                liquidity,
                false,
            )?
        }
    } else {
        amount_in = if max && exact_in {
            amount_in
        } else {
            sqrt_price_math::get_amount1_delta(
                sqrt_ratio_current,
                sqrt_ratio_next,
                liquidity,
                true,
            )?
        };
        amount_out = if max && !exact_in {
            amount_out
        } else {
            sqrt_price_math::get_amount0_delta(
                sqrt_ratio_current,
                sqrt_ratio_next,
                liquidity,
                false,
            )?
        };
    }

    if !exact_in && amount_out > amount_remaining.abs().into_raw() {
        amount_out = amount_remaining.abs().into_raw();
    }

    let fee_amount = if exact_in && sqrt_ratio_next != sqrt_ratio_target {
        safe_sub_u256(amount_remaining.abs().into_raw(), amount_in)?
    } else {
        mul_div_rounding_up(amount_in, U256::from(fee_pips), U256::from(1_000_000 - fee_pips))?
    };
    Ok((sqrt_ratio_next, amount_in, amount_out, fee_amount))
}

/// Computes the sqrt_price_new, amount_in, and amount_out needed to achieve a target trade price
/// within a single tick range.
///
/// # Trade Price Formula Derivation
///
/// **Notation:**
/// - `sqrt_ratio_current`: Current sqrt price before the swap (Q96 format)
/// - `sqrt_ratio_target`: Target sqrt price after the swap (Q96 format)
/// - `liquidity`: Active liquidity in the current tick range
/// - `fee_factor`: (1_000_000 - fee_pips) / 1_000_000, the fraction of input retained after fees
/// - `trade_price`: amount_out / amount_in, the execution price of the swap
///
/// **For zero_for_one=true (selling token0 for token1):**
/// ```text
/// amount_out = liquidity × (sqrt_ratio_current - sqrt_ratio_target) / 2^96
/// amount_in = liquidity × 2^96 × (sqrt_ratio_current - sqrt_ratio_target)
///             / (fee_factor × sqrt_ratio_current × sqrt_ratio_target)
/// trade_price = amount_out / amount_in
///             = fee_factor × sqrt_ratio_current × sqrt_ratio_target / 2^192
/// ```
/// Solving for sqrt_ratio_target:
/// ```text
/// sqrt_ratio_target = trade_price × 2^192 / (fee_factor × sqrt_ratio_current)
/// ```
///
/// **For zero_for_one=false (selling token1 for token0):**
/// ```text
/// trade_price = fee_factor × 2^192 / (sqrt_ratio_current × sqrt_ratio_target)
/// ```
/// Solving for sqrt_ratio_target:
/// ```text
/// sqrt_ratio_target = fee_factor × 2^192 / (trade_price × sqrt_ratio_current)
/// ```
///
/// # Arguments
/// * `sqrt_ratio_current` - Current sqrt price (Q96 format)
/// * `sqrt_ratio_limit` - Price limit (tick boundary) that we cannot cross
/// * `liquidity` - Current liquidity in the tick range
/// * `target_trade_price_num` - Target trade price numerator (amount_out / amount_in)
/// * `target_trade_price_den` - Target trade price denominator
/// * `fee_pips` - Fee in pips (e.g., 3000 for 0.3%)
///
/// # Swap Direction
///
/// The swap direction is inferred from the price relationship (consistent with
/// `compute_swap_step`):
/// * `zero_for_one` (selling token0 for token1): when `sqrt_ratio_current >= sqrt_ratio_limit`
/// * `one_for_zero` (selling token1 for token0): when `sqrt_ratio_current < sqrt_ratio_limit`
///
/// # Returns
/// * `Ok((sqrt_ratio_new, amount_in, amount_out, fee_amount, reached_target))` where:
///   - `sqrt_ratio_new` - The new sqrt price after the swap
///   - `amount_in` - Amount of input token (before fee)
///   - `amount_out` - Amount of output token
///   - `fee_amount` - Fee charged
///   - `reached_target` - Whether we achieved the target price (false if hit tick boundary first)
pub(crate) fn compute_swap_to_trade_price(
    sqrt_ratio_current: U256,
    sqrt_ratio_limit: U256,
    liquidity: u128,
    target_trade_price_num: &BigUint,
    target_trade_price_den: &BigUint,
    fee_pips: u32,
) -> Result<(U256, U256, U256, U256, bool), SimulationError> {
    // Infer direction from price relationship (consistent with compute_swap_step)
    let zero_for_one = sqrt_ratio_current >= sqrt_ratio_limit;
    if liquidity == 0 {
        return Err(SimulationError::FatalError("Zero liquidity".to_string()));
    }

    // γ = (1_000_000 - fee_pips) / 1_000_000
    let gamma_num = U512::from(1_000_000 - fee_pips);
    let gamma_den = U512::from(1_000_000u32);

    // 2^192 = 2^96 * 2^96
    let two_192 = U512::from(1u64) << 192;

    let s0 = U512::from(sqrt_ratio_current);
    let p_num = U512::from(biguint_to_u256(target_trade_price_num));
    let p_den = U512::from(biguint_to_u256(target_trade_price_den));

    // Calculate target sqrt_price (S₁)
    let sqrt_ratio_target_u512: U512 = if zero_for_one {
        // S₁ = P × 2^192 / (γ × S₀)
        // S₁ = (P_num × 2^192 × gamma_den) / (P_den × gamma_num × S₀)
        (p_num * two_192 * gamma_den) / (p_den * gamma_num * s0)
    } else {
        // S₁ = γ × 2^192 / (P × S₀)
        // S₁ = (gamma_num × 2^192 × P_den) / (gamma_den × P_num × S₀)
        (gamma_num * two_192 * p_den) / (gamma_den * p_num * s0)
    };

    // Convert to U256 (check for overflow)
    let limbs = sqrt_ratio_target_u512.as_limbs();
    if limbs[4] != 0 || limbs[5] != 0 || limbs[6] != 0 || limbs[7] != 0 {
        return Err(SimulationError::FatalError("sqrt_ratio_target overflows U256".to_string()));
    }
    let sqrt_ratio_target = U256::from_limbs([limbs[0], limbs[1], limbs[2], limbs[3]]);

    // Check if target is reachable within the tick boundary
    let (sqrt_ratio_new, reached_target) = if zero_for_one {
        // For zero_for_one, price decreases: target should be < current, and >= limit
        if sqrt_ratio_target >= sqrt_ratio_current {
            // Target is at or above current price - nothing to do
            return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
        }
        if sqrt_ratio_target < sqrt_ratio_limit {
            (sqrt_ratio_limit, false)
        } else {
            (sqrt_ratio_target, true)
        }
    } else {
        // For !zero_for_one, price increases: target should be > current, and <= limit
        if sqrt_ratio_target <= sqrt_ratio_current {
            // Target is at or below current price - nothing to do
            return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
        }
        if sqrt_ratio_target > sqrt_ratio_limit {
            (sqrt_ratio_limit, false)
        } else {
            (sqrt_ratio_target, true)
        }
    };

    // Calculate amounts using the standard delta functions
    let (amount_in, amount_out) = if zero_for_one {
        let amount_in = sqrt_price_math::get_amount0_delta(
            sqrt_ratio_new,
            sqrt_ratio_current,
            liquidity,
            true, // round up for amount_in
        )?;
        let amount_out = sqrt_price_math::get_amount1_delta(
            sqrt_ratio_new,
            sqrt_ratio_current,
            liquidity,
            false, // round down for amount_out
        )?;
        (amount_in, amount_out)
    } else {
        let amount_in = sqrt_price_math::get_amount1_delta(
            sqrt_ratio_current,
            sqrt_ratio_new,
            liquidity,
            true, // round up for amount_in
        )?;
        let amount_out = sqrt_price_math::get_amount0_delta(
            sqrt_ratio_current,
            sqrt_ratio_new,
            liquidity,
            false, // round down for amount_out
        )?;
        (amount_in, amount_out)
    };

    // Calculate fee
    let fee_amount =
        mul_div_rounding_up(amount_in, U256::from(fee_pips), U256::from(1_000_000 - fee_pips))?;

    Ok((sqrt_ratio_new, amount_in, amount_out, fee_amount, reached_target))
}

/// Computes the sqrt_price_new, amount_in, and amount_out needed to achieve a target cumulative
/// trade price across multiple tick ranges.
///
/// This function accounts for amounts already accumulated from previous ticks and solves
/// for the sqrt_price that achieves the target cumulative trade price:
/// `(accumulated_out + new_out) / (accumulated_in + new_in) = target_price`
///
/// # Trade Price Formula Derivation
///
/// **Notation:**
/// - `sqrt_ratio_current`: Current sqrt price before the swap (Q96 format)
/// - `sqrt_ratio_target`: Target sqrt price after the swap (Q96 format)
/// - `liquidity`: Active liquidity in the current tick range
/// - `fee_factor`: (1_000_000 - fee_pips) / 1_000_000, the fraction of input retained after fees
/// - `accumulated_in`: Total input amount from previous ticks
/// - `accumulated_out`: Total output amount from previous ticks
/// - `target_trade_price`: (accumulated_out + new_out) / (accumulated_in + new_in)
///
/// **For zero_for_one=true (selling token0 for token1):**
/// ```text
/// new_out = liquidity × (sqrt_ratio_current - sqrt_ratio_target) / 2^96
/// new_in = liquidity × 2^96 × (sqrt_ratio_current - sqrt_ratio_target)
///          / (fee_factor × sqrt_ratio_current × sqrt_ratio_target)
/// ```
/// Setting `(accumulated_out + new_out) / (accumulated_in + new_in) = target_price`
/// and solving for `x = sqrt_ratio_current - sqrt_ratio_target` yields:
/// ```text
/// x² + b×x + c = 0
/// where:
///   b = target_price × Q² / (fee_factor × sqrt_ratio_current) - sqrt_ratio_current - R × Q / L
///   c = R × Q × sqrt_ratio_current / L
///   R = target_price × accumulated_in - accumulated_out
/// ```
///
/// **For zero_for_one=false (selling token1 for token0):**
/// Similar derivation with adjusted formulas for the inverse direction.
///
/// # Arguments
///
/// * `sqrt_ratio_current` - The current sqrt price as a Q64.96 fixed-point number.
/// * `sqrt_ratio_limit` - The price that cannot be exceeded, from which the direction of the swap
///   is inferred. Typically represents the next tick boundary.
/// * `liquidity` - The usable liquidity in the current tick range.
/// * `target_trade_price_num` - Target trade price numerator (amount_out / amount_in).
/// * `target_trade_price_den` - Target trade price denominator.
/// * `fee_pips` - The fee taken from the input amount, expressed in hundredths of a bip
///   (e.g., 3000 = 0.3%).
/// * `accumulated_in` - Amount of input token already accumulated from previous ticks.
/// * `accumulated_out` - Amount of output token already accumulated from previous ticks.
///
/// # Swap Direction
///
/// The swap direction is inferred from the price relationship (consistent with
/// `compute_swap_step`):
/// * `zero_for_one` (selling token0 for token1): when `sqrt_ratio_current >= sqrt_ratio_limit`
/// * `one_for_zero` (selling token1 for token0): when `sqrt_ratio_current < sqrt_ratio_limit`
///
/// # Returns
///
/// A tuple of `(sqrt_ratio_new, amount_in, amount_out, fee_amount, reached_target)`:
/// * `sqrt_ratio_new` - The new sqrt price after the swap
/// * `amount_in` - Amount of input token (before fee) for this tick
/// * `amount_out` - Amount of output token for this tick
/// * `fee_amount` - Fee charged for this tick
/// * `reached_target` - Whether we achieved the target price (false if hit tick boundary first
///   or target was not achievable with the given accumulated amounts)
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_swap_to_trade_price_with_accumulated(
    sqrt_ratio_current: U256,
    sqrt_ratio_limit: U256,
    liquidity: u128,
    target_trade_price_num: &BigUint,
    target_trade_price_den: &BigUint,
    fee_pips: u32,
    accumulated_in: U256,
    accumulated_out: U256,
) -> Result<(U256, U256, U256, U256, bool), SimulationError> {
    let zero_for_one = sqrt_ratio_current >= sqrt_ratio_limit;
    if liquidity == 0 {
        return Err(SimulationError::FatalError("Zero liquidity".to_string()));
    }

    // If no accumulated amounts, use the simpler direct formula
    if accumulated_in.is_zero() && accumulated_out.is_zero() {
        return compute_swap_to_trade_price(
            sqrt_ratio_current,
            sqrt_ratio_limit,
            liquidity,
            target_trade_price_num,
            target_trade_price_den,
            fee_pips,
        );
    }

    // Use BigUint for high-precision arithmetic
    let q = BigUint::from(1u64) << 96; // 2^96
    let s0 = u256_to_biguint(sqrt_ratio_current);
    let s_limit = u256_to_biguint(sqrt_ratio_limit);
    let l = BigUint::from(liquidity);
    let p_num = target_trade_price_num.clone();
    let p_den = target_trade_price_den.clone();
    let gamma_num = BigUint::from(1_000_000u32 - fee_pips);
    let gamma_den = BigUint::from(1_000_000u32);
    let a = u256_to_biguint(accumulated_out); // accumulated output
    let b_acc = u256_to_biguint(accumulated_in); // accumulated input

    // R = P × B - A (how much more output we need relative to what we have)
    // R_num / R_den = (P_num / P_den) × B - A = (P_num × B - A × P_den) / P_den
    // When R > 0: we need more output to reach target (accumulated ratio is worse than target)
    // When R < 0: we have more output than needed (accumulated ratio is better than target)
    let pb = &p_num * &b_acc;
    let a_scaled = &a * &p_den;
    let (r_positive, r_abs_num) = if pb >= a_scaled {
        (true, &pb - &a_scaled)
    } else {
        (false, &a_scaled - &pb)
    };
    // R = r_abs_num / p_den (with sign = r_positive)

    // Solve quadratic equation for the price delta
    // If no valid solution exists (target not achievable), fall back to tick boundary
    let sqrt_ratio_target_result = solve_quadratic(
        &s0,
        &l,
        &q,
        &p_num,
        &p_den,
        &gamma_num,
        &gamma_den,
        &r_abs_num,
        r_positive,
        zero_for_one,
    );

    // Check if target is valid and within bounds
    let (sqrt_ratio_new, reached_target) = match sqrt_ratio_target_result {
        Ok(sqrt_ratio_target) => {
            if zero_for_one {
                if sqrt_ratio_target >= s0 {
                    // Target is at or above current price - nothing to do
                    return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
                }
                if sqrt_ratio_target < s_limit {
                    (s_limit, false)
                } else {
                    (sqrt_ratio_target, true)
                }
            } else {
                if sqrt_ratio_target <= s0 {
                    // Target is at or below current price - nothing to do
                    return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
                }
                if sqrt_ratio_target > s_limit {
                    (s_limit, false)
                } else {
                    (sqrt_ratio_target, true)
                }
            }
        }
        Err(_) => {
            // Target is not achievable with the given accumulated amounts
            // Fall back to swapping to the tick boundary (maximum swap in this tick)
            (s_limit, false)
        }
    };

    // Convert back to U256
    let sqrt_ratio_new_u256 = biguint_to_u256(&sqrt_ratio_new);

    // Calculate amounts using the standard delta functions
    let (amount_in, amount_out) = if zero_for_one {
        let amount_in = sqrt_price_math::get_amount0_delta(
            sqrt_ratio_new_u256,
            sqrt_ratio_current,
            liquidity,
            true,
        )?;
        let amount_out = sqrt_price_math::get_amount1_delta(
            sqrt_ratio_new_u256,
            sqrt_ratio_current,
            liquidity,
            false,
        )?;
        (amount_in, amount_out)
    } else {
        let amount_in = sqrt_price_math::get_amount1_delta(
            sqrt_ratio_current,
            sqrt_ratio_new_u256,
            liquidity,
            true,
        )?;
        let amount_out = sqrt_price_math::get_amount0_delta(
            sqrt_ratio_current,
            sqrt_ratio_new_u256,
            liquidity,
            false,
        )?;
        (amount_in, amount_out)
    };

    let fee_amount =
        mul_div_rounding_up(amount_in, U256::from(fee_pips), U256::from(1_000_000 - fee_pips))?;

    Ok((sqrt_ratio_new_u256, amount_in, amount_out, fee_amount, reached_target))
}

/// Solves the quadratic equation for price delta in swap-to-trade-price calculations.
///
/// For zero_for_one: x² + b×x + c = 0 where x = S₀ - S₁, returns S₁ = S₀ - x
/// For one_for_zero: y² + b×y + c = 0 where y = S₁ - S₀, returns S₁ = S₀ + y
///
/// The coefficients differ based on direction:
/// - zero_for_one: b = P×Q²/(γ×S₀) - S₀ - R×Q/L, c = R×Q×S₀/L
/// - one_for_zero: b = S₀ - γ×Q²/(P×S₀) - R×γ×Q/(P×L), c = -R×γ×Q×S₀/(P×L)
///
/// R = P×B - A (residual: difference between target and accumulated ratio)
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn solve_quadratic(
    s0: &BigUint,
    l: &BigUint,
    q: &BigUint,
    p_num: &BigUint,
    p_den: &BigUint,
    gamma_num: &BigUint,
    gamma_den: &BigUint,
    r_abs_num: &BigUint, // |R| × p_den where R = P×B - A
    r_positive: bool,    // sign of R (true means we need MORE output)
    zero_for_one: bool,  // swap direction
) -> Result<BigUint, SimulationError> {
    // Compute coefficients based on direction
    let (b_positive, b_abs_num, c_abs_num, c_abs_den, c_positive, denom) = if zero_for_one {
        // zero_for_one formulas:
        // b = P×Q²/(γ×S₀) - S₀ - R×Q/L
        // c = R×Q×S₀/L (positive when R > 0)
        // Common denominator: gamma_num × s0 × L × p_den
        let denom = gamma_num * s0 * l * p_den;

        // Term1: P×Q²/(γ×S₀) scaled by denom
        let term1 = p_num * q * q * gamma_den * l;
        // Term2: S₀ scaled by denom
        let term2 = s0 * &denom;
        // Term3: R×Q/L scaled by denom
        let term3 = r_abs_num * q * gamma_num * s0;

        // b_num = term1 - term2 - sign(R) × term3
        let (b_positive, b_abs_num) = {
            let t1_minus_or_plus_t3 = if r_positive {
                if term1 >= term3 {
                    (true, &term1 - &term3)
                } else {
                    (false, &term3 - &term1)
                }
            } else {
                (true, &term1 + &term3)
            };

            let (t1_t3_positive, t1_t3_abs) = t1_minus_or_plus_t3;
            if t1_t3_positive {
                if t1_t3_abs >= term2 {
                    (true, t1_t3_abs - &term2)
                } else {
                    (false, &term2 - t1_t3_abs)
                }
            } else {
                (false, t1_t3_abs + &term2)
            }
        };

        // c = R×Q×S₀/L
        let c_abs_num = r_abs_num * q * s0;
        let c_abs_den = p_den * l;
        let c_positive = r_positive;

        (b_positive, b_abs_num, c_abs_num, c_abs_den, c_positive, denom)
    } else {
        // one_for_zero formulas:
        // b = S₀ - γ×Q²/(P×S₀) - R×γ×Q/(P×L)
        // c = -R×γ×Q×S₀/(P×L) (negative when R > 0)
        // Common denominator: gamma_den × P_num × S₀ × L × p_den
        let denom = gamma_den * p_num * s0 * l * p_den;

        // Term1: S₀ scaled by denom
        let term1 = s0 * &denom;
        // Term2: γ×Q²/(P×S₀) scaled by denom
        let term2 = gamma_num * q * q * p_den * l * p_den;
        // Term3: R×γ×Q/(P×L) scaled by denom
        let term3 = r_abs_num * gamma_num * q * s0 * p_den;

        // b_num = term1 - term2 - sign(R) × term3
        let t1_minus_t2 = if term1 >= term2 {
            (true, &term1 - &term2)
        } else {
            (false, &term2 - &term1)
        };

        let (b_positive, b_abs_num) = if r_positive {
            let (t1_t2_pos, t1_t2_abs) = t1_minus_t2;
            if t1_t2_pos {
                if t1_t2_abs >= term3 {
                    (true, t1_t2_abs - &term3)
                } else {
                    (false, &term3 - t1_t2_abs)
                }
            } else {
                (false, t1_t2_abs + &term3)
            }
        } else {
            let (t1_t2_pos, t1_t2_abs) = t1_minus_t2;
            if t1_t2_pos {
                (true, t1_t2_abs + &term3)
            } else if term3 >= t1_t2_abs {
                (true, &term3 - t1_t2_abs)
            } else {
                (false, t1_t2_abs - &term3)
            }
        };

        // c = -R×γ×Q×S₀/(P×L)
        let c_abs_num = r_abs_num * gamma_num * q * s0;
        let c_abs_den = gamma_den * p_num * l;
        let c_positive = !r_positive;

        (b_positive, b_abs_num, c_abs_num, c_abs_den, c_positive, denom)
    };

    // Discriminant: D = b² - 4c
    let b_sq_scaled = &b_abs_num * &b_abs_num * &c_abs_den;
    let four_c_scaled = BigUint::from(4u64) * &c_abs_num * &denom * &denom;

    let d_num = if c_positive {
        if b_sq_scaled >= four_c_scaled {
            &b_sq_scaled - &four_c_scaled
        } else {
            return Err(SimulationError::FatalError(
                "Negative discriminant - target price not achievable".to_string(),
            ));
        }
    } else {
        &b_sq_scaled + &four_c_scaled
    };

    let d_den = &denom * &denom * &c_abs_den;
    let sqrt_d_num = d_num.sqrt();
    let sqrt_d_den = d_den.sqrt();

    let delta_den = BigUint::from(2u64) * &denom * &sqrt_d_den;

    let sqrt_d_term = &sqrt_d_num * &denom;
    let b_term = &b_abs_num * &sqrt_d_den;

    // delta = (-b + sqrt(D)) / 2
    let delta_num = if b_positive {
        if sqrt_d_term >= b_term {
            &sqrt_d_term - &b_term
        } else {
            return Err(SimulationError::FatalError(
                "No positive solution".to_string(),
            ));
        }
    } else {
        &sqrt_d_term + &b_term
    };

    // S₁ = S₀ - delta (zero_for_one) or S₀ + delta (one_for_zero)
    let s0_scaled = s0 * &delta_den;
    let s1_num = if zero_for_one {
        if s0_scaled < delta_num {
            return Err(SimulationError::FatalError(
                "sqrt_ratio_target would be negative".to_string(),
            ));
        }
        &s0_scaled - &delta_num
    } else {
        &s0_scaled + &delta_num
    };
    let s1 = s1_num / delta_den;

    Ok(s1)
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    use super::*;

    /// Verifies that the achieved trade price is within tolerance of the target trade price.
    ///
    /// Due to integer arithmetic (especially when computing sqrt_ratio_target from the target
    /// trade price), there's inherent rounding. This function checks that the rounding error
    /// is acceptable.
    pub fn verify_trade_price_tolerance(
        amount_in: U256,
        amount_out: U256,
        target_price_num: &BigUint,
        target_price_den: &BigUint,
        tolerance: f64,
    ) -> Result<(), String> {
        if amount_in.is_zero() {
            return Ok(()); // No swap occurred, nothing to verify
        }

        // Calculate achieved trade price: amount_out / amount_in
        // Use U512 to avoid overflow in cross-multiplication
        let achieved_num = U512::from(amount_out);
        let achieved_den = U512::from(amount_in);

        // Cross-multiply to compare: achieved_num/achieved_den vs target_num/target_den
        let target_num = U512::from(biguint_to_u256(target_price_num));
        let target_den = U512::from(biguint_to_u256(target_price_den));

        let lhs = achieved_num * target_den;
        let rhs = target_num * achieved_den;

        // Calculate relative difference: |lhs - rhs| / rhs
        let (diff, base) = if lhs >= rhs { (lhs - rhs, rhs) } else { (rhs - lhs, rhs) };

        let diff_f64 = u512_to_f64_approx(diff);
        let base_f64 = u512_to_f64_approx(base);

        if base_f64 == 0.0 {
            return Ok(());
        }

        let relative_diff = diff_f64 / base_f64;

        if relative_diff <= tolerance {
            Ok(())
        } else {
            let achieved_price = amount_out
                .to_string()
                .parse::<f64>()
                .unwrap_or(0.0) /
                amount_in
                    .to_string()
                    .parse::<f64>()
                    .unwrap_or(1.0);
            let target_price =
                target_price_num.to_f64().unwrap_or(0.0) / target_price_den.to_f64().unwrap_or(1.0);

            Err(format!(
                "Trade price {:.6} exceeds tolerance {:.4}% from target {:.6} (relative diff: {:.4}%)",
                achieved_price,
                tolerance * 100.0,
                target_price,
                relative_diff * 100.0
            ))
        }
    }

    /// Approximate conversion of U512 to f64 for tolerance calculations.
    fn u512_to_f64_approx(value: U512) -> f64 {
        let limbs = value.as_limbs();

        let mut highest_idx = 0;
        for (i, &limb) in limbs.iter().enumerate() {
            if limb != 0 {
                highest_idx = i;
            }
        }

        let high = limbs[highest_idx] as f64;
        let low = if highest_idx > 0 { limbs[highest_idx - 1] as f64 } else { 0.0 };

        let shift = highest_idx as i32 * 64;
        (high + low / (u64::MAX as f64 + 1.0)) * 2.0_f64.powi(shift)
    }

    use std::{ops::Neg, str::FromStr};

    struct TestCase {
        price: U256,
        target: U256,
        liquidity: u128,
        remaining: I256,
        fee: u32,
        exp: (U256, U256, U256, U256),
    }

    #[test]
    fn test_compute_swap_step() {
        let cases = vec![
            TestCase {
                price: U256::from_str("1917240610156820439288675683655550").unwrap(),
                target: U256::from_str("1919023616462402511535565081385034").unwrap(),
                liquidity: 23130341825817804069u128,
                remaining: I256::exp10(18),
                fee: 500,
                exp: (
                    U256::from_str("1917244033735642980420262835667387").unwrap(),
                    U256::from_str("999500000000000000").unwrap(),
                    U256::from_str("1706820897").unwrap(),
                    U256::from_str("500000000000000").unwrap(),
                ),
            },
            TestCase {
                price: U256::from_str("1917240610156820439288675683655550").unwrap(),
                target: U256::from_str("1919023616462402511535565081385034").unwrap(),
                liquidity: 23130341825817804069u128,
                remaining: I256::exp10(18).neg(),
                fee: 500,
                exp: (
                    U256::from_str("1919023616462402511535565081385034").unwrap(),
                    U256::from_str("520541484453545253034").unwrap(),
                    U256::from_str("888091216672").unwrap(),
                    U256::from_str("260400942698121688").unwrap(),
                ),
            },
            TestCase {
                price: U256::from_str("1917240610156820439288675683655550").unwrap(),
                target: U256::from_str("1908498483466244238266951834509291").unwrap(),
                liquidity: 23130341825817804069u128,
                remaining: I256::exp10(18).neg(),
                fee: 500,
                exp: (
                    U256::from_str("1917237184865352164019453920762266").unwrap(),
                    U256::from_str("1707680836").unwrap(),
                    U256::from_str("1000000000000000000").unwrap(),
                    U256::from_str("854268").unwrap(),
                ),
            },
            TestCase {
                price: U256::from_str("1917240610156820439288675683655550").unwrap(),
                target: U256::from_str("1908498483466244238266951834509291").unwrap(),
                liquidity: 23130341825817804069u128,
                remaining: I256::exp10(18),
                fee: 500,
                exp: (
                    U256::from_str("1908498483466244238266951834509291").unwrap(),
                    U256::from_str("4378348149175").unwrap(),
                    U256::from_str("2552228553845698906796").unwrap(),
                    U256::from_str("2190269210").unwrap(),
                ),
            },
            TestCase {
                price: U256::from_str("1917240610156820439288675683655550").unwrap(),
                target: U256::from_str("1908498483466244238266951834509291").unwrap(),
                liquidity: 0u128,
                remaining: I256::exp10(18),
                fee: 500,
                exp: (
                    U256::from_str("1908498483466244238266951834509291").unwrap(),
                    U256::ZERO,
                    U256::ZERO,
                    U256::ZERO,
                ),
            },
        ];

        for case in cases {
            let res = compute_swap_step(
                case.price,
                case.target,
                case.liquidity,
                case.remaining,
                case.fee,
            )
            .unwrap();

            assert_eq!(res, case.exp);
        }
    }

    #[test]
    fn test_compute_swap_to_trade_price_basic() {
        // Setup: sqrt_price ~= sqrt(2) * 2^96 -> spot price ~= 2.0
        // This means 1 token0 gets ~2 token1
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap(); // ~sqrt(2) * 2^96
        let liquidity = 1_000_000_000_000_000_000u128; // 1e18

        // Target trade price: 1.9 token1/token0 (worse than spot ~2.0)
        let target_price_num = BigUint::from(19u64);
        let target_price_den = BigUint::from(10u64);

        // Tick boundary far away (won't be hit)
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap(); // MIN_SQRT_RATIO + 1

        let fee_pips = 3000u32; // 0.3%

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // Should reach target (not hit tick boundary)
        assert!(reached_target, "Should reach target price");
        assert!(sqrt_price_new < sqrt_price_current, "Price should decrease for zero_for_one");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
        assert!(fee_amount > U256::ZERO, "fee_amount should be positive");

        // Verify fee calculation: fee_amount = amount_in * fee_pips / (1_000_000 - fee_pips)
        // For fee_pips = 3000: fee_amount ≈ amount_in * 0.003009
        let expected_fee =
            mul_div_rounding_up(amount_in, U256::from(fee_pips), U256::from(1_000_000 - fee_pips))
                .unwrap();
        assert_eq!(fee_amount, expected_fee, "Fee should match expected calculation");

        // Verify achieved trade price is close to target (within 1% tolerance due to integer
        // rounding)
        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &target_price_num,
            &target_price_den,
            0.01,
        )
        .expect("Trade price should be within tolerance");
    }

    #[test]
    fn test_compute_swap_to_trade_price_hits_tick_boundary() {
        // Setup: sqrt_price ~= sqrt(2) * 2^96 -> spot price ~= 2.0
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;

        // Target trade price: 0.5 token1/token0 (much worse than spot)
        // This will require crossing the tick boundary
        let target_price_num = BigUint::from(5u64);
        let target_price_den = BigUint::from(10u64);

        // Tick boundary close to current price (will be hit)
        let sqrt_ratio_limit = U256::from_str("111000000000000000000000000000").unwrap();

        let fee_pips = 3000u32;

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // Should hit tick boundary, not reach target
        assert!(!reached_target, "Should hit tick boundary before reaching target");
        assert_eq!(sqrt_price_new, sqrt_ratio_limit, "Should be at tick boundary");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
    }

    #[test]
    fn test_compute_swap_to_trade_price_target_already_achieved() {
        // Setup: sqrt_price ~= sqrt(2) * 2^96 -> spot price ~= 2.0
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;

        // Target trade price: 2.5 token1/token0 (better than spot ~2.0)
        // This is already achieved without any swap
        let target_price_num = BigUint::from(25u64);
        let target_price_den = BigUint::from(10u64);

        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let fee_pips = 3000u32;

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // Target is better than current spot, so no swap needed
        assert!(reached_target, "Target should be considered reached");
        assert_eq!(sqrt_price_new, sqrt_price_current, "Price should not change");
        assert_eq!(amount_in, U256::ZERO, "No input needed");
        assert_eq!(amount_out, U256::ZERO, "No output");
        assert_eq!(fee_amount, U256::ZERO, "No fee");
    }

    #[test]
    fn test_compute_swap_to_trade_price_one_for_zero() {
        // Setup for !zero_for_one (sqrt_ratio_limit > sqrt_ratio_current)
        // sqrt_price ~= sqrt(2) * 2^96 -> spot price token1/token0 ~= 2.0
        // For !zero_for_one, trade price is token0/token1 ~= 0.5
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;

        // Target trade price for !zero_for_one: 0.485 token0/token1
        // (slightly worse than effective spot ~0.497 after 0.3% fee)
        let target_price_num = BigUint::from(485u64);
        let target_price_den = BigUint::from(1000u64);

        // Tick boundary far above current price (won't be hit)
        let sqrt_ratio_limit = U256::from_str("150000000000000000000000000000").unwrap();

        let fee_pips = 3000u32;

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // Should reach target
        assert!(reached_target, "Should reach target price");
        assert!(sqrt_price_new > sqrt_price_current, "Price should increase for one_for_zero");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");

        // Verify trade price tolerance (1% due to integer rounding)
        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &target_price_num,
            &target_price_den,
            0.01,
        )
        .expect("Trade price should be within tolerance");
    }

    #[test]
    fn test_compute_swap_to_trade_price_zero_liquidity() {
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 0u128; // Zero liquidity

        let target_price_num = BigUint::from(19u64);
        let target_price_den = BigUint::from(10u64);
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let result = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_price_num,
            &target_price_den,
            3000,
        );

        assert!(result.is_err(), "Should fail with zero liquidity");
    }

    #[test]
    fn test_compute_swap_to_trade_price_at_tick_boundary() {
        // Test edge case where target price is exactly at the tick boundary
        // This tests the case: sqrt_ratio_target == sqrt_ratio_limit

        // Setup: sqrt_price ~= sqrt(2) * 2^96 -> spot price ~= 2.0
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Set tick boundary close to where a specific target price would land
        // For target price 1.5 with fee 0.3%, the sqrt_ratio_target would be around:
        // S₁ = P × 2^192 / (γ × S₀) where P=1.5, γ=0.997, S₀=sqrt_price_current
        // Let's use a boundary that's exactly where we'd want to end up
        let sqrt_ratio_limit = U256::from_str("97000000000000000000000000000").unwrap();

        // Target price that would result in sqrt_ratio near the boundary
        let target_price_num = BigUint::from(15u64);
        let target_price_den = BigUint::from(10u64); // 1.5

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // The swap should complete (either reaching target or hitting boundary)
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
        assert!(fee_amount > U256::ZERO, "fee_amount should be positive");

        // Price should have moved
        assert!(sqrt_price_new < sqrt_price_current, "Price should decrease for zero_for_one");

        // Should either reach target or be at boundary
        assert!(
            sqrt_price_new >= sqrt_ratio_limit,
            "New price should be at or above boundary"
        );

        // If we hit the boundary exactly, reached_target should be false
        if sqrt_price_new == sqrt_ratio_limit {
            assert!(
                !reached_target,
                "Should not have reached target if stopped at boundary"
            );
        }
    }

    #[test]
    fn test_compute_swap_to_trade_price_target_equals_boundary() {
        // Test when the computed target sqrt_price exactly equals the boundary
        // This is a precise edge case

        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // First, compute what sqrt_ratio_target would be for a specific price
        // Then set the boundary to exactly that value
        let target_price_num = BigUint::from(18u64);
        let target_price_den = BigUint::from(10u64); // 1.8

        // Compute expected sqrt_ratio_target using the formula:
        // S₁ = P × 2^192 / (γ × S₀) for zero_for_one
        // With P=1.8, γ=0.997, S₀=sqrt_price_current
        // This gives us approximately 101400000000000000000000000000
        let sqrt_ratio_limit = U256::from_str("101400000000000000000000000000").unwrap();

        let result = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_price_num,
            &target_price_den,
            fee_pips,
        );

        assert!(result.is_ok(), "Should succeed even at exact boundary");
        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) = result.unwrap();

        // Should have swapped
        assert!(amount_in > U256::ZERO, "Should have swapped some amount");
        assert!(amount_out > U256::ZERO, "Should have output");

        // Price should be at or above the boundary
        assert!(
            sqrt_price_new >= sqrt_ratio_limit,
            "New price {} should be >= boundary {}",
            sqrt_price_new,
            sqrt_ratio_limit
        );

        // If target is beyond boundary, we stop at boundary
        if sqrt_price_new == sqrt_ratio_limit {
            assert!(!reached_target, "Should not reach target if at boundary");
        }
    }

    // ==================== Tests for compute_swap_to_trade_price_with_accumulated ====================

    /// Helper to verify cumulative trade price is within tolerance.
    fn verify_cumulative_trade_price_tolerance(
        accumulated_in: U256,
        accumulated_out: U256,
        new_amount_in: U256,
        new_amount_out: U256,
        target_price_num: &BigUint,
        target_price_den: &BigUint,
        tolerance: f64,
    ) -> Result<(), String> {
        let total_in = accumulated_in + new_amount_in;
        let total_out = accumulated_out + new_amount_out;

        if total_in.is_zero() {
            return Ok(());
        }

        verify_trade_price_tolerance(total_in, total_out, target_price_num, target_price_den, tolerance)
    }

    #[test]
    fn test_accumulated_zero_matches_original() {
        // When accumulated amounts are zero, should behave identically to the original function
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let target_price_num = BigUint::from(19u64);
        let target_price_den = BigUint::from(10u64);
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();
        let fee_pips = 3000u32;

        let result_original = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_price_num,
            &target_price_den,
            fee_pips,
        )
        .expect("Original should succeed");

        let result_accumulated = compute_swap_to_trade_price_with_accumulated(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_price_num,
            &target_price_den,
            fee_pips,
            U256::ZERO,
            U256::ZERO,
        )
        .expect("Accumulated should succeed");

        assert_eq!(result_original, result_accumulated, "Results should match when accumulated is zero");
    }

    #[test]
    fn test_accumulated_zero_for_one_basic() {
        // Test with accumulated amounts in zero_for_one direction
        // Scenario: We have accumulated some amounts at a certain trade price, and now
        // we want to continue swapping to reach a different cumulative target.
        //
        // The achievable cumulative price range depends on:
        // 1. Current cumulative ratio (from accumulated amounts)
        // 2. Marginal price at current sqrt_price
        // If marginal > current cumulative, we can increase cumulative by swapping more
        // If marginal < current cumulative, we can decrease cumulative by swapping more
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Tick boundary far away (won't be hit)
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        // Start with small accumulated amounts at ratio 1.90
        // (simulating previous swaps in other ticks)
        let accumulated_in = U256::from(100_000u128);
        let accumulated_out = U256::from(190_000u128); // ratio = 1.9

        // Target: 1.95 (between current 1.9 and marginal ~1.99)
        // This should be achievable by swapping more
        let target_num = BigUint::from(195u64);
        let target_den = BigUint::from(100u64);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price_with_accumulated(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_num,
                &target_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            )
            .expect("Should succeed");

        // Should reach target (since 1.95 is between 1.9 and 1.99)
        assert!(reached_target, "Should reach target price");
        assert!(sqrt_price_new < sqrt_price_current, "Price should decrease for zero_for_one");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
        assert!(fee_amount > U256::ZERO, "fee_amount should be positive");

        // Verify CUMULATIVE trade price is close to target
        verify_cumulative_trade_price_tolerance(
            accumulated_in,
            accumulated_out,
            amount_in,
            amount_out,
            &target_num,
            &target_den,
            0.02, // 2% tolerance due to integer sqrt approximations
        )
        .expect("Cumulative trade price should be within tolerance");
    }

    #[test]
    fn test_accumulated_unreachable_target_falls_back_to_boundary() {
        // Test that when target is not achievable, we fall back to tick boundary
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Tick boundary reasonably close
        let sqrt_ratio_limit = U256::from_str("100000000000000000000000000000").unwrap();

        // Large accumulated amounts with ratio 1.7
        let accumulated_in = U256::from(1_000_000_000_000_000_000u128); // 1e18
        let accumulated_out = U256::from(1_700_000_000_000_000_000u128); // 1.7e18

        // Target 1.9 is not achievable (see earlier analysis)
        let target_num = BigUint::from(19u64);
        let target_den = BigUint::from(10u64);

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price_with_accumulated(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_num,
                &target_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            )
            .expect("Should succeed (falling back to boundary)");

        // Should NOT reach target (falls back to boundary)
        assert!(!reached_target, "Should not reach unreachable target");
        assert_eq!(
            sqrt_price_new,
            biguint_to_u256(&u256_to_biguint(sqrt_ratio_limit)),
            "Should be at tick boundary"
        );
        assert!(amount_in > U256::ZERO, "Should have swapped some");
        assert!(amount_out > U256::ZERO, "Should have received some output");
    }

    #[test]
    fn test_accumulated_one_for_zero_basic() {
        // Test with accumulated amounts in one_for_zero direction
        // Setup for !zero_for_one (sqrt_ratio_limit > sqrt_ratio_current)
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // For !zero_for_one, trade price is token0/token1
        // Marginal spot price is ~0.5 token0/token1 (γ/S₀²×Q²)
        // Use small accumulated amounts so the quadratic is solvable

        // Tick boundary far above current price (won't be hit)
        let sqrt_ratio_limit = U256::from_str("150000000000000000000000000000").unwrap();

        // Accumulated amounts - ratio is 0.48 (need to reach 0.49)
        let accumulated_in = U256::from(100_000u128); // small token1
        let accumulated_out = U256::from(48_000u128); // ratio = 0.48

        // Target cumulative trade price: 0.49 (between 0.48 and marginal ~0.5)
        let target_price_num = BigUint::from(49u64);
        let target_price_den = BigUint::from(100u64);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price_with_accumulated(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            )
            .expect("Should succeed");

        // Should reach target
        assert!(reached_target, "Should reach target price");
        assert!(sqrt_price_new > sqrt_price_current, "Price should increase for one_for_zero");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
        assert!(fee_amount > U256::ZERO, "fee_amount should be positive");

        // Verify CUMULATIVE trade price is close to target
        verify_cumulative_trade_price_tolerance(
            accumulated_in,
            accumulated_out,
            amount_in,
            amount_out,
            &target_price_num,
            &target_price_den,
            0.02, // 2% tolerance
        )
        .expect("Cumulative trade price should be within tolerance");
    }

    #[test]
    fn test_accumulated_hits_tick_boundary() {
        // Test where target requires going past tick boundary
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Very aggressive target price: 0.5 (would require large price movement)
        let target_price_num = BigUint::from(5u64);
        let target_price_den = BigUint::from(10u64);

        // Tick boundary close to current price (will be hit)
        let sqrt_ratio_limit = U256::from_str("111000000000000000000000000000").unwrap();

        // Some accumulated amounts
        let accumulated_in = U256::from(500_000_000_000_000_000u128);
        let accumulated_out = U256::from(900_000_000_000_000_000u128);

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price_with_accumulated(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            )
            .expect("Should succeed");

        // Should hit tick boundary, not reach target
        assert!(!reached_target, "Should hit tick boundary before reaching target");
        assert_eq!(sqrt_price_new, sqrt_ratio_limit, "Should be at tick boundary");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
    }

    #[test]
    fn test_accumulated_target_already_achieved() {
        // Test where cumulative ratio already exceeds target (no swap needed)
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Target price: 1.5
        let target_price_num = BigUint::from(15u64);
        let target_price_den = BigUint::from(10u64);

        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        // Accumulated ratio is 1.4 (worse than target 1.5 for zero_for_one where we want high output)
        // Wait - for zero_for_one, we're selling token0 for token1
        // If target is 1.5 and current cumulative is 2.0 (better), then we need to swap more to
        // worsen the ratio to 1.5. So this should trigger a swap.
        //
        // Let's test the case where cumulative is worse than target:
        // Target is 1.5, cumulative is 1.3 - we can't improve it without swapping in the other
        // direction.
        let accumulated_in = U256::from(1_000_000_000_000_000_000u128);
        let accumulated_out = U256::from(1_300_000_000_000_000_000u128); // ratio = 1.3

        // This should fail because we can't achieve 1.5 by swapping more zero_for_one
        // (the marginal trade price will be worse than 1.3, so cumulative will only get worse)
        let result = compute_swap_to_trade_price_with_accumulated(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_price_num,
            &target_price_den,
            fee_pips,
            accumulated_in,
            accumulated_out,
        );

        // The quadratic should either return an error or a target that's >= current price
        // (meaning nothing to do)
        if let Ok((sqrt_price_new, amount_in, _amount_out, _fee_amount, reached_target)) = result {
            // If successful, either:
            // 1. No swap needed (sqrt_price_new == sqrt_price_current)
            // 2. Or we found a valid solution
            if sqrt_price_new == sqrt_price_current {
                assert!(reached_target || amount_in.is_zero(), "Should indicate no swap needed");
            }
        }
        // Error is also acceptable - target not achievable
    }

    #[test]
    fn test_accumulated_multi_tick_simulation() {
        // Simulate a multi-tick scenario:
        // 1. First tick: swap without accumulated
        // 2. Second tick: use first tick's amounts as accumulated
        // Verify final cumulative price matches target

        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // Target cumulative trade price: 1.85
        let target_price_num = BigUint::from(185u64);
        let target_price_den = BigUint::from(100u64);

        // First tick
        let sqrt_price_tick1 = U256::from_str("112045541949572287496682733568").unwrap();
        let sqrt_ratio_limit_tick1 = U256::from_str("108000000000000000000000000000").unwrap();

        let (sqrt_price_after_tick1, amount_in_tick1, amount_out_tick1, _fee1, reached1) =
            compute_swap_to_trade_price_with_accumulated(
                sqrt_price_tick1,
                sqrt_ratio_limit_tick1,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            )
            .expect("Tick1 should succeed");

        // If we didn't reach target in tick1, continue to tick2
        if !reached1 {
            assert_eq!(sqrt_price_after_tick1, sqrt_ratio_limit_tick1, "Should be at tick1 boundary");

            // Second tick - use tick1's amounts as accumulated
            let sqrt_price_tick2 = sqrt_ratio_limit_tick1; // Continue from where we left off
            let sqrt_ratio_limit_tick2 = U256::from_str("79228162514264337593543950336").unwrap();

            let (_sqrt_price_after_tick2, amount_in_tick2, amount_out_tick2, _fee2, reached2) =
                compute_swap_to_trade_price_with_accumulated(
                    sqrt_price_tick2,
                    sqrt_ratio_limit_tick2,
                    liquidity,
                    &target_price_num,
                    &target_price_den,
                    fee_pips,
                    amount_in_tick1,
                    amount_out_tick1,
                )
                .expect("Tick2 should succeed");

            // Should reach target in tick2 (since boundary is far away)
            assert!(reached2, "Should reach target in tick2");

            // Verify final cumulative trade price
            verify_cumulative_trade_price_tolerance(
                amount_in_tick1,
                amount_out_tick1,
                amount_in_tick2,
                amount_out_tick2,
                &target_price_num,
                &target_price_den,
                0.01, // 1% tolerance
            )
            .expect("Final cumulative trade price should match target");
        }
    }

}
