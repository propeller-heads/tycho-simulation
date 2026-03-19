use alloy::primitives::{I256, U256};
use num_bigint::BigUint;
use ruint::aliases::{U1024, U2048};
use tycho_common::simulation::errors::SimulationError;

use super::sqrt_price_math;
use crate::evm::protocol::{
    safe_math::safe_sub_u256,
    utils::solidity_math::{mul_div, mul_div_rounding_up},
};

fn u1024_to_u256(val: U1024) -> U256 {
    let bytes = val.to_le_bytes::<128>();
    U256::from_le_slice(&bytes[..32])
}

fn u2048_to_u1024(val: U2048) -> U1024 {
    let bytes = val.to_le_bytes::<256>();
    U1024::from_le_bytes::<128>(bytes[..128].try_into().unwrap())
}

/// Computes the result of swapping some amount in, or amount out, given the parameters of the swap
///
/// The fee, plus the amount in, will never exceed the amount remaining if the swap's
/// `amountSpecified` is positive
///
/// # Arguments
///
/// * `sqrt_ratio_current` - The current sqrt price as a Q64.96 fixed-point number.
/// * `sqrt_ratio_target` - The price that cannot be exceeded, from which the direction of the swap
///   is inferred. Typically represents the next tick boundary.
/// * `liquidity` - The usable liquidity in the current tick range.
/// * `amount_remaining` - How much input or output amount is remaining to be swapped in/out.
///   Positive values indicate exact input swaps (amount_specified > 0), negative values indicate
///   exact output swaps (amount_specified < 0).
/// * `fee_pips` - The fee taken from the input amount, expressed in hundredths of a bip (e.g., 3000
///   = 0.3%).
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

/// Computes the sqrt_price_new, amount_in, and amount_out needed to achieve a target cumulative
/// trade price, optionally accounting for amounts already accumulated from previous ticks.
///
/// This function solves for the sqrt_price that achieves the target cumulative trade price:
/// `(accumulated_out + new_out) / (accumulated_in + new_in) = target_price`
///
/// For single-tick swaps without accumulated amounts, pass `accumulated_in = 0` and
/// `accumulated_out = 0`.
///
/// # Price Types and Fee Handling
///
/// **IMPORTANT:** This function expects a **FORMULA target price** (amount_out /
/// amount_in_pre_fee). The caller is responsible for converting the user's target price to the
/// formula price:
///
/// - **User price**: `amount_out / amount_in_with_fee` - what the user pays/receives
/// - **Formula price**: `amount_out / amount_in_pre_fee` - used in Uniswap math
/// - **Conversion**: `formula_price = user_price / (1 - fee)` `formula_price = user_price ×
///   1_000_000 / (1_000_000 - fee_pips)`
///
/// The `fee_pips` parameter is used ONLY to calculate the fee_amount on the returned amounts,
/// NOT for target price conversion (that's the caller's responsibility).
///
/// # Trade Price Formula Derivation
///
/// **Notation:**
/// - `S₀`: sqrt_ratio_current (Q96 format)
/// - `S₁`: sqrt_ratio_target (Q96 format)
/// - `L`: liquidity in the current tick range
/// - `Q`: 2^96 (Q96 scaling factor)
/// - `P`: Formula target price (amount_out / amount_in_pre_fee)
/// - `A_in`, `A_out`: accumulated_in and accumulated_out from previous ticks
/// - `R`: Residual = P × A_in - A_out (how much more output we need)
///
/// **For zero_for_one=true (selling token0 for token1):**
/// ```text
/// new_in = L × Q × δ / (S₀ × S₁)   where δ = S₀ - S₁
/// new_out = L × δ / Q
/// ```
/// Solving `(A_out + new_out) / (A_in + new_in) = P` yields quadratic in δ:
/// ```text
/// δ² + b×δ + c = 0
/// where:
///   b = P×Q²/S₀ - S₀ - R×Q/L
///   c = R×Q×S₀/L
/// ```
///
/// **For zero_for_one=false (selling token1 for token0):**
/// ```text
/// δ² + b×δ + c = 0
/// where:
///   b = S₀ - Q²/(P×S₀) + R×Q/(P×L)
///   c = R×Q×S₀/(P×L)
/// ```
///
/// # Arguments
///
/// * `sqrt_ratio_current` - The current sqrt price as a Q64.96 fixed-point number.
/// * `sqrt_ratio_limit` - The price that cannot be exceeded, from which the direction of the swap
///   is inferred. Typically represents the next tick boundary.
/// * `liquidity` - The usable liquidity in the current tick range.
/// * `target_price_num` - **Formula** target price numerator (amount_out / amount_in_pre_fee).
/// * `target_price_den` - **Formula** target price denominator.
/// * `fee_pips` - The fee in pips (e.g., 3000 = 0.3%). Used ONLY for fee_amount calculation.
/// * `accumulated_in` - **Pre-fee** input amount already accumulated from previous ticks (0 if
///   none).
/// * `accumulated_out` - Output amount already accumulated from previous ticks (0 if none).
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
/// * `reached_target` - Whether we achieved the target price (false if hit tick boundary first or
///   target was not achievable with the given accumulated amounts)
type SignedU1024 = (bool, U1024);
type SignedU2048 = (bool, U2048);

fn biguint_to_u1024(value: &BigUint) -> U1024 {
    let bytes = value.to_bytes_le();
    let mut buf = [0u8; 128];
    let len = bytes.len().min(128);
    buf[..len].copy_from_slice(&bytes[..len]);
    U1024::from_le_bytes(buf)
}

fn signed_sub_u1024(a: U1024, b: U1024) -> SignedU1024 {
    if a >= b {
        (true, a - b)
    } else {
        (false, b - a)
    }
}

fn signed_sub_u2048(a: U2048, b: U2048) -> SignedU2048 {
    if a >= b {
        (true, a - b)
    } else {
        (false, b - a)
    }
}

/// a_sign*a_mag - b_sign*b_mag
fn signed_sub_signed_u1024(
    (a_pos, a_mag): SignedU1024,
    (b_pos, b_mag): SignedU1024,
) -> SignedU1024 {
    match (a_pos, b_pos) {
        (true, true) => signed_sub_u1024(a_mag, b_mag),
        (false, false) => {
            let (pos, mag) = signed_sub_u1024(a_mag, b_mag);
            (!pos, mag)
        }
        (true, false) => (true, a_mag + b_mag),
        (false, true) => (false, a_mag + b_mag),
    }
}

/// a_sign*a_mag + b_sign*b_mag
fn signed_add_signed_u1024(a: SignedU1024, (b_pos, b_mag): SignedU1024) -> SignedU1024 {
    signed_sub_signed_u1024(a, (!b_pos, b_mag))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_swap_to_trade_price(
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

    let s0 = U1024::from(sqrt_ratio_current);
    let s_limit = U1024::from(sqrt_ratio_limit);
    let price_num = biguint_to_u1024(target_trade_price_num);
    let price_den = biguint_to_u1024(target_trade_price_den);

    // Zero-accumulated shortcut: single division instead of full quadratic.
    // For z4o: T = S₀×S/Q² → S_target = P_num × Q¹⁹² / (P_den × S₀)
    // For o4z: T = Q²/(S₀×S) → S_target = P_den × Q¹⁹² / (P_num × S₀)
    let sqrt_ratio_target_result = if accumulated_in.is_zero() && accumulated_out.is_zero() {
        let q192 = U1024::from(1u64) << 192;
        let (num_price, den_price) =
            if zero_for_one { (price_num, price_den) } else { (price_den, price_num) };
        let numerator = num_price * q192;
        let denominator = den_price * s0;
        if denominator.is_zero() {
            Err(SimulationError::FatalError("Zero denominator in shortcut".to_string()))
        } else {
            let s_target = numerator / denominator;
            if s_target > U1024::from(U256::MAX) {
                Err(SimulationError::FatalError("sqrt_ratio_target exceeds U256 range".to_string()))
            } else {
                Ok(s_target)
            }
        }
    } else {
        let liq = U1024::from(liquidity);
        let q96 = U1024::from(1u64) << 96;
        let acc_in = U1024::from(accumulated_in);
        let acc_out = U1024::from(accumulated_out);

        let pos_term = price_num * acc_in;
        let neg_term = acc_out * price_den;
        let (residual_pos, residual_mag) = signed_sub_u1024(pos_term, neg_term);

        compute_sqrt_ratio_target_ruint(
            s0,
            liq,
            q96,
            price_num,
            price_den,
            (residual_pos, residual_mag),
            zero_for_one,
        )
    };

    let (sqrt_ratio_new, reached_target) = match sqrt_ratio_target_result {
        Ok(s_target) => {
            if zero_for_one {
                if s_target >= s0 {
                    return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
                }
                if s_target < s_limit {
                    (s_limit, false)
                } else {
                    (s_target, true)
                }
            } else {
                if s_target <= s0 {
                    return Ok((sqrt_ratio_current, U256::ZERO, U256::ZERO, U256::ZERO, true));
                }
                if s_target > s_limit {
                    (s_limit, false)
                } else {
                    (s_target, true)
                }
            }
        }
        Err(_) => (s_limit, false),
    };

    let sqrt_ratio_new_u256 = u1024_to_u256(sqrt_ratio_new);

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

/// Solves x² + bx + c = 0 for the positive root using ruint fixed-width integers.
/// b and c are represented as signed fractions: b = b_num/b_den, c = c_num/c_den.
fn solve_quadratic_ruint(
    (b_pos, b_mag): SignedU1024,
    b_den: U1024,
    (c_pos, c_mag): SignedU1024,
    c_den: U1024,
) -> Result<U1024, SimulationError> {
    // D = b² - 4c = (b_num² * c_den - 4 * c_num * b_den²) / (b_den² * c_den)
    let b_den_wide = U2048::from(b_den);
    let b_mag_wide = U2048::from(b_mag);
    let c_mag_wide = U2048::from(c_mag);
    let c_den_wide = U2048::from(c_den);

    let b_den_sq = b_den_wide * b_den_wide;

    // b_num² is always positive
    let b_sq_term = b_mag_wide * b_mag_wide * c_den_wide;
    // four_c: magnitude = 4 * c_mag * b_den², sign = c_pos
    let four_c_mag = (c_mag_wide * b_den_sq) << 2;

    // d_num = b_sq_term - (c_pos ? four_c_mag : -four_c_mag)
    let (d_pos, d_mag) = if c_pos {
        signed_sub_u2048(b_sq_term, four_c_mag)
    } else {
        (true, b_sq_term + four_c_mag)
    };

    if !d_pos {
        return Err(SimulationError::FatalError(
            "Negative discriminant - no real solution".to_string(),
        ));
    }

    let d_den_val = b_den_sq * c_den_wide;
    let sqrt_d_num = d_mag.root(2);
    let sqrt_d_den = d_den_val.root(2);

    // Use numerically stable formula based on sign of b
    let (x_pos, x_mag, x_den_val) = if b_pos {
        // b >= 0: x = -2c / (b + sqrt(D))
        // x_num = -2 * c_num * b_den * sqrt_d_den, sign = !c_pos
        // x_den = c_den * (b_num * sqrt_d_den + sqrt_d_num * b_den), sign = positive (both
        // positive)
        let x_num_mag = (c_mag_wide * b_den_wide * sqrt_d_den) << 1;
        let x_num_pos = !c_pos;
        let x_den_val = c_den_wide * (b_mag_wide * sqrt_d_den + sqrt_d_num * b_den_wide);
        // x_den is always positive when b >= 0
        (x_num_pos, x_num_mag, x_den_val)
    } else {
        // b < 0: x = (-b + sqrt(D)) / 2
        // x_num = |b_num| * sqrt_d_den + sqrt_d_num * b_den (both positive)
        // x_den = 2 * b_den * sqrt_d_den
        let x_num_mag = b_mag_wide * sqrt_d_den + sqrt_d_num * b_den_wide;
        let x_den_val = (b_den_wide * sqrt_d_den) << 1;
        (true, x_num_mag, x_den_val)
    };

    if !x_pos && x_mag != U2048::ZERO {
        return Err(SimulationError::FatalError("No positive solution".to_string()));
    }

    // Round to nearest: (x_mag + x_den_val/2) / x_den_val
    let result_wide = (x_mag + x_den_val / U2048::from(2u64)) / x_den_val;

    Ok(u2048_to_u1024(result_wide))
}

/// Computes quadratic coefficients using ruint.
/// Returns (b_num_signed, b_den, c_num_signed, c_den).
fn compute_quadratic_coefficients_ruint(
    s0: U1024,
    liq: U1024,
    q96: U1024,
    price_num: U1024,
    price_den: U1024,
    residual: SignedU1024,
    zero_for_one: bool,
) -> (SignedU1024, U1024, SignedU1024, U1024) {
    let q96_sq = q96 * q96;
    let q96_times_s0 = q96 * s0;
    let (res_pos, res_mag) = residual;

    if zero_for_one {
        // b_den = S₀ × L × P_den
        let b_den = s0 * liq * price_den;
        // term1 = P_num × Q² × L (positive)
        let term1 = price_num * q96_sq * liq;
        // term2 = S₀ × b_den = S₀² × L × P_den (positive)
        let term2 = s0 * b_den;
        // term3 = residual × Q × S₀ (signed, sign = res_pos)
        let term3_mag = res_mag * q96_times_s0;

        // b_num = term1 - term2 - term3
        let step1 = signed_sub_u1024(term1, term2);
        let b_num = signed_sub_signed_u1024(step1, (res_pos, term3_mag));

        // c_num = residual × Q × S₀, c_den = P_den × L
        let c_num = (res_pos, res_mag * q96_times_s0);
        let c_den = price_den * liq;

        (b_num, b_den, c_num, c_den)
    } else {
        // b_den = P_num × S₀ × L × P_den
        let b_den = price_num * s0 * liq * price_den;
        // term1 = S₀ × b_den = S₀² × P_num × L × P_den (positive)
        let term1 = s0 * b_den;
        // term2 = Q² × P_den × L × P_den (positive)
        let term2 = q96_sq * price_den * liq * price_den;
        // term3 = residual × Q × S₀ × P_den (signed, sign = res_pos)
        let term3_mag = res_mag * q96_times_s0 * price_den;

        // b_num = term1 - term2 + term3
        let step1 = signed_sub_u1024(term1, term2);
        let b_num = signed_add_signed_u1024(step1, (res_pos, term3_mag));

        // c_num = residual × Q × S₀, c_den = P_num × L
        let c_num = (res_pos, res_mag * q96_times_s0);
        let c_den = price_num * liq;

        (b_num, b_den, c_num, c_den)
    }
}

/// Computes the target sqrt_ratio using ruint types.
fn compute_sqrt_ratio_target_ruint(
    s0: U1024,
    liq: U1024,
    q96: U1024,
    price_num: U1024,
    price_den: U1024,
    residual: SignedU1024,
    zero_for_one: bool,
) -> Result<U1024, SimulationError> {
    let (b_num, b_den, c_num, c_den) = compute_quadratic_coefficients_ruint(
        s0,
        liq,
        q96,
        price_num,
        price_den,
        residual,
        zero_for_one,
    );

    let delta = solve_quadratic_ruint(b_num, b_den, c_num, c_den)?;

    let result = if zero_for_one {
        if s0 < delta {
            return Err(SimulationError::FatalError(
                "sqrt_ratio_target would be negative".to_string(),
            ));
        }
        s0 - delta
    } else {
        s0 + delta
    };

    if result > U1024::from(U256::MAX) {
        return Err(SimulationError::FatalError("sqrt_ratio_target exceeds U256 range".to_string()));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{Sign, I256, U512};
    use num_traits::ToPrimitive;

    use super::*;
    use crate::evm::protocol::u256_num::{biguint_to_u256, u256_to_biguint};

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
        // This is the USER target price (amount_out / amount_in_with_fee)
        let user_target_num = BigUint::from(19u64);
        let user_target_den = BigUint::from(10u64);

        // Tick boundary far away (won't be hit)
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap(); // MIN_SQRT_RATIO + 1

        let fee_pips = 3000u32; // 0.3%

        // Convert user target to formula target: formula = user / (1 - fee)
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
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

        // Verify achieved trade price is close to formula target (within 0.001% tolerance)
        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &formula_target_num,
            &formula_target_den,
            0.00001,
        )
        .expect("Trade price should be within tolerance");

        // Cross-validate: use compute_swap_step as an independent oracle.
        // Feed sqrt_price_new as the target — compute_swap_step will compute amounts
        // using the standard Uniswap delta functions, independently from the quadratic solver.
        let large_amount =
            I256::checked_from_sign_and_abs(Sign::Positive, U256::from(u128::MAX)).unwrap();
        let (step_sqrt, step_in, step_out, _step_fee) = compute_swap_step(
            sqrt_price_current,
            sqrt_price_new,
            liquidity,
            large_amount,
            fee_pips,
        )
        .expect("compute_swap_step should succeed");
        assert_eq!(step_sqrt, sqrt_price_new, "compute_swap_step should arrive at same sqrt_price");
        assert_eq!(step_in, amount_in, "compute_swap_step amount_in should match");
        assert_eq!(step_out, amount_out, "compute_swap_step amount_out should match");
    }

    #[test]
    fn test_compute_swap_to_trade_price_hits_tick_boundary() {
        // Setup: sqrt_price ~= sqrt(2) * 2^96 -> spot price ~= 2.0
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;

        // Target trade price: 0.5 token1/token0 (much worse than spot)
        // This will require crossing the tick boundary
        let user_target_num = BigUint::from(5u64);
        let user_target_den = BigUint::from(10u64);

        // Tick boundary close to current price (will be hit)
        let sqrt_ratio_limit = U256::from_str("111000000000000000000000000000").unwrap();

        let fee_pips = 3000u32;

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
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
        let user_target_num = BigUint::from(25u64);
        let user_target_den = BigUint::from(10u64);

        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let fee_pips = 3000u32;

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
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
        let user_target_num = BigUint::from(485u64);
        let user_target_den = BigUint::from(1000u64);

        // Tick boundary far above current price (won't be hit)
        let sqrt_ratio_limit = U256::from_str("150000000000000000000000000000").unwrap();

        let fee_pips = 3000u32;

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // Should reach target
        assert!(reached_target, "Should reach target price");
        assert!(sqrt_price_new > sqrt_price_current, "Price should increase for one_for_zero");
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");

        // Verify trade price tolerance (0.001% with quadratic formula)
        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &formula_target_num,
            &formula_target_den,
            0.00001,
        )
        .expect("Trade price should be within tolerance");

        // Cross-validate with compute_swap_step (independent oracle)
        let large_amount =
            I256::checked_from_sign_and_abs(Sign::Positive, U256::from(u128::MAX)).unwrap();
        let (step_sqrt, step_in, step_out, _step_fee) = compute_swap_step(
            sqrt_price_current,
            sqrt_price_new,
            liquidity,
            large_amount,
            fee_pips,
        )
        .expect("compute_swap_step should succeed");
        assert_eq!(step_sqrt, sqrt_price_new, "compute_swap_step should arrive at same sqrt_price");
        assert_eq!(step_in, amount_in, "compute_swap_step amount_in should match");
        assert_eq!(step_out, amount_out, "compute_swap_step amount_out should match");
    }

    #[test]
    fn test_compute_swap_to_trade_price_zero_liquidity() {
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 0u128; // Zero liquidity

        let formula_target_num = BigUint::from(19_000_000u64);
        let formula_target_den = BigUint::from(9_970_000u64); // ~1.9 / 0.997
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let result = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &formula_target_num,
            &formula_target_den,
            3000,
            U256::ZERO,
            U256::ZERO,
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
        let sqrt_ratio_limit = U256::from_str("97000000000000000000000000000").unwrap();

        // Target price that would result in sqrt_ratio near the boundary
        let user_target_num = BigUint::from(15u64);
        let user_target_den = BigUint::from(10u64); // 1.5

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            )
            .expect("compute_swap_to_trade_price should succeed");

        // The swap should complete (either reaching target or hitting boundary)
        assert!(amount_in > U256::ZERO, "amount_in should be positive");
        assert!(amount_out > U256::ZERO, "amount_out should be positive");
        assert!(fee_amount > U256::ZERO, "fee_amount should be positive");

        // Price should have moved
        assert!(sqrt_price_new < sqrt_price_current, "Price should decrease for zero_for_one");

        // Should either reach target or be at boundary
        assert!(sqrt_price_new >= sqrt_ratio_limit, "New price should be at or above boundary");

        // If we hit the boundary exactly, reached_target should be false
        if sqrt_price_new == sqrt_ratio_limit {
            assert!(!reached_target, "Should not have reached target if stopped at boundary");
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
        let user_target_num = BigUint::from(18u64);
        let user_target_den = BigUint::from(10u64); // 1.8

        // Compute expected sqrt_ratio_target using the formula
        let sqrt_ratio_limit = U256::from_str("101400000000000000000000000000").unwrap();

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let result = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &formula_target_num,
            &formula_target_den,
            fee_pips,
            U256::ZERO,
            U256::ZERO,
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

    // ==================== Tests for compute_swap_to_trade_price with accumulated amounts
    // ====================

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

        verify_trade_price_tolerance(
            total_in,
            total_out,
            target_price_num,
            target_price_den,
            tolerance,
        )
    }

    #[test]
    fn test_accumulated_zero_gives_same_result() {
        // When accumulated amounts are zero, test that the function works correctly
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let user_target_num = BigUint::from(19u64);
        let user_target_den = BigUint::from(10u64);
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();
        let fee_pips = 3000u32;

        // Convert user target to formula target
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let result = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &formula_target_num,
            &formula_target_den,
            fee_pips,
            U256::ZERO,
            U256::ZERO,
        )
        .expect("Should succeed");

        // Should reach target and have positive amounts
        assert!(result.4, "Should reach target");
        assert!(result.1 > U256::ZERO, "amount_in should be positive");
        assert!(result.2 > U256::ZERO, "amount_out should be positive");
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
            compute_swap_to_trade_price(
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
            0.005, // 0.5% tolerance (integer sqrt introduces small rounding)
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
            compute_swap_to_trade_price(
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
            compute_swap_to_trade_price(
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
            0.005, // 0.5% tolerance (integer sqrt introduces small rounding)
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
            compute_swap_to_trade_price(
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
    fn test_accumulated_target_unreachable_due_to_worse_cumulative() {
        // Accumulated ratio (1.3) is worse than target (1.5) for zero_for_one.
        // The marginal price (pre-fee ~2.0) is better, so swapping more will pull
        // the cumulative ratio TOWARD the target. The function should either:
        // - Find the exact delta that achieves 1.5 cumulatively, or
        // - Fall back to the tick boundary if the target requires too much movement.
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        let target_price_num = BigUint::from(15u64);
        let target_price_den = BigUint::from(10u64);

        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let accumulated_in = U256::from(1_000_000_000_000_000_000u128);
        let accumulated_out = U256::from(1_300_000_000_000_000_000u128); // ratio = 1.3

        let (sqrt_price_new, amount_in, amount_out, _fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_price_num,
                &target_price_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            )
            .expect("Should succeed — marginal price is better than target");

        assert!(amount_in > U256::ZERO, "Should swap some amount to improve cumulative ratio");
        assert!(amount_out > U256::ZERO, "Should receive output");
        assert!(sqrt_price_new < sqrt_price_current, "Price should decrease for zero_for_one");

        let new_out = 1_300_000_000_000_000_000u128 + amount_out.to::<u128>();
        let new_in = 1_000_000_000_000_000_000u128 + amount_in.to::<u128>();
        let new_ratio = new_out as f64 / new_in as f64;
        assert!(new_ratio > 1.3, "Cumulative ratio should improve from 1.3, got {new_ratio:.6}");

        if reached_target {
            verify_cumulative_trade_price_tolerance(
                accumulated_in,
                accumulated_out,
                amount_in,
                amount_out,
                &target_price_num,
                &target_price_den,
                0.005,
            )
            .expect("If target reached, cumulative price should match");
        }
    }

    #[test]
    fn test_negative_residual_accumulated_out_exceeds_target() {
        // Negative residual: accumulated_out / accumulated_in > target
        // The algorithm should find a delta where new marginal swaps pull cumulative
        // ratio down toward the target.
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        // Accumulated ratio = 2.1 (out/in), target = 1.95
        // So residual = 1.95 * 100_000 - 210_000 = -15_000 (negative)
        let accumulated_in = U256::from(100_000u128);
        let accumulated_out = U256::from(210_000u128);

        let target_num = BigUint::from(195u64);
        let target_den = BigUint::from(100u64);

        let (_sp, amount_in, amount_out, _fee, reached_target) = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit,
            liquidity,
            &target_num,
            &target_den,
            fee_pips,
            accumulated_in,
            accumulated_out,
        )
        .expect("Should succeed with negative residual");

        assert!(reached_target, "Should reach target");
        assert!(amount_in > U256::ZERO, "Should swap some amount");
        assert!(amount_out > U256::ZERO, "Should produce output");

        // New marginal price is worse than current cumulative, pulling it toward target
        let total_in = 100_000u128 + amount_in.to::<u128>();
        let total_out = 210_000u128 + amount_out.to::<u128>();
        let cumulative_ratio = total_out as f64 / total_in as f64;
        let target_ratio = 1.95;
        let relative_diff = (cumulative_ratio - target_ratio).abs() / target_ratio;
        assert!(
            relative_diff < 0.005,
            "Cumulative ratio {cumulative_ratio:.6} should be close to target {target_ratio:.6}, diff: {relative_diff:.6}"
        );
    }

    #[test]
    fn test_rounding_direction_is_conservative() {
        // The achieved trade price should never be BETTER than the target.
        // For zero_for_one: actual amount_out/amount_in ≤ target
        // For one_for_zero: actual amount_out/amount_in ≤ target
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;

        // zero_for_one: target = 1.9
        let user_target_num = BigUint::from(19u64);
        let user_target_den = BigUint::from(10u64);
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);
        let sqrt_ratio_limit_z = U256::from_str("79228162514264337593543950336").unwrap();

        let (_sp, amount_in_z, amount_out_z, _fee, reached) = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit_z,
            liquidity,
            &formula_target_num,
            &formula_target_den,
            fee_pips,
            U256::ZERO,
            U256::ZERO,
        )
        .expect("zero_for_one should succeed");

        assert!(reached, "Should reach target");
        // achieved = amount_out / amount_in; must be ≤ target (formula target)
        let achieved_cross =
            U512::from(amount_out_z) * U512::from(biguint_to_u256(&formula_target_den));
        let target_cross =
            U512::from(biguint_to_u256(&formula_target_num)) * U512::from(amount_in_z);
        assert!(
            achieved_cross <= target_cross,
            "zero_for_one: achieved price must not exceed target (conservative rounding)"
        );

        // one_for_zero: target = 0.485
        let user_target_num_o = BigUint::from(485u64);
        let user_target_den_o = BigUint::from(1000u64);
        let formula_target_num_o = &user_target_num_o * BigUint::from(1_000_000u32);
        let formula_target_den_o = &user_target_den_o * BigUint::from(1_000_000u32 - fee_pips);
        let sqrt_ratio_limit_o = U256::from_str("150000000000000000000000000000").unwrap();

        let (_sp, amount_in_o, amount_out_o, _fee, reached) = compute_swap_to_trade_price(
            sqrt_price_current,
            sqrt_ratio_limit_o,
            liquidity,
            &formula_target_num_o,
            &formula_target_den_o,
            fee_pips,
            U256::ZERO,
            U256::ZERO,
        )
        .expect("one_for_zero should succeed");

        assert!(reached, "Should reach target");
        let achieved_cross_o =
            U512::from(amount_out_o) * U512::from(biguint_to_u256(&formula_target_den_o));
        let target_cross_o =
            U512::from(biguint_to_u256(&formula_target_num_o)) * U512::from(amount_in_o);
        assert!(
            achieved_cross_o <= target_cross_o,
            "one_for_zero: achieved price must not exceed target (conservative rounding)"
        );
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
            compute_swap_to_trade_price(
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
            assert_eq!(
                sqrt_price_after_tick1, sqrt_ratio_limit_tick1,
                "Should be at tick1 boundary"
            );

            // Second tick - use tick1's amounts as accumulated
            let sqrt_price_tick2 = sqrt_ratio_limit_tick1; // Continue from where we left off
            let sqrt_ratio_limit_tick2 = U256::from_str("79228162514264337593543950336").unwrap();

            let (_sqrt_price_after_tick2, amount_in_tick2, amount_out_tick2, _fee2, reached2) =
                compute_swap_to_trade_price(
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

    #[test]
    fn test_zero_accumulated_shortcut_matches_full_solver_z4o() {
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let user_target_num = BigUint::from(19u64);
        let user_target_den = BigUint::from(10u64);
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            )
            .expect("Should succeed with zero accumulated");

        assert!(reached_target, "Should reach target");
        assert!(amount_in > U256::ZERO);
        assert!(amount_out > U256::ZERO);
        assert!(fee_amount > U256::ZERO);

        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &formula_target_num,
            &formula_target_den,
            0.00001,
        )
        .expect("Shortcut should match target within tolerance");

        let large_amount =
            I256::checked_from_sign_and_abs(Sign::Positive, U256::from(u128::MAX)).unwrap();
        let (step_sqrt, step_in, step_out, _) = compute_swap_step(
            sqrt_price_current,
            sqrt_price_new,
            liquidity,
            large_amount,
            fee_pips,
        )
        .expect("compute_swap_step should succeed");
        assert_eq!(step_sqrt, sqrt_price_new);
        assert_eq!(step_in, amount_in);
        assert_eq!(step_out, amount_out);
    }

    #[test]
    fn test_zero_accumulated_shortcut_matches_full_solver_o4z() {
        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;
        let sqrt_ratio_limit = U256::from_str("150000000000000000000000000000").unwrap();

        let user_target_num = BigUint::from(485u64);
        let user_target_den = BigUint::from(1000u64);
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let (sqrt_price_new, amount_in, amount_out, fee_amount, reached_target) =
            compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            )
            .expect("Should succeed with zero accumulated");

        assert!(reached_target, "Should reach target");
        assert!(amount_in > U256::ZERO);
        assert!(amount_out > U256::ZERO);
        assert!(fee_amount > U256::ZERO);

        verify_trade_price_tolerance(
            amount_in,
            amount_out,
            &formula_target_num,
            &formula_target_den,
            0.00001,
        )
        .expect("Shortcut should match target within tolerance");

        let large_amount =
            I256::checked_from_sign_and_abs(Sign::Positive, U256::from(u128::MAX)).unwrap();
        let (step_sqrt, step_in, step_out, _) = compute_swap_step(
            sqrt_price_current,
            sqrt_price_new,
            liquidity,
            large_amount,
            fee_pips,
        )
        .expect("compute_swap_step should succeed");
        assert_eq!(step_sqrt, sqrt_price_new);
        assert_eq!(step_in, amount_in);
        assert_eq!(step_out, amount_out);
    }

    #[test]
    #[ignore]
    fn bench_compute_swap_to_trade_price() {
        use std::time::Instant;

        let sqrt_price_current = U256::from_str("112045541949572287496682733568").unwrap();
        let liquidity = 1_000_000_000_000_000_000u128;
        let fee_pips = 3000u32;
        let sqrt_ratio_limit = U256::from_str("79228162514264337593543950336").unwrap();

        let user_target_num = BigUint::from(19u64);
        let user_target_den = BigUint::from(10u64);
        let formula_target_num = &user_target_num * BigUint::from(1_000_000u32);
        let formula_target_den = &user_target_den * BigUint::from(1_000_000u32 - fee_pips);

        let iterations = 10_000;

        // Benchmark zero-accumulated (shortcut path)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &formula_target_num,
                &formula_target_den,
                fee_pips,
                U256::ZERO,
                U256::ZERO,
            ));
        }
        let zero_accum_ns = start.elapsed().as_nanos() / iterations;

        // Benchmark with accumulated (full quadratic path)
        let accumulated_in = U256::from(100_000u128);
        let accumulated_out = U256::from(190_000u128);
        let target_num = BigUint::from(195u64);
        let target_den = BigUint::from(100u64);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(compute_swap_to_trade_price(
                sqrt_price_current,
                sqrt_ratio_limit,
                liquidity,
                &target_num,
                &target_den,
                fee_pips,
                accumulated_in,
                accumulated_out,
            ));
        }
        let accum_ns = start.elapsed().as_nanos() / iterations;

        println!("\n=== swap_to_trade_price benchmark ({iterations} iterations) ===");
        println!("Zero-accumulated (shortcut): {} ns/iter", zero_accum_ns);
        println!("With accumulated (quadratic): {} ns/iter", accum_ns);
    }
}
