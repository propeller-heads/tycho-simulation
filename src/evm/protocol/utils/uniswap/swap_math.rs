use alloy::primitives::{I256, U256, U512};
use num_bigint::BigUint;
use tycho_common::simulation::errors::SimulationError;

use super::sqrt_price_math;
use crate::evm::protocol::{
    safe_math::safe_sub_u256,
    u256_num::biguint_to_u256,
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
/// For zero_for_one=true (selling token0 for token1):
/// - amount_out = L × (S₀ - S₁) / 2^96
/// - amount_in = [L × 2^96 × (S₀ - S₁) / (S₀ × S₁)] / γ
/// - Trade price P = amount_out / amount_in = γ × S₀ × S₁ / 2^192
/// - Solving: S₁ = P × 2^192 / (γ × S₀)
///
/// For zero_for_one=false (selling token1 for token0):
/// - Trade price P = γ × 2^192 / (S₀ × S₁)
/// - Solving: S₁ = γ × 2^192 / (P × S₀)
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
}
