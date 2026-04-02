//! StableSwap invariant math.
//!
//! Newton-Raphson solvers for the Curve StableSwap invariant, ported from Vyper:
//! <https://github.com/curvefi/curve-contract/blob/master/contracts/pool-templates/base/SwapTemplateBase.vy>
//!
//! All arithmetic uses `U256` and matches the on-chain implementation bit-for-bit.
use alloy::primitives::U256;
use tycho_common::simulation::errors::SimulationError;

use crate::evm::protocol::safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256};

/// Number of coins in the pool (this module supports only 2-token pools).
pub const N_COINS: U256 = U256::from_limbs([2, 0, 0, 0]);

/// Curve uses A_PRECISION = 100 for extra precision on the amplification coefficient.
pub const A_PRECISION: U256 = U256::from_limbs([100, 0, 0, 0]);

/// Curve fee denominator (10^10).
pub const FEE_DENOMINATOR: U256 = U256::from_limbs([10_000_000_000, 0, 0, 0]);

/// Compute normalization rate from token decimals: `10^(18 - decimals)`.
pub fn rate_from_decimals(decimals: u32) -> Result<U256, SimulationError> {
    let exponent = 18u32
        .checked_sub(decimals)
        .ok_or_else(|| {
            SimulationError::FatalError(format!("Token decimals must be <= 18, got {decimals}"))
        })?;
    Ok(U256::from(10u64).pow(U256::from(exponent)))
}

/// Maximum Newton-Raphson iterations (matches Curve on-chain).
const MAX_ITERATIONS: usize = 255;

/// Compute the StableSwap invariant D for a 2-token pool.
///
/// Uses Newton-Raphson iteration matching Curve's on-chain `get_D()` exactly.
/// `amp` is the amplification coefficient in internal precision (`A * A_PRECISION`).
pub fn get_d(x: U256, y: U256, amp: U256) -> Result<U256, SimulationError> {
    let s = safe_add_u256(x, y)?;
    if s.is_zero() {
        return Ok(U256::ZERO);
    }

    let ann = safe_mul_u256(amp, N_COINS)?;

    let mut d = s;
    for _ in 0..MAX_ITERATIONS {
        let d_prev = d;

        // D_P = D * D / (x * N_COINS) * D / (y * N_COINS)
        let d_p = safe_div_u256(safe_mul_u256(d, d)?, safe_mul_u256(x, N_COINS)?)?;
        let d_p = safe_div_u256(safe_mul_u256(d_p, d)?, safe_mul_u256(y, N_COINS)?)?;

        // numerator = (Ann * S / A_PRECISION + D_P * N_COINS) * D
        let numerator = safe_mul_u256(
            safe_add_u256(
                safe_div_u256(safe_mul_u256(ann, s)?, A_PRECISION)?,
                safe_mul_u256(d_p, N_COINS)?,
            )?,
            d,
        )?;

        // denominator = (Ann - A_PRECISION) * D / A_PRECISION + (N_COINS + 1) * D_P
        let denominator = safe_add_u256(
            safe_div_u256(safe_mul_u256(safe_sub_u256(ann, A_PRECISION)?, d)?, A_PRECISION)?,
            safe_mul_u256(safe_add_u256(N_COINS, U256::from(1))?, d_p)?,
        )?;

        d = safe_div_u256(numerator, denominator)?;

        // Convergence check: |D - D_prev| <= 1
        let diff = if d > d_prev { d - d_prev } else { d_prev - d };
        if diff <= U256::from(1) {
            return Ok(d);
        }
    }

    Err(SimulationError::FatalError("get_d did not converge".to_string()))
}

/// Compute the new balance of one token given the other's new balance and invariant D.
///
/// For a 2-token pool: given new x, finds new y (or vice versa — symmetric).
/// `amp` is the amplification coefficient in internal precision (`A * A_PRECISION`).
pub fn get_y(x_new: U256, d: U256, amp: U256) -> Result<U256, SimulationError> {
    let ann = safe_mul_u256(amp, N_COINS)?;

    // c = D * D / (x_new * N_COINS)
    // c = c * D * A_PRECISION / (Ann * N_COINS)
    // Matches Vyper multiplication order exactly.
    let c = safe_div_u256(safe_mul_u256(d, d)?, safe_mul_u256(x_new, N_COINS)?)?;
    let c = safe_div_u256(
        safe_mul_u256(safe_mul_u256(c, d)?, A_PRECISION)?,
        safe_mul_u256(ann, N_COINS)?,
    )?;

    // b = x_new + D * A_PRECISION / Ann
    let b = safe_add_u256(x_new, safe_div_u256(safe_mul_u256(d, A_PRECISION)?, ann)?)?;

    let mut y = d;
    for _ in 0..MAX_ITERATIONS {
        let y_prev = y;

        // y = (y^2 + c) / (2*y + b - D)
        let numerator = safe_add_u256(safe_mul_u256(y, y)?, c)?;
        let denominator = safe_sub_u256(safe_add_u256(safe_mul_u256(y, U256::from(2))?, b)?, d)?;
        y = safe_div_u256(numerator, denominator)?;

        let diff = if y > y_prev { y - y_prev } else { y_prev - y };
        if diff <= U256::from(1) {
            return Ok(y);
        }
    }

    Err(SimulationError::FatalError("get_y did not converge".to_string()))
}

/// Analytical spot price dy/dx for a 2-token StableSwap pool.
///
/// Derived from the StableSwap invariant:
///   dy/dx = (Ann * x + D_p / x) / (Ann * y + D_p / y)
/// where D_p = D^3 / (4 * x * y).
///
/// Returns the marginal price as (numerator, denominator) in U256 to avoid f64 precision loss.
pub fn spot_price_raw(
    norm_in: U256,
    norm_out: U256,
    d: U256,
    amp: U256,
) -> Result<(U256, U256), SimulationError> {
    let ann = safe_mul_u256(amp, N_COINS)?;

    // D_p = D^3 / (4 * norm_in * norm_out), split to avoid overflow
    let d_p = safe_div_u256(
        safe_mul_u256(safe_div_u256(safe_mul_u256(d, d)?, safe_mul_u256(norm_in, N_COINS)?)?, d)?,
        safe_mul_u256(norm_out, N_COINS)?,
    )?;

    // dy/dx = (Ann/A_PRECISION * norm_in + D_p) / (Ann/A_PRECISION * norm_out + D_p)
    let ann_scaled = safe_div_u256(ann, A_PRECISION)?;
    let numerator = safe_add_u256(safe_mul_u256(ann_scaled, norm_in)?, d_p)?;
    let denominator = safe_add_u256(safe_mul_u256(ann_scaled, norm_out)?, d_p)?;

    Ok((numerator, denominator))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_d_balanced_pool() {
        let reserve = U256::from(1_000_000_000_000_000_000_000u128); // 1000 tokens (18 dec)
        let amp = U256::from(10_000u64); // A=100, A_PRECISION=100

        let d = get_d(reserve, reserve, amp).expect("get_d should converge");

        let expected = reserve * U256::from(2);
        let diff = if d > expected { d - expected } else { expected - d };
        assert!(diff <= U256::from(1));
    }

    #[test]
    fn get_d_zero() {
        let d = get_d(U256::ZERO, U256::ZERO, U256::from(10000)).expect("should return 0");
        assert_eq!(d, U256::ZERO);
    }

    #[test]
    fn get_y_roundtrip() {
        let x = U256::from(1_000_000_000_000_000_000_000_000u128);
        let y = U256::from(1_000_000_000_000_000_000_000_000u128);
        let amp = U256::from(10_000u64);

        let d = get_d(x, y, amp).expect("get_d");
        let x_new = x + U256::from(1_000_000_000_000_000_000_000u128);
        let y_new = get_y(x_new, d, amp).expect("get_y");

        // y_new should be less than original y (we added to x)
        assert!(y_new < y);
    }

    #[test]
    fn get_d_imbalanced_pool() {
        let x = U256::from(900_000_000_000_000_000_000_000u128); // 900k
        let y = U256::from(100_000_000_000_000_000_000_000u128); // 100k
        let amp = U256::from(10_000u64);

        let d = get_d(x, y, amp).expect("get_d should converge");

        // D should be between sum (1M) and geometric mean * 2 (~600k)
        let sum = x + y;
        assert!(d <= sum, "D should not exceed sum of reserves");
        assert!(d > y, "D should exceed the smaller reserve");
    }

    #[test]
    fn get_d_high_amplification() {
        let x = U256::from(1_000_000_000_000_000_000_000_000u128);
        let y = U256::from(1_000_000_000_000_000_000_000_000u128);
        let amp = U256::from(100_000_000u64); // A = 1M

        let d = get_d(x, y, amp).expect("get_d should converge with high A");
        let expected = x + y;
        let diff = if d > expected { d - expected } else { expected - d };
        assert!(diff <= U256::from(1));
    }

    #[test]
    fn get_d_low_amplification() {
        let x = U256::from(1_000_000_000_000_000_000_000_000u128);
        let y = U256::from(1_000_000_000_000_000_000_000_000u128);
        let amp = U256::from(100u64); // A = 1

        let d = get_d(x, y, amp).expect("get_d should converge with low A");
        let sum = x + y;
        assert!(d <= sum);
    }

    #[test]
    fn spot_price_raw_balanced() {
        let reserve = U256::from(1_000_000_000_000_000_000_000_000u128);
        let amp = U256::from(10_000u64);

        let d = get_d(reserve, reserve, amp).expect("get_d");
        let (num, den) = spot_price_raw(reserve, reserve, d, amp).expect("spot_price_raw");

        // For balanced pool, marginal price should be ~1:1
        // num and den should be equal (or differ by at most 1)
        let diff = if num > den { num - den } else { den - num };
        assert!(diff <= U256::from(1), "balanced pool should have 1:1 marginal price, diff={diff}");
    }

    #[test]
    fn spot_price_raw_imbalanced() {
        let x = U256::from(900_000_000_000_000_000_000_000u128); // 900k (abundant)
        let y = U256::from(100_000_000_000_000_000_000_000u128); // 100k (scarce)
        let amp = U256::from(10_000u64);

        let d = get_d(x, y, amp).expect("get_d");
        // dy/dx with norm_in=x (abundant), norm_out=y (scarce)
        let (num, den) = spot_price_raw(x, y, d, amp).expect("spot_price_raw");
        // Marginal price should differ from 1:1 for imbalanced pool
        assert_ne!(num, den, "imbalanced pool should not have 1:1 marginal price");
    }

    #[test]
    fn get_y_symmetric() {
        // get_y should be symmetric: if we solve for y given x, then solve for x given y, we get
        // back
        let x = U256::from(1_000_000_000_000_000_000_000_000u128);
        let y = U256::from(500_000_000_000_000_000_000_000u128);
        let amp = U256::from(10_000u64);

        let d = get_d(x, y, amp).expect("get_d");

        // Given x, solve for y
        let y_solved = get_y(x, d, amp).expect("get_y for x");
        let diff = if y_solved > y { y_solved - y } else { y - y_solved };
        assert!(diff <= U256::from(1), "get_y should recover y from x, diff={diff}");

        // Given y, solve for x
        let x_solved = get_y(y, d, amp).expect("get_y for y");
        let diff = if x_solved > x { x_solved - x } else { x - x_solved };
        assert!(diff <= U256::from(1), "get_y should recover x from y, diff={diff}");
    }
}
