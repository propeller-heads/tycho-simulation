use alloy::primitives::U256;
use num_bigint::BigUint;
use num_traits::Zero;
use tycho_common::{simulation::errors::SimulationError, Bytes};

use crate::evm::protocol::{
    safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256},
    u256_num::u256_to_biguint,
};

pub fn cfmm_get_amount_out(
    amount_in: U256,
    zero2one: bool,
    reserve0: U256,
    reserve1: U256,
    fee_bps: u32,
    decimals0: u8,
    decimals1: u8,
) -> Result<U256, SimulationError> {
    if amount_in.is_zero() {
        return Err(SimulationError::InvalidInput("Amount in cannot be zero".to_string(), None));
    }

    if reserve0.is_zero() || reserve1.is_zero() {
        return Err(SimulationError::RecoverableError("No liquidity".to_string()));
    }

    let xy = _k(reserve0, reserve1, decimals0, decimals1)?;
    let e18 = U256::from(10u128.pow(18));
    let decimals0_scale = U256::from(10u128.pow(decimals0 as u32));
    let decimals1_scale = U256::from(10u128.pow(decimals1 as u32));

    let reserve0_normalized = safe_div_u256(safe_mul_u256(reserve0, e18)?, decimals0_scale)?;
    let reserve1_normalized = safe_div_u256(safe_mul_u256(reserve1, e18)?, decimals1_scale)?;

    let (reserve_in, reserve_out, decimals_in, decimals_out) = if zero2one {
        (reserve0_normalized, reserve1_normalized, decimals0, decimals1)
    } else {
        (reserve1_normalized, reserve0_normalized, decimals1, decimals0)
    };

    let fee_multiplier = U256::from(10000 - fee_bps);
    let amount_in_with_fee =
        safe_div_u256(safe_mul_u256(amount_in, fee_multiplier)?, U256::from(10000))?;

    let decimals_in_scale = U256::from(10u128.pow(decimals_in as u32));
    let amount_in_normalized =
        safe_div_u256(safe_mul_u256(amount_in_with_fee, e18)?, decimals_in_scale)?;

    let x0 = safe_add_u256(amount_in_normalized, reserve_in)?;
    let y_new = _get_y(x0, xy, reserve_out)?;
    let y_diff = safe_sub_u256(reserve_out, y_new)?;
    let decimals_out_scale = U256::from(10u128.pow(decimals_out as u32));
    let amount_out = safe_div_u256(safe_mul_u256(y_diff, decimals_out_scale)?, e18)?;

    Ok(amount_out)
}

/// f(x0, y) = (x0 * y / 1e18) * ((x0² + y²) / 1e18) / 1e18
fn _f(x0: U256, y: U256) -> Result<U256, SimulationError> {
    let e18 = U256::from(10u128.pow(18));

    // _a = (x0 * y) / 1e18
    let _a = safe_div_u256(safe_mul_u256(x0, y)?, e18)?;

    // _b = ((x0 * x0) / 1e18 + (y * y) / 1e18)
    let x0_squared = safe_div_u256(safe_mul_u256(x0, x0)?, e18)?;
    let y_squared = safe_div_u256(safe_mul_u256(y, y)?, e18)?;
    let _b = safe_add_u256(x0_squared, y_squared)?;

    // return (_a * _b) / 1e18
    safe_div_u256(safe_mul_u256(_a, _b)?, e18)
}

/// d(x0, y) = 3 * x0 * y² / 1e18² + x0³ / 1e18²
fn _d(x0: U256, y: U256) -> Result<U256, SimulationError> {
    let e18 = U256::from(10u128.pow(18));

    // 3 * x0 * ((y * y) / 1e18) / 1e18
    let y_squared = safe_div_u256(safe_mul_u256(y, y)?, e18)?;
    let term1 = safe_div_u256(safe_mul_u256(safe_mul_u256(U256::from(3), x0)?, y_squared)?, e18)?;

    // ((((x0 * x0) / 1e18) * x0) / 1e18)
    let x0_squared = safe_div_u256(safe_mul_u256(x0, x0)?, e18)?;
    let term2 = safe_div_u256(safe_mul_u256(x0_squared, x0)?, e18)?;

    safe_add_u256(term1, term2)
}

/// k = (x * y / 1e18) * ((x² + y²) / 1e18) / 1e18
fn _k(x: U256, y: U256, decimals0: u8, decimals1: u8) -> Result<U256, SimulationError> {
    let e18 = U256::from(10u128.pow(18));
    let decimals0_scale = U256::from(10u128.pow(decimals0 as u32));
    let decimals1_scale = U256::from(10u128.pow(decimals1 as u32));

    let _x = safe_div_u256(safe_mul_u256(x, e18)?, decimals0_scale)?;
    let _y = safe_div_u256(safe_mul_u256(y, e18)?, decimals1_scale)?;

    // _a = (_x * _y) / 1e18
    let _a = safe_div_u256(safe_mul_u256(_x, _y)?, e18)?;

    // _b = ((_x * _x) / 1e18 + (_y * _y) / 1e18)
    let _b = safe_add_u256(
        safe_div_u256(safe_mul_u256(_x, _x)?, e18)?,
        safe_div_u256(safe_mul_u256(_y, _y)?, e18)?,
    )?;

    // return (_a * _b) / 1e18
    safe_div_u256(safe_mul_u256(_a, _b)?, e18)
}

/// f(x0, y) = xy
fn _get_y(x0: U256, xy: U256, mut y: U256) -> Result<U256, SimulationError> {
    let e18 = U256::from(10u128.pow(18));

    for _ in 0..255 {
        let k = _f(x0, y)?;

        if k < xy {
            let d = _d(x0, y)?;
            if d.is_zero() {
                return Err(SimulationError::FatalError("Division by zero in _get_y".to_string()));
            }

            // dy = ((xy - k) * 1e18) / _d(x0, y)
            let diff = safe_sub_u256(xy, k)?;
            let mut dy = safe_div_u256(safe_mul_u256(diff, e18)?, d)?;

            if dy.is_zero() {
                if k == xy {
                    return Ok(y);
                }

                let y_plus_1 = safe_add_u256(y, U256::from(1))?;
                if _f(x0, y_plus_1)? > xy {
                    return Ok(y_plus_1);
                }

                dy = U256::from(1);
            }
            y = safe_add_u256(y, dy)?;
        } else {
            let d = _d(x0, y)?;
            if d.is_zero() {
                return Err(SimulationError::FatalError("Division by zero in _get_y".to_string()));
            }

            // dy = ((k - xy) * 1e18) / _d(x0, y)
            let diff = safe_sub_u256(k, xy)?;
            let mut dy = safe_div_u256(safe_mul_u256(diff, e18)?, d)?;

            if dy.is_zero() {
                if k == xy {
                    return Ok(y);
                }
                let y_minus_1 = safe_sub_u256(y, U256::from(1))?;
                if _f(x0, y_minus_1)? < xy {
                    return Ok(y);
                }
                dy = U256::from(1);
            }
            y = safe_sub_u256(y, dy)?;
        }
    }

    Err(SimulationError::FatalError(
        "Failed to converge in _get_y after 255 iterations".to_string(),
    ))
}

pub fn cfmm_get_limits(
    sell_token: Bytes,
    buy_token: Bytes,
    reserve0: U256,
    reserve1: U256,
    decimals0: u8,
    decimals1: u8,
) -> Result<(BigUint, BigUint), SimulationError> {
    if reserve0.is_zero() || reserve1.is_zero() {
        return Ok((BigUint::zero(), BigUint::zero()));
    }

    let zero_for_one = sell_token < buy_token;
    let (reserve_in, reserve_out, decimals_in, decimals_out) = if zero_for_one {
        (reserve0, reserve1, decimals0, decimals1)
    } else {
        (reserve1, reserve0, decimals1, decimals0)
    };

    // Calculate the invariant k = (x * y / 1e18) * ((x^2 + y^2) / 1e18) / 1e18
    let xy = _k(reserve0, reserve1, decimals0, decimals1)?;

    let e18 = U256::from(10u128.pow(18));
    let decimals_in_scale = U256::from(10u128.pow(decimals_in as u32));
    let decimals_out_scale = U256::from(10u128.pow(decimals_out as u32));

    // Normalize reserves to 18 decimals
    let reserve_in_normalized = safe_div_u256(safe_mul_u256(reserve_in, e18)?, decimals_in_scale)?;
    let reserve_out_normalized =
        safe_div_u256(safe_mul_u256(reserve_out, e18)?, decimals_out_scale)?;

    // Soft limit for amount_in is the amount to get a 90% price impact.
    // For stable swap curves, we need a larger multiplier than constant product (2.16x)
    // because the curve is flatter. We use 3x as a conservative estimate.
    // Target: (reserve_out - amount_out) / (reserve_in + amount_in) = 0.1 × (reserve_out /
    // reserve_in)
    let amount_in_estimate =
        safe_div_u256(safe_mul_u256(reserve_in, U256::from(300))?, U256::from(100))?;

    // Normalize amount_in to 18 decimals
    let amount_in_normalized =
        safe_div_u256(safe_mul_u256(amount_in_estimate, e18)?, decimals_in_scale)?;

    // Calculate new reserve_in after the swap
    let x0 = safe_add_u256(reserve_in_normalized, amount_in_normalized)?;

    // Use Newton's method to solve for the new reserve_out (y) that maintains the invariant
    let y_new = _get_y(x0, xy, reserve_out_normalized)?;

    // Calculate the output amount (normalized)
    let amount_out_normalized = safe_sub_u256(reserve_out_normalized, y_new)?;

    // Denormalize amount_out back to original token decimals
    let amount_out = safe_div_u256(safe_mul_u256(amount_out_normalized, decimals_out_scale)?, e18)?;

    Ok((u256_to_biguint(amount_in_estimate), u256_to_biguint(amount_out)))
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use super::*;

    #[test]
    fn test_cfmm_get_amount_out() {
        assert_eq!(
            cfmm_get_amount_out(
                U256::from_str("2000000000000000000").unwrap(),
                true,
                U256::from_str("2642455102346776307825").unwrap(),
                U256::from_str("3320301880379841502303").unwrap(),
                5,
                18,
                18
            ).unwrap(),
            U256::from_str("2004830151166915124").unwrap()
        )
    }
}