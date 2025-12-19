use alloy::primitives::{Sign, I256, U256};
use num_bigint::BigUint;
use tycho_common::{
    simulation::{errors::SimulationError, protocol_sim::Price},
    Bytes,
};

use crate::evm::protocol::{
    u256_num::u256_to_biguint,
    utils::uniswap::{sqrt_price_math::get_sqrt_price_limit, SwapResults},
};

// U160_MAX = 2^160 - 1, used for "infinite" swap amounts in swap_to_price
const U160_MAX: U256 = U256::from_limbs([u64::MAX, u64::MAX, u64::MAX >> 32, 0]); // 2^160 - 1

/// Calculates the exact amount of token_in required to move the pool's marginal price down to
/// a target price for Concentrated Liquidity Market Makers (CLMM).
///
/// This function encapsulates the common logic for swap_to_price across UniswapV3 and UniswapV4,
/// handling differences through the provided closure.
///
/// # Algorithm
///
/// Unlike constant product AMMs, CLMMs have concentrated liquidity that varies across price ranges.
/// The swap is executed by iterating through liquidity ticks until the target price is reached.
/// The function uses a large amount (U160_MAX) to find the maximum available liquidity that can
/// be swapped while respecting the target price limit.
///
/// # Arguments
/// * `sqrt_price` - Current sqrt price of the pool in Q96 format
/// * `token_in` - Token being sold
/// * `token_out` - Token being bought
/// * `target_price` - Target marginal (spot) price as token_out/token_in (tycho convention)
/// * `fee_pips` - Total fee in pips (1/1_000_000)
/// * `amount_sign` - Sign for the amount_specified (Positive for V3, Negative for V4)
/// * `swap_fn` - Closure that performs the actual swap operation
///
/// # Returns
/// A tuple containing (amount_in, amount_out, SwapResults)
///
/// # Errors
/// Returns `InvalidInput` if the target price is unreachable (already below current spot price)
#[allow(clippy::too_many_arguments)]
pub fn clmm_swap_to_price<F>(
    sqrt_price: U256,
    token_in: &Bytes,
    token_out: &Bytes,
    target_price: &Price,
    fee_pips: u32,
    amount_sign: Sign,
    swap_fn: F,
) -> Result<(BigUint, BigUint, SwapResults), SimulationError>
where
    F: FnOnce(bool, I256, U256) -> Result<SwapResults, SimulationError>,
{
    let zero_for_one = token_in < token_out;

    let sqrt_price_limit =
        get_sqrt_price_limit(token_in, token_out, target_price, U256::from(fee_pips))?;

    // Validate price limit is compatible with swap direction
    if zero_for_one && sqrt_price_limit >= sqrt_price {
        return Err(SimulationError::InvalidInput(
            "Target price is unreachable (already below current spot price)".to_string(),
            None,
        ));
    }
    if !zero_for_one && sqrt_price_limit <= sqrt_price {
        return Err(SimulationError::InvalidInput(
            "Target price is unreachable (already below current spot price)".to_string(),
            None,
        ));
    }

    // Use U160_MAX as "infinite" amount to find maximum available liquidity
    let amount_specified =
        I256::checked_from_sign_and_abs(amount_sign, U160_MAX).ok_or_else(|| {
            SimulationError::InvalidInput("I256 overflow: U160_MAX".to_string(), None)
        })?;

    // Call the provided swap function
    // The swap function should already ensure that the sqrt price result is on the correct side of
    // the sqrt_price_limit
    let result = swap_fn(zero_for_one, amount_specified, sqrt_price_limit)?;

    // Calculate amount_in from amount consumed: amount_in = amount_specified - amount_remaining
    let amount_in = (result.amount_specified - result.amount_remaining)
        .abs()
        .into_raw();

    if amount_in == U256::ZERO {
        return Ok((BigUint::ZERO, BigUint::ZERO, SwapResults::default()));
    }

    // Use the accumulated amount_calculated for output
    let amount_out = result
        .amount_calculated
        .abs()
        .into_raw();

    Ok((u256_to_biguint(amount_in), u256_to_biguint(amount_out), result))
}

/// Calculates the exact amount of token_in required such that the trade's execution price
/// equals the limit price for Concentrated Liquidity Market Makers (CLMM).
///
/// # Algorithm
///
/// Unlike `clmm_swap_to_price` which targets the *marginal* (spot) price after the swap,
/// this function targets the *trade* (average execution) price of the swap itself.
///
/// The trade price is defined as: `P = amount_in / amount_out`
///
/// Since CLMM has concentrated liquidity that varies across price ranges, there's no
/// closed-form solution. Instead, we execute the swap step-by-step until the accumulated
/// trade price reaches the limit.
///
/// ## Trade Price vs Spot Price
/// - **Spot price**: The marginal price at a specific point (price for infinitesimal swap)
/// - **Trade price**: The average price of an executed swap (total_in / total_out)
/// - Trade price accounts for price impact across the entire swap
///
/// # Arguments
/// * `sqrt_price` - Current sqrt price of the pool in Q96 format
/// * `token_in` - Token being sold
/// * `token_out` - Token being bought
/// * `limit_price` - Limit trade price as token_out/token_in (tycho convention)
/// * `fee_pips` - Total fee in pips (1/1_000_000)
/// * `tolerance` - Relative tolerance for trade price (e.g., 0.01 = 1%)
/// * `amount_sign` - Sign for the amount_specified (Positive for V3, Negative for V4)
/// * `swap_step_fn` - Closure that performs one step of the swap and checks trade price
///
/// # Returns
/// A tuple containing (amount_in, amount_out, SwapResults)
///
/// # Behavior
/// - If limit is achievable: swaps to achieve the limit price (within tolerance)
/// - If limit is worse than available: swaps all available liquidity (doesn't error)
/// - Errors only if limit is better than effective spot (impossible to achieve)
///
/// # Errors
/// Returns `InvalidInput` if limit is better than effective spot price (impossible to achieve)
#[allow(clippy::too_many_arguments)]
pub fn clmm_swap_to_trade_price<F>(
    sqrt_price: U256,
    token_in: &Bytes,
    token_out: &Bytes,
    limit_price: &Price,
    fee_pips: u32,
    tolerance: f64,
    amount_sign: Sign,
    swap_step_fn: F,
) -> Result<(BigUint, BigUint, SwapResults), SimulationError>
where
    F: FnOnce(bool, &Price, f64, Sign) -> Result<(U256, U256, SwapResults), SimulationError>,
{
    let zero_for_one = token_in < token_out;

    // Validate that limit trade price is not better than effective spot price
    // (which would be impossible to achieve)

    use num_traits::ToPrimitive;
    let limit_price_f64 = limit_price.numerator.to_f64().unwrap_or(0.0)
        / limit_price.denominator.to_f64().unwrap_or(1.0);

    // Validate against effective spot price (spot price after fees)
    // The best achievable trade price is the current spot price minus fees
    use crate::evm::protocol::u256_num::u256_to_f64;
    let sqrt_price_f64 = u256_to_f64(sqrt_price)?;
    let q96 = 2_f64.powi(96);
    let raw_spot = (sqrt_price_f64 / q96).powi(2); // This is token1/token0
    let spot_price_f64 = if zero_for_one {
        raw_spot // For zero_for_one: Price = token_out/token_in = token1/token0
    } else {
        1.0 / raw_spot // For !zero_for_one: Price = token_out/token_in = token0/token1
    };

    // Apply fees to get effective spot (best achievable trade price)
    // Using same precision as get_sqrt_price_limit: fee_precision = 1_000_000
    let fee_multiplier = 1.0 - (fee_pips as f64 / 1_000_000.0);
    let effective_spot_price = spot_price_f64 * fee_multiplier;

    // Check if limit is better than effective spot (impossible to achieve)
    if limit_price_f64 > effective_spot_price {
        return Err(SimulationError::InvalidInput(
            format!(
                "Limit trade price {:.6} is better than effective spot price {:.6} (spot {:.6} × {:.6}) - impossible to achieve",
                limit_price_f64, effective_spot_price, spot_price_f64, fee_multiplier
            ),
            None,
        ));
    }

    // Execute the swap with trade price checking
    // If limit is achievable: swaps to achieve the limit price
    // If limit is worse than available: swaps all available liquidity (natural behavior)
    let (amount_in, amount_out, result) = swap_step_fn(zero_for_one, limit_price, tolerance, amount_sign)?;

    if amount_in == U256::ZERO {
        return Ok((BigUint::ZERO, BigUint::ZERO, SwapResults::default()));
    }

    Ok((u256_to_biguint(amount_in), u256_to_biguint(amount_out), result))
}
