use alloy::primitives::{Sign, I256, U256, U512};
use num_bigint::BigUint;
use tycho_common::{
    simulation::{errors::SimulationError, protocol_sim::Price},
    Bytes,
};

use crate::evm::protocol::{
    u256_num::{biguint_to_u256, u256_to_biguint},
    utils::uniswap::{sqrt_price_math::get_sqrt_price_limit, SwapResults, SwapToTradePriceResult},
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
/// The trade price is defined as: `P = amount_out / amount_in`
///
/// This uses an analytical formula to compute the exact sqrt_price needed to achieve
/// the target trade price within each tick range, iterating through tick boundaries
/// as needed.
///
/// ## Trade Price vs Spot Price
/// - **Spot price**: The marginal price at a specific point (price for infinitesimal swap)
/// - **Trade price**: The average price of an executed swap (total_out / total_in)
/// - Trade price accounts for price impact across the entire swap
///
/// # Arguments
/// * `sqrt_price` - Current sqrt price of the pool in Q96 format
/// * `token_in` - Token being sold
/// * `token_out` - Token being bought
/// * `limit_price` - Limit trade price as token_out/token_in (tycho convention)
/// * `fee_pips` - Total fee in pips (1/1_000_000)
/// * `swap_fn` - Closure that performs the swap to achieve target trade price
///
/// # Returns
/// A tuple containing (amount_in, amount_out, SwapResults)
///
/// # Behavior
/// - If limit is achievable: swaps to achieve the exact limit price
/// - If limit is worse than available: swaps all available liquidity (doesn't error)
/// - Errors only if limit is better than effective spot (impossible to achieve)
///
/// # Errors
/// Returns `InvalidInput` if limit is better than effective spot price (impossible to achieve)
pub fn clmm_swap_to_trade_price<F>(
    sqrt_price: U256,
    token_in: &Bytes,
    token_out: &Bytes,
    limit_price: &Price,
    fee_pips: u32,
    swap_fn: F,
) -> Result<(BigUint, BigUint, SwapResults), SimulationError>
where
    F: FnOnce(bool, &Price) -> Result<SwapToTradePriceResult, SimulationError>,
{
    let zero_for_one = token_in < token_out;

    // Validate that limit trade price is not better than effective spot price
    // (which would be impossible to achieve)
    //
    // We use cross-multiplication to avoid f64 precision issues:
    //
    // For zero_for_one:
    //   spot_price = sqrt_price² / 2^192
    //   effective_spot = spot_price × (1_000_000 - fee_pips) / 1_000_000
    //   Check: limit_num/limit_den > effective_spot
    //   Cross-mult: limit_num × 2^192 × 1_000_000 > sqrt_price² × (1_000_000 - fee_pips) × limit_den
    //
    // For !zero_for_one:
    //   spot_price = 2^192 / sqrt_price²
    //   effective_spot = spot_price × (1_000_000 - fee_pips) / 1_000_000
    //   Check: limit_num/limit_den > effective_spot
    //   Cross-mult: limit_num × sqrt_price² × 1_000_000 > 2^192 × (1_000_000 - fee_pips) × limit_den

    let limit_num = U512::from(biguint_to_u256(&limit_price.numerator));
    let limit_den = U512::from(biguint_to_u256(&limit_price.denominator));
    let sqrt_price_512 = U512::from(sqrt_price);
    let sqrt_price_squared = sqrt_price_512.checked_mul(sqrt_price_512).ok_or_else(|| {
        SimulationError::FatalError("Overflow in sqrt_price squared calculation".to_string())
    })?;
    let two_192 = U512::from(1u64) << 192;
    let fee_factor = U512::from(1_000_000 - fee_pips);
    let fee_precision = U512::from(1_000_000u32);

    let limit_is_better_than_spot = if zero_for_one {
        // limit_num × 2^192 × 1_000_000 > sqrt_price² × (1_000_000 - fee_pips) × limit_den
        let lhs = limit_num
            .checked_mul(two_192)
            .and_then(|x| x.checked_mul(fee_precision));
        let rhs = sqrt_price_squared
            .checked_mul(fee_factor)
            .and_then(|x| x.checked_mul(limit_den));

        match (lhs, rhs) {
            (Some(l), Some(r)) => l > r,
            _ => {
                return Err(SimulationError::FatalError(
                    "Overflow in limit price comparison".to_string(),
                ))
            }
        }
    } else {
        // limit_num × sqrt_price² × 1_000_000 > 2^192 × (1_000_000 - fee_pips) × limit_den
        let lhs = limit_num
            .checked_mul(sqrt_price_squared)
            .and_then(|x| x.checked_mul(fee_precision));
        let rhs = two_192
            .checked_mul(fee_factor)
            .and_then(|x| x.checked_mul(limit_den));

        match (lhs, rhs) {
            (Some(l), Some(r)) => l > r,
            _ => {
                return Err(SimulationError::FatalError(
                    "Overflow in limit price comparison".to_string(),
                ))
            }
        }
    };

    if limit_is_better_than_spot {
        return Err(SimulationError::InvalidInput(
            "Limit trade price is better than effective spot price - impossible to achieve"
                .to_string(),
            None,
        ));
    }

    // Execute the swap using analytical calculation
    // If limit is achievable: swaps to achieve the exact limit price
    // If limit is worse than available: swaps all available liquidity (natural behavior)
    let result = swap_fn(zero_for_one, limit_price)?;

    if result.amount_in == U256::ZERO {
        return Ok((BigUint::ZERO, BigUint::ZERO, SwapResults::default()));
    }

    Ok((u256_to_biguint(result.amount_in), u256_to_biguint(result.amount_out), result.swap_state))
}
