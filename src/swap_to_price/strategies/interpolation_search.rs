use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};

use crate::swap_to_price::{
    within_tolerance, SwapToPriceResult, SwapToPriceError, SwapToPriceStrategy,
    SWAP_TO_PRICE_MAX_ITERATIONS,
};

/// Debug assertions for interpolation function invariants
#[inline]
fn debug_assert_interpolation_invariants(
    low_amount: &BigUint,
    low_price: f64,
    high_amount: &BigUint,
    high_price: f64,
    target_price: f64,
) {
    debug_assert!(
        low_amount < high_amount,
        "low_amount must be < high_amount"
    );
    debug_assert!(
        low_price < target_price || within_tolerance(low_price, target_price),
        "low_price ({}) must be < target_price ({}) or within tolerance",
        low_price, target_price
    );
    debug_assert!(
        target_price < high_price || within_tolerance(target_price, high_price),
        "target_price ({}) must be < high_price ({}) or within tolerance",
        target_price, high_price
    );
}

/// Trait for different interpolation strategies for bounded search to calculate the next amount to test
pub trait InterpolationFunction {
    /// Calculate the interpolated amount based on current bounds and prices
    ///
    /// # Arguments
    /// * `low_amount` - Lower bound amount
    /// * `low_price` - Price at lower bound
    /// * `high_amount` - Upper bound amount
    /// * `high_price` - Price at upper bound
    /// * `target_price` - Target price we're searching for
    ///
    /// # Returns
    /// The next amount to test
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint;
}

/// Binary interpolation: always picks the midpoint, ignoring price information
pub struct BinaryInterpolation;

impl InterpolationFunction for BinaryInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);
        (low_amount + high_amount) / 2u32
    }
}

/// Linear interpolation: assumes price changes linearly with amount
pub struct LinearInterpolation;

impl InterpolationFunction for LinearInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let binary_midpoint = (low_amount + high_amount) / 2u32;

        // Linear interpolation formula:
        // mid = low + (high - low) * (target_price - low_price) / (high_price - low_price)
        let amount_range = high_amount - low_amount;
        let price_range = high_price - low_price;
        let price_offset = target_price - low_price;

        // Calculate interpolation ratio
        let ratio = price_offset / price_range;

        // Clamp ratio to [0, 1] to handle floating point errors
        let ratio = ratio.clamp(0.0, 1.0);

        // Convert to BigUint: mid = low + amount_range * ratio
        // Fall back to binary midpoint if conversion fails
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => return binary_midpoint,
        };
        let offset = (amount_range_f64 * ratio) as u128;

        // If interpolation gives no progress (offset == 0), fall back to binary midpoint
        if offset == 0 {
            return binary_midpoint;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Sqrt price interpolation: matches the x*y=k AMM curve
///
/// For constant product AMMs like UniswapV2, price grows quadratically with amount:
///   price(amount) = p₀ * (1 + amount/reserve)²
///
/// Therefore amount is proportional to sqrt(price), so we interpolate on sqrt(price):
///   ratio = (sqrt(target) - sqrt(low)) / (sqrt(high) - sqrt(low))
pub struct SqrtPriceInterpolation;

impl InterpolationFunction for SqrtPriceInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let binary_midpoint = (low_amount + high_amount) / 2u32;

        // Square root interpolation formula:
        // ratio = (sqrt(target_price) - sqrt(low_price)) / (sqrt(high_price) - sqrt(low_price))
        let sqrt_low = low_price.sqrt();
        let sqrt_high = high_price.sqrt();
        let sqrt_target = target_price.sqrt();

        let sqrt_range = sqrt_high - sqrt_low;
        let sqrt_offset = sqrt_target - sqrt_low;

        // Calculate interpolation ratio
        let ratio = sqrt_offset / sqrt_range;

        // Clamp ratio to [0, 1] to handle floating point errors
        let ratio = ratio.clamp(0.0, 1.0);

        // Convert to BigUint: mid = low + amount_range * ratio
        let amount_range = high_amount - low_amount;
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => return binary_midpoint,
        };
        let offset = (amount_range_f64 * ratio) as u128;

        // If interpolation gives no progress (offset == 0), fall back to binary midpoint
        if offset == 0 {
            return binary_midpoint;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Logarithmic bisection: uses sqrt of amount range as step size
///
/// Instead of halving the range (binary search), this takes sqrt-sized steps.
/// This narrows huge ranges faster by taking progressively smaller relative steps.
/// Adaptively probes from the end closer to the target price.
pub struct LogarithmicBisection;

impl InterpolationFunction for LogarithmicBisection {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let amount_range = high_amount - low_amount;

        // Use sqrt of amount_range as step size
        // For BigUint, we use integer sqrt
        let step = amount_range.sqrt();

        // If step is 0, fall back to midpoint
        if step == BigUint::from(0u32) {
            return (low_amount + high_amount) / 2u32;
        }

        // Adaptive: probe from the end closer to the target price
        let dist_to_low = target_price - low_price;
        let dist_to_high = high_price - target_price;

        if dist_to_low <= dist_to_high {
            // Target closer to low - probe from low going up
            low_amount + &step
        } else {
            // Target closer to high - probe from high going down
            high_amount - &step
        }
    }
}

/// Log price interpolation: interpolates in logarithmic price space
///
/// Formula: ratio = (log(target) - log(low)) / (log(high) - log(low))
///
/// Compresses huge price ranges into manageable log units.
/// 50 orders of magnitude becomes ~157 log2 units.
///
/// **Known limitation**: When target is very close to one bound in log space,
/// this strategy makes slow progress because it always probes near that bound.
pub struct LogPriceInterpolation;

impl InterpolationFunction for LogPriceInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let binary_midpoint = (low_amount + high_amount) / 2u32;

        // Log interpolation: ratio = (log(target) - log(low)) / (log(high) - log(low))
        let log_low = low_price.ln();
        let log_high = high_price.ln();
        let log_target = target_price.ln();

        let log_range = log_high - log_low;
        let log_offset = log_target - log_low;

        let ratio = (log_offset / log_range).clamp(0.0, 1.0);

        // DEBUG
        // eprintln!("  [log_price] log_low={:.6}, log_high={:.6}, log_target={:.6}", log_low, log_high, log_target);
        // eprintln!("  [log_price] log_range={:.6e}, log_offset={:.6e}, ratio={:.6e}", log_range, log_offset, ratio);

        let amount_range = high_amount - low_amount;
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => {
                // eprintln!("  [log_price] amount_range too large, falling back to binary");
                return binary_midpoint;
            }
        };

        let offset = (amount_range_f64 * ratio) as u128;
        // eprintln!("  [log_price] amount_range={:.6e}, offset={}", amount_range_f64, offset);

        if offset == 0 {
            // eprintln!("  [log_price] offset is 0, falling back to binary");
            return binary_midpoint;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Binary search in log-amount space
///
/// Instead of interpolating on price, we binary search where:
///   log_mid = (log(low_amount) + log(high_amount)) / 2
///   mid = exp(log_mid) = sqrt(low_amount * high_amount)  (geometric mean)
///
/// This narrows the amount range by sqrt each iteration, which is O(log log N)
/// for finding the right order of magnitude.
pub struct LogAmountBinarySearch;

impl InterpolationFunction for LogAmountBinarySearch {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        _low_price: f64,
        high_amount: &BigUint,
        _high_price: f64,
        _target_price: f64,
    ) -> BigUint {
        // Geometric mean = sqrt(low * high)
        // This is equivalent to midpoint in log-amount space
        let product = low_amount * high_amount;
        let mid = product.sqrt();

        // If low_amount is 0, sqrt(0 * high) = 0, which won't make progress
        // Fall back to sqrt(high) in that case
        if mid <= *low_amount {
            // Take sqrt of high_amount as the step from 0
            let sqrt_high = high_amount.sqrt();
            if sqrt_high > BigUint::from(0u32) {
                // eprintln!("  [log_amount] low=0, using sqrt(high)={}", &sqrt_high);
                return sqrt_high;
            }
            return (low_amount + high_amount) / 2u32;
        }

        // eprintln!("  [log_amount] geometric_mean={}", &mid);
        mid
    }
}

/// Exponential probing: doubles step size from low bound until overshoot
///
/// This is the fastest way to find the right order of magnitude.
/// Each iteration covers more ground than the last.
///
/// The step size is tracked externally, so this struct stores state.
pub struct ExponentialProbing {
    /// Current step multiplier (doubles each iteration until overshoot)
    pub step_multiplier: f64,
}

impl Default for ExponentialProbing {
    fn default() -> Self {
        Self { step_multiplier: 0.001 } // Start with 0.1% of range
    }
}

impl InterpolationFunction for ExponentialProbing {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let amount_range = high_amount - low_amount;
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => return (low_amount + high_amount) / 2u32,
        };

        // Use a fixed small multiplier - the search loop will narrow the range
        // We probe at step_multiplier of the current range from low
        let offset = (amount_range_f64 * self.step_multiplier) as u128;

        // eprintln!("  [exp_probe] range={:.6e}, multiplier={}, offset={}", amount_range_f64, self.step_multiplier, offset);

        if offset == 0 {
            return (low_amount + high_amount) / 2u32;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Secant method: uses two recent points to estimate where target is
///
/// Given two points (amount1, price1) and (amount2, price2), estimate
/// the amount that would give target_price using linear extrapolation:
///   amount = amount1 + (target - price1) * (amount2 - amount1) / (price2 - price1)
///
/// This is similar to Newton's method but doesn't require the derivative.
pub struct SecantMethod;

impl InterpolationFunction for SecantMethod {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let binary_midpoint = (low_amount + high_amount) / 2u32;

        // Secant formula: estimate amount using linear interpolation between two known points
        // amount = low + (target - low_price) * (high - low) / (high_price - low_price)
        let price_range = high_price - low_price;
        let price_offset = target_price - low_price;

        if price_range.abs() < f64::EPSILON {
            return binary_midpoint;
        }

        let ratio = (price_offset / price_range).clamp(0.0, 1.0);

        // eprintln!("  [secant] price_range={:.6e}, price_offset={:.6e}, ratio={:.6e}", price_range, price_offset, ratio);

        let amount_range = high_amount - low_amount;
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => return binary_midpoint,
        };

        let offset = (amount_range_f64 * ratio) as u128;

        if offset == 0 {
            return binary_midpoint;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Two-phase approach: aggressive narrowing then fine interpolation
///
/// Phase 1: Use geometric mean (log-amount binary search) to quickly narrow
///          the range until it's within 10x of target in price space
/// Phase 2: Switch to log-price interpolation for fine-grained search
pub struct TwoPhaseSearch;

impl InterpolationFunction for TwoPhaseSearch {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        let binary_midpoint = (low_amount + high_amount) / 2u32;

        // Check if we're in phase 1 (wide range) or phase 2 (narrow range)
        // Phase 1: price ratio > 10x, use geometric mean for fast narrowing
        // Phase 2: price ratio <= 10x, use log-price interpolation
        let price_ratio = high_price / low_price;

        if price_ratio > 10.0 {
            // Phase 1: Use geometric mean (midpoint in log-amount space)
            // eprintln!("  [two_phase] Phase 1: price_ratio={:.2e}, using geometric mean", price_ratio);

            // Geometric mean = sqrt(low * high)
            let product = low_amount * high_amount;
            let mid = product.sqrt();

            // Handle low_amount = 0 case
            if mid <= *low_amount {
                let sqrt_high = high_amount.sqrt();
                if sqrt_high > BigUint::from(0u32) {
                    return sqrt_high;
                }
                return binary_midpoint;
            }

            return mid;
        }

        // Phase 2: Use log-price interpolation for fine search
        // eprintln!("  [two_phase] Phase 2: price_ratio={:.2e}, using log-price interpolation", price_ratio);

        let log_low = low_price.ln();
        let log_high = high_price.ln();
        let log_target = target_price.ln();

        let log_range = log_high - log_low;
        let log_offset = log_target - log_low;

        let ratio = (log_offset / log_range).clamp(0.0, 1.0);

        let amount_range = high_amount - low_amount;
        let amount_range_f64 = match amount_range.to_f64() {
            Some(v) => v,
            None => return binary_midpoint,
        };

        let offset = (amount_range_f64 * ratio) as u128;

        if offset == 0 {
            return binary_midpoint;
        }

        low_amount + BigUint::from(offset)
    }
}

/// Bounded linear interpolation: limits deviation from binary search midpoint
/// This prevents worst-case performance on highly non-linear curves
pub struct BoundedLinearInterpolation {
    /// Maximum deviation from binary midpoint (0.0 to 1.0)
    /// 0.5 = can deviate up to 50% of the range from midpoint
    pub max_deviation: f64,
}

impl InterpolationFunction for BoundedLinearInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> BigUint {
        debug_assert_interpolation_invariants(low_amount, low_price, high_amount, high_price, target_price);

        // Calculate binary search midpoint
        let binary_mid = (low_amount + high_amount) / 2u32;

        // First, get linear interpolation result
        let linear = LinearInterpolation;
        let interpolated = linear.interpolate(
            low_amount,
            low_price,
            high_amount,
            high_price,
            target_price,
        );

        // Calculate the range
        let range = high_amount - low_amount;
        let max_offset = match range.to_f64() {
            Some(v) => (v * self.max_deviation) as u128,
            None => return binary_mid,
        };

        // Clamp interpolated result to be within max_deviation of binary midpoint
        let diff = if interpolated > binary_mid {
            &interpolated - &binary_mid
        } else {
            &binary_mid - &interpolated
        };

        if diff > BigUint::from(max_offset) {
            // Clamp to max deviation
            if interpolated > binary_mid {
                &binary_mid + BigUint::from(max_offset)
            } else {
                &binary_mid - BigUint::from(max_offset)
            }
        } else {
            interpolated
        }
    }
}

/// Interpolation search strategy for finding amount_in to reach target price
pub struct InterpolationSearchStrategy<F: InterpolationFunction> {
    pub interpolation_fn: F,
}

impl<F: InterpolationFunction> InterpolationSearchStrategy<F> {
    pub fn new(interpolation_fn: F) -> Self {
        Self { interpolation_fn }
    }
}

impl<F: InterpolationFunction> SwapToPriceStrategy for InterpolationSearchStrategy<F> {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let max_iterations = SWAP_TO_PRICE_MAX_ITERATIONS;

        // Step 1: Get spot price
        let spot_price = state.spot_price(token_out, token_in)?;

        // Step 2: Check if we're already at target (within tolerance)
        if within_tolerance(spot_price, target_price) {
            // Already at target, no swap needed
            let minimal_result = state.get_amount_out(BigUint::from(1u32), token_in, token_out)?;
            return Ok(SwapToPriceResult {
                amount_in: BigUint::from(0u32),
                actual_price: spot_price,
                gas: BigUint::from(0u32),
                new_state: minimal_result.new_state,
                iterations: 0,
            });
        }

        // Step 3: Get limits
        let (max_amount_in, _max_amount_out) =
            state.get_limits(token_in.address.clone(), token_out.address.clone())?;

        // Step 4: Validate target price is above spot
        if target_price < spot_price {
            return Err(SwapToPriceError::TargetBelowSpot {
                target: target_price,
                spot: spot_price,
            });
        }

        // Step 5: Calculate limit price by simulating max swap
        let limit_price = state
            .get_amount_out(max_amount_in.clone(), token_in, token_out)?
            .new_state
            .spot_price(token_out, token_in)?;

        // Step 6: Validate limit price is valid (should be > spot price)
        if limit_price <= spot_price {
            return Err(SwapToPriceError::TargetAboveLimit {
                target: target_price,
                spot: spot_price,
                limit: limit_price,
            });
        }

        // Step 7: Validate target is reachable within pool limits
        if target_price > limit_price {
            return Err(SwapToPriceError::TargetAboveLimit {
                target: target_price,
                spot: spot_price,
                limit: limit_price,
            });
        }

        // Step 8: Interpolation search
        let mut low = BigUint::from(0u32);
        let mut low_price = spot_price;
        let mut high = max_amount_in.clone();
        let mut high_price = limit_price;
        let mut actual_iterations = 0;

        for iterations in 1..=max_iterations {
            actual_iterations = iterations;

            // // DEBUG: Log bounds
            // eprintln!(
            //     "[iter {}] low={}, low_price={:.6e}, high={}, high_price={:.6e}, target={:.6e}",
            //     iterations, &low, low_price, &high, high_price, target_price
            // );

            // Calculate interpolated midpoint
            let mid = self
                .interpolation_fn
                .interpolate(
                    &low,
                    low_price,
                    &high,
                    high_price,
                    target_price,
                );

            // // DEBUG: Log interpolate result
            // eprintln!("[iter {}] mid={}", iterations, &mid);

            // Ensure we make progress
            if mid <= low || mid >= high {
                // No progress possible, try boundaries
                // eprintln!("[iter {}] NO PROGRESS: mid <= low ({}) or mid >= high ({})", iterations, mid <= low, mid >= high);
                break;
            }

            // Calculate price at midpoint
            let result = state.get_amount_out(mid.clone(), token_in, token_out)?;
            let price = result.new_state.spot_price(token_out, token_in)?;

            // // DEBUG: Log price result
            // eprintln!("[iter {}] price={:.6e}, diff from target={:.6e}", iterations, price, price - target_price);

            // Check if we're within tolerance
            if within_tolerance(price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: mid,
                    actual_price: price,
                    gas: result.gas.clone(),
                    new_state: result.new_state,
                    iterations,
                });
            }

            // Adjust search range
            if price < target_price {
                low = mid;
                low_price = price;
            } else {
                high = mid;
                high_price = price;
            }

            // Check for convergence
            if &high - &low <= BigUint::from(1u32) {
                break;
            }
        }

        // Get prices at both boundaries
        let low_result = if low > BigUint::from(0u32) {
            Some(state.get_amount_out(low.clone(), token_in, token_out)?)
        } else {
            None
        };

        let high_result = if high != low && high > BigUint::from(0u32) {
            Some(state.get_amount_out(high.clone(), token_in, token_out)?)
        } else {
            None
        };

        // Check if low boundary is within tolerance
        if let Some(ref res) = low_result {
            let price = res.new_state.spot_price(token_out, token_in)?;
            if within_tolerance(price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: low.clone(),
                    actual_price: price,
                    gas: res.gas.clone(),
                    new_state: res.new_state.clone(),
                    iterations: actual_iterations,
                });
            }
        }

        // Check if high boundary is within tolerance
        if let Some(ref res) = high_result {
            let price = res.new_state.spot_price(token_out, token_in)?;
            if within_tolerance(price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: high.clone(),
                    actual_price: price,
                    gas: res.gas.clone(),
                    new_state: res.new_state.clone(),
                    iterations: actual_iterations,
                });
            }
        }

        // Neither boundary is within tolerance - return "best achievable"
        // When search converges to adjacent amounts, the pool can't represent a more precise price.
        // Return whichever boundary is closer to the target.
        match (&low_result, &high_result) {
            (Some(low_res), Some(high_res)) => {
                let lp = low_res.new_state.spot_price(token_out, token_in)?;
                let hp = high_res.new_state.spot_price(token_out, token_in)?;

                // Return the one closer to target as "best achievable"
                let low_diff = (lp - target_price).abs();
                let high_diff = (hp - target_price).abs();

                if low_diff <= high_diff {
                    Ok(SwapToPriceResult {
                        amount_in: low.clone(),
                        actual_price: lp,
                        gas: low_res.gas.clone(),
                        new_state: low_res.new_state.clone(),
                        iterations: actual_iterations,
                    })
                } else {
                    Ok(SwapToPriceResult {
                        amount_in: high.clone(),
                        actual_price: hp,
                        gas: high_res.gas.clone(),
                        new_state: high_res.new_state.clone(),
                        iterations: actual_iterations,
                    })
                }
            }
            (Some(low_res), None) => {
                let lp = low_res.new_state.spot_price(token_out, token_in)?;
                Ok(SwapToPriceResult {
                    amount_in: low.clone(),
                    actual_price: lp,
                    gas: low_res.gas.clone(),
                    new_state: low_res.new_state.clone(),
                    iterations: actual_iterations,
                })
            }
            (None, Some(high_res)) => {
                let hp = high_res.new_state.spot_price(token_out, token_in)?;
                Ok(SwapToPriceResult {
                    amount_in: high.clone(),
                    actual_price: hp,
                    gas: high_res.gas.clone(),
                    new_state: high_res.new_state.clone(),
                    iterations: actual_iterations,
                })
            }
            (None, None) => {
                // No valid boundary results - this shouldn't happen in practice
                Err(SwapToPriceError::ConvergenceFailure(actual_iterations))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::U256;
    use rstest::rstest;
    use std::str::FromStr;
    use tycho_common::models::Chain;
    use crate::evm::protocol::uniswap_v2::state::UniswapV2State;

    // Test helper functions
    fn create_token0() -> Token {
        Token::new(
            &tycho_common::Bytes::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .unwrap(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn create_token1() -> Token {
        Token::new(
            &tycho_common::Bytes::from_str("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
                .unwrap(),
            "WETH",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn create_pool() -> UniswapV2State {
        UniswapV2State::new(
            U256::from_str("6005747565594546069633144").unwrap(),
            U256::from_str("2148576922062920125253").unwrap(),
        )
    }

    // Test all interpolation strategies
    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_binary_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(BinaryInterpolation);
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Binary: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_linear_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Linear: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_bounded_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(BoundedLinearInterpolation {
            max_deviation: 0.5,
        });
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Bounded: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[test]
    fn test_target_equals_spot_price() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy
            .swap_to_price(&pool, spot_price, &token0, &token1)
            .unwrap();

        assert_eq!(result.amount_in, BigUint::from(0u32));
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_target_below_spot_returns_error() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let target_price = spot_price * 0.99;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy.swap_to_price(&pool, target_price, &token0, &token1);

        assert!(matches!(result, Err(SwapToPriceError::TargetBelowSpot { .. })));
    }

    #[test]
    fn test_target_above_limit_returns_error() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let target_price = spot_price * 1000.0;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy.swap_to_price(&pool, target_price, &token0, &token1);

        assert!(matches!(
            result,
            Err(SwapToPriceError::TargetAboveLimit { .. })
        ));
    }
}
