//! Brent's method for finding swap amounts to reach target prices.
//!
//! This module provides two search modes:
//! - [`swap_to_price`]: Find amount to move **spot price** to target
//! - [`query_supply`]: Find max trade where **trade price** ≤ target

use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive};
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

// =============================================================================
// Constants
// =============================================================================

/// Tolerance for price convergence (0.001%)
const TOLERANCE: f64 = 0.00001;

/// Maximum iterations before giving up
const MAX_ITERATIONS: u32 = 100;

/// IQI acceptance threshold as fraction of bracket size.
/// IQI estimate is accepted if it improves the bracket by at least this fraction.
/// Benchmark result: 0.01 (1%) gives best performance (3.9 mean iterations).
const IQI_THRESHOLD: f64 = 0.01;

/// Minimum divisor to avoid division by zero
const MIN_DIVISOR: f64 = 1e-12;

// =============================================================================
// Types
// =============================================================================

/// Which price metric to track during the search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriceMetric {
    /// Track the resulting spot price after the swap
    SpotPrice,
    /// Track the trade price (execution price = amount_in / amount_out)
    TradePrice,
}

/// Configuration for the search algorithm
struct SearchConfig {
    metric: PriceMetric,
}

impl SearchConfig {
    fn swap_to_price() -> Self {
        Self { metric: PriceMetric::SpotPrice }
    }

    fn query_supply() -> Self {
        Self { metric: PriceMetric::TradePrice }
    }
}

/// A point in the search history
#[derive(Debug, Clone)]
struct HistoryPoint {
    amount_f64: f64,
    price: f64,
}

/// Result of a swap_to_price operation
#[derive(Debug, Clone)]
pub struct SwapToPriceResult {
    /// The amount of input token needed to achieve the target price
    pub amount_in: BigUint,
    /// The amount of output token received
    pub amount_out: BigUint,
    /// The actual final price achieved (spot price)
    pub actual_price: f64,
    /// Gas cost of the operation
    pub gas: BigUint,
    /// The updated protocol state after the swap
    pub new_state: Box<dyn ProtocolSim>,
    /// Number of get_amount_out calls (iterations) needed
    pub iterations: u32,
}

/// Result of a query_supply operation
#[derive(Debug, Clone)]
pub struct QuerySupplyResult {
    /// The amount of input token for this trade
    pub amount_in: BigUint,
    /// The amount of output token received
    pub amount_out: BigUint,
    /// The trade price (execution price = amount_in / amount_out)
    pub trade_price: f64,
    /// Gas cost of the operation
    pub gas: BigUint,
    /// The updated protocol state after the swap
    pub new_state: Box<dyn ProtocolSim>,
    /// Number of get_amount_out calls (iterations) needed
    pub iterations: u32,
}

/// Error types for Brent search operations
#[derive(Debug, thiserror::Error)]
pub enum BrentSearchError {
    #[error("Target price {target} is below spot price {spot}. Cannot reach target by trading in this direction.")]
    TargetBelowSpot { target: f64, spot: f64 },
    #[error("Target price {target} is above limit price {limit} (spot: {spot}). Pool doesn't have enough liquidity to reach target.")]
    TargetAboveLimit { target: f64, spot: f64, limit: f64 },
    #[error("Failed to converge within {iterations} iterations. Target: {target_price:.6e}, best: {best_price:.6e} (diff: {error_bps:.2}bps), amount: {amount}")]
    ConvergenceFailure {
        iterations: u32,
        target_price: f64,
        best_price: f64,
        error_bps: f64,
        amount: String,
    },
    #[error("Simulation error: {0}")]
    SimulationError(#[from] SimulationError),
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if actual price is within tolerance of target price (one-sided)
///
/// The target_price is a hard upper limit we must not exceed. Returns true if:
/// - `actual_price <= target_price` (hard upper limit), AND
/// - `actual_price >= target_price * (1 - TOLERANCE)` (within tolerance below)
pub fn within_tolerance(actual_price: f64, target_price: f64) -> bool {
    if actual_price > target_price {
        return false;
    }
    let lower_bound = target_price * (1.0 - TOLERANCE);
    actual_price >= lower_bound
}

/// Geometric mean of two BigUint values (bisection in log space)
fn geometric_mean(a: &BigUint, b: &BigUint) -> BigUint {
    let a_f64 = a.to_f64().unwrap_or(0.0);
    let b_f64 = b.to_f64().unwrap_or(f64::MAX);

    if a_f64 <= 0.0 || b_f64 <= 0.0 {
        return (a + b) / 2u32;
    }

    let result = (a_f64 * b_f64).sqrt();
    BigUint::from_f64(result).unwrap_or_else(|| (a + b) / 2u32)
}

/// Ensure the next amount is safely within bounds and different from current bounds
fn safe_next_amount(amount: BigUint, low: &BigUint, high: &BigUint) -> Option<BigUint> {
    if &amount <= low || &amount >= high {
        return None;
    }

    let one = BigUint::from(1u32);
    if &amount == &(low + &one) || &amount == &(high - &one) {
        return Some(amount);
    }

    Some(amount)
}

/// Inverse Quadratic Interpolation from 3 points
fn iqi(a1: f64, p1: f64, a2: f64, p2: f64, a3: f64, p3: f64, target: f64) -> Option<f64> {
    let denom1 = (p1 - p2) * (p1 - p3);
    let denom2 = (p2 - p1) * (p2 - p3);
    let denom3 = (p3 - p1) * (p3 - p2);

    if denom1.abs() < MIN_DIVISOR || denom2.abs() < MIN_DIVISOR || denom3.abs() < MIN_DIVISOR {
        return None;
    }

    let t1 = (target - p2) * (target - p3) / denom1;
    let t2 = (target - p1) * (target - p3) / denom2;
    let t3 = (target - p1) * (target - p2) / denom3;

    let result = a1 * t1 + a2 * t2 + a3 * t3;

    if result.is_finite() && result > 0.0 {
        Some(result)
    } else {
        None
    }
}

/// Secant method estimate from 2 points
fn secant(a1: f64, p1: f64, a2: f64, p2: f64, target: f64) -> Option<f64> {
    let dp = p2 - p1;
    if dp.abs() < MIN_DIVISOR {
        return None;
    }
    let result = a2 - (p2 - target) * (a2 - a1) / dp;
    if result.is_finite() && result > 0.0 {
        Some(result)
    } else {
        None
    }
}

/// Compute the next amount using Brent's method (1% bracket threshold variant)
fn brent_next_amount(
    history: &[HistoryPoint],
    low: &BigUint,
    high: &BigUint,
    target: f64,
) -> BigUint {
    let fallback = geometric_mean(low, high);
    let low_f64 = low.to_f64().unwrap_or(0.0);
    let high_f64 = high.to_f64().unwrap_or(f64::MAX);

    // Try IQI if we have 3+ points
    if history.len() >= 3 {
        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2];
        let p3 = &history[n - 1];

        if let Some(estimate) = iqi(
            p1.amount_f64,
            p1.price,
            p2.amount_f64,
            p2.price,
            p3.amount_f64,
            p3.price,
            target,
        ) {
            // Accept if estimate is within bounds AND improves bracket by >1%
            let bracket_size = high_f64 - low_f64;

            if estimate > low_f64 && estimate < high_f64 {
                let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                if iqi_improvement > bracket_size * IQI_THRESHOLD {
                    if let Some(amount) = BigUint::from_f64(estimate) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
            }
        }
    }

    // Try secant if we have 2+ points
    if history.len() >= 2 {
        let n = history.len();
        let p1 = &history[n - 2];
        let p2 = &history[n - 1];

        if let Some(estimate) = secant(p1.amount_f64, p1.price, p2.amount_f64, p2.price, target) {
            if estimate > low_f64 && estimate < high_f64 {
                if let Some(amount) = BigUint::from_f64(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }
    }

    // Fall back to geometric mean (bisection in log space)
    fallback
}

// =============================================================================
// Core Search Algorithm
// =============================================================================

/// Run the search algorithm with Brent's method
fn run_search(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    config: SearchConfig,
) -> Result<SwapToPriceResult, BrentSearchError> {
    // Step 1: Get spot price
    let spot_price = state.spot_price(token_out, token_in)?;

    // Step 2: Check if we're already at target (for spot price metric)
    if config.metric == PriceMetric::SpotPrice && within_tolerance(spot_price, target_price) {
        let minimal_result = state.get_amount_out(BigUint::from(1u32), token_in, token_out)?;
        return Ok(SwapToPriceResult {
            amount_in: BigUint::from(0u32),
            amount_out: BigUint::from(0u32),
            actual_price: spot_price,
            gas: BigUint::from(0u32),
            new_state: minimal_result.new_state,
            iterations: 0,
        });
    }

    // Step 3: Get limits
    let (max_amount_in, _) =
        state.get_limits(token_in.address.clone(), token_out.address.clone())?;

    // Step 4: Validate target price is above spot
    if target_price < spot_price {
        return Err(BrentSearchError::TargetBelowSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 5: Calculate limit price (spot price at max trade)
    let limit_result = state.get_amount_out(max_amount_in.clone(), token_in, token_out)?;
    let limit_spot_price = limit_result.new_state.spot_price(token_out, token_in)?;

    // Step 6: Validate limit price
    if limit_spot_price <= spot_price {
        return Err(BrentSearchError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 7: Validate target is reachable
    if target_price > limit_spot_price {
        return Err(BrentSearchError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 8: Initialize search
    let mut low = BigUint::from(0u32);
    let mut high = max_amount_in.clone();
    let mut history: Vec<HistoryPoint> = Vec::with_capacity(MAX_ITERATIONS as usize);

    // Add initial bounds to history
    history.push(HistoryPoint {
        amount_f64: 0.0,
        price: spot_price,
    });
    if let Some(high_f64) = high.to_f64() {
        history.push(HistoryPoint {
            amount_f64: high_f64,
            price: limit_spot_price,
        });
    }

    // Track best result
    let mut best_result: Option<SwapToPriceResult> = None;
    let mut best_error = f64::MAX;

    // Step 9: Main search loop
    for iteration in 0..MAX_ITERATIONS {
        // Calculate next amount using Brent's method
        let next_amount = brent_next_amount(&history, &low, &high, target_price);

        // Simulate the swap
        let result = state.get_amount_out(next_amount.clone(), token_in, token_out)?;

        // Get the price based on the metric
        let current_price = match config.metric {
            PriceMetric::SpotPrice => result.new_state.spot_price(token_out, token_in)?,
            PriceMetric::TradePrice => {
                let amount_in_f64 = next_amount.to_f64().unwrap_or(0.0);
                let amount_out_f64 = result.amount.to_f64().unwrap_or(1.0);
                if amount_out_f64 > 0.0 {
                    amount_in_f64 / amount_out_f64
                } else {
                    f64::MAX
                }
            }
        };

        // Add to history
        if let Some(amount_f64) = next_amount.to_f64() {
            history.push(HistoryPoint {
                amount_f64,
                price: current_price,
            });
        }

        // Calculate error
        let error = (current_price - target_price).abs() / target_price;

        // Track best result
        if error < best_error {
            best_error = error;
            best_result = Some(SwapToPriceResult {
                amount_in: next_amount.clone(),
                amount_out: result.amount.clone(),
                actual_price: current_price,
                gas: result.gas.clone(),
                new_state: result.new_state.clone(),
                iterations: iteration + 1,
            });
        }

        // Check convergence
        if within_tolerance(current_price, target_price) {
            return Ok(SwapToPriceResult {
                amount_in: next_amount,
                amount_out: result.amount,
                actual_price: current_price,
                gas: result.gas,
                new_state: result.new_state,
                iterations: iteration + 1,
            });
        }

        // Update bounds
        if current_price < target_price {
            low = next_amount;
        } else {
            high = next_amount;
        }

        // Check if we've converged to adjacent integers (precision limit)
        if &high - &low <= BigUint::from(1u32) {
            if let Some(best) = best_result {
                return Ok(best);
            }
        }
    }

    // Return convergence failure with best result info
    let best = best_result.unwrap_or(SwapToPriceResult {
        amount_in: BigUint::from(0u32),
        amount_out: BigUint::from(0u32),
        actual_price: spot_price,
        gas: BigUint::from(0u32),
        new_state: state.clone_box(),
        iterations: MAX_ITERATIONS,
    });

    Err(BrentSearchError::ConvergenceFailure {
        iterations: MAX_ITERATIONS,
        target_price,
        best_price: best.actual_price,
        error_bps: best_error * 10000.0,
        amount: best.amount_in.to_string(),
    })
}

// =============================================================================
// Public API
// =============================================================================

/// Find the amount of input token needed to reach a target spot price using Brent's method.
///
/// This finds how much `token_in` to sell for `token_out` to reach a target price.
/// When you sell `token_in` for `token_out`, you make `token_out` more expensive
/// (scarcer in the pool), so the spot price of `token_out` increases.
///
/// # Price Units
/// The `target_price` represents: **How many `token_in` needed to buy 1 `token_out`**
///
/// In other words: `target_price = spot_price(token_out, token_in)`
///
/// # Example
/// ```text
/// Pool: 6M DAI, 2k WETH
/// Current: spot_price(WETH, DAI) = 2795 DAI/WETH
/// Target:  3000 DAI/WETH (WETH becomes more expensive)
///
/// Call: swap_to_price(state, 3000.0, &DAI, &WETH)
/// Result: How much DAI to sell for WETH to reach 3000 DAI/WETH
/// ```
pub fn swap_to_price(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
) -> Result<SwapToPriceResult, BrentSearchError> {
    run_search(state, target_price, token_in, token_out, SearchConfig::swap_to_price())
}

/// Find the maximum trade where the trade price stays at or below the target.
///
/// This finds the largest amount of `token_in` that can be traded for `token_out`
/// while keeping the execution price (amount_in / amount_out) at or below `target_price`.
///
/// # Difference from swap_to_price
/// - **Metric**: Tracks trade price (execution price), not spot price
pub fn query_supply(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
) -> Result<QuerySupplyResult, BrentSearchError> {
    let result = run_search(state, target_price, token_in, token_out, SearchConfig::query_supply())?;
    Ok(QuerySupplyResult {
        amount_in: result.amount_in,
        amount_out: result.amount_out,
        trade_price: result.actual_price,
        gas: result.gas,
        new_state: result.new_state,
        iterations: result.iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_within_tolerance() {
        // TOLERANCE is 0.00001 (0.001%)

        // Exactly at target - should pass
        assert!(within_tolerance(1.0, 1.0));

        // Just below target (within tolerance) - should pass
        assert!(within_tolerance(0.999995, 1.0));
        assert!(within_tolerance(0.99999, 1.0));

        // Above target - should NEVER pass (hard limit)
        assert!(!within_tolerance(1.000001, 1.0));
        assert!(!within_tolerance(1.00001, 1.0));

        // Too far below target - should NOT pass
        assert!(!within_tolerance(0.9999, 1.0));
    }

    #[test]
    fn test_geometric_mean() {
        let a = BigUint::from(100u32);
        let b = BigUint::from(400u32);
        let result = geometric_mean(&a, &b);
        // sqrt(100 * 400) = sqrt(40000) = 200
        assert_eq!(result, BigUint::from(200u32));
    }

    #[test]
    fn test_iqi() {
        // Linear case: should give exact result
        let result = iqi(0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.5);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!((r - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_secant() {
        // Linear case
        let result = secant(0.0, 0.0, 2.0, 2.0, 1.0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!((r - 1.0).abs() < 0.001);
    }
}
