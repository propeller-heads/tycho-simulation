//! Brent-style method for finding swap amounts to reach target prices.
//!
//! This module provides two search modes:
//! - [`swap_to_price`]: Find amount to move **spot price** to target
//! - [`query_supply`]: Find max trade where **trade price** ≤ target
//!
//! # Implementation Notes
//!
//! This implementation is inspired by Brent's method but adapted for the
//! discrete, non-linear nature of AMM price curves. It combines:
//! - Inverse Quadratic Interpolation (IQI) for fast convergence
//! - Secant method as a fallback
//! - Geometric mean bisection (log-space) as the final fallback
//!
//! Unlike classic Brent which uses complex state tracking and x-space
//! convergence, this uses a simpler history-based approach with
//! **price-space convergence**.
//!
//! ## References
//!
//! - Brent, R. P. (1973). "Algorithms for Minimization without Derivatives."
//!   Prentice-Hall. Chapter 4.
//!
//! Reference implementations:
//! - SciPy brentq: <https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/brentq.c>
//!
//! ## Algorithm
//!
//! At each iteration:
//! 1. Try IQI if 3+ history points exist and the estimate is within bounds
//! 2. Try secant method if 2+ history points exist
//! 3. Fall back to geometric mean (bisection in log-space)
//! 4. Update bracket based on whether price is above/below target
//!
//! ## Convergence
//!
//! Terminates when:
//! 1. The price is within tolerance of the target
//! 2. The discrete precision limit is reached: `high - low <= 1` (BigUint)

use num_bigint::BigUint;
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration parameters for Brent-style root-finding algorithm.
///
/// These parameters control convergence behavior, numerical stability,
/// and iteration limits.
#[derive(Debug, Clone, Copy)]
pub struct BrentConfig {
    /// Relative tolerance for price convergence.
    /// The algorithm converges when `|actual - target| / target < tolerance`.
    /// Default: 0.00001 (0.001%)
    pub tolerance: f64,

    /// Maximum number of iterations before giving up.
    /// Default: 100
    pub max_iterations: u32,

    /// Minimum divisor to avoid division by zero in numerical computations.
    /// Default: 1e-12
    pub min_divisor: f64,

    /// IQI acceptance threshold: fraction of bracket size.
    /// IQI estimate is accepted if it improves the bracket by at least this fraction.
    /// Default: 0.01 (1%)
    pub iqi_threshold: f64,
}

impl Default for BrentConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.00001,
            max_iterations: 30,
            min_divisor: 1e-12,
            iqi_threshold: 0.01,
        }
    }
}

impl BrentConfig {
    /// Creates a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the minimum divisor for numerical stability.
    pub fn with_min_divisor(mut self, min_divisor: f64) -> Self {
        self.min_divisor = min_divisor;
        self
    }

    /// Sets the IQI acceptance threshold.
    pub fn with_iqi_threshold(mut self, iqi_threshold: f64) -> Self {
        self.iqi_threshold = iqi_threshold;
        self
    }
}

// =============================================================================
// Types
// =============================================================================

/// Which price metric to track during the search.
///
/// All prices are in units of **token_out per token_in**.
/// Both metrics DECREASE as amount_in increases due to slippage.
/// Valid targets: `limit_price <= target < spot_price`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriceMetric {
    /// Track the resulting spot price (marginal rate) after the swap.
    SpotPrice,
    /// Track the trade price (execution price = amount_out / amount_in).
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

/// Calculate trade price (execution price) normalized by token decimals.
///
/// Trade price = (amount_out / amount_in) * 10^(decimals_in - decimals_out)
///
/// This gives price in units of **token_out per token_in**.
fn calculate_trade_price(
    amount_in: f64,
    amount_out: f64,
    decimals_in: u32,
    decimals_out: u32,
) -> f64 {
    if amount_in <= 0.0 {
        return f64::MAX;
    }
    let decimal_adjustment = 10_f64.powi(decimals_in as i32 - decimals_out as i32);
    (amount_out / amount_in) * decimal_adjustment
}

/// A point in the search history (amount, price)
#[derive(Debug, Clone, Copy)]
struct HistoryPoint {
    amount: f64,
    price: f64,
}

/// Ring buffer of last 3 history points for IQI/secant interpolation.
///
/// Points are stored in insertion order (oldest first when full).
/// We only need 3 points for IQI and 2 for secant, so no need to store full history.
#[derive(Debug, Clone, Default)]
struct RecentHistory {
    points: [Option<HistoryPoint>; 3],
    count: usize,
}

impl RecentHistory {
    fn new() -> Self {
        Self { points: [None; 3], count: 0 }
    }

    fn push(&mut self, amount: f64, price: f64) {
        let point = HistoryPoint { amount, price };
        if self.count < 3 {
            self.points[self.count] = Some(point);
            self.count += 1;
        } else {
            // Shift left and add at end (drop oldest)
            self.points[0] = self.points[1];
            self.points[1] = self.points[2];
            self.points[2] = Some(point);
        }
    }

    fn len(&self) -> usize {
        self.count
    }

    /// Get the last N points (most recent last)
    fn last_n(&self, n: usize) -> Vec<HistoryPoint> {
        let n = n.min(self.count);
        let start = self.count.saturating_sub(n);
        (start..self.count)
            .filter_map(|i| self.points[i])
            .collect()
    }
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
    #[error("Target price {target} is above spot price {spot}. Target must be below spot (prices decrease with amount).")]
    TargetAboveSpot { target: f64, spot: f64 },
    #[error("Target price {target} is below limit price {limit} (spot: {spot}). Pool cannot reach such a low price.")]
    TargetBelowLimit { target: f64, spot: f64, limit: f64 },
    #[error("Limit price {limit} is at or above spot price {spot}. Expected limit < spot since prices decrease.")]
    LimitAboveSpot { limit: f64, spot: f64 },
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

/// Check if actual price is within tolerance of target price (one-sided).
///
/// The target_price is a hard upper limit we must not exceed. Returns true if:
/// - `actual_price <= target_price` (hard upper limit), AND
/// - `actual_price >= target_price * (1 - tolerance)` (within tolerance below)
pub fn within_tolerance(actual_price: f64, target_price: f64, tolerance: f64) -> bool {
    if actual_price > target_price {
        return false;
    }
    let lower_bound = target_price * (1.0 - tolerance);
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

/// Inverse Quadratic Interpolation from 3 points
fn iqi(
    a1: f64,
    p1: f64,
    a2: f64,
    p2: f64,
    a3: f64,
    p3: f64,
    target: f64,
    min_divisor: f64,
) -> Option<f64> {
    let denom1 = (p1 - p2) * (p1 - p3);
    let denom2 = (p2 - p1) * (p2 - p3);
    let denom3 = (p3 - p1) * (p3 - p2);

    if denom1.abs() < min_divisor || denom2.abs() < min_divisor || denom3.abs() < min_divisor {
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

/// Secant method estimate
fn secant(a1: f64, p1: f64, a2: f64, p2: f64, target: f64, min_divisor: f64) -> Option<f64> {
    let dp = p2 - p1;
    if dp.abs() < min_divisor {
        return None;
    }
    let result = a2 - (p2 - target) * (a2 - a1) / dp;
    if result.is_finite() && result > 0.0 {
        Some(result)
    } else {
        None
    }
}

/// Ensure the next amount is safely within bounds and different from current bounds
fn safe_next_amount(amount: BigUint, low: &BigUint, high: &BigUint) -> Option<BigUint> {
    if &amount <= low || &amount >= high {
        return None;
    }
    Some(amount)
}

/// Compute the next amount using Brent-style method.
///
/// Tries IQI first, then secant, then falls back to geometric mean.
fn brent_next_amount(
    history: &RecentHistory,
    low: &BigUint,
    high: &BigUint,
    target: f64,
    config: &BrentConfig,
) -> BigUint {
    let fallback = geometric_mean(low, high);
    let low_f64 = low.to_f64().unwrap_or(0.0);
    let high_f64 = high.to_f64().unwrap_or(f64::MAX);

    // Try IQI if we have 3 points
    if history.len() >= 3 {
        let pts = history.last_n(3);
        if pts.len() == 3 {
            if let Some(estimate) = iqi(
                pts[0].amount,
                pts[0].price,
                pts[1].amount,
                pts[1].price,
                pts[2].amount,
                pts[2].price,
                target,
                config.min_divisor,
            ) {
                // Accept if within bounds and making reasonable progress
                let bracket_size = high_f64 - low_f64;
                if estimate > low_f64 && estimate < high_f64 {
                    let improvement = (estimate - low_f64).min(high_f64 - estimate);
                    if improvement > bracket_size * config.iqi_threshold {
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }
    }

    // Try secant if we have 2+ points
    if history.len() >= 2 {
        let pts = history.last_n(2);
        if pts.len() == 2 {
            if let Some(estimate) =
                secant(pts[0].amount, pts[0].price, pts[1].amount, pts[1].price, target, config.min_divisor)
            {
                if estimate > low_f64 && estimate < high_f64 {
                    if let Some(amount) = BigUint::from_f64(estimate) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
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

/// Run the search algorithm with Brent-style method.
///
/// Uses a simpler history-based approach that works better for AMM price curves.
fn run_search(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    search_config: SearchConfig,
    config: &BrentConfig,
) -> Result<SwapToPriceResult, BrentSearchError> {
    // Step 1: Get spot price (token_out per token_in)
    let spot_price = state.spot_price(token_in, token_out)?;

    // Step 2: Check if we're already at target (for spot price metric)
    if search_config.metric == PriceMetric::SpotPrice
        && within_tolerance(spot_price, target_price, config.tolerance)
    {
        let minimal_result = state.get_amount_out(BigUint::one(), token_in, token_out)?;
        return Ok(SwapToPriceResult {
            amount_in: BigUint::zero(),
            amount_out: BigUint::zero(),
            actual_price: spot_price,
            gas: BigUint::zero(),
            new_state: minimal_result.new_state,
            iterations: 0,
        });
    }

    // Step 3: Get limits
    let (max_amount_in, _) =
        state.get_limits(token_in.address.clone(), token_out.address.clone())?;

    // Step 4: Validate target price is below spot price
    // With prices in token_out/token_in units, prices DECREASE as amount increases.
    // spot_price is the maximum achievable (at amount=0), so target must be below it.
    if target_price > spot_price {
        return Err(BrentSearchError::TargetAboveSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 5: Calculate limit price (metric-specific)
    let limit_result = state.get_amount_out(max_amount_in.clone(), token_in, token_out)?;

    let limit_price = match search_config.metric {
        PriceMetric::SpotPrice => limit_result.new_state.spot_price(token_in, token_out)?,
        PriceMetric::TradePrice => {
            let max_in_f64 = max_amount_in.to_f64().unwrap_or(f64::MAX);
            let max_out_f64 = limit_result.amount.to_f64().unwrap_or(0.0);
            calculate_trade_price(max_in_f64, max_out_f64, token_in.decimals, token_out.decimals)
        }
    };

    // Step 6: Validate limit price (should be below spot since prices decrease)
    if limit_price >= spot_price {
        return Err(BrentSearchError::LimitAboveSpot {
            limit: limit_price,
            spot: spot_price,
        });
    }

    // Step 7: Validate target is reachable (must be >= limit_price)
    if target_price < limit_price {
        return Err(BrentSearchError::TargetBelowLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_price,
        });
    }

    // Step 8: Initialize search state
    let mut low = BigUint::zero();
    let mut low_price = spot_price;
    let mut high = max_amount_in.clone();
    let mut high_price = limit_price;
    let mut history = RecentHistory::new();

    // Add initial bounds to history
    // For amount=0, use spot_price as approximation (once low moves, we get real trade prices)
    history.push(0.0, spot_price);
    if let Some(high_f64) = high.to_f64() {
        history.push(high_f64, limit_price);
    }

    // Track best result
    let mut best_result: Option<SwapToPriceResult> = None;
    let mut best_error = f64::MAX;

    // Step 9: Main search loop
    for iteration in 0..config.max_iterations {
        // Calculate next amount using Brent-style method
        let next_amount = brent_next_amount(&history, &low, &high, target_price, config);

        // Simulate the swap
        let result = state.get_amount_out(next_amount.clone(), token_in, token_out)?;

        // Get the price based on the metric
        let current_price = match search_config.metric {
            PriceMetric::SpotPrice => result.new_state.spot_price(token_in, token_out)?,
            PriceMetric::TradePrice => {
                let amount_in_f64 = next_amount.to_f64().unwrap_or(0.0);
                let amount_out_f64 = result.amount.to_f64().unwrap_or(0.0);
                calculate_trade_price(
                    amount_in_f64,
                    amount_out_f64,
                    token_in.decimals,
                    token_out.decimals,
                )
            }
        };

        // Add to history
        if let Some(amount_f64) = next_amount.to_f64() {
            history.push(amount_f64, current_price);
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
        if within_tolerance(current_price, target_price, config.tolerance) {
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
        // Price decreases with amount: price > target → need more amount, price < target → need less
        if current_price > target_price {
            low = next_amount;
            low_price = current_price;
        } else {
            high = next_amount;
            high_price = current_price;
        }

        // Suppress unused variable warnings
        let _ = (low_price, high_price);

        // Check if we've converged to adjacent integers (precision limit)
        // Evaluate both boundaries to find if either is within tolerance
        if &high - &low <= BigUint::one() {
            let mut best_boundary: Option<SwapToPriceResult> = None;
            let mut best_boundary_error = f64::MAX;

            for boundary_amount in [&low, &high] {
                if boundary_amount.is_zero() {
                    // For TradePrice metric, amount=0 means target is at or below spot price.
                    // Return spot_price as the theoretical best trade price.
                    if search_config.metric == PriceMetric::TradePrice {
                        return Ok(SwapToPriceResult {
                            amount_in: BigUint::zero(),
                            amount_out: BigUint::zero(),
                            actual_price: spot_price,
                            gas: BigUint::zero(),
                            new_state: state.clone_box(),
                            iterations: iteration + 1,
                        });
                    }
                    continue;
                }
                let result = state.get_amount_out(boundary_amount.clone(), token_in, token_out)?;
                let price = match search_config.metric {
                    PriceMetric::SpotPrice => result.new_state.spot_price(token_in, token_out)?,
                    PriceMetric::TradePrice => {
                        let amount_in_f64 = boundary_amount.to_f64().unwrap_or(0.0);
                        let amount_out_f64 = result.amount.to_f64().unwrap_or(0.0);
                        calculate_trade_price(
                            amount_in_f64,
                            amount_out_f64,
                            token_in.decimals,
                            token_out.decimals,
                        )
                    }
                };

                // Check if this boundary is within tolerance - return immediately
                if within_tolerance(price, target_price, config.tolerance) {
                    return Ok(SwapToPriceResult {
                        amount_in: boundary_amount.clone(),
                        amount_out: result.amount,
                        actual_price: price,
                        gas: result.gas,
                        new_state: result.new_state,
                        iterations: iteration + 1,
                    });
                }

                // Track the best boundary (closest to target without exceeding)
                let error = if price <= target_price {
                    (target_price - price) / target_price
                } else {
                    (price - target_price) / target_price + 1000.0
                };
                if error < best_boundary_error {
                    best_boundary_error = error;
                    best_boundary = Some(SwapToPriceResult {
                        amount_in: boundary_amount.clone(),
                        amount_out: result.amount,
                        actual_price: price,
                        gas: result.gas,
                        new_state: result.new_state,
                        iterations: iteration + 1,
                    });
                }
            }

            // Return the best boundary result, or the tracked best_result
            if let Some(result) = best_boundary {
                return Ok(result);
            }
            if let Some(best) = best_result {
                return Ok(best);
            }
        }
    }

    // Return convergence failure with best result info
    let best = best_result.unwrap_or(SwapToPriceResult {
        amount_in: BigUint::zero(),
        amount_out: BigUint::zero(),
        actual_price: spot_price,
        gas: BigUint::zero(),
        new_state: state.clone_box(),
        iterations: config.max_iterations,
    });

    Err(BrentSearchError::ConvergenceFailure {
        iterations: config.max_iterations,
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
/// Call: swap_to_price(state, 3000.0, &DAI, &WETH, None)
/// Result: How much DAI to sell for WETH to reach 3000 DAI/WETH
/// ```
pub fn swap_to_price(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    config: Option<BrentConfig>,
) -> Result<SwapToPriceResult, BrentSearchError> {
    let config = config.unwrap_or_default();
    run_search(state, target_price, token_in, token_out, SearchConfig::swap_to_price(), &config)
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
    config: Option<BrentConfig>,
) -> Result<QuerySupplyResult, BrentSearchError> {
    let config = config.unwrap_or_default();
    let result =
        run_search(state, target_price, token_in, token_out, SearchConfig::query_supply(), &config)?;
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

    const DEFAULT_TOLERANCE: f64 = 0.00001;

    // =========================================================================
    // Tests for within_tolerance
    // =========================================================================

    #[test]
    fn test_within_tolerance() {
        // Exactly at target - should pass
        assert!(within_tolerance(1.0, 1.0, DEFAULT_TOLERANCE));

        // Just below target (within tolerance) - should pass
        assert!(within_tolerance(0.999995, 1.0, DEFAULT_TOLERANCE));
        assert!(within_tolerance(0.99999, 1.0, DEFAULT_TOLERANCE));

        // Above target - should NEVER pass (hard limit)
        assert!(!within_tolerance(1.000001, 1.0, DEFAULT_TOLERANCE));
        assert!(!within_tolerance(1.00001, 1.0, DEFAULT_TOLERANCE));

        // Too far below target - should NOT pass
        assert!(!within_tolerance(0.9999, 1.0, DEFAULT_TOLERANCE));
    }

    #[test]
    fn test_geometric_mean() {
        let a = BigUint::from(100u32);
        let b = BigUint::from(400u32);
        let result = geometric_mean(&a, &b);
        // sqrt(100 * 400) = sqrt(40000) = 200
        assert_eq!(result, BigUint::from(200u32));
    }

    // =========================================================================
    // Tests for Brent's root-finding algorithm
    // =========================================================================

    /// Pure Brent root-finding (for testing the algorithm itself)
    ///
    /// This implementation follows the classic Brent-Dekker algorithm as described
    /// in Brent (1973) and implemented in SciPy/Numerical Recipes.
    fn find_root_brent<F>(f: F, mut a: f64, mut b: f64, xtol: f64, max_iter: u32) -> (f64, u32)
    where
        F: Fn(f64) -> f64,
    {
        let rtol = 4.0 * f64::EPSILON;
        let mut fa = f(a);
        let mut fb = f(b);

        // Ensure fa and fb have opposite signs
        assert!(
            fa * fb <= 0.0,
            "f(a)={} and f(b)={} must have opposite signs",
            fa,
            fb
        );

        // c is the "contrapoint" - it brackets the root with b
        // Initially c = a (so [b, c] brackets the root)
        let mut c = a;
        let mut fc = fa;

        // d is the step size, e is the step before d (for safety checks)
        let mut d = b - a;
        let mut e = d;

        for iteration in 0..max_iter {
            // Ensure the root is bracketed by [b, c] with |f(b)| <= |f(c)|
            if (fb > 0.0) == (fc > 0.0) {
                // b and c have the same sign, so reset c to a
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }

            if fc.abs() < fb.abs() {
                // Swap so that b has the smaller function value
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            // Convergence check
            let tol1 = 2.0 * rtol * b.abs() + xtol / 2.0;
            let m = (c - b) / 2.0; // midpoint

            if m.abs() <= tol1 || fb == 0.0 {
                return (b, iteration);
            }

            // Decide whether to use interpolation or bisection
            if e.abs() >= tol1 && fa.abs() > fb.abs() {
                // Try interpolation
                let (p, q) = if (a - c).abs() < f64::EPSILON {
                    // Linear interpolation (secant method)
                    // We have two points: (a, fa) and (b, fb)
                    // Linear interpolation gives: x = b - fb * (b - a) / (fb - fa)
                    let s = fb / fa;
                    let p = 2.0 * m * s;
                    let q = 1.0 - s;
                    (p, q)
                } else {
                    // Inverse quadratic interpolation
                    // We have three points: (a, fa), (b, fb), (c, fc)
                    let q = fa / fc;
                    let r = fb / fc;
                    let s = fb / fa;
                    let p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                    let q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                    (p, q)
                };

                // Adjust signs so q > 0
                let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

                // Safety check: interpolated step must:
                // 1. Keep the new point within the bracket (between b and c)
                // 2. Not be larger than half the previous step (ensure shrinking)
                let min_bound = 3.0 * m * q - (tol1 * q).abs();
                let prev_step_bound = (e * q).abs();

                if 2.0 * p.abs() < min_bound.min(prev_step_bound) {
                    // Accept interpolation
                    e = d;
                    d = p / q;
                } else {
                    // Reject interpolation, use bisection
                    d = m;
                    e = m;
                }
            } else {
                // Bisection
                d = m;
                e = m;
            }

            // Update a (save the old b)
            a = b;
            fa = fb;

            // Update b
            if d.abs() > tol1 {
                b += d;
            } else {
                // Take a minimal step in the direction of the root
                b += if m > 0.0 { tol1 } else { -tol1 };
            }
            fb = f(b);
        }

        (b, max_iter)
    }

    /// Test Brent's method on f(x) = x^3 - 2x - 5
    /// Root ≈ 2.0946
    #[test]
    fn test_brent_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let expected_root = 2.0945514815423265;

        let (root, iterations) = find_root_brent(f, 2.0, 3.0, 1e-10, 100);

        assert!(
            (root - expected_root).abs() < 1e-8,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
        assert!(iterations < 20, "Should converge quickly, took {} iterations", iterations);
    }

    /// Test on f(x) = x^2 - 2, root at sqrt(2) ≈ 1.4142
    #[test]
    fn test_brent_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let expected_root = std::f64::consts::SQRT_2;

        let (root, iterations) = find_root_brent(f, 1.0, 2.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
        assert!(iterations < 50, "Should converge reasonably fast, took {} iterations", iterations);
    }

    /// Test on f(x) = sin(x), root at π ≈ 3.1416
    #[test]
    fn test_brent_sine() {
        let f = |x: f64| x.sin();
        let expected_root = std::f64::consts::PI;

        let (root, iterations) = find_root_brent(f, 3.0, 4.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
        assert!(iterations < 50, "Should converge reasonably fast, took {} iterations", iterations);
    }

    /// Test on f(x) = (x - 3)^3, root at x = 3 (triple root)
    #[test]
    fn test_brent_triple_root() {
        let f = |x: f64| (x - 3.0).powi(3);
        let expected_root = 3.0;

        let (root, iterations) = find_root_brent(f, 0.0, 10.0, 1e-10, 100);

        // Triple roots converge more slowly
        assert!(
            (root - expected_root).abs() < 1e-3,
            "Expected root = {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }

    /// Test on f(x) = x^9, root at x = 0
    #[test]
    fn test_brent_high_power() {
        let f = |x: f64| x.powi(9);

        let (root, iterations) = find_root_brent(f, -1.0, 1.0, 1e-10, 100);

        assert!(root.abs() < 1e-3, "Expected root ≈ 0, got {} after {} iterations", root, iterations);
    }

    /// Test on exp(x) - 2, root at ln(2) ≈ 0.693
    #[test]
    fn test_brent_exponential() {
        let f = |x: f64| x.exp() - 2.0;
        let expected_root = 2.0_f64.ln();

        let (root, iterations) = find_root_brent(f, 0.0, 1.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
        assert!(iterations < 50, "Should converge reasonably fast, took {} iterations", iterations);
    }

    /// Test on a function with a root very close to one bracket endpoint
    #[test]
    fn test_brent_root_near_endpoint() {
        let f = |x: f64| x - 0.001;
        let expected_root = 0.001;

        let (root, iterations) = find_root_brent(f, 0.0, 1.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }

    /// Test comparing Brent's method efficiency against bisection
    /// Brent should converge in a reasonable number of iterations
    #[test]
    fn test_brent_efficiency() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;

        let (root, brent_iterations) = find_root_brent(f, 2.0, 3.0, 1e-12, 100);

        // Bisection would take log2(1.0 / 1e-12) ≈ 40 iterations
        // Brent should converge in a comparable or better number
        assert!(brent_iterations < 50, "Brent took {} iterations, should converge", brent_iterations);
        assert!((root - 2.0945514815423265).abs() < 1e-10);
    }

    /// Test on polynomial: x^2 - x - 2 = (x-2)(x+1), root at x = 2
    #[test]
    fn test_brent_quadratic() {
        let f = |x: f64| x * x - x - 2.0;
        let expected_root = 2.0;

        let (root, iterations) = find_root_brent(f, 1.5, 3.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root = {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }

    /// Stress test: high-degree polynomial (x - 1.5)^15
    #[test]
    fn test_brent_degree_15() {
        let expected_root = 1.5;
        let f = |x: f64| (x - expected_root).powi(15);

        let (root, iterations) = find_root_brent(f, -20.0, 20.0, 1e-12, 200);

        // High-degree polynomials are very flat near roots
        assert!(
            (root - expected_root).abs() < 0.5,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }
}
