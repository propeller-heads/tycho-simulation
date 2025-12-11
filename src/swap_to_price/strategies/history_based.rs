//! History-based swap-to-price strategies
//!
//! These strategies use the history of (amount, price) pairs to estimate
//! derivatives and converge faster than simple interpolation methods.

use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};

use crate::swap_to_price::{
    within_tolerance, PriceMetric, ProtocolSimExt, QuerySupplyResult, SwapToPriceError,
    SwapToPriceResult, SWAP_TO_PRICE_MAX_ITERATIONS,
};

// =============================================================================
// Shared Utilities
// =============================================================================

/// Minimum threshold for denominators and differences to avoid division by near-zero.
/// Using 1e-12 as a practical threshold - much larger than MIN_DIVISOR (~2.2e-16)
/// but small enough to not reject valid numerical operations.
pub const MIN_DIVISOR: f64 = 1e-12;

/// Compute geometric mean of two BigUints: sqrt(low * high)
/// This is the midpoint in log-amount space.
pub fn geometric_mean(low: &BigUint, high: &BigUint) -> BigUint {
    if low.is_zero() {
        // sqrt(0 * high) = 0, which won't make progress
        // Return sqrt(high) instead
        return high.sqrt();
    }
    (low * high).sqrt()
}

/// Ensure the candidate amount is within bounds and makes progress
pub fn safe_next_amount(candidate: BigUint, low: &BigUint, high: &BigUint) -> Option<BigUint> {
    if candidate <= *low || candidate >= *high {
        None // Out of bounds, caller should fall back
    } else {
        Some(candidate)
    }
}

/// A point in the search history
#[derive(Clone, Debug)]
pub struct HistoryPoint {
    pub amount_f64: f64,
    pub price: f64,
}

// =============================================================================
// Common search loop infrastructure
// =============================================================================

/// Configuration for the search behavior
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Which price metric to track (spot price or trade price)
    pub metric: PriceMetric,
    /// Whether to return error on convergence failure (true) or best result (false)
    pub error_on_failure: bool,
}

impl SearchConfig {
    /// Config for swap_to_price: track spot price, error on convergence failure
    pub fn swap_to_price() -> Self {
        Self {
            metric: PriceMetric::SpotPrice,
            error_on_failure: true,
        }
    }

    /// Config for query_supply: track trade price, return best result on failure
    pub fn query_supply() -> Self {
        Self {
            metric: PriceMetric::TradePrice,
            error_on_failure: false,
        }
    }
}

/// Calculate trade price from amounts
fn calc_trade_price(amount_in: &BigUint, amount_out: &BigUint) -> Option<f64> {
    let in_f64 = amount_in.to_f64()?;
    let out_f64 = amount_out.to_f64()?;
    if out_f64 == 0.0 {
        return None;
    }
    Some(in_f64 / out_f64)
}

/// Tracked result during search
#[derive(Clone)]
struct TrackedResult {
    amount_in: BigUint,
    amount_out: BigUint,
    spot_price: f64,
    trade_price: f64,
    gas: BigUint,
    new_state: Box<dyn ProtocolSim>,
}

/// Run a price-targeting search with a custom next_amount function
///
/// This function supports both swap_to_price and query_supply via SearchConfig:
/// - `metric`: Which price to track (spot or trade)
/// - `error_on_failure`: Whether to return error or best result on convergence failure
pub fn run_search<F>(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    config: SearchConfig,
    mut next_amount_fn: F,
) -> Result<SwapToPriceResult, SwapToPriceError>
where
    F: FnMut(&[HistoryPoint], &BigUint, f64, &BigUint, f64, f64) -> BigUint,
{
    let max_iterations = SWAP_TO_PRICE_MAX_ITERATIONS;

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
        return Err(SwapToPriceError::TargetBelowSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 5: Calculate limit price (spot price at max trade)
    let limit_result = state.get_amount_out(max_amount_in.clone(), token_in, token_out)?;
    let limit_spot_price = limit_result.new_state.spot_price(token_out, token_in)?;

    // Step 6: Validate limit price
    if limit_spot_price <= spot_price {
        return Err(SwapToPriceError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 7: Validate target is reachable (using spot price as proxy)
    if target_price > limit_spot_price {
        return Err(SwapToPriceError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 8: Initialize search
    let mut low = BigUint::from(0u32);
    let mut low_price = spot_price; // This is spot price for bounds tracking
    let mut high = max_amount_in.clone();
    let mut high_price = limit_spot_price;
    let mut history: Vec<HistoryPoint> = Vec::with_capacity(max_iterations as usize);

    // Add initial bounds to history
    history.push(HistoryPoint {
        amount_f64: 0.0,
        price: low_price,
    });
    if let Some(high_f64) = high.to_f64() {
        history.push(HistoryPoint {
            amount_f64: high_f64,
            price: high_price,
        });
    }

    let mut actual_iterations = 0;

    // Track the best result seen during the entire search
    let mut best_result: Option<TrackedResult> = None;
    let mut best_error = f64::MAX;

    for iterations in 1..=max_iterations {
        actual_iterations = iterations;

        // Get next amount to try
        let mid = next_amount_fn(&history, &low, low_price, &high, high_price, target_price);

        // Ensure progress
        if mid <= low || mid >= high {
            break;
        }

        // Calculate result at mid
        let result = state.get_amount_out(mid.clone(), token_in, token_out)?;
        let new_spot_price = result.new_state.spot_price(token_out, token_in)?;
        let trade_price = calc_trade_price(&mid, &result.amount).unwrap_or(f64::MAX);

        // Get the price we're tracking based on metric
        let tracked_price = match config.metric {
            PriceMetric::SpotPrice => new_spot_price,
            PriceMetric::TradePrice => trade_price,
        };

        // Track the best result seen (closest to target without exceeding)
        let error = if tracked_price <= target_price {
            (target_price - tracked_price) / target_price
        } else {
            // Penalize exceeding target heavily
            (tracked_price - target_price) / target_price + 1000.0
        };

        if error < best_error {
            best_error = error;
            best_result = Some(TrackedResult {
                amount_in: mid.clone(),
                amount_out: result.amount.clone(),
                spot_price: new_spot_price,
                trade_price,
                gas: result.gas.clone(),
                new_state: result.new_state.clone(),
            });
        }

        // Add to history (always use spot price for interpolation - it's monotonic)
        if let Some(mid_f64) = mid.to_f64() {
            history.push(HistoryPoint {
                amount_f64: mid_f64,
                price: new_spot_price,
            });
        }

        // Check convergence
        if within_tolerance(tracked_price, target_price) {
            return Ok(SwapToPriceResult {
                amount_in: mid,
                amount_out: result.amount,
                actual_price: tracked_price,
                gas: result.gas,
                new_state: result.new_state,
                iterations,
            });
        }

        // Update bounds (always based on spot price for monotonicity)
        if new_spot_price < target_price {
            low = mid;
            low_price = new_spot_price;
        } else {
            high = mid;
            high_price = new_spot_price;
        }

        // Check for convergence to adjacent amounts
        if &high - &low <= BigUint::from(1u32) {
            break;
        }
    }

    // Search ended without convergence
    // Get best result from boundaries if we have them
    let low_result = if low > BigUint::from(0u32) {
        let res = state.get_amount_out(low.clone(), token_in, token_out)?;
        let spot = res.new_state.spot_price(token_out, token_in)?;
        let trade = calc_trade_price(&low, &res.amount).unwrap_or(f64::MAX);
        Some(TrackedResult {
            amount_in: low.clone(),
            amount_out: res.amount.clone(),
            spot_price: spot,
            trade_price: trade,
            gas: res.gas,
            new_state: res.new_state,
        })
    } else {
        None
    };

    let high_result = if high != low && high > BigUint::from(0u32) {
        let res = state.get_amount_out(high.clone(), token_in, token_out)?;
        let spot = res.new_state.spot_price(token_out, token_in)?;
        let trade = calc_trade_price(&high, &res.amount).unwrap_or(f64::MAX);
        Some(TrackedResult {
            amount_in: high.clone(),
            amount_out: res.amount.clone(),
            spot_price: spot,
            trade_price: trade,
            gas: res.gas,
            new_state: res.new_state,
        })
    } else {
        None
    };

    // Check boundaries for convergence
    for boundary_result in [&low_result, &high_result].into_iter().flatten() {
        let tracked_price = match config.metric {
            PriceMetric::SpotPrice => boundary_result.spot_price,
            PriceMetric::TradePrice => boundary_result.trade_price,
        };
        if within_tolerance(tracked_price, target_price) {
            return Ok(SwapToPriceResult {
                amount_in: boundary_result.amount_in.clone(),
                amount_out: boundary_result.amount_out.clone(),
                actual_price: tracked_price,
                gas: boundary_result.gas.clone(),
                new_state: boundary_result.new_state.clone(),
                iterations: actual_iterations,
            });
        }
    }

    // Update best_result with boundaries if they're better
    for boundary_result in [low_result, high_result].into_iter().flatten() {
        let tracked_price = match config.metric {
            PriceMetric::SpotPrice => boundary_result.spot_price,
            PriceMetric::TradePrice => boundary_result.trade_price,
        };
        let error = if tracked_price <= target_price {
            (target_price - tracked_price) / target_price
        } else {
            (tracked_price - target_price) / target_price + 1000.0
        };
        if error < best_error {
            best_error = error;
            best_result = Some(boundary_result);
        }
    }

    // Final handling depends on config
    match best_result {
        Some(result) => {
            let tracked_price = match config.metric {
                PriceMetric::SpotPrice => result.spot_price,
                PriceMetric::TradePrice => result.trade_price,
            };

            if config.error_on_failure && !within_tolerance(tracked_price, target_price) {
                // swap_to_price: return error if not converged
                let error_bps =
                    ((tracked_price - target_price).abs() / target_price) * 10000.0;
                Err(SwapToPriceError::ConvergenceFailure {
                    iterations: actual_iterations,
                    target_price,
                    best_price: tracked_price,
                    error_bps,
                    amount: result.amount_in.to_string(),
                })
            } else {
                // query_supply: return best result regardless of convergence
                Ok(SwapToPriceResult {
                    amount_in: result.amount_in,
                    amount_out: result.amount_out,
                    actual_price: tracked_price,
                    gas: result.gas,
                    new_state: result.new_state,
                    iterations: actual_iterations,
                })
            }
        }
        None => {
            // No valid result found at all
            Err(SwapToPriceError::ConvergenceFailure {
                iterations: actual_iterations,
                target_price,
                best_price: spot_price,
                error_bps: ((spot_price - target_price).abs() / target_price) * 10000.0,
                amount: "0".to_string(),
            })
        }
    }
}

// =============================================================================
// Strategy 1: Inverse Quadratic Interpolation (IQI)
// =============================================================================

/// Inverse Quadratic Interpolation strategy
///
/// Uses Lagrange interpolation in price space to estimate the amount
/// that would give the target price. Bootstraps with geometric mean.
pub struct IqiStrategy;

impl IqiStrategy {
    /// Compute IQI estimate from 3 points
    pub fn iqi(
        a1: f64,
        p1: f64,
        a2: f64,
        p2: f64,
        a3: f64,
        p3: f64,
        target: f64,
    ) -> Option<f64> {
        // Lagrange basis polynomials in price space
        let denom1 = (p1 - p2) * (p1 - p3);
        let denom2 = (p2 - p1) * (p2 - p3);
        let denom3 = (p3 - p1) * (p3 - p2);

        // Check for division by zero (duplicate prices)
        if denom1.abs() < MIN_DIVISOR
            || denom2.abs() < MIN_DIVISOR
            || denom3.abs() < MIN_DIVISOR
        {
            return None;
        }

        let l1 = (target - p2) * (target - p3) / denom1;
        let l2 = (target - p1) * (target - p3) / denom2;
        let l3 = (target - p1) * (target - p2) / denom3;

        let result = a1 * l1 + a2 * l2 + a3 * l3;

        if result.is_finite() && result > 0.0 {
            Some(result)
        } else {
            None
        }
    }

    /// IQI next amount function for use with run_search
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Need at least 3 points for IQI
        if history.len() < 3 {
            return fallback;
        }

        // Get the 3 most recent points with distinct prices
        let points: Vec<&HistoryPoint> = history
            .iter()
            .filter(|p| p.amount_f64 > 0.0 && p.price > 0.0)
            .collect();

        if points.len() < 3 {
            return fallback;
        }

        // Use last 3 points
        let n = points.len();
        let p1 = points[n - 3];
        let p2 = points[n - 2];
        let p3 = points[n - 1];

        // Try IQI
        if let Some(estimate) = Self::iqi(
            p1.amount_f64,
            p1.price,
            p2.amount_f64,
            p2.price,
            p3.amount_f64,
            p3.price,
            target,
        ) {
            if let Some(amount) = BigUint::from_f64(estimate) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for IqiStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::swap_to_price(),
            Self::next_amount,
        )
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let result = run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::query_supply(),
            Self::next_amount,
        )?;
        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}

// =============================================================================
// Strategy 2: Brent's Method
// =============================================================================

/// IQI acceptance threshold: fraction of bracket size.
/// IQI estimate is accepted if it improves the bracket by at least this fraction.
/// Benchmark result: 0.01 (1%) gives best performance (3.9 mean iterations).
const IQI_THRESHOLD: f64 = 0.01;

/// Brent's method: robust combination of IQI, secant, and bisection
///
/// Tries IQI first, falls back to secant, then bisection.
/// Guaranteed to make progress while achieving superlinear convergence when possible.
pub struct BrentStrategy;

impl BrentStrategy {
    /// Secant method estimate
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

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
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

            if let Some(estimate) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                // Brent's acceptance criterion: result must be within bounds
                // and must reduce the bracket by at least the threshold compared to bisection
                let bracket_size = high_f64 - low_f64;

                if estimate > low_f64 && estimate < high_f64 {
                    // Check if IQI is making reasonable progress
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

            if let Some(estimate) =
                Self::secant(p1.amount_f64, p1.price, p2.amount_f64, p2.price, target)
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

        // Fall back to geometric mean (bisection in log space)
        fallback
    }
}

impl ProtocolSimExt for BrentStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::swap_to_price(),
            Self::next_amount,
        )
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let result = run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::query_supply(),
            Self::next_amount,
        )?;
        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}

// =============================================================================
// Strategy 2b: Brent's Original Method (classic acceptance criterion)
// =============================================================================

/// Classic Brent step fraction: 0.5 (new step must be < half previous step)
const STEP_FRACTION: f64 = 0.5;

/// Brent's original method with the classic "half previous step" acceptance criterion.
///
/// This uses the original Brent acceptance check: the new step must be smaller than
/// half the previous step. This ensures accelerating convergence and prevents oscillation.
///
/// Benchmark result: Performs worse than `BrentStrategy` (5.0 vs 3.9 mean iterations).
/// The eff_tol check was found to make no difference and has been removed.
///
/// Compare with `BrentStrategy` which uses a simpler "1% of bracket" threshold.
pub struct BrentOriginalStrategy;

impl BrentOriginalStrategy {
    /// Secant method estimate
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

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
        prev_step: &mut f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);
        let bracket_size = high_f64 - low_f64;
        let midpoint = (low_f64 + high_f64) / 2.0;

        // Try IQI if we have 3+ points
        if history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            if let Some(estimate) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                if estimate > low_f64 && estimate < high_f64 {
                    // Classic Brent acceptance: new step < half previous step
                    let new_step = (estimate - midpoint).abs();

                    if new_step < *prev_step * STEP_FRACTION {
                        *prev_step = new_step;
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

            if let Some(estimate) =
                Self::secant(p1.amount_f64, p1.price, p2.amount_f64, p2.price, target)
            {
                if estimate > low_f64 && estimate < high_f64 {
                    let new_step = (estimate - midpoint).abs();

                    if new_step < *prev_step * STEP_FRACTION {
                        *prev_step = new_step;
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        // Fall back to bisection - update prev_step to bracket_size / 2
        *prev_step = bracket_size / 2.0;
        fallback
    }
}

impl ProtocolSimExt for BrentOriginalStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let mut prev_step = f64::MAX; // Start with large step so first IQI/secant is accepted
        run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::swap_to_price(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let mut prev_step = f64::MAX;
        let result = run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::query_supply(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )?;
        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}

// =============================================================================
// Strategy 2c: Brent AND (both criteria must be met)
// =============================================================================

/// Brent's method requiring BOTH acceptance criteria:
/// - 1% bracket threshold (our modification)
/// - Half previous step (classical Brent)
///
/// This is more conservative - falls back to bisection more often.
pub struct BrentAndStrategy;

impl BrentAndStrategy {
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

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
        prev_step: &mut f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);
        let bracket_size = high_f64 - low_f64;
        let midpoint = (low_f64 + high_f64) / 2.0;

        // Try IQI if we have 3+ points
        if history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            if let Some(estimate) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                if estimate > low_f64 && estimate < high_f64 {
                    let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    let new_step = (estimate - midpoint).abs();

                    // AND: both criteria must be met
                    let bracket_ok = iqi_improvement > bracket_size * IQI_THRESHOLD;
                    let step_ok = new_step < *prev_step * STEP_FRACTION;

                    if bracket_ok && step_ok {
                        *prev_step = new_step;
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

            if let Some(estimate) =
                Self::secant(p1.amount_f64, p1.price, p2.amount_f64, p2.price, target)
            {
                if estimate > low_f64 && estimate < high_f64 {
                    let secant_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    let new_step = (estimate - midpoint).abs();

                    let bracket_ok = secant_improvement > bracket_size * IQI_THRESHOLD;
                    let step_ok = new_step < *prev_step * STEP_FRACTION;

                    if bracket_ok && step_ok {
                        *prev_step = new_step;
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        // Fall back to bisection
        *prev_step = bracket_size / 2.0;
        fallback
    }
}

impl ProtocolSimExt for BrentAndStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let mut prev_step = f64::MAX;
        run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::swap_to_price(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let mut prev_step = f64::MAX;
        let result = run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::query_supply(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )?;
        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}

// =============================================================================
// Strategy 2d: Brent OR (either criterion suffices)
// =============================================================================

/// Brent's method accepting IQI if EITHER criterion is met:
/// - 1% bracket threshold (our modification)
/// - Half previous step (classical Brent)
///
/// This is more permissive - accepts more IQI steps.
pub struct BrentOrStrategy;

impl BrentOrStrategy {
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

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
        prev_step: &mut f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);
        let bracket_size = high_f64 - low_f64;
        let midpoint = (low_f64 + high_f64) / 2.0;

        // Try IQI if we have 3+ points
        if history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            if let Some(estimate) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                if estimate > low_f64 && estimate < high_f64 {
                    let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    let new_step = (estimate - midpoint).abs();

                    // OR: either criterion suffices
                    let bracket_ok = iqi_improvement > bracket_size * IQI_THRESHOLD;
                    let step_ok = new_step < *prev_step * STEP_FRACTION;

                    if bracket_ok || step_ok {
                        *prev_step = new_step;
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

            if let Some(estimate) =
                Self::secant(p1.amount_f64, p1.price, p2.amount_f64, p2.price, target)
            {
                if estimate > low_f64 && estimate < high_f64 {
                    let secant_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    let new_step = (estimate - midpoint).abs();

                    let bracket_ok = secant_improvement > bracket_size * IQI_THRESHOLD;
                    let step_ok = new_step < *prev_step * STEP_FRACTION;

                    if bracket_ok || step_ok {
                        *prev_step = new_step;
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        // Fall back to bisection
        *prev_step = bracket_size / 2.0;
        fallback
    }
}

impl ProtocolSimExt for BrentOrStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let mut prev_step = f64::MAX;
        run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::swap_to_price(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let mut prev_step = f64::MAX;
        let result = run_search(
            state,
            target_price,
            token_in,
            token_out,
            SearchConfig::query_supply(),
            |history, low, low_price, high, high_price, target| {
                Self::next_amount(history, low, low_price, high, high_price, target, &mut prev_step)
            },
        )?;
        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}

// =============================================================================
// Strategy 3: EVM Chandrupatla (uses crate::evm::chandrupatla module)
// =============================================================================

/// Wrapper for the EVM Chandrupatla implementation.
///
/// This strategy delegates to the `crate::evm::chandrupatla` module which
/// implements Chandrupatla's algorithm following the TensorFlow Probability
/// and SciPy reference implementations.
///
/// Unlike the archived `ChandrupatlaStrategy`, this version:
/// - Uses the correct xi/phi criterion: `(1 - √(1-ξ)) < φ < √ξ`
/// - Properly maintains the three-point bracket state (x1, x2, x3)
/// - Supports configurable tolerance, max_iterations, min_divisor, and t_min
pub struct EVMChandrupatlaStrategy {
    /// Configuration for the Chandrupatla algorithm
    pub config: crate::evm::chandrupatla::ChandrupatlaConfig,
}

impl Default for EVMChandrupatlaStrategy {
    fn default() -> Self {
        Self {
            config: crate::evm::chandrupatla::ChandrupatlaConfig::default(),
        }
    }
}

impl EVMChandrupatlaStrategy {
    /// Create with custom configuration
    pub fn with_config(config: crate::evm::chandrupatla::ChandrupatlaConfig) -> Self {
        Self { config }
    }
}

impl ProtocolSimExt for EVMChandrupatlaStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let result = crate::evm::chandrupatla::swap_to_price(
            state,
            target_price,
            token_in,
            token_out,
            Some(self.config),
        )
        .map_err(|e| match e {
            crate::evm::chandrupatla::ChandrupatlaSearchError::TargetBelowSpot { target, spot } => {
                SwapToPriceError::TargetBelowSpot { target, spot }
            }
            crate::evm::chandrupatla::ChandrupatlaSearchError::TargetAboveLimit {
                target,
                spot,
                limit,
            } => SwapToPriceError::TargetAboveLimit {
                target,
                spot,
                limit,
            },
            crate::evm::chandrupatla::ChandrupatlaSearchError::ConvergenceFailure {
                iterations,
                target_price,
                best_price,
                error_bps,
                amount,
            } => SwapToPriceError::ConvergenceFailure {
                iterations,
                target_price,
                best_price,
                error_bps,
                amount,
            },
            crate::evm::chandrupatla::ChandrupatlaSearchError::SimulationError(e) => {
                SwapToPriceError::SimulationError(e)
            }
        })?;

        Ok(SwapToPriceResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            actual_price: result.actual_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }

    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError> {
        let result = crate::evm::chandrupatla::query_supply(
            state,
            target_price,
            token_in,
            token_out,
            Some(self.config),
        )
        .map_err(|e| match e {
            crate::evm::chandrupatla::ChandrupatlaSearchError::TargetBelowSpot { target, spot } => {
                SwapToPriceError::TargetBelowSpot { target, spot }
            }
            crate::evm::chandrupatla::ChandrupatlaSearchError::TargetAboveLimit {
                target,
                spot,
                limit,
            } => SwapToPriceError::TargetAboveLimit {
                target,
                spot,
                limit,
            },
            crate::evm::chandrupatla::ChandrupatlaSearchError::ConvergenceFailure {
                iterations,
                target_price,
                best_price,
                error_bps,
                amount,
            } => SwapToPriceError::ConvergenceFailure {
                iterations,
                target_price,
                best_price,
                error_bps,
                amount,
            },
            crate::evm::chandrupatla::ChandrupatlaSearchError::SimulationError(e) => {
                SwapToPriceError::SimulationError(e)
            }
        })?;

        Ok(QuerySupplyResult {
            amount_in: result.amount_in,
            amount_out: result.amount_out,
            trade_price: result.trade_price,
            gas: result.gas,
            new_state: result.new_state,
            iterations: result.iterations,
        })
    }
}
