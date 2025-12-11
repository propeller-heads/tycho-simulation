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
const MIN_DIVISOR: f64 = 1e-12;

/// Compute geometric mean of two BigUints: sqrt(low * high)
/// This is the midpoint in log-amount space.
fn geometric_mean(low: &BigUint, high: &BigUint) -> BigUint {
    if low.is_zero() {
        // sqrt(0 * high) = 0, which won't make progress
        // Return sqrt(high) instead
        return high.sqrt();
    }
    (low * high).sqrt()
}

/// Ensure the candidate amount is within bounds and makes progress
fn safe_next_amount(candidate: BigUint, low: &BigUint, high: &BigUint) -> Option<BigUint> {
    if candidate <= *low || candidate >= *high {
        None // Out of bounds, caller should fall back
    } else {
        Some(candidate)
    }
}

/// A point in the search history
#[derive(Clone, Debug)]
pub struct HistoryPoint {
    amount_f64: f64,
    price: f64,
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
    fn iqi(
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
}

impl BrentStrategy {
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
                // and must reduce the bracket by at least half compared to bisection
                let bracket_size = high_f64 - low_f64;

                if estimate > low_f64 && estimate < high_f64 {
                    // Check if IQI is making reasonable progress
                    let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    if iqi_improvement > bracket_size * 0.01 {
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
// Strategy 3: Newton with Central Difference
// =============================================================================

/// Newton's method with central difference derivative estimation
///
/// Uses (p3 - p1) / (a3 - a1) as the derivative estimate at the middle point.
pub struct NewtonCentralStrategy;

impl NewtonCentralStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Need at least 3 points
        if history.len() < 3 {
            return fallback;
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2]; // Middle point - we'll apply Newton here
        let p3 = &history[n - 1];

        // Central difference derivative
        let da = p3.amount_f64 - p1.amount_f64;
        let dp = p3.price - p1.price;

        if da.abs() < MIN_DIVISOR || dp.abs() < MIN_DIVISOR {
            return fallback;
        }

        let derivative = dp / da; // d(price)/d(amount)
        let error = p2.price - target;

        // Newton step: a_new = a2 - error / derivative
        let estimate = p2.amount_f64 - error / derivative;

        if estimate.is_finite() && estimate > 0.0 {
            if let Some(amount) = BigUint::from_f64(estimate) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for NewtonCentralStrategy {
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
// Strategy 4: Quadratic Regression
// =============================================================================

/// Quadratic regression strategy
///
/// Fits price = a*x^2 + b*x + c to all history points using least squares,
/// then solves for x where the quadratic equals target_price.
pub struct QuadraticRegressionStrategy;

impl QuadraticRegressionStrategy {
    /// Fit quadratic and solve for target
    fn fit_and_solve(history: &[HistoryPoint], target: f64) -> Option<f64> {
        let n = history.len();
        if n < 3 {
            return None;
        }

        // Least squares: solve (X^T X) * beta = X^T * y
        // where X = [x^2, x, 1] and y = price

        // Compute sums for normal equations
        let mut sum_x4 = 0.0;
        let mut sum_x3 = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_x = 0.0;
        let mut sum_1 = 0.0;
        let mut sum_x2y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_y = 0.0;

        for p in history.iter() {
            let x = p.amount_f64;
            let y = p.price;
            let x2 = x * x;

            sum_x4 += x2 * x2;
            sum_x3 += x2 * x;
            sum_x2 += x2;
            sum_x += x;
            sum_1 += 1.0;
            sum_x2y += x2 * y;
            sum_xy += x * y;
            sum_y += y;
        }

        // Solve 3x3 system using Cramer's rule
        // | sum_x4  sum_x3  sum_x2 | | a |   | sum_x2y |
        // | sum_x3  sum_x2  sum_x  | | b | = | sum_xy  |
        // | sum_x2  sum_x   sum_1  | | c |   | sum_y   |

        let det = sum_x4 * (sum_x2 * sum_1 - sum_x * sum_x)
                - sum_x3 * (sum_x3 * sum_1 - sum_x * sum_x2)
                + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);

        if det.abs() < MIN_DIVISOR {
            return None;
        }

        let a = (sum_x2y * (sum_x2 * sum_1 - sum_x * sum_x)
               - sum_x3 * (sum_xy * sum_1 - sum_x * sum_y)
               + sum_x2 * (sum_xy * sum_x - sum_x2 * sum_y)) / det;

        let b = (sum_x4 * (sum_xy * sum_1 - sum_x * sum_y)
               - sum_x2y * (sum_x3 * sum_1 - sum_x * sum_x2)
               + sum_x2 * (sum_x3 * sum_y - sum_xy * sum_x2)) / det;

        let c = (sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy)
               - sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y)
               + sum_x2y * (sum_x3 * sum_x - sum_x2 * sum_x2)) / det;

        // Solve a*x^2 + b*x + (c - target) = 0
        let c_adjusted = c - target;

        if a.abs() < MIN_DIVISOR {
            // Linear case
            if b.abs() < MIN_DIVISOR {
                return None;
            }
            return Some(-c_adjusted / b);
        }

        // Quadratic formula
        let discriminant = b * b - 4.0 * a * c_adjusted;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let x1 = (-b + sqrt_disc) / (2.0 * a);
        let x2 = (-b - sqrt_disc) / (2.0 * a);

        // Return the positive root that's closest to our search range
        let valid_roots: Vec<f64> = [x1, x2]
            .iter()
            .filter(|&&x| x.is_finite() && x > 0.0)
            .copied()
            .collect();

        if valid_roots.is_empty() {
            None
        } else if valid_roots.len() == 1 {
            Some(valid_roots[0])
        } else {
            // Pick the one closest to the last point
            let last_x = history.last().map(|p| p.amount_f64).unwrap_or(0.0);
            if (valid_roots[0] - last_x).abs() < (valid_roots[1] - last_x).abs() {
                Some(valid_roots[0])
            } else {
                Some(valid_roots[1])
            }
        }
    }
}

impl QuadraticRegressionStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Need at least 4 points for meaningful regression
        if history.len() < 4 {
            return fallback;
        }

        if let Some(estimate) = Self::fit_and_solve(history, target) {
            if let Some(amount) = BigUint::from_f64(estimate) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for QuadraticRegressionStrategy {
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
// Strategy 5: Weighted Regression
// =============================================================================

/// Weighted quadratic regression strategy
///
/// Same as QuadraticRegressionStrategy but gives higher weight to recent points.
pub struct WeightedRegressionStrategy {
    /// Decay factor for weighting (0.5 = 50% weight reduction per point back)
    pub decay: f64,
}

impl Default for WeightedRegressionStrategy {
    fn default() -> Self {
        Self { decay: 0.7 }
    }
}

impl WeightedRegressionStrategy {
    /// Fit weighted quadratic and solve for target
    fn fit_and_solve(history: &[HistoryPoint], target: f64, decay: f64) -> Option<f64> {
        let n = history.len();
        if n < 3 {
            return None;
        }

        // Weighted least squares with exponential decay
        let mut sum_x4 = 0.0;
        let mut sum_x3 = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_x = 0.0;
        let mut sum_w = 0.0;
        let mut sum_x2y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_y = 0.0;

        for (i, p) in history.iter().enumerate() {
            let weight = decay.powi((n - 1 - i) as i32);
            let x = p.amount_f64;
            let y = p.price;
            let x2 = x * x;

            sum_x4 += weight * x2 * x2;
            sum_x3 += weight * x2 * x;
            sum_x2 += weight * x2;
            sum_x += weight * x;
            sum_w += weight;
            sum_x2y += weight * x2 * y;
            sum_xy += weight * x * y;
            sum_y += weight * y;
        }

        // Solve weighted 3x3 system
        let det = sum_x4 * (sum_x2 * sum_w - sum_x * sum_x)
                - sum_x3 * (sum_x3 * sum_w - sum_x * sum_x2)
                + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2);

        if det.abs() < MIN_DIVISOR {
            return None;
        }

        let a = (sum_x2y * (sum_x2 * sum_w - sum_x * sum_x)
               - sum_x3 * (sum_xy * sum_w - sum_x * sum_y)
               + sum_x2 * (sum_xy * sum_x - sum_x2 * sum_y)) / det;

        let b = (sum_x4 * (sum_xy * sum_w - sum_x * sum_y)
               - sum_x2y * (sum_x3 * sum_w - sum_x * sum_x2)
               + sum_x2 * (sum_x3 * sum_y - sum_xy * sum_x2)) / det;

        let c = (sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy)
               - sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y)
               + sum_x2y * (sum_x3 * sum_x - sum_x2 * sum_x2)) / det;

        // Solve quadratic
        let c_adjusted = c - target;

        if a.abs() < MIN_DIVISOR {
            if b.abs() < MIN_DIVISOR {
                return None;
            }
            return Some(-c_adjusted / b);
        }

        let discriminant = b * b - 4.0 * a * c_adjusted;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let x1 = (-b + sqrt_disc) / (2.0 * a);
        let x2 = (-b - sqrt_disc) / (2.0 * a);

        let valid_roots: Vec<f64> = [x1, x2]
            .iter()
            .filter(|&&x| x.is_finite() && x > 0.0)
            .copied()
            .collect();

        if valid_roots.is_empty() {
            None
        } else if valid_roots.len() == 1 {
            Some(valid_roots[0])
        } else {
            let last_x = history.last().map(|p| p.amount_f64).unwrap_or(0.0);
            if (valid_roots[0] - last_x).abs() < (valid_roots[1] - last_x).abs() {
                Some(valid_roots[0])
            } else {
                Some(valid_roots[1])
            }
        }
    }
}

impl WeightedRegressionStrategy {
    fn make_next_amount(
        decay: f64,
    ) -> impl FnMut(&[HistoryPoint], &BigUint, f64, &BigUint, f64, f64) -> BigUint {
        move |history, low, _low_price, high, _high_price, target| {
            let fallback = geometric_mean(low, high);

            if history.len() < 4 {
                return fallback;
            }

            if let Some(estimate) = Self::fit_and_solve(history, target, decay) {
                if let Some(amount) = BigUint::from_f64(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }

            fallback
        }
    }
}

impl ProtocolSimExt for WeightedRegressionStrategy {
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
            Self::make_next_amount(self.decay),
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
            Self::make_next_amount(self.decay),
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
// Strategy 6: Piecewise Linear
// =============================================================================

/// Piecewise linear interpolation strategy
///
/// Sorts history by amount, finds the segment containing the target price,
/// and linearly interpolates within that segment.
pub struct PiecewiseLinearStrategy;

impl PiecewiseLinearStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        if history.len() < 2 {
            return fallback;
        }

        // Sort by amount
        let mut sorted: Vec<&HistoryPoint> = history.iter().collect();
        sorted.sort_by(|a, b| {
            a.amount_f64
                .partial_cmp(&b.amount_f64)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find segment containing target price
        for window in sorted.windows(2) {
            let p1 = window[0];
            let p2 = window[1];

            // Check if target is in this segment
            let (min_p, max_p) = if p1.price < p2.price {
                (p1.price, p2.price)
            } else {
                (p2.price, p1.price)
            };

            if target >= min_p && target <= max_p {
                // Linear interpolation
                let dp = p2.price - p1.price;
                if dp.abs() < MIN_DIVISOR {
                    continue;
                }

                let ratio = (target - p1.price) / dp;
                let estimate = p1.amount_f64 + ratio * (p2.amount_f64 - p1.amount_f64);

                if estimate.is_finite() && estimate > 0.0 {
                    if let Some(amount) = BigUint::from_f64(estimate) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for PiecewiseLinearStrategy {
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
// Strategy 7: IqiV2 - Better Point Selection
// =============================================================================

/// Improved IQI strategy with better point selection
///
/// Instead of using the last 3 points, selects points that bracket the target:
/// - At least one point with price below target
/// - At least one point with price above target
/// - Third point closest to target on the side with fewer points
pub struct IqiV2Strategy;

impl IqiV2Strategy {
    /// Select 3 points that bracket the target well for quadratic fit
    fn select_bracketing_points<'a>(
        history: &'a [HistoryPoint],
        target: f64,
    ) -> Option<(&'a HistoryPoint, &'a HistoryPoint, &'a HistoryPoint)> {
        // Separate points by whether price is below or above target
        let mut below: Vec<_> = history.iter().filter(|p| p.price < target && p.amount_f64 > 0.0).collect();
        let mut above: Vec<_> = history.iter().filter(|p| p.price >= target && p.amount_f64 > 0.0).collect();

        // Need at least one on each side
        if below.is_empty() || above.is_empty() {
            return None;
        }

        // Sort by distance to target
        below.sort_by(|a, b| {
            (target - a.price).abs().partial_cmp(&(target - b.price).abs()).unwrap()
        });
        above.sort_by(|a, b| {
            (a.price - target).abs().partial_cmp(&(b.price - target).abs()).unwrap()
        });

        // Pick: closest below, closest above, and one more from the larger side
        let p_below = below[0];
        let p_above = above[0];

        let p_third = if below.len() > above.len() && below.len() > 1 {
            below[1]
        } else if above.len() > 1 {
            above[1]
        } else if below.len() > 1 {
            below[1]
        } else {
            return None; // Only 2 usable points
        };

        // Return sorted by amount for consistent IQI calculation
        let mut points = [p_below, p_above, p_third];
        points.sort_by(|a, b| a.amount_f64.partial_cmp(&b.amount_f64).unwrap());

        Some((points[0], points[1], points[2]))
    }
}

impl IqiV2Strategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Need at least 3 points
        if history.len() < 3 {
            return fallback;
        }

        // Try to select bracketing points
        if let Some((p1, p2, p3)) = Self::select_bracketing_points(history, target) {
            if let Some(estimate) = IqiStrategy::iqi(
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
        }

        fallback
    }
}

impl ProtocolSimExt for IqiV2Strategy {
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
// Strategy 8: Chandrupatla - "Try if it looks good" policy
// =============================================================================

/// Chandrupatla's method: selective IQI with geometric criterion
///
/// Unlike Brent which always tries IQI then checks, Chandrupatla first
/// checks if IQI is likely to help based on the geometry of the points.
/// This avoids wasting iterations on bad IQI estimates.
pub struct ChandrupatlaStrategy;

impl ChandrupatlaStrategy {
    /// Decide whether to use IQI based on Chandrupatla's criterion
    ///
    /// Returns true if IQI is likely to give a good estimate
    fn should_use_iqi(low_p: f64, mid_p: f64, high_p: f64, target: f64) -> bool {
        // Normalize to [0, 1] interval
        let range = high_p - low_p;
        if range.abs() < MIN_DIVISOR {
            return false;
        }

        let t = (target - low_p) / range;
        let phi = (mid_p - low_p) / range;

        // Chandrupatla's criterion: use IQI if the midpoint is positioned
        // such that interpolation won't overshoot
        // The key insight: if phi is close to t, linear interpolation is good
        // If phi is far from t but on the same side, IQI might overshoot

        // Use bisection if:
        // 1. phi and t are on opposite sides of 0.5 (midpoint poorly placed)
        // 2. phi is very close to 0 or 1 (degenerate case)
        if phi < MIN_DIVISOR || phi > 1.0 - MIN_DIVISOR {
            return false;
        }

        // Chandrupatla's actual criterion based on quadratic vertex position
        let xi = (t - phi) / (1.0 - phi);
        let use_iqi = xi.abs() <= 0.5 && (1.0 - xi).abs() <= 0.5;

        use_iqi
    }
}

impl ChandrupatlaStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);

        // Need at least 3 points for IQI consideration
        if history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            // Check Chandrupatla's criterion before trying IQI
            if Self::should_use_iqi(low_price, p2.price, high_price, target) {
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
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        // Try linear interpolation (secant) as fallback before geometric mean
        if history.len() >= 2 {
            let n = history.len();
            let p1 = &history[n - 2];
            let p2 = &history[n - 1];

            let dp = p2.price - p1.price;
            if dp.abs() > MIN_DIVISOR {
                let estimate =
                    p1.amount_f64 + (target - p1.price) * (p2.amount_f64 - p1.amount_f64) / dp;
                if estimate.is_finite() && estimate > low_f64 && estimate < high_f64 {
                    if let Some(amount) = BigUint::from_f64(estimate) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for ChandrupatlaStrategy {
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
// Strategy 9: NewtonLog - Log-space derivative for exponential curves
// =============================================================================

/// Newton's method with log-space derivative estimation
///
/// AMM price curves are often exponential (price = k/x² for constant product).
/// Using d(log p)/d(log a) instead of dp/da gives better estimates.
pub struct NewtonLogStrategy;

impl NewtonLogStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        _low_price: f64,
        high: &BigUint,
        _high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Need at least 3 points
        if history.len() < 3 {
            return fallback;
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2]; // Middle point - apply Newton here
        let p3 = &history[n - 1];

        // All values must be positive for log
        if p1.amount_f64 <= 0.0
            || p1.price <= 0.0
            || p2.amount_f64 <= 0.0
            || p2.price <= 0.0
            || p3.amount_f64 <= 0.0
            || p3.price <= 0.0
            || target <= 0.0
        {
            return fallback;
        }

        // Log-space derivative: d(ln p) / d(ln a) = (a/p) * (dp/da)
        // Estimate using central difference in log space
        let log_a1 = p1.amount_f64.ln();
        let log_a3 = p3.amount_f64.ln();
        let log_p1 = p1.price.ln();
        let log_p3 = p3.price.ln();

        let d_log_a = log_a3 - log_a1;
        let d_log_p = log_p3 - log_p1;

        if d_log_a.abs() < MIN_DIVISOR {
            return fallback;
        }

        // Log-space derivative (elasticity)
        let elasticity = d_log_p / d_log_a;

        if elasticity.abs() < MIN_DIVISOR {
            return fallback;
        }

        // Newton step in log space
        let log_a2 = p2.amount_f64.ln();
        let log_p2 = p2.price.ln();
        let log_target = target.ln();
        let log_error = log_p2 - log_target;

        let log_estimate = log_a2 - log_error / elasticity;
        let estimate = log_estimate.exp();

        if estimate.is_finite() && estimate > 0.0 {
            if let Some(amount) = BigUint::from_f64(estimate) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for NewtonLogStrategy {
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
// Strategy 10: ConvexSearch - AMM-aware convex interpolation
// =============================================================================

/// AMM-aware search exploiting convexity of price curves
///
/// For AMMs with convex price curves (constant product x*y=k):
/// - Linear interpolation gives a LOWER bound for the root
/// - Geometric mean gives an UPPER bound
/// This strategy blends them based on the observed curvature.
pub struct ConvexSearchStrategy;

impl ConvexSearchStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);

        // Need valid bounds
        if high_f64 <= low_f64 || high_price <= low_price {
            return geometric_mean(low, high);
        }

        // Linear interpolation (lower bound for convex function root)
        let linear =
            low_f64 + (target - low_price) * (high_f64 - low_f64) / (high_price - low_price);

        // Geometric mean (upper bound for convex)
        let geometric = (low_f64 * high_f64).sqrt();

        // Estimate curvature from history if available
        let blend = if history.len() >= 3 {
            // Measure how "curved" the function is
            // For a truly linear function, midpoint price would be average
            // For convex, midpoint price < average
            let n = history.len();
            let mid = &history[n - 1];

            let expected_linear_price = low_price
                + (mid.amount_f64 - low_f64) / (high_f64 - low_f64) * (high_price - low_price);
            let actual_price = mid.price;

            // Curvature estimate: how much does actual deviate from linear?
            let curvature = (expected_linear_price - actual_price) / expected_linear_price;

            // Blend: 0 = pure linear, 1 = pure geometric
            // More curvature → more geometric
            curvature.abs().min(1.0).max(0.0)
        } else {
            // Default: slight preference for geometric (AMMs are usually convex)
            0.3
        };

        // Blend linear and geometric based on curvature
        let estimate = linear * (1.0 - blend) + geometric * blend;

        // Ensure within bounds
        let clamped = estimate.max(low_f64 + 1.0).min(high_f64 - 1.0);

        if clamped.is_finite() && clamped > 0.0 {
            if let Some(amount) = BigUint::from_f64(clamped) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        geometric_mean(low, high)
    }
}

impl ProtocolSimExt for ConvexSearchStrategy {
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
// Strategy 11: ITP (Interpolate, Truncate, Project)
// =============================================================================

/// ITP (Interpolate, Truncate, Project) method
///
/// Based on Oliveira and Takahashi (2021), this method achieves superlinear
/// convergence while maintaining worst-case bisection guarantees.
///
/// The algorithm works in three steps:
/// 1. Interpolate: Use regula falsi to estimate root location
/// 2. Truncate: Perturb estimate towards center to avoid extrapolation issues
/// 3. Project: Project onto a neighborhood of the bisection midpoint
///
/// Reference: "An Enhancement of the Bisection Method Average Performance
/// Preserving Minmax Optimality" - ACM TOMS 47(1), 2021
pub struct ItpStrategy {
    /// κ₁ scaling factor for truncation (default: 0.2 / (b - a) normalized)
    pub k1: f64,
    /// κ₂ exponent for truncation (default: 2.0)
    pub k2: f64,
    /// n₀ slack variable for projection (default: 1)
    pub n0: u32,
}

impl Default for ItpStrategy {
    fn default() -> Self {
        Self {
            k1: 0.2,
            k2: 2.0,
            n0: 1,
        }
    }
}

impl ItpStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);

        // Need valid bounds
        if high_f64 <= low_f64 || (high_price - low_price).abs() < MIN_DIVISOR {
            return fallback;
        }

        // ITP parameters (using defaults)
        let k1 = 0.2;
        let k2 = 2.0;
        let n0 = 1u32;

        let bracket = high_f64 - low_f64;

        // Step 1: Interpolation (regula falsi)
        let x_half = (low_f64 + high_f64) / 2.0; // bisection point
        let x_f = low_f64 + (target - low_price) * (high_f64 - low_f64) / (high_price - low_price);

        // Step 2: Truncation
        // σ = sign(x_half - x_f)
        let sigma = if x_half >= x_f { 1.0 } else { -1.0 };

        // δ = min(κ₁ * |b - a|^κ₂, |x_half - x_f|)
        let delta = (k1 * bracket.powf(k2)).min((x_half - x_f).abs());

        // x_t = x_f + σ * δ
        let x_t = x_f + sigma * delta;

        // Step 3: Projection
        // Calculate the iteration number from history
        let j = history.len().saturating_sub(2) as u32; // subtract initial bounds

        // n_half = ceil(log2((b-a) / 2ε)) - but we use tolerance from search
        // For simplicity, estimate based on bracket size
        let epsilon = 1.0; // minimum amount unit
        let n_half = ((bracket / (2.0 * epsilon)).log2().ceil() as u32).max(1);

        // ρ = min(ε * 2^(n_half + n0 - j) - (b-a)/2, |x_t - x_half|)
        let rho_bound = epsilon * 2.0_f64.powi((n_half + n0).saturating_sub(j) as i32) - bracket / 2.0;
        let rho = rho_bound.max(0.0).min((x_t - x_half).abs());

        // x_ITP = x_half - σ * ρ (project towards bisection)
        let x_itp = x_half - sigma * rho;

        // Ensure within bounds
        let estimate = x_itp.max(low_f64 + 1.0).min(high_f64 - 1.0);

        if estimate.is_finite() && estimate > 0.0 {
            if let Some(amount) = BigUint::from_f64(estimate) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for ItpStrategy {
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
// Strategy 12: Hybrid (Lloyd's improved method)
// =============================================================================

/// Hybrid root-finding strategy based on Anthony Lloyd's improvements
///
/// This method combines quadratic interpolation, inverse quadratic interpolation,
/// and bisection with two key improvements over Brent's method:
///
/// 1. Hybrid switching: If the bracket reduces by less than 40%, switch to
///    quadratic interpolation; otherwise try IQI or bisection.
///
/// 2. Boundary optimization: Instead of evaluating boundary points directly,
///    evaluate points 20% in from the boundaries first.
///
/// Reference: https://anthonylloyd.github.io/blog/2021/06/03/Root-finding
pub struct HybridStrategy;

impl HybridStrategy {
    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);
        let low_f64 = low.to_f64().unwrap_or(0.0);
        let high_f64 = high.to_f64().unwrap_or(f64::MAX);

        // Need valid bounds
        if high_f64 <= low_f64 {
            return fallback;
        }

        let bracket = high_f64 - low_f64;

        // Improvement 1: Boundary optimization
        // Evaluate points 20% in from the boundary instead of the boundary itself
        let boundary_offset = 0.2;

        // Check if we should use boundary optimization (early iterations)
        if history.len() <= 3 {
            // Calculate 20% in from the appropriate boundary based on target
            let low_inner = low_f64 + boundary_offset * bracket;
            let high_inner = high_f64 - boundary_offset * bracket;

            // Choose based on where target lies relative to current prices
            let estimate = if (target - low_price).abs() < (target - high_price).abs() {
                // Target closer to low price, try low_inner
                low_inner
            } else {
                // Target closer to high price, try high_inner
                high_inner
            };

            if estimate.is_finite() && estimate > low_f64 && estimate < high_f64 {
                if let Some(amount) = BigUint::from_f64(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // Improvement 2: Hybrid switching based on 40% reduction criterion
        // Check the last bracket reduction
        let use_quadratic = if history.len() >= 4 {
            let n = history.len();
            let prev_bracket = (history[n - 2].amount_f64 - history[n - 3].amount_f64).abs();
            let curr_bracket = bracket;
            // If bracket didn't reduce by at least 40%, use quadratic
            prev_bracket > 0.0 && curr_bracket / prev_bracket > 0.6
        } else {
            false
        };

        if use_quadratic && history.len() >= 3 {
            // Use quadratic interpolation (similar to Muller's method)
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            // Fit parabola through 3 points and find where it equals target
            let h1 = p2.amount_f64 - p1.amount_f64;
            let h2 = p3.amount_f64 - p2.amount_f64;
            let d1 = (p2.price - p1.price) / h1;
            let d2 = (p3.price - p2.price) / h2;

            if (h1 + h2).abs() > MIN_DIVISOR {
                let a = (d2 - d1) / (h1 + h2);
                let b = a * h2 + d2;
                let c = p3.price - target;

                // Quadratic formula
                let discriminant = b * b - 4.0 * a * c;
                if discriminant >= 0.0 && a.abs() > MIN_DIVISOR {
                    let sqrt_disc = discriminant.sqrt();
                    // Choose the root closer to p3
                    let x1 = p3.amount_f64 + (-b + sqrt_disc) / (2.0 * a);
                    let x2 = p3.amount_f64 + (-b - sqrt_disc) / (2.0 * a);

                    let estimate = if (x1 - p3.amount_f64).abs() < (x2 - p3.amount_f64).abs() {
                        x1
                    } else {
                        x2
                    };

                    if estimate.is_finite() && estimate > low_f64 && estimate < high_f64 {
                        if let Some(amount) = BigUint::from_f64(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        // Try IQI if we have enough points
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
                    if let Some(amount) = BigUint::from_f64(estimate) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
            }
        }

        // Try linear interpolation (secant)
        if (high_price - low_price).abs() > MIN_DIVISOR {
            let estimate =
                low_f64 + (target - low_price) * (high_f64 - low_f64) / (high_price - low_price);
            if estimate.is_finite() && estimate > low_f64 && estimate < high_f64 {
                if let Some(amount) = BigUint::from_f64(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // Fall back to geometric mean (bisection in log space)
        fallback
    }
}

impl ProtocolSimExt for HybridStrategy {
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
// AMM-Aware Strategies - Domain-specific optimizations
// =============================================================================

// =============================================================================
// Strategy 13: Curvature Detection Strategy
// =============================================================================

/// Curvature-aware adaptive strategy for AMM price curves
///
/// This strategy estimates the local curvature of the price curve from sampled
/// points and dynamically selects the optimal numerical method:
/// - Nearly linear curves (StableSwap middle region): Use secant method
/// - Convex curves (constant product, power law): Use IQI
/// - Unknown/mixed: Use geometric mean as safe fallback
///
/// The curvature ratio measures how much the curve deviates from linear:
/// - |curvature| < 0.005: Nearly linear → secant optimal
/// - curvature < -0.005: Convex → IQI optimal
/// - Otherwise: Use safe fallback
pub struct CurvatureAdaptiveStrategy;

impl CurvatureAdaptiveStrategy {
    /// Estimate curvature ratio from 3 history points
    ///
    /// Returns a normalized curvature measure:
    /// - Negative: convex curve (price decreases faster than linear)
    /// - Positive: concave curve (price decreases slower than linear)
    /// - Near zero: approximately linear
    fn estimate_curvature_ratio(history: &[HistoryPoint]) -> f64 {
        if history.len() < 3 {
            return 0.5; // Unknown: assume moderate curvature
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2];
        let p3 = &history[n - 1];

        // Avoid division by zero
        let da = p3.amount_f64 - p1.amount_f64;
        if da.abs() < MIN_DIVISOR {
            return 0.5;
        }

        // Expected linear interpolation at midpoint
        let t = (p2.amount_f64 - p1.amount_f64) / da;
        let expected_linear = p1.price + t * (p3.price - p1.price);

        // Curvature ratio: how much actual deviates from linear
        // Positive means price is higher than linear (concave)
        // Negative means price is lower than linear (convex)
        if expected_linear.abs() < MIN_DIVISOR {
            return 0.5;
        }

        (p2.price - expected_linear) / expected_linear.abs()
    }

    /// Secant method estimate from last 2 points
    fn secant_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 2 {
            return None;
        }

        let n = history.len();
        let p1 = &history[n - 2];
        let p2 = &history[n - 1];

        let dp = p2.price - p1.price;
        if dp.abs() < MIN_DIVISOR {
            return None;
        }

        let result = p2.amount_f64 - (p2.price - target) * (p2.amount_f64 - p1.amount_f64) / dp;

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

        if history.len() < 2 {
            return fallback;
        }

        let curvature = Self::estimate_curvature_ratio(history);

        // Nearly linear (StableSwap middle region): Use secant
        if curvature.abs() < 0.005 {
            if let Some(est) = Self::secant_estimate(history, target) {
                if let Some(amount) = BigUint::from_f64(est) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // Convex curve (constant product, power law): Use IQI
        if curvature < -0.005 && history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            if let Some(est) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                if let Some(amount) = BigUint::from_f64(est) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // Concave or unknown: Use geometric mean (safe)
        fallback
    }
}

impl ProtocolSimExt for CurvatureAdaptiveStrategy {
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
// Strategy 14: Elasticity-Aware Newton Strategy
// =============================================================================

/// Elasticity-aware Newton's method for power-law AMM curves
///
/// For power-law curves like Balancer weighted pools, the price elasticity
/// is approximately constant:
///   ε = d(ln p) / d(ln a) ≈ -(w_in + w_out) / w_in
///
/// This enables fast convergence by estimating elasticity from sampled points
/// and using Newton's method in log-space:
///   log(a_new) = log(a) - (log(p) - log(target)) / elasticity
///
/// This is particularly effective for:
/// - Balancer weighted pools (power law: Πx_i^w_i = k)
/// - Constant product AMMs (elasticity ≈ -2)
/// - Any AMM with approximately constant elasticity
pub struct ElasticityNewtonStrategy;

impl ElasticityNewtonStrategy {
    /// Estimate elasticity from log-log slope of price vs amount
    fn estimate_elasticity(history: &[HistoryPoint]) -> Option<f64> {
        if history.len() < 2 {
            return None;
        }

        // Find two points with sufficient separation
        let p1 = history.first()?;
        let p2 = history.last()?;

        // Need positive values for log
        if p1.amount_f64 <= 0.0 || p2.amount_f64 <= 0.0 || p1.price <= 0.0 || p2.price <= 0.0 {
            return None;
        }

        let log_amount_diff = p2.amount_f64.ln() - p1.amount_f64.ln();
        if log_amount_diff.abs() < 0.01 {
            return None; // Insufficient separation
        }

        let log_price_diff = p2.price.ln() - p1.price.ln();
        let elasticity = log_price_diff / log_amount_diff;

        // Sanity check: elasticity should be negative for typical AMMs
        // (more input → lower output price)
        if elasticity.is_finite() && elasticity.abs() > 0.01 && elasticity.abs() < 100.0 {
            Some(elasticity)
        } else {
            None
        }
    }

    /// Newton step in log-space using elasticity
    fn elasticity_newton_step(
        current_amount: f64,
        current_price: f64,
        target: f64,
        elasticity: f64,
    ) -> Option<f64> {
        if current_amount <= 0.0 || current_price <= 0.0 || target <= 0.0 {
            return None;
        }

        let log_price_error = current_price.ln() - target.ln();
        let log_amount_correction = log_price_error / elasticity;
        let log_new_amount = current_amount.ln() - log_amount_correction;

        let new_amount = log_new_amount.exp();

        if new_amount.is_finite() && new_amount > 0.0 {
            Some(new_amount)
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

        if history.len() < 2 {
            return fallback;
        }

        // Estimate elasticity from history
        if let Some(elasticity) = Self::estimate_elasticity(history) {
            // Use the most recent point for Newton step
            let latest = history.last().unwrap();

            if let Some(est) =
                Self::elasticity_newton_step(latest.amount_f64, latest.price, target, elasticity)
            {
                if let Some(amount) = BigUint::from_f64(est) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // Fallback to geometric mean
        fallback
    }
}

impl ProtocolSimExt for ElasticityNewtonStrategy {
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
// Strategy 15: StableSwap-Aware Strategy
// =============================================================================

/// Two-phase search strategy optimized for StableSwap curves (Curve, Balancer Stable)
///
/// StableSwap curves have distinct regions:
/// 1. **Stable region** (around balanced reserves): Nearly linear, use secant
/// 2. **Transition region** (moderate imbalance): Mixed behavior, adaptive
/// 3. **Edge region** (high imbalance): Constant-product-like, use IQI
///
/// The phase is determined by the price change ratio:
/// - |price_ratio - 1| < 0.01: Stable region → secant
/// - |price_ratio - 1| < 0.10: Transition → adaptive (check curvature)
/// - Otherwise: Edge region → IQI
pub struct StableSwapAwareStrategy;

impl StableSwapAwareStrategy {
    /// Determine which region we're in based on price ratio
    fn determine_region(current_price: f64, target_price: f64) -> StableSwapRegion {
        if current_price <= 0.0 || target_price <= 0.0 {
            return StableSwapRegion::Unknown;
        }

        let price_ratio = target_price / current_price;

        if (0.99..=1.01).contains(&price_ratio) {
            StableSwapRegion::Stable
        } else if (0.90..=1.10).contains(&price_ratio) {
            StableSwapRegion::Transition
        } else {
            StableSwapRegion::Edge
        }
    }

    /// Secant method for linear regions
    fn secant_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 2 {
            return None;
        }

        let n = history.len();
        let p1 = &history[n - 2];
        let p2 = &history[n - 1];

        let dp = p2.price - p1.price;
        if dp.abs() < MIN_DIVISOR {
            return None;
        }

        let result = p2.amount_f64 - (p2.price - target) * (p2.amount_f64 - p1.amount_f64) / dp;

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

        if history.is_empty() {
            return fallback;
        }

        let current_price = history.last().map(|p| p.price).unwrap_or(0.0);
        let region = Self::determine_region(current_price, target);

        match region {
            StableSwapRegion::Stable => {
                // Nearly linear - secant is optimal
                if let Some(est) = Self::secant_estimate(history, target) {
                    if let Some(amount) = BigUint::from_f64(est) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
            }
            StableSwapRegion::Transition => {
                // Mixed behavior - try secant first, then IQI
                if let Some(est) = Self::secant_estimate(history, target) {
                    if let Some(amount) = BigUint::from_f64(est) {
                        if let Some(safe) = safe_next_amount(amount, low, high) {
                            return safe;
                        }
                    }
                }
                // Fall through to IQI below
                if history.len() >= 3 {
                    let n = history.len();
                    if let Some(est) = IqiStrategy::iqi(
                        history[n - 3].amount_f64,
                        history[n - 3].price,
                        history[n - 2].amount_f64,
                        history[n - 2].price,
                        history[n - 1].amount_f64,
                        history[n - 1].price,
                        target,
                    ) {
                        if let Some(amount) = BigUint::from_f64(est) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
            StableSwapRegion::Edge | StableSwapRegion::Unknown => {
                // Constant-product-like - IQI is better
                if history.len() >= 3 {
                    let n = history.len();
                    if let Some(est) = IqiStrategy::iqi(
                        history[n - 3].amount_f64,
                        history[n - 3].price,
                        history[n - 2].amount_f64,
                        history[n - 2].price,
                        history[n - 1].amount_f64,
                        history[n - 1].price,
                        target,
                    ) {
                        if let Some(amount) = BigUint::from_f64(est) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }
        }

        fallback
    }
}

/// Region classification for StableSwap curves
#[derive(Debug, Clone, Copy, PartialEq)]
enum StableSwapRegion {
    /// Near balance: nearly linear behavior
    Stable,
    /// Moderate imbalance: mixed behavior
    Transition,
    /// High imbalance: constant-product-like
    Edge,
    /// Cannot determine
    Unknown,
}

impl ProtocolSimExt for StableSwapAwareStrategy {
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
// Strategy 16: Blended IQI-Secant Strategy
// =============================================================================

/// Blended estimate strategy combining IQI and secant based on curvature
///
/// When uncertain about curve type, this strategy computes both IQI and secant
/// estimates and blends them based on observed curvature:
/// - High curvature → favor IQI (quadratic fit)
/// - Low curvature → favor secant (linear fit)
///
/// The blend weight is computed as:
///   iqi_weight = min(|curvature| * 100, 1.0)
///   estimate = iqi * iqi_weight + secant * (1 - iqi_weight)
///
/// This provides smooth transitions between methods and handles mixed curves.
pub struct BlendedIqiSecantStrategy;

impl BlendedIqiSecantStrategy {
    /// Estimate curvature ratio from 3 history points
    fn estimate_curvature_ratio(history: &[HistoryPoint]) -> f64 {
        if history.len() < 3 {
            return 0.5; // Unknown: moderate curvature assumed
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2];
        let p3 = &history[n - 1];

        let da = p3.amount_f64 - p1.amount_f64;
        if da.abs() < MIN_DIVISOR {
            return 0.5;
        }

        let t = (p2.amount_f64 - p1.amount_f64) / da;
        let expected_linear = p1.price + t * (p3.price - p1.price);

        if expected_linear.abs() < MIN_DIVISOR {
            return 0.5;
        }

        (p2.price - expected_linear) / expected_linear.abs()
    }

    /// Secant estimate from last 2 points
    fn secant_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 2 {
            return None;
        }

        let n = history.len();
        let p1 = &history[n - 2];
        let p2 = &history[n - 1];

        let dp = p2.price - p1.price;
        if dp.abs() < MIN_DIVISOR {
            return None;
        }

        let result = p2.amount_f64 - (p2.price - target) * (p2.amount_f64 - p1.amount_f64) / dp;

        if result.is_finite() && result > 0.0 {
            Some(result)
        } else {
            None
        }
    }

    /// IQI estimate from last 3 points
    fn iqi_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 3 {
            return None;
        }

        let n = history.len();
        IqiStrategy::iqi(
            history[n - 3].amount_f64,
            history[n - 3].price,
            history[n - 2].amount_f64,
            history[n - 2].price,
            history[n - 1].amount_f64,
            history[n - 1].price,
            target,
        )
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

        // Need at least 2 points for secant
        if history.len() < 2 {
            return fallback;
        }

        let secant_est = Self::secant_estimate(history, target);
        let iqi_est = Self::iqi_estimate(history, target);

        // If we have both estimates, blend them
        if let (Some(secant), Some(iqi)) = (secant_est, iqi_est) {
            let curvature = Self::estimate_curvature_ratio(history);

            // Blend based on curvature confidence
            // Higher |curvature| → more weight to IQI
            // Lower |curvature| → more weight to secant
            let iqi_weight = (curvature.abs() * 100.0).min(1.0);
            let secant_weight = 1.0 - iqi_weight;

            let blended = iqi * iqi_weight + secant * secant_weight;

            if blended.is_finite() && blended > 0.0 {
                if let Some(amount) = BigUint::from_f64(blended) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // If only one estimate available, use it
        if let Some(iqi) = iqi_est {
            if let Some(amount) = BigUint::from_f64(iqi) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        if let Some(secant) = secant_est {
            if let Some(amount) = BigUint::from_f64(secant) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for BlendedIqiSecantStrategy {
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
// Strategy 17: Precision-Limit-Aware Strategy
// =============================================================================

/// Strategy with early detection of integer precision limits
///
/// Some pools (especially stablecoin pairs on Curve) have limited price precision
/// due to integer math. This strategy detects when we've reached the precision
/// floor (adjacent integers bracket the target but neither achieves it) and
/// terminates early to avoid wasting iterations.
///
/// The strategy uses IQI as the primary method but adds precision limit detection
/// to the search loop decision making.
pub struct PrecisionLimitAwareStrategy;

impl PrecisionLimitAwareStrategy {
    /// Check if we've hit the precision limit
    ///
    /// Returns true if low and high are adjacent integers and the target
    /// price is between their prices (meaning we can't get any closer).
    fn at_precision_limit(
        low: &BigUint,
        high: &BigUint,
        low_price: f64,
        high_price: f64,
        target: f64,
    ) -> bool {
        // Check if adjacent (difference of 1 or less)
        if high <= low {
            return false;
        }

        let diff = high - low;
        if diff > BigUint::from(2u32) {
            return false; // Not adjacent, still room to search
        }

        // Check if target is bracketed by the prices
        let (min_price, max_price) = if low_price < high_price {
            (low_price, high_price)
        } else {
            (high_price, low_price)
        };

        // Target is between min and max but we can't subdivide further
        target >= min_price && target <= max_price
    }

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // Early termination check: if we're at precision limit, return midpoint
        // The run_search loop will detect no progress and terminate
        if Self::at_precision_limit(low, high, low_price, high_price, target) {
            // Return the bound that's closer to target
            let low_error = (low_price - target).abs();
            let high_error = (high_price - target).abs();
            // Return a value that won't make progress, triggering termination
            // but with the better bound being the "best result"
            if low_error < high_error {
                return low.clone() + BigUint::from(1u32);
            } else {
                return high.clone() - BigUint::from(1u32);
            }
        }

        // Standard IQI-based search
        if history.len() >= 3 {
            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];
            let p3 = &history[n - 1];

            if let Some(est) = IqiStrategy::iqi(
                p1.amount_f64,
                p1.price,
                p2.amount_f64,
                p2.price,
                p3.amount_f64,
                p3.price,
                target,
            ) {
                if let Some(amount) = BigUint::from_f64(est) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for PrecisionLimitAwareStrategy {
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
// Strategy 18: Improved Brent's Method (BrentV2)
// =============================================================================

/// Improved Brent's method with curvature-aware acceptance threshold
///
/// The original Brent implementation uses a fixed 1% threshold for accepting
/// IQI estimates. This improved version adapts the threshold based on observed
/// curve curvature:
/// - For nearly linear curves: conservative threshold (10%) - prefer secant
/// - For curved regions: aggressive threshold (0.1%) - trust IQI more
///
/// This addresses the issue where Brent's fixed threshold rejects good IQI
/// estimates on convex curves (like constant product AMMs) while being too
/// permissive on linear curves (like StableSwap middle region).
pub struct BrentV2Strategy;

impl BrentV2Strategy {
    /// Estimate curvature ratio
    fn estimate_curvature_ratio(history: &[HistoryPoint]) -> f64 {
        if history.len() < 3 {
            return 0.01; // Default: moderate curvature
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2];
        let p3 = &history[n - 1];

        let da = p3.amount_f64 - p1.amount_f64;
        if da.abs() < MIN_DIVISOR {
            return 0.01;
        }

        let t = (p2.amount_f64 - p1.amount_f64) / da;
        let expected_linear = p1.price + t * (p3.price - p1.price);

        if expected_linear.abs() < MIN_DIVISOR {
            return 0.01;
        }

        ((p2.price - expected_linear) / expected_linear.abs()).abs()
    }

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
        let bracket_size = high_f64 - low_f64;

        // Curvature-aware acceptance threshold
        let curvature = Self::estimate_curvature_ratio(history);
        let threshold = if curvature < 0.01 {
            0.1 // Nearly linear: be conservative with IQI (10% threshold)
        } else if curvature < 0.05 {
            0.05 // Moderate curvature: balanced threshold (5%)
        } else {
            0.001 // High curvature: trust IQI more (0.1% threshold)
        };

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
                    // Curvature-aware acceptance criterion
                    let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                    if iqi_improvement > bracket_size * threshold {
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

        fallback
    }
}

impl ProtocolSimExt for BrentV2Strategy {
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
// Strategy 19: Improved Blended Strategy for VM Pools
// =============================================================================

/// Improved blended strategy optimized for VM-simulated AMM pools
///
/// This is an enhanced version of BlendedIqiSecantStrategy with:
/// 1. **Linear bootstrap**: Assumes linear curve initially (optimal for VM pools)
/// 2. **Sigmoid blending**: Smoother transition between secant and IQI
/// 3. **Precision limit detection**: Early termination when hitting integer precision
///
/// The sigmoid-based blend weight provides smoother transitions:
/// - curvature ≈ 0: weight ≈ 0 (pure secant, optimal for linear curves)
/// - curvature = 0.02: weight ≈ 0.63
/// - curvature = 0.05: weight ≈ 0.92
/// - curvature ≥ 0.10: weight ≈ 1.0 (pure IQI, optimal for curved)
///
/// This is the recommended strategy for EVMPoolState (Curve, Balancer, etc.)
pub struct BlendedIqiSecantV2Strategy;

impl BlendedIqiSecantV2Strategy {
    /// Estimate curvature ratio from 3 history points
    ///
    /// Returns a normalized curvature measure:
    /// - Near zero: approximately linear (use secant)
    /// - Large magnitude: significantly curved (use IQI)
    fn estimate_curvature_ratio(history: &[HistoryPoint]) -> f64 {
        if history.len() < 3 {
            // IMPROVEMENT: Assume linear for VM pools (Curve/Balancer stable regions)
            // This is better than 0.5 because VM pools are mostly linear
            return 0.0;
        }

        let n = history.len();
        let p1 = &history[n - 3];
        let p2 = &history[n - 2];
        let p3 = &history[n - 1];

        let da = p3.amount_f64 - p1.amount_f64;
        if da.abs() < f64::EPSILON {
            return 0.0;
        }

        // Expected linear interpolation at midpoint
        let t = (p2.amount_f64 - p1.amount_f64) / da;
        let expected_linear = p1.price + t * (p3.price - p1.price);

        if expected_linear.abs() < f64::EPSILON {
            return 0.0;
        }

        // Curvature ratio: deviation from linear, normalized
        (p2.price - expected_linear) / expected_linear.abs()
    }

    /// Check if we've hit the integer precision limit
    ///
    /// Returns true if low and high are adjacent (or very close) and the target
    /// price is bracketed, meaning we can't get any closer due to integer math.
    fn at_precision_limit(
        low: &BigUint,
        high: &BigUint,
        low_price: f64,
        high_price: f64,
        target: f64,
    ) -> bool {
        if high <= low {
            return false;
        }

        let diff = high - low;
        if diff > BigUint::from(2u32) {
            return false; // Not adjacent, still room to search
        }

        // Check if target is bracketed by the prices
        let (min_price, max_price) = if low_price < high_price {
            (low_price, high_price)
        } else {
            (high_price, low_price)
        };

        // Target is between min and max but we can't subdivide further
        target >= min_price && target <= max_price
    }

    /// Secant estimate from last 2 points (linear interpolation)
    fn secant_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 2 {
            return None;
        }

        let n = history.len();
        let p1 = &history[n - 2];
        let p2 = &history[n - 1];

        let dp = p2.price - p1.price;
        if dp.abs() < f64::EPSILON {
            return None;
        }

        let result = p2.amount_f64 - (p2.price - target) * (p2.amount_f64 - p1.amount_f64) / dp;

        if result.is_finite() && result > 0.0 {
            Some(result)
        } else {
            None
        }
    }

    /// IQI estimate from last 3 points (quadratic interpolation)
    fn iqi_estimate(history: &[HistoryPoint], target: f64) -> Option<f64> {
        if history.len() < 3 {
            return None;
        }

        let n = history.len();
        IqiStrategy::iqi(
            history[n - 3].amount_f64,
            history[n - 3].price,
            history[n - 2].amount_f64,
            history[n - 2].price,
            history[n - 1].amount_f64,
            history[n - 1].price,
            target,
        )
    }

    fn next_amount(
        history: &[HistoryPoint],
        low: &BigUint,
        low_price: f64,
        high: &BigUint,
        high_price: f64,
        target: f64,
    ) -> BigUint {
        let fallback = geometric_mean(low, high);

        // IMPROVEMENT: Early termination on precision limit
        // This prevents wasted iterations when we've hit the integer floor
        if Self::at_precision_limit(low, high, low_price, high_price, target) {
            // Return the bound closer to target to trigger termination
            let low_error = (low_price - target).abs();
            let high_error = (high_price - target).abs();
            if low_error < high_error {
                return low.clone() + BigUint::from(1u32);
            } else {
                return high.clone() - BigUint::from(1u32);
            }
        }

        // Need at least 2 points for secant
        if history.len() < 2 {
            return fallback;
        }

        let secant_est = Self::secant_estimate(history, target);
        let iqi_est = Self::iqi_estimate(history, target);

        // If we have both estimates, blend them with sigmoid weighting
        if let (Some(secant), Some(iqi)) = (secant_est, iqi_est) {
            let curvature = Self::estimate_curvature_ratio(history);

            // IMPROVEMENT: Sigmoid-like blending instead of linear ramp
            // This provides smoother transitions and reduces oscillation
            //
            // Formula: iqi_weight = 1 - exp(-|curvature| * k)
            // With k=50:
            //   curvature=0.00 → weight=0.00 (pure secant)
            //   curvature=0.01 → weight=0.39
            //   curvature=0.02 → weight=0.63
            //   curvature=0.05 → weight=0.92
            //   curvature=0.10 → weight=0.99 (nearly pure IQI)
            let iqi_weight = 1.0 - (-curvature.abs() * 50.0).exp();
            let secant_weight = 1.0 - iqi_weight;

            let blended = iqi * iqi_weight + secant * secant_weight;

            if blended.is_finite() && blended > 0.0 {
                if let Some(amount) = BigUint::from_f64(blended) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }
        }

        // If only IQI available (3+ points but secant failed), use it
        if let Some(iqi) = iqi_est {
            if let Some(amount) = BigUint::from_f64(iqi) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        // If only secant available (2 points), use it
        if let Some(secant) = secant_est {
            if let Some(amount) = BigUint::from_f64(secant) {
                if let Some(safe) = safe_next_amount(amount, low, high) {
                    return safe;
                }
            }
        }

        fallback
    }
}

impl ProtocolSimExt for BlendedIqiSecantV2Strategy {
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
