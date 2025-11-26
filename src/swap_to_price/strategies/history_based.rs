//! History-based swap-to-price strategies
//!
//! These strategies use the history of (amount, price) pairs to estimate
//! derivatives and converge faster than simple interpolation methods.

use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};

use crate::swap_to_price::{
    within_tolerance, SwapToPriceError, SwapToPriceResult, SwapToPriceStrategy,
    SWAP_TO_PRICE_MAX_ITERATIONS,
};

// =============================================================================
// Shared Utilities
// =============================================================================

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

/// Convert BigUint to f64 for math operations
fn amount_to_f64(amount: &BigUint) -> Option<f64> {
    amount.to_f64()
}

/// Convert f64 back to BigUint (truncates to integer)
fn f64_to_amount(value: f64) -> Option<BigUint> {
    if value < 0.0 || !value.is_finite() {
        None
    } else {
        Some(BigUint::from(value as u128))
    }
}

/// A point in the search history
#[derive(Clone, Debug)]
struct HistoryPoint {
    amount: BigUint,
    amount_f64: f64,
    price: f64,
}

// =============================================================================
// Common search loop infrastructure
// =============================================================================

/// Run a swap-to-price search with a custom next_amount function
fn run_search<F>(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    mut next_amount_fn: F,
) -> Result<SwapToPriceResult, SwapToPriceError>
where
    F: FnMut(&[HistoryPoint], &BigUint, f64, &BigUint, f64, f64) -> BigUint,
{
    let max_iterations = SWAP_TO_PRICE_MAX_ITERATIONS;

    // Step 1: Get spot price
    let spot_price = state.spot_price(token_out, token_in)?;

    // Step 2: Check if we're already at target
    if within_tolerance(spot_price, target_price) {
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
    let (max_amount_in, _) = state.get_limits(token_in.address.clone(), token_out.address.clone())?;

    // Step 4: Validate target price is above spot
    if target_price < spot_price {
        return Err(SwapToPriceError::TargetBelowSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 5: Calculate limit price
    let limit_price = state
        .get_amount_out(max_amount_in.clone(), token_in, token_out)?
        .new_state
        .spot_price(token_out, token_in)?;

    // Step 6: Validate limit price
    if limit_price <= spot_price {
        return Err(SwapToPriceError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_price,
        });
    }

    // Step 7: Validate target is reachable
    if target_price > limit_price {
        return Err(SwapToPriceError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_price,
        });
    }

    // Step 8: Initialize search
    let mut low = BigUint::from(0u32);
    let mut low_price = spot_price;
    let mut high = max_amount_in.clone();
    let mut high_price = limit_price;
    let mut history: Vec<HistoryPoint> = Vec::with_capacity(max_iterations as usize);

    // Add initial bounds to history
    history.push(HistoryPoint {
        amount: low.clone(),
        amount_f64: 0.0,
        price: low_price,
    });
    if let Some(high_f64) = amount_to_f64(&high) {
        history.push(HistoryPoint {
            amount: high.clone(),
            amount_f64: high_f64,
            price: high_price,
        });
    }

    let mut actual_iterations = 0;

    for iterations in 1..=max_iterations {
        actual_iterations = iterations;

        // Get next amount to try
        let mid = next_amount_fn(&history, &low, low_price, &high, high_price, target_price);

        // Ensure progress
        if mid <= low || mid >= high {
            break;
        }

        // Calculate price at mid
        let result = state.get_amount_out(mid.clone(), token_in, token_out)?;
        let price = result.new_state.spot_price(token_out, token_in)?;

        // Add to history
        if let Some(mid_f64) = amount_to_f64(&mid) {
            history.push(HistoryPoint {
                amount: mid.clone(),
                amount_f64: mid_f64,
                price,
            });
        }

        // Check convergence
        if within_tolerance(price, target_price) {
            return Ok(SwapToPriceResult {
                amount_in: mid,
                actual_price: price,
                gas: result.gas.clone(),
                new_state: result.new_state,
                iterations,
            });
        }

        // Update bounds
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

    // Neither boundary is within tolerance - check if we can return "best achievable"
    // If target is bracketed between low_price and high_price, return the closer one
    if let (Some(low_res), Some(high_res)) = (&low_result, &high_result) {
        let lp = low_res.new_state.spot_price(token_out, token_in)?;
        let hp = high_res.new_state.spot_price(token_out, token_in)?;

        // Check if target is bracketed (between low_price and high_price)
        let is_bracketed = (lp <= target_price && target_price <= hp)
            || (hp <= target_price && target_price <= lp);

        if is_bracketed {
            // Return the one closer to target as "best achievable"
            let low_diff = (lp - target_price).abs();
            let high_diff = (hp - target_price).abs();

            return if low_diff <= high_diff {
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
            };
        }
    }

    // Target not bracketed - genuine failure
    Err(SwapToPriceError::ConvergenceFailure(actual_iterations))
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
        a1: f64, p1: f64,
        a2: f64, p2: f64,
        a3: f64, p3: f64,
        target: f64,
    ) -> Option<f64> {
        // Lagrange basis polynomials in price space
        let denom1 = (p1 - p2) * (p1 - p3);
        let denom2 = (p2 - p1) * (p2 - p3);
        let denom3 = (p3 - p1) * (p3 - p2);

        // Check for division by zero (duplicate prices)
        if denom1.abs() < f64::EPSILON || denom2.abs() < f64::EPSILON || denom3.abs() < f64::EPSILON {
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
}

impl SwapToPriceStrategy for IqiStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(state, target_price, token_in, token_out, |history, low, low_price, high, high_price, target| {
            let fallback = geometric_mean(low, high);

            // Need at least 3 points for IQI
            if history.len() < 3 {
                return fallback;
            }

            // Get the 3 most recent points with distinct prices
            let mut points: Vec<&HistoryPoint> = history.iter()
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
                p1.amount_f64, p1.price,
                p2.amount_f64, p2.price,
                p3.amount_f64, p3.price,
                target,
            ) {
                if let Some(amount) = f64_to_amount(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }

            fallback
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
        if dp.abs() < f64::EPSILON {
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

impl SwapToPriceStrategy for BrentStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(state, target_price, token_in, token_out, |history, low, low_price, high, high_price, target| {
            let fallback = geometric_mean(low, high);
            let low_f64 = amount_to_f64(low).unwrap_or(0.0);
            let high_f64 = amount_to_f64(high).unwrap_or(f64::MAX);

            // Try IQI if we have 3+ points
            if history.len() >= 3 {
                let n = history.len();
                let p1 = &history[n - 3];
                let p2 = &history[n - 2];
                let p3 = &history[n - 1];

                if let Some(estimate) = IqiStrategy::iqi(
                    p1.amount_f64, p1.price,
                    p2.amount_f64, p2.price,
                    p3.amount_f64, p3.price,
                    target,
                ) {
                    // Brent's acceptance criterion: result must be within bounds
                    // and must reduce the bracket by at least half compared to bisection
                    let bisection_point = (low_f64 + high_f64) / 2.0;
                    let bracket_size = high_f64 - low_f64;

                    if estimate > low_f64 && estimate < high_f64 {
                        // Check if IQI is making reasonable progress
                        let iqi_improvement = (estimate - low_f64).min(high_f64 - estimate);
                        if iqi_improvement > bracket_size * 0.01 {
                            if let Some(amount) = f64_to_amount(estimate) {
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

                if let Some(estimate) = Self::secant(
                    p1.amount_f64, p1.price,
                    p2.amount_f64, p2.price,
                    target,
                ) {
                    if estimate > low_f64 && estimate < high_f64 {
                        if let Some(amount) = f64_to_amount(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }

            // Fall back to geometric mean (bisection in log space)
            fallback
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

impl SwapToPriceStrategy for NewtonCentralStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(state, target_price, token_in, token_out, |history, low, _low_price, high, _high_price, target| {
            let fallback = geometric_mean(low, high);

            // Need at least 3 points
            if history.len() < 3 {
                return fallback;
            }

            let n = history.len();
            let p1 = &history[n - 3];
            let p2 = &history[n - 2];  // Middle point - we'll apply Newton here
            let p3 = &history[n - 1];

            // Central difference derivative
            let da = p3.amount_f64 - p1.amount_f64;
            let dp = p3.price - p1.price;

            if da.abs() < f64::EPSILON || dp.abs() < f64::EPSILON {
                return fallback;
            }

            let derivative = dp / da;  // d(price)/d(amount)
            let error = p2.price - target;

            // Newton step: a_new = a2 - error / derivative
            let estimate = p2.amount_f64 - error / derivative;

            if estimate.is_finite() && estimate > 0.0 {
                if let Some(amount) = f64_to_amount(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }

            fallback
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

        if det.abs() < f64::EPSILON {
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

        if a.abs() < f64::EPSILON {
            // Linear case
            if b.abs() < f64::EPSILON {
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

impl SwapToPriceStrategy for QuadraticRegressionStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(state, target_price, token_in, token_out, |history, low, _low_price, high, _high_price, target| {
            let fallback = geometric_mean(low, high);

            // Need at least 4 points for meaningful regression
            if history.len() < 4 {
                return fallback;
            }

            if let Some(estimate) = Self::fit_and_solve(history, target) {
                if let Some(amount) = f64_to_amount(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }

            fallback
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

        if det.abs() < f64::EPSILON {
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

        if a.abs() < f64::EPSILON {
            if b.abs() < f64::EPSILON {
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

impl SwapToPriceStrategy for WeightedRegressionStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let decay = self.decay;
        run_search(state, target_price, token_in, token_out, move |history, low, _low_price, high, _high_price, target| {
            let fallback = geometric_mean(low, high);

            if history.len() < 4 {
                return fallback;
            }

            if let Some(estimate) = Self::fit_and_solve(history, target, decay) {
                if let Some(amount) = f64_to_amount(estimate) {
                    if let Some(safe) = safe_next_amount(amount, low, high) {
                        return safe;
                    }
                }
            }

            fallback
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

impl SwapToPriceStrategy for PiecewiseLinearStrategy {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        run_search(state, target_price, token_in, token_out, |history, low, _low_price, high, _high_price, target| {
            let fallback = geometric_mean(low, high);

            if history.len() < 2 {
                return fallback;
            }

            // Sort by amount
            let mut sorted: Vec<&HistoryPoint> = history.iter().collect();
            sorted.sort_by(|a, b| a.amount_f64.partial_cmp(&b.amount_f64).unwrap_or(std::cmp::Ordering::Equal));

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
                    if dp.abs() < f64::EPSILON {
                        continue;
                    }

                    let ratio = (target - p1.price) / dp;
                    let estimate = p1.amount_f64 + ratio * (p2.amount_f64 - p1.amount_f64);

                    if estimate.is_finite() && estimate > 0.0 {
                        if let Some(amount) = f64_to_amount(estimate) {
                            if let Some(safe) = safe_next_amount(amount, low, high) {
                                return safe;
                            }
                        }
                    }
                }
            }

            fallback
        })
    }
}
