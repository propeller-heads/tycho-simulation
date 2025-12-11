//! Brent's method for finding swap amounts to reach target prices.
//!
//! This module provides two search modes:
//! - [`swap_to_price`]: Find amount to move **spot price** to target
//! - [`query_supply`]: Find max trade where **trade price** ≤ target
//!
//! # Implementation Notes
//!
//! This implementation follows Brent's 1973 algorithm, which combines three
//! root-finding techniques: bisection, secant method, and inverse quadratic
//! interpolation (IQI). The algorithm is also known as the van Wijngaarden-Dekker-Brent
//! method.
//!
//! ## References
//!
//! - Brent, R. P. (1973). "Algorithms for Minimization without Derivatives."
//!   Prentice-Hall. Chapter 4.
//! - SciPy: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html>
//! - argmin-rs: <https://github.com/argmin-rs/argmin/blob/main/crates/argmin/src/solver/brent/brentroot.rs>
//!
//! ## Algorithm
//!
//! The algorithm maintains a bracket [a, b] where f(a) and f(b) have opposite signs.
//! At each iteration:
//! 1. Try inverse quadratic interpolation if three distinct function values exist
//! 2. Otherwise, try the secant method (linear interpolation)
//! 3. Apply safety conditions - if they fail, use bisection instead
//! 4. Update the bracket based on the sign of f at the new point
//!
//! The safety conditions ensure that:
//! - The new point stays within the bracket
//! - The step size doesn't grow too large
//! - Progress is being made toward convergence

use num_bigint::BigUint;
use num_traits::{FromPrimitive, ToPrimitive};
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration parameters for Brent's root-finding algorithm.
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

    /// Absolute tolerance for the x values (amounts).
    /// Used in convergence checking: |b - a| < xtol + rtol * |b|
    /// Default: 2e-12
    pub xtol: f64,

    /// Relative tolerance for the x values (amounts).
    /// Used in convergence checking: |b - a| < xtol + rtol * |b|
    /// Default: 4 * f64::EPSILON
    pub rtol: f64,
}

impl Default for BrentConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.00001,
            max_iterations: 100,
            xtol: 2e-12,
            rtol: 4.0 * f64::EPSILON,
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

    /// Sets the absolute x tolerance.
    pub fn with_xtol(mut self, xtol: f64) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the relative x tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }
}

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

// =============================================================================
// Core Search Algorithm
// =============================================================================

/// Evaluate the objective function: f(x) = price(x) - target_price
///
/// We're finding the root of f(x) = 0, i.e., where price equals target.
fn evaluate_objective(
    state: &dyn ProtocolSim,
    amount: &BigUint,
    token_in: &Token,
    token_out: &Token,
    target_price: f64,
    config: &SearchConfig,
) -> Result<(f64, f64, BigUint, BigUint, Box<dyn ProtocolSim>), BrentSearchError> {
    let result = state.get_amount_out(amount.clone(), token_in, token_out)?;

    let price = match config.metric {
        PriceMetric::SpotPrice => result.new_state.spot_price(token_out, token_in)?,
        PriceMetric::TradePrice => {
            let amount_in_f64 = amount.to_f64().unwrap_or(0.0);
            let amount_out_f64 = result.amount.to_f64().unwrap_or(1.0);
            if amount_out_f64 > 0.0 {
                amount_in_f64 / amount_out_f64
            } else {
                f64::MAX
            }
        }
    };

    // f(x) = price - target (we want to find where this equals 0)
    let f_value = price - target_price;

    Ok((price, f_value, result.amount, result.gas, result.new_state))
}

/// Run the search algorithm with Brent's method.
fn run_search(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    search_config: SearchConfig,
    config: &BrentConfig,
) -> Result<SwapToPriceResult, BrentSearchError> {
    // Step 1: Get spot price
    let spot_price = state.spot_price(token_out, token_in)?;

    // Step 2: Check if we're already at target (for spot price metric)
    if search_config.metric == PriceMetric::SpotPrice
        && within_tolerance(spot_price, target_price, config.tolerance)
    {
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
        return Err(BrentSearchError::TargetBelowSpot { target: target_price, spot: spot_price });
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

    // Step 8: Initialize Brent state
    // We frame this as root-finding: f(x) = price(x) - target_price = 0
    //
    // Initial bracket: [0, max_amount_in]
    // f(0) = spot_price - target_price < 0  (spot is below target)
    // f(max) = limit_price - target_price > 0  (limit is above target)

    let a = 0.0_f64;
    let fa = spot_price - target_price; // negative

    let b = max_amount_in.to_f64().unwrap_or(f64::MAX);
    let fb = limit_spot_price - target_price; // positive

    // Ensure |f(b)| <= |f(a)| by swapping if necessary
    let (mut a, mut fa, mut b, mut fb) = if fa.abs() < fb.abs() {
        (b, fb, a, fa)
    } else {
        (a, fa, b, fb)
    };

    // Initialize c = a (the contrapoint)
    let mut c = a;
    let mut fc = fa;

    // d and e track step sizes for safety conditions
    let mut d = b - a;
    let mut e = d;

    // Track best result for discrete (BigUint) precision
    let mut best_result: Option<SwapToPriceResult> = None;
    let mut best_error = f64::MAX;

    // Also track bracket in BigUint for precision
    let mut low = BigUint::from(0u32);
    let mut high = max_amount_in.clone();

    // Step 9: Main Brent loop
    for iteration in 0..config.max_iterations {
        // Check convergence on bracket width
        let tol1 = config.rtol * b.abs() + config.xtol / 2.0;
        let xm = (a - b) / 2.0;

        if xm.abs() <= tol1 || fb == 0.0 {
            if let Some(result) = best_result {
                return Ok(result);
            }
        }

        // Compute next point using Brent's method
        let s = if e.abs() < tol1 || fc.abs() <= fb.abs() {
            // Bisection step
            e = xm;
            d = xm;
            b + xm
        } else {
            // Try interpolation
            let (p, q) = if (a - c).abs() < f64::EPSILON {
                // Secant method
                let s_val = fb / fa;
                let p = 2.0 * xm * s_val;
                let q = 1.0 - s_val;
                (p, q)
            } else {
                // Inverse quadratic interpolation
                let s_val = fb / fa;
                let q_val = fa / fc;
                let r = fb / fc;
                let p = s_val * (2.0 * xm * q_val * (q_val - r) - (b - c) * (r - 1.0));
                let q = (q_val - 1.0) * (r - 1.0) * (s_val - 1.0);
                (p, q)
            };

            // Adjust signs
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };

            let s_val = e;
            e = d;

            // Safety conditions
            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min(s_val * q).abs() {
                // Accept interpolation
                d = p / q;
                b + d
            } else {
                // Bisection
                d = xm;
                e = d;
                b + xm
            }
        };

        // Ensure the step is at least tol1
        let s = if (s - b).abs() < tol1 {
            b + tol1.copysign(xm)
        } else {
            s
        };

        // Convert to BigUint (our discrete domain)
        let amount_new = if s <= 0.0 {
            BigUint::from(1u32)
        } else {
            BigUint::from_f64(s).unwrap_or_else(|| geometric_mean(&low, &high))
        };

        // Ensure we're within bounds and making progress
        let amount_new = if &amount_new <= &low {
            &low + BigUint::from(1u32)
        } else if &amount_new >= &high {
            &high - BigUint::from(1u32)
        } else {
            amount_new
        };

        // Check if we've hit precision limit
        if &high - &low <= BigUint::from(1u32) {
            if let Some(result) = best_result {
                return Ok(result);
            }
        }

        // Evaluate objective at new point
        let (price_new, f_new, amount_out, gas, new_state) = evaluate_objective(
            state,
            &amount_new,
            token_in,
            token_out,
            target_price,
            &search_config,
        )?;

        let s_f64 = amount_new.to_f64().unwrap_or(s);

        // Calculate error and track best
        let error = (price_new - target_price).abs() / target_price;
        if error < best_error {
            best_error = error;
            best_result = Some(SwapToPriceResult {
                amount_in: amount_new.clone(),
                amount_out: amount_out.clone(),
                actual_price: price_new,
                gas: gas.clone(),
                new_state: new_state.clone(),
                iterations: iteration + 1,
            });
        }

        // Check convergence
        if within_tolerance(price_new, target_price, config.tolerance) {
            return Ok(SwapToPriceResult {
                amount_in: amount_new,
                amount_out,
                actual_price: price_new,
                gas,
                new_state,
                iterations: iteration + 1,
            });
        }

        // Update Brent state
        // c becomes the old contrapoint
        c = a;
        fc = fa;

        // Update bracket based on sign of f_new
        if (f_new > 0.0) == (fb > 0.0) {
            // f_new has same sign as fb: replace b
            a = b;
            fa = fb;
        }

        b = s_f64;
        fb = f_new;

        // Ensure |f(b)| <= |f(a)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        // Update BigUint bounds
        if price_new < target_price {
            low = amount_new;
        } else {
            high = amount_new;
        }
    }

    // Return convergence failure with best result info
    let best = best_result.unwrap_or(SwapToPriceResult {
        amount_in: BigUint::from(0u32),
        amount_out: BigUint::from(0u32),
        actual_price: spot_price,
        gas: BigUint::from(0u32),
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
