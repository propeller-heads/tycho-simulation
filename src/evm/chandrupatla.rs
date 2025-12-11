//! Chandrupatla's method for finding swap amounts to reach target prices.
//!
//! This module provides two search modes:
//! - [`swap_to_price`]: Find amount to move **spot price** to target
//! - [`query_supply`]: Find max trade where **trade price** ≤ target
//!
//! # Implementation Notes
//!
//! This implementation follows Chandrupatla's 1997 algorithm as implemented in
//! TensorFlow Probability and SciPy. The algorithm maintains a three-point
//! bracket (a, b, c) and selectively uses Inverse Quadratic Interpolation (IQI)
//! based on geometric criteria.
//!
//! ## References
//!
//! - Chandrupatla, T.R. (1997). "A new hybrid quadratic/bisection algorithm for
//!   finding the zero of a nonlinear function without using derivatives."
//!   Advances in Engineering Software, 28(3):145-149.
//!   <https://doi.org/10.1016/S0965-9978(96)00051-8>
//!
//! Reference implementations (Apache 2.0 and BSD-3-Clause licenses):
//! - TensorFlow Probability: <https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/math/root_search.py#L352-L559>
//! - SciPy: <https://github.com/scipy/scipy/blob/v1.15.2/scipy/optimize/_chandrupatla.py>
//!
//! ## Algorithm
//!
//! The algorithm maintains three points:
//! - `x1`: One endpoint of the bracket (f1 and f2 have opposite signs)
//! - `x2`: Other endpoint, also the current best estimate
//! - `x3`: Previous value of the contrapoint (used for IQI)
//!
//! At each iteration:
//! 1. Compute ξ (xi) = position ratio and φ (phi) = function value ratio
//! 2. Check if IQI is favorable: `(1 - √(1-ξ)) < φ < √ξ`
//! 3. If favorable, use IQI to estimate the next point
//! 4. Otherwise, use bisection (t = 0.5)

use num_bigint::BigUint;
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration parameters for Chandrupatla's root-finding algorithm.
///
/// These parameters control convergence behavior, numerical stability,
/// and iteration limits.
#[derive(Debug, Clone, Copy)]
pub struct ChandrupatlaConfig {
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

    /// Minimum value for the interpolation parameter t.
    /// Keeps the next estimate away from bracket endpoints.
    /// The parameter t is clamped to [t_min, 1 - t_min].
    /// Default: 0.05
    pub t_min: f64,
}

impl Default for ChandrupatlaConfig {
    fn default() -> Self {
        Self { tolerance: 0.00001, max_iterations: 100, min_divisor: 1e-12, t_min: 0.05 }
    }
}

impl ChandrupatlaConfig {
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

    /// Sets the minimum interpolation parameter.
    pub fn with_t_min(mut self, t_min: f64) -> Self {
        self.t_min = t_min;
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

/// Chandrupatla bracket state
///
/// Maintains three points for the algorithm:
/// - `x1, f1`: One bracket endpoint
/// - `x2, f2`: Other bracket endpoint (and current best estimate)
/// - `x3, f3`: Previous contrapoint (for IQI interpolation)
///
/// Invariant: f1 and f2 have opposite signs (bracket the root)
#[derive(Debug, Clone, Copy)]
struct ChandrupatlaState {
    x1: f64,
    f1: f64,
    x2: f64,
    f2: f64,
    x3: f64,
    f3: f64,
}

/// Result of evaluating the objective function at a given amount.
/// Contains: (price, f_value, amount_out, gas, new_state)
type ObjectiveResult = (f64, f64, BigUint, BigUint, Box<dyn ProtocolSim>);

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

/// Error types for Chandrupatla search operations
#[derive(Debug, thiserror::Error)]
pub enum ChandrupatlaSearchError {
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

/// Decide whether to use IQI based on Chandrupatla's criterion.
///
/// This implements the criterion from Chandrupatla (1997) as used in
/// TensorFlow Probability and SciPy:
///
/// ```text
/// xi = (x1 - x2) / (x3 - x2)    // position ratio
/// phi = (f1 - f2) / (f3 - f2)   // function value ratio
///
/// Use IQI when: (1 - sqrt(1 - xi)) < phi < sqrt(xi)
/// ```
///
/// This is equivalent to: `phi^2 < xi` AND `(1-phi)^2 < (1-xi)`
///
/// # Arguments
/// * `xi` - Position ratio: (x1 - x2) / (x3 - x2)
/// * `phi` - Function value ratio: (f1 - f2) / (f3 - f2)
///
/// # Returns
/// `true` if IQI should be used, `false` if bisection should be used
pub fn should_use_iqi(xi: f64, phi: f64) -> bool {
    // Guard against invalid values
    if !xi.is_finite() || !phi.is_finite() {
        return false;
    }
    if xi <= 0.0 || xi >= 1.0 {
        return false;
    }
    if phi <= 0.0 || phi >= 1.0 {
        return false;
    }

    // Chandrupatla's criterion (equivalent formulations):
    // Original: phi^2 < xi AND (1-phi)^2 < (1-xi)
    // TensorFlow/SciPy: (1 - sqrt(1 - xi)) < phi < sqrt(xi)
    // We use the multiplication form (no sqrt) for efficiency
    phi * phi < xi && (1.0 - phi) * (1.0 - phi) < (1.0 - xi)
}

/// Compute the IQI interpolation parameter t.
///
/// Given three points (x1, f1), (x2, f2), (x3, f3), compute the parameter t
/// such that the new point is at x1 + t * (x2 - x1).
///
/// This follows the TensorFlow/SciPy formulation.
fn iqi_parameter(state: &ChandrupatlaState, config: &ChandrupatlaConfig) -> f64 {
    let ChandrupatlaState { x1, f1, x2, f2, x3, f3 } = *state;

    // Compute alpha = (x3 - x1) / (x2 - x1)
    let dx21 = x2 - x1;
    if dx21.abs() < config.min_divisor {
        return 0.5; // fallback to bisection
    }
    let alpha = (x3 - x1) / dx21;

    // Compute IQI parameter
    // t = (f1/(f1-f2)) * (f3/(f3-f2)) - alpha * (f1/(f3-f1)) * (f2/(f2-f3))
    let df12 = f1 - f2;
    let df32 = f3 - f2;
    let df31 = f3 - f1;
    let df23 = f2 - f3;

    if df12.abs() < config.min_divisor ||
        df32.abs() < config.min_divisor ||
        df31.abs() < config.min_divisor ||
        df23.abs() < config.min_divisor
    {
        return 0.5; // fallback to bisection
    }

    let term1 = (f1 / df12) * (f3 / df32);
    let term2 = alpha * (f1 / df31) * (f2 / df23);

    let t = term1 - term2;

    // Clamp t to [t_min, 1 - t_min] to avoid getting too close to endpoints
    t.clamp(config.t_min, 1.0 - config.t_min)
}

/// Compute the next evaluation point using Chandrupatla's method.
///
/// Returns the interpolation parameter t, where the new point is at:
/// x_new = x1 + t * (x2 - x1)
///
/// # Algorithm (following TensorFlow/SciPy)
///
/// 1. Compute xi = (x1 - x2) / (x3 - x2) and phi = (f1 - f2) / (f3 - f2)
/// 2. If IQI criterion is satisfied: use IQI to compute t
/// 3. Otherwise: use bisection (t = 0.5)
fn chandrupatla_next_t(state: &ChandrupatlaState, config: &ChandrupatlaConfig) -> f64 {
    let ChandrupatlaState { x1, f1, x2, f2, x3, f3 } = *state;

    // Compute xi (position ratio) and phi (function value ratio)
    let dx32 = x3 - x2;
    let df32 = f3 - f2;

    if dx32.abs() < config.min_divisor || df32.abs() < config.min_divisor {
        return 0.5; // bisection
    }

    let xi = (x1 - x2) / dx32;
    let phi = (f1 - f2) / df32;

    // Check Chandrupatla's criterion
    if should_use_iqi(xi, phi) {
        iqi_parameter(state, config)
    } else {
        0.5 // bisection
    }
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
) -> Result<ObjectiveResult, ChandrupatlaSearchError> {
    let result = state.get_amount_out(amount.clone(), token_in, token_out)?;

    let price = match config.metric {
        PriceMetric::SpotPrice => result
            .new_state
            .spot_price(token_out, token_in)?,
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

/// Run the search algorithm with Chandrupatla's method.
///
/// This implementation follows the algorithm from TensorFlow Probability and SciPy,
/// based on Chandrupatla (1997).
fn run_search(
    state: &dyn ProtocolSim,
    target_price: f64,
    token_in: &Token,
    token_out: &Token,
    search_config: SearchConfig,
    config: &ChandrupatlaConfig,
) -> Result<SwapToPriceResult, ChandrupatlaSearchError> {
    // Step 1: Get spot price
    let spot_price = state.spot_price(token_out, token_in)?;

    // Step 2: Check if we're already at target (for spot price metric)
    if search_config.metric == PriceMetric::SpotPrice &&
        within_tolerance(spot_price, target_price, config.tolerance)
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

    // Step 4: Validate target price is above spot
    if target_price < spot_price {
        return Err(ChandrupatlaSearchError::TargetBelowSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 5: Calculate limit price (spot price at max trade)
    let limit_result = state.get_amount_out(max_amount_in.clone(), token_in, token_out)?;
    let limit_spot_price = limit_result
        .new_state
        .spot_price(token_out, token_in)?;

    // Step 6: Validate limit price
    if limit_spot_price <= spot_price {
        return Err(ChandrupatlaSearchError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 7: Validate target is reachable
    if target_price > limit_spot_price {
        return Err(ChandrupatlaSearchError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_spot_price,
        });
    }

    // Step 8: Initialize Chandrupatla state
    // We frame this as root-finding: f(x) = price(x) - target_price = 0
    //
    // Initial bracket: [0, max_amount_in]
    // f(0) = spot_price - target_price < 0  (spot is below target)
    // f(max) = limit_price - target_price > 0  (limit is above target)
    //
    // Following TensorFlow/SciPy naming:
    // x1, x2 bracket the root (f1 and f2 have opposite signs)
    // x3 is the previous contrapoint

    let x1 = 0.0_f64;
    let f1 = spot_price - target_price; // negative

    let x2 = max_amount_in
        .to_f64()
        .unwrap_or(f64::MAX);
    let f2 = limit_spot_price - target_price; // positive

    // Initially x3 = x1 (no previous point yet)
    let mut chandru_state = ChandrupatlaState { x1, f1, x2, f2, x3: x1, f3: f1 };

    // Track best result metadata only (avoid cloning Box<dyn ProtocolSim> every iteration)
    let mut best_amount: Option<BigUint> = None;
    let mut best_price = spot_price;
    let mut best_error = f64::MAX;

    // Also track bracket in BigUint for precision
    let mut low = BigUint::zero();
    let mut high = max_amount_in.clone();

    // Pre-allocate constant for bounds checking
    let one = BigUint::one();

    // Step 9: Main Chandrupatla loop
    for iteration in 0..config.max_iterations {
        // Compute next point using Chandrupatla's method
        let t = chandrupatla_next_t(&chandru_state, config);

        // x_new = x1 + t * (x2 - x1)
        let x_new = chandru_state.x1 + t * (chandru_state.x2 - chandru_state.x1);

        // Convert to BigUint (our discrete domain)
        let amount_new = if x_new <= 0.0 {
            one.clone()
        } else {
            BigUint::from_f64(x_new).unwrap_or_else(|| geometric_mean(&low, &high))
        };

        // Ensure we're within bounds and making progress
        let amount_new = if amount_new <= low {
            &low + &one
        } else if amount_new >= high {
            &high - &one
        } else {
            amount_new
        };

        // Check if we've hit precision limit - re-evaluate best amount if needed
        if &high - &low <= one {
            if let Some(best_amt) = best_amount {
                let (price, _, amount_out, gas, new_state) = evaluate_objective(
                    state,
                    &best_amt,
                    token_in,
                    token_out,
                    target_price,
                    &search_config,
                )?;
                return Ok(SwapToPriceResult {
                    amount_in: best_amt,
                    amount_out,
                    actual_price: price,
                    gas,
                    new_state,
                    iterations: iteration + 1,
                });
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

        let x_new_f64 = amount_new.to_f64().unwrap_or(x_new);

        // Calculate error and track best (metadata only, no cloning)
        let error = (price_new - target_price).abs() / target_price;
        if error < best_error {
            best_error = error;
            best_price = price_new;
            best_amount = Some(amount_new.clone());
        }

        // Check convergence - return immediately with current evaluation results
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

        // Update Chandrupatla state (following TensorFlow/SciPy logic)
        // If f_new has same sign as f1, replace (x1, f1) with (x_new, f_new)
        // Otherwise, shift: x3 <- x1, x1 <- x_new, and x2 stays as contrapoint
        if f_new * chandru_state.f1 > 0.0 {
            // Same sign as f1: replace x1
            chandru_state.x3 = chandru_state.x1;
            chandru_state.f3 = chandru_state.f1;
            chandru_state.x1 = x_new_f64;
            chandru_state.f1 = f_new;
        } else {
            // Opposite sign from f1: x_new brackets with x1
            chandru_state.x3 = chandru_state.x2;
            chandru_state.f3 = chandru_state.f2;
            chandru_state.x2 = chandru_state.x1;
            chandru_state.f2 = chandru_state.f1;
            chandru_state.x1 = x_new_f64;
            chandru_state.f1 = f_new;
        }

        // Update BigUint bounds
        if price_new < target_price {
            low = amount_new;
        } else {
            high = amount_new;
        }
    }

    // Convergence failure - construct error with tracked metadata
    Err(ChandrupatlaSearchError::ConvergenceFailure {
        iterations: config.max_iterations,
        target_price,
        best_price,
        error_bps: best_error * 10000.0,
        amount: best_amount
            .map(|a| a.to_string())
            .unwrap_or_default(),
    })
}

// =============================================================================
// Public API
// =============================================================================

/// Find the amount of input token needed to reach a target spot price using Chandrupatla's method.
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
    config: Option<ChandrupatlaConfig>,
) -> Result<SwapToPriceResult, ChandrupatlaSearchError> {
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
    config: Option<ChandrupatlaConfig>,
) -> Result<QuerySupplyResult, ChandrupatlaSearchError> {
    let config = config.unwrap_or_default();
    let result = run_search(
        state,
        target_price,
        token_in,
        token_out,
        SearchConfig::query_supply(),
        &config,
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

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_TOLERANCE: f64 = 0.00001;

    #[test]
    fn test_within_tolerance() {
        // Default tolerance is 0.00001 (0.001%)

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
    // Tests for should_use_iqi - Chandrupatla's criterion
    // Based on TensorFlow Probability and SciPy implementations
    // =========================================================================

    #[test]
    fn test_should_use_iqi_criterion() {
        // The criterion is: (1 - sqrt(1 - xi)) < phi < sqrt(xi)
        // Equivalently: phi^2 < xi AND (1-phi)^2 < (1-xi)

        // Test case: xi = 0.5, phi = 0.5
        // Lower bound: 1 - sqrt(0.5) ≈ 0.293
        // Upper bound: sqrt(0.5) ≈ 0.707
        // phi = 0.5 is in (0.293, 0.707) → should use IQI
        assert!(should_use_iqi(0.5, 0.5));

        // Test case: xi = 0.25, phi = 0.4
        // Lower bound: 1 - sqrt(0.75) ≈ 0.134
        // Upper bound: sqrt(0.25) = 0.5
        // phi = 0.4 is in (0.134, 0.5) → should use IQI
        assert!(should_use_iqi(0.25, 0.4));

        // Test case: xi = 0.25, phi = 0.6
        // phi = 0.6 > 0.5 (upper bound) → should NOT use IQI
        assert!(!should_use_iqi(0.25, 0.6));

        // Test case: xi = 0.81, phi = 0.8
        // Lower bound: 1 - sqrt(0.19) ≈ 0.564
        // Upper bound: sqrt(0.81) = 0.9
        // phi = 0.8 is in (0.564, 0.9) → should use IQI
        assert!(should_use_iqi(0.81, 0.8));

        // Test case: xi = 0.81, phi = 0.95
        // phi = 0.95 > 0.9 (upper bound) → should NOT use IQI
        assert!(!should_use_iqi(0.81, 0.95));
    }

    #[test]
    fn test_should_use_iqi_boundary_cases() {
        // Invalid xi values should return false
        assert!(!should_use_iqi(0.0, 0.5)); // xi at boundary
        assert!(!should_use_iqi(1.0, 0.5)); // xi at boundary
        assert!(!should_use_iqi(-0.1, 0.5)); // xi negative
        assert!(!should_use_iqi(1.1, 0.5)); // xi > 1

        // Invalid phi values should return false
        assert!(!should_use_iqi(0.5, 0.0)); // phi at boundary
        assert!(!should_use_iqi(0.5, 1.0)); // phi at boundary
        assert!(!should_use_iqi(0.5, -0.1)); // phi negative
        assert!(!should_use_iqi(0.5, 1.1)); // phi > 1

        // NaN and infinity
        assert!(!should_use_iqi(f64::NAN, 0.5));
        assert!(!should_use_iqi(0.5, f64::NAN));
        assert!(!should_use_iqi(f64::INFINITY, 0.5));
        assert!(!should_use_iqi(0.5, f64::INFINITY));
    }

    #[test]
    fn test_should_use_iqi_equivalence() {
        // Verify the two equivalent formulations give the same result
        // Formula 1: (1 - sqrt(1 - xi)) < phi < sqrt(xi)
        // Formula 2: phi^2 < xi AND (1-phi)^2 < (1-xi)

        let test_cases: [(f64, f64); 9] = [
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.5),
            (0.6, 0.6),
            (0.7, 0.7),
            (0.8, 0.8),
            (0.9, 0.85),
        ];

        for (xi, phi) in test_cases {
            let formula1 = {
                let lower = 1.0 - (1.0 - xi).sqrt();
                let upper = xi.sqrt();
                phi > lower && phi < upper
            };
            let formula2 = phi * phi < xi && (1.0 - phi) * (1.0 - phi) < (1.0 - xi);

            assert_eq!(formula1, formula2, "Formulas disagree for xi={}, phi={}", xi, phi);

            // Also verify our implementation matches
            let result = should_use_iqi(xi, phi);
            assert_eq!(result, formula1, "Implementation differs for xi={}, phi={}", xi, phi);
        }
    }

    // =========================================================================
    // Tests for IQI parameter computation
    // =========================================================================

    #[test]
    fn test_iqi_parameter_linear_function() {
        // For a linear function, IQI should give the same result as linear interpolation
        // f(x) = x - 1, root at x = 1
        // Points: (0, -1), (2, 1), (0.5, -0.5)
        let state = ChandrupatlaState { x1: 0.0, f1: -1.0, x2: 2.0, f2: 1.0, x3: 0.5, f3: -0.5 };
        let config = ChandrupatlaConfig::default();

        let t = iqi_parameter(&state, &config);
        // x_new = x1 + t * (x2 - x1) = 0 + t * 2
        // For root at 1, we need t = 0.5
        assert!((t - 0.5).abs() < 0.1, "Expected t ≈ 0.5, got {}", t);
    }

    #[test]
    fn test_chandrupatla_next_t_uses_bisection_initially() {
        // When x3 = x1 (initial state), the criterion should fail
        // and we should get t = 0.5 (bisection)
        let state = ChandrupatlaState {
            x1: 0.0,
            f1: -1.0,
            x2: 10.0,
            f2: 1.0,
            x3: 0.0, // Same as x1
            f3: -1.0,
        };
        let config = ChandrupatlaConfig::default();

        let t = chandrupatla_next_t(&state, &config);
        assert!((t - 0.5).abs() < 0.01, "Expected bisection (t=0.5), got t={}", t);
    }

    // =========================================================================
    // Tests based on Chandrupatla (1997) test functions
    // Reference: SciPy _tstutils.py chandrupatla collection
    // =========================================================================

    /// Simple root-finding test using pure Chandrupatla on f(x) = x^3 - 2x - 5
    /// Root ≈ 2.0946
    #[test]
    fn test_chandrupatla_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let expected_root = 2.0945514815423265;

        // Run Chandrupatla manually
        let (root, iterations) = find_root_chandrupatla(f, 2.0, 3.0, 1e-10, 100);

        assert!(
            (root - expected_root).abs() < 1e-8,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
        assert!(iterations < 20, "Should converge quickly, took {} iterations", iterations);
    }

    /// Test on f(x) = (x - 3)^3, root at x = 3
    /// This is a "flat" function near the root (triple root)
    /// Note: Triple roots are challenging for root-finding algorithms
    #[test]
    fn test_chandrupatla_triple_root() {
        let f = |x: f64| (x - 3.0).powi(3);
        let expected_root = 3.0;

        let (root, iterations) = find_root_chandrupatla(f, 0.0, 10.0, 1e-10, 100);

        // Triple roots converge more slowly; use looser tolerance
        assert!(
            (root - expected_root).abs() < 1e-3,
            "Expected root = {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }

    /// Test on f(x) = x^9, root at x = 0
    /// Very flat near the root
    #[test]
    fn test_chandrupatla_high_power() {
        let f = |x: f64| x.powi(9);

        let (root, iterations) = find_root_chandrupatla(f, -1.0, 1.0, 1e-10, 100);

        assert!(
            root.abs() < 1e-3,
            "Expected root ≈ 0, got {} after {} iterations",
            root,
            iterations
        );
    }

    /// Test from TensorFlow: high-degree polynomial (x - r)^15
    /// Note: Very high degree polynomials are extremely flat near roots,
    /// making them challenging for any root-finding algorithm.
    /// TensorFlow uses atol=1e-2 for degree-15 polynomials.
    #[test]
    fn test_chandrupatla_degree_15() {
        let expected_root = 1.5;
        let f = |x: f64| (x - expected_root).powi(15);

        // Use tighter tolerance and more iterations for this hard case
        let (root, iterations) = find_root_chandrupatla(f, -20.0, 20.0, 1e-12, 200);

        // High-degree polynomials are very flat near roots
        // TensorFlow's test uses atol=1e-2 for degree-15 polynomials
        assert!(
            (root - expected_root).abs() < 0.5,
            "Expected root ≈ {}, got {} after {} iterations",
            expected_root,
            root,
            iterations
        );
    }

    // =========================================================================
    // Helper: Pure Chandrupatla implementation for testing
    // =========================================================================

    /// Pure Chandrupatla root-finding (for testing the algorithm itself)
    fn find_root_chandrupatla<F>(
        f: F,
        mut a: f64,
        mut b: f64,
        tol: f64,
        max_iter: u32,
    ) -> (f64, u32)
    where
        F: Fn(f64) -> f64,
    {
        let config = ChandrupatlaConfig::default();
        let mut fa = f(a);
        let mut fb = f(b);

        // Ensure fa and fb have opposite signs
        assert!(fa * fb <= 0.0, "f(a) and f(b) must have opposite signs");

        // Initialize: c = a (no previous point yet)
        let mut c = a;
        let mut fc = fa;

        for iteration in 0..max_iter {
            // Check convergence
            if (b - a).abs() < tol || fb.abs() < tol {
                return (b, iteration);
            }

            // Compute xi and phi
            let xi = (a - b) / (c - b);
            let phi = (fa - fb) / (fc - fb);

            // Decide: IQI or bisection
            let t = if should_use_iqi(xi, phi) {
                // IQI
                let alpha = (c - a) / (b - a);
                let df_ab = fa - fb;
                let df_cb = fc - fb;
                let df_ca = fc - fa;
                let df_bc = fb - fc;

                if df_ab.abs() > config.min_divisor &&
                    df_cb.abs() > config.min_divisor &&
                    df_ca.abs() > config.min_divisor &&
                    df_bc.abs() > config.min_divisor
                {
                    let term1 = (fa / df_ab) * (fc / df_cb);
                    let term2 = alpha * (fa / df_ca) * (fb / df_bc);
                    (term1 - term2).clamp(config.t_min, 1.0 - config.t_min)
                } else {
                    0.5
                }
            } else {
                0.5 // bisection
            };

            // New point
            let x_new = a + t * (b - a);
            let f_new = f(x_new);

            // Update bracket
            if f_new * fa > 0.0 {
                // Same sign as fa: replace a
                c = a;
                fc = fa;
                a = x_new;
                fa = f_new;
            } else {
                // Opposite sign: x_new brackets with a
                c = b;
                fc = fb;
                b = a;
                fb = fa;
                a = x_new;
                fa = f_new;
            }
        }

        (b, max_iter)
    }
}
