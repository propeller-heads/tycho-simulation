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
// Constants
// =============================================================================

/// Minimum value for the interpolation parameter t.
/// Keeps the next estimate away from bracket endpoints.
/// The parameter t is clamped to [T_MIN, 1 - T_MIN].
const T_MIN: f64 = 0.05;

/// Fallback value for t when IQI cannot be computed (bisection).
const T_FALLBACK: f64 = 0.5;

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
}

impl Default for ChandrupatlaConfig {
    fn default() -> Self {
        Self { tolerance: 0.00001, max_iterations: 100, min_divisor: 1e-12 }
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

/// Calculate trade price (execution price) normalized by token decimals.
///
/// Trade price = (amount_in / amount_out) * 10^(decimals_out - decimals_in)
///
/// This normalizes raw token amounts to a human-readable price.
/// For example, swapping 1000 USDC (6 dec) for 0.5 WETH (18 dec):
/// - Raw: 1000e6 / 0.5e18 = 2e-9
/// - Normalized: 2e-9 * 10^(18-6) = 2000 USDC per WETH
fn calculate_trade_price(
    amount_in: f64,
    amount_out: f64,
    decimals_in: u32,
    decimals_out: u32,
) -> f64 {
    if amount_out <= 0.0 {
        return f64::MAX;
    }
    let decimal_adjustment = 10_f64.powi(decimals_out as i32 - decimals_in as i32);
    (amount_in / amount_out) * decimal_adjustment
}

/// Chandrupatla bracket state
///
/// Maintains three points for the algorithm:
/// - `x1, f1`: One bracket endpoint
/// - `x2, f2`: Other bracket endpoint (and current best estimate)
/// - `x3, f3`: Previous contrapoint (for IQI interpolation)
///
/// Invariant: f1 and f2 have opposite signs (bracket the root)
///
/// # Precision Note
///
/// The x values (amounts) are stored as `f64` for efficient interpolation arithmetic.
/// This means amounts larger than 2^53 (~9 × 10^15) cannot be represented exactly -
/// consecutive integers may round to the same f64 value. For 18-decimal tokens,
/// this corresponds to ~9 million tokens.
///
/// The algorithm maintains separate `BigUint` bounds (`low`, `high`) to ensure
/// evaluation points are always valid integers, but the interpolation itself
/// may lose precision for very large amounts. In practice, the algorithm will
/// still converge but may not find the mathematically optimal integer amount
/// for extremely large trades.
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
    #[error("Limit price {limit} is at or below spot price {spot}. Trading in this direction does not increase price.")]
    LimitBelowSpot { limit: f64, spot: f64 },
    #[error("Maximum output amount at limit is zero. Pool has no liquidity for output token.")]
    ZeroOutputAmountAtLimit,
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
    if a.is_zero() || b.is_zero() {
        return (a + b) / 2u32;
    }
    (a * b).sqrt()
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
        return T_FALLBACK;
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
        return T_FALLBACK;
    }

    let term1 = (f1 / df12) * (f3 / df32);
    let term2 = alpha * (f1 / df31) * (f2 / df23);

    let t = term1 - term2;

    // Clamp t to [t_min, 1 - t_min] to avoid getting too close to endpoints
    t.clamp(T_MIN, 1.0 - T_MIN)
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
/// 3. Otherwise: use t fallback (default 0.5 for bisection)
fn chandrupatla_next_t(state: &ChandrupatlaState, config: &ChandrupatlaConfig) -> f64 {
    let ChandrupatlaState { x1, f1, x2, f2, x3, f3 } = *state;

    // Compute xi (position ratio) and phi (function value ratio)
    let dx = x3 - x2;
    let df = f3 - f2;

    if dx.abs() < config.min_divisor || df.abs() < config.min_divisor {
        return T_FALLBACK;
    }

    let xi = (x1 - x2) / dx;
    let phi = (f1 - f2) / df;

    // Check Chandrupatla's criterion
    if should_use_iqi(xi, phi) {
        iqi_parameter(state, config)
    } else {
        T_FALLBACK
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
            let amount_in_f64 = amount.to_f64().unwrap_or(f64::MAX);
            let amount_out_f64 = result.amount.to_f64().unwrap_or(f64::MAX);
            calculate_trade_price(
                amount_in_f64,
                amount_out_f64,
                token_in.decimals,
                token_out.decimals,
            )
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

    // Step 3: Validate target price is above spot
    if target_price < spot_price {
        return Err(ChandrupatlaSearchError::TargetBelowSpot {
            target: target_price,
            spot: spot_price,
        });
    }

    // Step 4: Get limits
    let (max_amount_in, max_amount_out) =
        state.get_limits(token_in.address.clone(), token_out.address.clone())?;

    // Step 5: Calculate limit price (price at max trade)
    let limit_price = match search_config.metric {
        PriceMetric::TradePrice => {
            if max_amount_out.is_zero() {
                return Err(ChandrupatlaSearchError::ZeroOutputAmountAtLimit);
            }
            let max_in_f64 = max_amount_in.to_f64().unwrap_or(f64::MAX);
            let max_out_f64 = max_amount_out.to_f64().unwrap_or(f64::MAX);
            calculate_trade_price(
                max_in_f64,
                max_out_f64,
                token_in.decimals,
                token_out.decimals,
            )
        }
        PriceMetric::SpotPrice => {
            // Spot price after max trade requires simulation
            let limit_result = state.get_amount_out(max_amount_in.clone(), token_in, token_out)?;
            limit_result.new_state.spot_price(token_out, token_in)?
        }
    };

    // Step 6: Validate limit price
    if limit_price <= spot_price {
        return Err(ChandrupatlaSearchError::LimitBelowSpot { limit: limit_price, spot: spot_price });
    }

    // Step 7: Validate target is reachable
    if target_price > limit_price {
        return Err(ChandrupatlaSearchError::TargetAboveLimit {
            target: target_price,
            spot: spot_price,
            limit: limit_price,
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
    let f2 = limit_price - target_price; // positive

    // Initially x3 = x1 (no previous point yet)
    let mut chandru_state = ChandrupatlaState { x1, f1, x2, f2, x3: x1, f3: f1 };

    // Track best result metadata
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

        // Convert to BigUint
        let amount_new = if x_new <= 0.0 {
            BigUint::one()
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
        if (&high - &low).is_one() {
            let final_amount = best_amount.unwrap_or(amount_new.clone());
            let (price, _, amount_out, gas, new_state) = evaluate_objective(
                state,
                &final_amount,
                token_in,
                token_out,
                target_price,
                &search_config,
            )?;
            return Ok(SwapToPriceResult {
                amount_in: final_amount,
                amount_out,
                actual_price: price,
                gas,
                new_state,
                iterations: iteration + 1,
            });
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
    // Integration tests with standard test functions from SciPy
    // =========================================================================

    /// Run the Chandrupatla algorithm on a scalar function.
    fn run_chandrupatla<F>(f: F, a: f64, b: f64, tol: f64, max_iter: u32) -> (f64, f64, u32)
    where
        F: Fn(f64) -> f64,
    {
        let config = ChandrupatlaConfig::default();
        let mut x1 = a;
        let mut f1 = f(a);
        let mut x2 = b;
        let mut f2 = f(b);

        assert!(
            f1 * f2 <= 0.0,
            "f(a) and f(b) must have opposite signs: f({})={}, f({})={}",
            a, f1, b, f2
        );

        // Initialize x3 = x1 (no previous point yet)
        let mut x3 = x1;
        let mut f3 = f1;

        for iteration in 0..max_iter {
            // Check convergence
            if (x2 - x1).abs() < tol || f2.abs() < tol {
                let best_x = if f1.abs() < f2.abs() { x1 } else { x2 };
                let best_f = if f1.abs() < f2.abs() { f1 } else { f2 };
                return (best_x, best_f, iteration);
            }

            let state = ChandrupatlaState { x1, f1, x2, f2, x3, f3 };
            let t = chandrupatla_next_t(&state, &config);

            // Compute new point
            let x_new = x1 + t * (x2 - x1);
            let f_new = f(x_new);

            // Update bracket
            if f_new * f1 > 0.0 {
                // Same sign as f1: replace x1
                x3 = x1;
                f3 = f1;
                x1 = x_new;
                f1 = f_new;
            } else {
                // Opposite sign: x_new brackets with x1
                x3 = x2;
                f3 = f2;
                x2 = x1;
                f2 = f1;
                x1 = x_new;
                f1 = f_new;
            }
        }

        let best_x = if f1.abs() < f2.abs() { x1 } else { x2 };
        let best_f = if f1.abs() < f2.abs() { f1 } else { f2 };
        (best_x, best_f, max_iter)
    }

    /// SciPy fun1: x³ - 2x - 5, root ≈ 2.0945514815423265
    #[test]
    fn test_scipy_fun1_cubic() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let expected_root = 2.0945514815423265;

        let (root, f_root, iters) = run_chandrupatla(f, 2.0, 3.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} (f={}) after {} iterations",
            expected_root, root, f_root, iters
        );
        assert!(iters <= 12, "Should converge in ≤12 iterations, took {}", iters);
    }

    /// SciPy fun2: 1 - 1/x², root = 1 (singularity at 0)
    #[test]
    fn test_scipy_fun2_rational() {
        let f = |x: f64| 1.0 - 1.0 / (x * x);
        let expected_root = 1.0;

        let (root, f_root, iters) = run_chandrupatla(f, 0.5, 2.0, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root = {}, got {} (f={}) after {} iterations",
            expected_root, root, f_root, iters
        );
    }

    /// SciPy fun3: (x-3)³, root = 3 (triple root - challenging)
    #[test]
    fn test_scipy_fun3_triple_root() {
        let f = |x: f64| (x - 3.0).powi(3);
        let expected_root = 3.0;

        let (root, f_root, iters) = run_chandrupatla(f, 0.0, 6.0, 1e-12, 100);

        // Triple roots are flat, requiring looser tolerance
        assert!(
            (root - expected_root).abs() < 1e-3,
            "Expected root = {}, got {} (f={}) after {} iterations",
            expected_root, root, f_root, iters
        );
    }

    /// SciPy fun5: x⁹, root = 0 (very flat near root)
    #[test]
    fn test_scipy_fun5_high_power() {
        let f = |x: f64| x.powi(9);

        let (root, f_root, iters) = run_chandrupatla(f, -1.0, 1.0, 1e-12, 100);

        // High-power functions are flat near roots
        assert!(
            root.abs() < 1e-3,
            "Expected root ≈ 0, got {} (f={}) after {} iterations",
            root, f_root, iters
        );
    }

    /// SciPy fun9: exp(x) - 2 - 0.01/x² + 0.000002/x³
    #[test]
    fn test_scipy_fun9_mixed() {
        let f = |x: f64| x.exp() - 2.0 - 0.01 / (x * x) + 0.000002 / (x * x * x);
        let expected_root = 0.7032048403631358;

        let (root, f_root, iters) = run_chandrupatla(f, 0.5, 1.5, 1e-12, 100);

        assert!(
            (root - expected_root).abs() < 1e-10,
            "Expected root ≈ {}, got {} (f={}) after {} iterations",
            expected_root, root, f_root, iters
        );
    }

    /// Test with wide brackets (stress test from SciPy)
    #[test]
    fn test_wide_bracket() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let expected_root = 2.0945514815423265;

        // Very wide bracket: [-1e6, 1e6]
        let (root, f_root, iters) = run_chandrupatla(f, -1e6, 1e6, 1e-10, 100);

        assert!(
            (root - expected_root).abs() < 1e-8,
            "Expected root ≈ {}, got {} (f={}) after {} iterations",
            expected_root, root, f_root, iters
        );
        // Should still converge reasonably fast even with wide bracket
        assert!(iters <= 60, "Wide bracket should converge in ≤60 iterations, took {}", iters);
    }

    /// Test iteration count validation (like TensorFlow's test_chandrupatla_max_iterations)
    #[test]
    fn test_max_iterations_respected() {
        let f = |x: f64| x.powi(19); // Very flat, will need many iterations

        for max_iter in [5, 10, 15] {
            let (_, _, iters) = run_chandrupatla(f, -1.0, 1.0, 1e-15, max_iter);
            assert!(
                iters <= max_iter,
                "Should respect max_iterations={}, but ran {} iterations",
                max_iter, iters
            );
        }
    }

    /// Test convergence with very tight tolerance
    #[test]
    fn test_high_precision() {
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let expected_root = 2.0945514815423265;

        let (root, f_root, _iters) = run_chandrupatla(f, 2.0, 3.0, 1e-14, 100);

        assert!(
            (root - expected_root).abs() < 1e-12,
            "High precision: expected {}, got {} (f={})",
            expected_root, root, f_root
        );
        assert!(
            f_root.abs() < 1e-12,
            "Function value should be near zero: f({}) = {}",
            root, f_root
        );
    }

    // =========================================================================
    // Tests for trade price calculation (query_supply decimal normalization)
    // =========================================================================

    #[test]
    fn test_calculate_trade_price_same_decimals() {
        // Both tokens have 18 decimals (e.g., WETH <-> DAI)
        // 1e18 in for 2e18 out = 0.5 price
        let price = calculate_trade_price(1e18, 2e18, 18, 18);
        assert!((price - 0.5).abs() < 1e-10, "Expected 0.5, got {}", price);

        // 2e18 in for 1e18 out = 2.0 price
        let price = calculate_trade_price(2e18, 1e18, 18, 18);
        assert!((price - 2.0).abs() < 1e-10, "Expected 2.0, got {}", price);
    }

    #[test]
    fn test_calculate_trade_price_usdc_to_weth() {
        // USDC (6 dec) -> WETH (18 dec)
        // Swap 2000 USDC for 1 WETH
        // Raw amounts: 2000e6 in, 1e18 out
        // Expected price: 2000 USDC per WETH
        let amount_in = 2000.0 * 1e6; // 2000 USDC in raw
        let amount_out = 1.0 * 1e18; // 1 WETH in raw

        let price = calculate_trade_price(amount_in, amount_out, 6, 18);
        assert!(
            (price - 2000.0).abs() < 1e-6,
            "Expected 2000, got {}",
            price
        );
    }

    #[test]
    fn test_calculate_trade_price_weth_to_usdc() {
        // WETH (18 dec) -> USDC (6 dec)
        // Swap 1 WETH for 2000 USDC
        // Raw amounts: 1e18 in, 2000e6 out
        // Expected price: 0.0005 WETH per USDC
        let amount_in = 1.0 * 1e18; // 1 WETH in raw
        let amount_out = 2000.0 * 1e6; // 2000 USDC in raw

        let price = calculate_trade_price(amount_in, amount_out, 18, 6);
        assert!(
            (price - 0.0005).abs() < 1e-10,
            "Expected 0.0005, got {}",
            price
        );
    }

    #[test]
    fn test_calculate_trade_price_wbtc_to_usdc() {
        // WBTC (8 dec) -> USDC (6 dec)
        // Swap 1 WBTC for 50000 USDC
        // Raw amounts: 1e8 in, 50000e6 out
        // Expected price: 0.00002 WBTC per USDC
        let amount_in = 1.0 * 1e8; // 1 WBTC in raw
        let amount_out = 50000.0 * 1e6; // 50000 USDC in raw

        let price = calculate_trade_price(amount_in, amount_out, 8, 6);
        assert!(
            (price - 0.00002).abs() < 1e-12,
            "Expected 0.00002, got {}",
            price
        );
    }

    #[test]
    fn test_calculate_trade_price_zero_output() {
        let price = calculate_trade_price(1e18, 0.0, 18, 18);
        assert_eq!(price, f64::MAX);

        let price = calculate_trade_price(1e18, -1.0, 18, 18);
        assert_eq!(price, f64::MAX);
    }

    #[test]
    fn test_search_config_metrics() {
        let swap_config = SearchConfig::swap_to_price();
        assert_eq!(swap_config.metric, PriceMetric::SpotPrice);

        let query_config = SearchConfig::query_supply();
        assert_eq!(query_config.metric, PriceMetric::TradePrice);
    }
}
