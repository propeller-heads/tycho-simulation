//! Extension traits and utilities for tycho-simulation
//!
//! This module provides extended functionality for working with protocol simulations,
//! including utilities for finding swap amounts to reach specific target prices.
//!
//! # Two Methods
//!
//! - **`swap_to_price`**: Find amount to move **spot price** to target. Returns error if
//!   not within tolerance after max iterations.
//! - **`query_supply`**: Find maximum trade where **trade price** stays at/below target.
//!   Always returns best valid trade, even if far from target.

pub mod strategies;

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

pub const SWAP_TO_PRICE_TOLERANCE: f64 = 0.00001; // 0.001%
pub const SWAP_TO_PRICE_MAX_ITERATIONS: u32 = 30;

/// Which price metric to track during the search.
///
/// All prices are in units of **token_out per token_in** (how much token_out you get per token_in).
/// Both metrics DECREASE as amount_in increases due to slippage.
/// Valid targets: `limit_price <= target < spot_price`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceMetric {
    /// Track the resulting spot price (marginal rate) after the swap.
    /// Used by `swap_to_price` - finds amount to move pool's marginal price down to target.
    SpotPrice,

    /// Track the trade price (execution price = amount_out / amount_in).
    /// Used by `query_supply` - finds max trade where average price stays at or above target.
    TradePrice,
}

/// Check if actual price is within tolerance of target price (one-sided)
///
/// The target_price is a hard upper limit we must not exceed. This function returns true if:
/// - `actual_price <= target_price` (hard upper limit), AND
/// - `actual_price >= target_price * (1 - SWAP_TO_PRICE_TOLERANCE)` (within tolerance below)
///
/// This applies to both `swap_to_price` (tracking spot price) and `query_supply` (tracking
/// trade price). In both cases, we're finding an amount where the resulting price approaches
/// the target from below without exceeding it.
pub fn within_tolerance(actual_price: f64, target_price: f64) -> bool {
    // actual_price must not exceed target (hard limit)
    if actual_price > target_price {
        return false;
    }
    // actual_price must be within tolerance below target
    let lower_bound = target_price * (1.0 - SWAP_TO_PRICE_TOLERANCE);
    actual_price >= lower_bound
}

/// Result of a price-targeting operation (swap_to_price or query_supply)
///
/// This result may represent either:
/// - **Converged**: `actual_price` is within `SWAP_TO_PRICE_TOLERANCE` of target
/// - **Best achievable**: `actual_price` is the closest the pool can represent
///
/// # Best Achievable Results
///
/// When a pool has limited price precision (e.g., stablecoin pairs on Curve),
/// the search may converge to adjacent integer amounts where neither achieves
/// the exact target price. In this case, the result with the price closest to
/// the target is returned.
///
/// Callers can check if the result is within tolerance using:
/// ```ignore
/// use tycho_simulation::swap_to_price::within_tolerance;
/// let is_exact = within_tolerance(result.actual_price, target_price);
/// ```
#[derive(Debug, Clone)]
pub struct SwapToPriceResult {
    /// The amount of input token needed to achieve the target price
    pub amount_in: BigUint,
    /// The amount of output token received
    pub amount_out: BigUint,
    /// The actual final price achieved (spot price or trade price depending on method).
    /// May differ from target if pool precision limits convergence (best achievable).
    /// Use `within_tolerance(actual_price, target_price)` to check if exact.
    pub actual_price: f64,
    /// Gas cost of the operation
    pub gas: BigUint,
    /// The updated protocol state after the swap
    pub new_state: Box<dyn ProtocolSim>,
    /// Number of get_amount_out calls (iterations) needed
    pub iterations: u32,
}

impl SwapToPriceResult {
    /// Calculate the trade price (execution price) for this result
    ///
    /// Trade price = amount_in / amount_out (how much you pay per unit received)
    pub fn trade_price(&self) -> Option<f64> {
        let amount_in_f64 = self.amount_in.to_f64()?;
        let amount_out_f64 = self.amount_out.to_f64()?;
        if amount_out_f64 == 0.0 {
            return None;
        }
        Some(amount_in_f64 / amount_out_f64)
    }
}

/// Result of a query_supply operation
///
/// This represents the maximum trade where the trade price (execution price)
/// stays at or below the target price.
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

/// Error types for swap-to-price operations
#[derive(Debug, thiserror::Error)]
pub enum SwapToPriceError {
    #[error("Target price {target} is above spot price {spot}. Target must be below spot (prices decrease with amount).")]
    TargetAboveSpot { target: f64, spot: f64 },
    #[error("Target price {target} is below limit price {limit} (spot: {spot}). Pool cannot reach such a low price.")]
    TargetBelowLimit { target: f64, spot: f64, limit: f64 },
    #[error("Limit price {limit} is at or above spot price {spot}. Expected limit < spot since prices decrease.")]
    LimitAboveSpot { limit: f64, spot: f64 },
    #[error("Target price {target} is below searchable range (min searchable price: {searchable_price}). Binary search with {max_iterations} iterations cannot converge.")]
    TargetBelowSearchable {
        target: f64,
        searchable_price: f64,
        max_iterations: u32,
    },
    #[error("Failed to converge within {iterations} iterations. Target: {target_price:.6e}, best: {best_price:.6e} (diff: {error_bps:.2}bps), amount: {amount}")]
    ConvergenceFailure {
        iterations: u32,
        target_price: f64,
        best_price: f64,
        /// Difference from target in basis points
        error_bps: f64,
        /// The amount that produced the best_price
        amount: String,
    },
    #[error("Simulation error: {0}")]
    SimulationError(#[from] SimulationError),
    #[error("Other error: {0}")]
    Other(String),
}

/// Extension trait for ProtocolSim with price-targeting search strategies
///
/// This trait provides two methods for finding trade amounts based on price targets:
///
/// - **`swap_to_price`**: Find amount to move spot price to target (strict convergence)
/// - **`query_supply`**: Find max trade at/below target trade price (best effort)
pub trait ProtocolSimExt {
    /// Calculate the amount of input token needed to reach a target spot price
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
    ///
    /// # Returns
    /// - `Ok(result)` if converged within tolerance
    /// - `Err(ConvergenceFailure)` if max iterations reached without convergence
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError>;

    /// Find the maximum trade where the trade price stays at or below the target
    ///
    /// This finds the largest amount of `token_in` that can be traded for `token_out`
    /// while keeping the execution price (amount_in / amount_out) at or below `target_price`.
    ///
    /// # Difference from swap_to_price
    /// - **Metric**: Tracks trade price (execution price), not spot price
    /// - **On max iterations**: Returns best valid trade found, never errors for convergence
    ///
    /// # Returns
    /// Always returns a result (may return zero trade if target is below spot)
    fn query_supply(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<QuerySupplyResult, SwapToPriceError>;
}

// Keep the old trait name as an alias for backward compatibility
pub use ProtocolSimExt as SwapToPriceStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_within_tolerance() {
        // SWAP_TO_PRICE_TOLERANCE is 0.00001 (0.001%)
        // Target is a hard upper limit - tolerance only applies below

        // Exactly at target - should pass
        assert!(within_tolerance(1.0, 1.0));

        // Just below target (within tolerance) - should pass
        assert!(within_tolerance(0.999995, 1.0)); // 0.0005% below
        assert!(within_tolerance(0.99999, 1.0)); // 0.001% below (at boundary)

        // Above target - should NEVER pass (hard limit)
        assert!(!within_tolerance(1.000001, 1.0)); // even tiny amount above
        assert!(!within_tolerance(1.000005, 1.0)); // 0.0005% above
        assert!(!within_tolerance(1.00001, 1.0)); // 0.001% above

        // Too far below target - should NOT pass (not close enough)
        assert!(!within_tolerance(0.9999, 1.0)); // 0.01% below

        // Test with larger numbers
        let target = 3000.0;
        assert!(within_tolerance(target, target)); // exact
        assert!(within_tolerance(target * 0.999995, target)); // within tolerance below
        assert!(!within_tolerance(target * 1.000001, target)); // above target (fails)
        assert!(!within_tolerance(target * 0.9999, target)); // too far below
    }

    #[test]
    fn test_within_tolerance_edge_cases() {
        // Very close to boundary
        let target = 1000.0;
        let tolerance = SWAP_TO_PRICE_TOLERANCE;

        // Just inside tolerance
        let just_inside = target * (1.0 - tolerance * 0.99);
        assert!(within_tolerance(just_inside, target));

        // Just outside tolerance
        let just_outside = target * (1.0 - tolerance * 1.01);
        assert!(!within_tolerance(just_outside, target));

        // Just above target (should always fail)
        let just_above = target * (1.0 + tolerance * 0.01);
        assert!(!within_tolerance(just_above, target));
    }
}
