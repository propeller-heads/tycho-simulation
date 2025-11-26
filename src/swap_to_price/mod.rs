//! Extension traits and utilities for tycho-simulation
//!
//! This module provides extended functionality for working with protocol simulations,
//! including utilities for finding swap amounts to reach specific target prices.

pub mod strategies;

use num_bigint::BigUint;
use tycho_common::{
    models::token::Token,
    simulation::{errors::SimulationError, protocol_sim::ProtocolSim},
};

pub const SWAP_TO_PRICE_TOLERANCE: f64 = 0.0001; // 0.01% tolerance
pub const SWAP_TO_PRICE_MAX_ITERATIONS: u32 = 256;

/// Check if actual price is within tolerance of target price
///
/// Returns true if the relative difference between actual and target is <= SWAP_TO_PRICE_TOLERANCE
pub fn within_tolerance(actual_price: f64, target_price: f64) -> bool {
    let diff = (actual_price - target_price).abs();
    let relative_diff = diff / target_price;
    relative_diff <= SWAP_TO_PRICE_TOLERANCE
}

/// Result of a swap_to_price operation
#[derive(Debug, Clone)]
pub struct SwapToPriceResult {
    /// The amount of input token needed to achieve the target price
    pub amount_in: BigUint,
    /// The actual final price achieved (may differ slightly from target)
    pub actual_price: f64,
    /// Gas cost of the operation
    pub gas: BigUint,
    /// The updated protocol state after the swap
    pub new_state: Box<dyn ProtocolSim>,
    /// Number of get_amount_out calls (iterations) needed
    /// TODO: This is a temporary variable for benchmarking purposes
    pub iterations: u32,
}

/// Error types for swap-to-price operations
#[derive(Debug, thiserror::Error)]
pub enum SwapToPriceError {
    #[error("Target price {target} is below spot price {spot}. Cannot reach target by trading in this direction.")]
    TargetBelowSpot { target: f64, spot: f64 },
    #[error("Target price {target} is above limit price {limit} (spot: {spot}). Pool doesn't have enough liquidity to reach target.")]
    TargetAboveLimit { target: f64, spot: f64, limit: f64 },
    #[error("Target price {target} is above searchable range (max searchable price: {searchable_price}). Binary search with {max_iterations} iterations cannot converge beyond 2^{max_iterations} precision. Consider increasing max iterations or using a different strategy.")]
    TargetAboveSearchable {
        target: f64,
        searchable_price: f64,
        max_iterations: u32,
    },
    #[error("Failed to converge within {0} iterations")]
    ConvergenceFailure(u32),
    #[error("Simulation error: {0}")]
    SimulationError(#[from] SimulationError),
}

/// Trait for different swap-to-price strategies
pub trait SwapToPriceStrategy {
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
    /// # Arguments
    /// * `state` - The current protocol state
    /// * `target_price` - The desired price of `token_out` in terms of `token_in`
    ///   (i.e., `spot_price(token_out, token_in)`)
    /// * `token_in` - The token being sold (quote token in the price)
    /// * `token_out` - The token being bought (base token in the price)
    ///
    /// # Returns
    /// The amount in needed and the resulting state, or an error
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_within_tolerance() {
        // 0.1% difference, should be within 0.1% tolerance
        assert!(within_tolerance(1.001, 1.0));

        // 0.05% difference, should be within tolerance
        assert!(within_tolerance(1.0005, 1.0));

        // 2% difference, should NOT be within 0.1% tolerance
        assert!(!within_tolerance(1.02, 1.0));

        // Exactly at tolerance boundary (0.1%)
        assert!(within_tolerance(1.001, 1.0));

        // Test with larger numbers
        assert!(within_tolerance(2000.0, 2001.0));
        assert!(!within_tolerance(2000.0, 2100.0));
    }
}
