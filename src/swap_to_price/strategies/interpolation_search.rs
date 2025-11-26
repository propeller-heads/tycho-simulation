use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};

use crate::swap_to_price::{
    within_tolerance, SwapToPriceResult, SwapToPriceError, SwapToPriceStrategy,
    SWAP_TO_PRICE_MAX_ITERATIONS,
};

/// Trait for different interpolation strategies for bounded search to calculate the next amount to test
pub trait InterpolationFunction {
    /// Calculate the interpolated amount based on current bounds and prices
    ///
    /// # Arguments
    /// * `low_amount` - Lower bound amount
    /// * `low_price` - Price at lower bound
    /// * `high_amount` - Upper bound amount
    /// * `high_price` - Price at upper bound
    /// * `target_price` - Target price we're searching for
    ///
    /// # Returns
    /// The next amount to test, or None if interpolation fails
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> Option<BigUint>;
}

/// Binary interpolation: always picks the midpoint, ignoring price information
/// This is equivalent to traditional binary search
pub struct BinaryInterpolation;

impl InterpolationFunction for BinaryInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        _low_price: f64,
        high_amount: &BigUint,
        _high_price: f64,
        _target_price: f64,
    ) -> Option<BigUint> {
        // Binary search: always take the midpoint
        Some((low_amount + high_amount) / 2u32)
    }
}

/// Linear interpolation: assumes price changes linearly with amount
pub struct LinearInterpolation;

impl InterpolationFunction for LinearInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> Option<BigUint> {
        // Check for (almost) zero price range
        if (high_price - low_price).abs() < 1e-15 {
            // Fall back to binary search midpoint
            return Some((low_amount + high_amount) / 2u32);
        }

        // Linear interpolation formula:
        // mid = low + (high - low) * (target_price - low_price) / (high_price - low_price)
        let amount_range = high_amount - low_amount;
        let price_range = high_price - low_price;
        let price_offset = target_price - low_price;

        // Calculate interpolation ratio
        let ratio = price_offset / price_range;

        // Clamp ratio to [0, 1] to handle floating point errors
        let ratio = ratio.clamp(0.0, 1.0);

        // Convert to BigUint: mid = low + amount_range * ratio
        let amount_range_f64 = amount_range.to_f64()?;
        let offset = (amount_range_f64 * ratio) as u128;

        Some(low_amount + BigUint::from(offset))
    }
}

/// Bounded linear interpolation: limits deviation from binary search midpoint
/// This prevents worst-case performance on highly non-linear curves
pub struct BoundedLinearInterpolation {
    /// Maximum deviation from binary midpoint (0.0 to 1.0)
    /// 0.5 = can deviate up to 50% of the range from midpoint
    pub max_deviation: f64,
}

impl InterpolationFunction for BoundedLinearInterpolation {
    fn interpolate(
        &self,
        low_amount: &BigUint,
        low_price: f64,
        high_amount: &BigUint,
        high_price: f64,
        target_price: f64,
    ) -> Option<BigUint> {
        // First, get linear interpolation result
        let linear = LinearInterpolation;
        let interpolated = linear.interpolate(
            low_amount,
            low_price,
            high_amount,
            high_price,
            target_price,
        )?;

        // Calculate binary search midpoint
        let binary_mid = (low_amount + high_amount) / 2u32;

        // Calculate the range
        let range = high_amount - low_amount;
        let max_offset = (range.to_f64()? * self.max_deviation) as u128;

        // Clamp interpolated result to be within max_deviation of binary midpoint
        let diff = if interpolated > binary_mid {
            &interpolated - &binary_mid
        } else {
            &binary_mid - &interpolated
        };

        if diff > BigUint::from(max_offset) {
            // Clamp to max deviation
            if interpolated > binary_mid {
                Some(&binary_mid + BigUint::from(max_offset))
            } else {
                Some(&binary_mid - BigUint::from(max_offset))
            }
        } else {
            Some(interpolated)
        }
    }
}

/// Interpolation search strategy for finding amount_in to reach target price
pub struct InterpolationSearchStrategy<F: InterpolationFunction> {
    pub interpolation_fn: F,
}

impl<F: InterpolationFunction> InterpolationSearchStrategy<F> {
    pub fn new(interpolation_fn: F) -> Self {
        Self { interpolation_fn }
    }
}

impl<F: InterpolationFunction> SwapToPriceStrategy for InterpolationSearchStrategy<F> {
    fn swap_to_price(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<SwapToPriceResult, SwapToPriceError> {
        let max_iterations = SWAP_TO_PRICE_MAX_ITERATIONS;

        // Step 1: Get spot price
        let spot_price = state.spot_price(token_out, token_in)?;

        // Step 2: Check if we're already at target (within tolerance)
        if within_tolerance(spot_price, target_price) {
            // Already at target, no swap needed
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
        let (max_amount_in, _max_amount_out) =
            state.get_limits(token_in.address.clone(), token_out.address.clone())?;

        // Step 4: Validate target price is above spot
        if target_price < spot_price {
            return Err(SwapToPriceError::TargetBelowSpot {
                target: target_price,
                spot: spot_price,
            });
        }

        // Step 5: Calculate limit price by simulating max swap
        let limit_price = state
            .get_amount_out(max_amount_in.clone(), token_in, token_out)?
            .new_state
            .spot_price(token_out, token_in)?;

        // Step 6: Validate limit price is valid (should be > spot price)
        if limit_price <= spot_price {
            return Err(SwapToPriceError::TargetAboveLimit {
                target: target_price,
                spot: spot_price,
                limit: limit_price,
            });
        }

        // Step 7: Validate target is reachable within pool limits
        if target_price > limit_price {
            return Err(SwapToPriceError::TargetAboveLimit {
                target: target_price,
                spot: spot_price,
                limit: limit_price,
            });
        }

        // Step 8: Interpolation search
        let mut low = BigUint::from(0u32);
        let mut low_price = spot_price;
        let mut high = max_amount_in.clone();
        let mut high_price = limit_price;
        let mut actual_iterations = 0;

        for iterations in 1..=max_iterations {
            actual_iterations = iterations;

            // Calculate interpolated midpoint
            let mid = self
                .interpolation_fn
                .interpolate(
                    &low,
                    low_price,
                    &high,
                    high_price,
                    target_price,
                )
                .ok_or_else(|| SwapToPriceError::ConvergenceFailure(iterations))?;

            // Ensure we make progress
            if mid <= low || mid >= high {
                // No progress possible, try boundaries
                break;
            }

            // Calculate price at midpoint
            let result = state.get_amount_out(mid.clone(), token_in, token_out)?;
            let price = result.new_state.spot_price(token_out, token_in)?;

            // Check if we're within tolerance
            if within_tolerance(price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: mid,
                    actual_price: price,
                    gas: result.gas.clone(),
                    new_state: result.new_state,
                    iterations,
                });
            }

            // Adjust search range
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

        // Check low boundary
        if low > BigUint::from(0u32) {
            let low_result = state.get_amount_out(low.clone(), token_in, token_out)?;
            let low_price = low_result.new_state.spot_price(token_out, token_in)?;
            if within_tolerance(low_price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: low,
                    actual_price: low_price,
                    gas: low_result.gas,
                    new_state: low_result.new_state,
                    iterations: actual_iterations,
                });
            }
        }

        // Check high boundary if different from low
        if high != low && high > BigUint::from(0u32) {
            let high_result = state.get_amount_out(high.clone(), token_in, token_out)?;
            let high_price = high_result.new_state.spot_price(token_out, token_in)?;
            if within_tolerance(high_price, target_price) {
                return Ok(SwapToPriceResult {
                    amount_in: high,
                    actual_price: high_price,
                    gas: high_result.gas,
                    new_state: high_result.new_state,
                    iterations: actual_iterations,
                });
            }
        }

        // If we reach here, we failed to converge within tolerance
        Err(SwapToPriceError::ConvergenceFailure(actual_iterations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::primitives::U256;
    use rstest::rstest;
    use std::str::FromStr;
    use tycho_common::models::Chain;
    use crate::evm::protocol::uniswap_v2::state::UniswapV2State;

    // Test helper functions
    fn create_token0() -> Token {
        Token::new(
            &tycho_common::Bytes::from_str("0x6b175474e89094c44da98b954eedeac495271d0f")
                .unwrap(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn create_token1() -> Token {
        Token::new(
            &tycho_common::Bytes::from_str("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")
                .unwrap(),
            "WETH",
            18,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        )
    }

    fn create_pool() -> UniswapV2State {
        UniswapV2State::new(
            U256::from_str("6005747565594546069633144").unwrap(),
            U256::from_str("2148576922062920125253").unwrap(),
        )
    }

    // Test all interpolation strategies
    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_binary_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(BinaryInterpolation);
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Binary: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_linear_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Linear: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(10)]
    #[case(50)]
    #[case(100)]
    fn test_bounded_interpolation_price_increase(#[case] bps: u32) {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let multiplier = 1.0 + (bps as f64 / 10000.0);
        let target_price = spot_price * multiplier;

        let strategy = InterpolationSearchStrategy::new(BoundedLinearInterpolation {
            max_deviation: 0.5,
        });
        let result = strategy
            .swap_to_price(&pool, target_price, &token0, &token1)
            .unwrap();

        assert!(
            within_tolerance(result.actual_price, target_price),
            "Bounded: Price {} not within tolerance of target {} ({} bps)",
            result.actual_price,
            target_price,
            bps
        );
        assert!(result.amount_in > BigUint::from(0u32));
    }

    #[test]
    fn test_target_equals_spot_price() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy
            .swap_to_price(&pool, spot_price, &token0, &token1)
            .unwrap();

        assert_eq!(result.amount_in, BigUint::from(0u32));
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_target_below_spot_returns_error() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let target_price = spot_price * 0.99;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy.swap_to_price(&pool, target_price, &token0, &token1);

        assert!(matches!(result, Err(SwapToPriceError::TargetBelowSpot { .. })));
    }

    #[test]
    fn test_target_above_limit_returns_error() {
        let token0 = create_token0();
        let token1 = create_token1();
        let pool = create_pool();

        let spot_price = pool.spot_price(&token1, &token0).unwrap();
        let target_price = spot_price * 1000.0;

        let strategy = InterpolationSearchStrategy::new(LinearInterpolation);
        let result = strategy.swap_to_price(&pool, target_price, &token0, &token1);

        assert!(matches!(
            result,
            Err(SwapToPriceError::TargetAboveLimit { .. })
        ));
    }
}
