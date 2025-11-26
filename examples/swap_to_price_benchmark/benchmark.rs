use crate::snapshot;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;
use tycho_common::models::token::Token;

#[cfg(feature = "swap_to_price")]
use tycho_simulation::swap_to_price::{
    strategies::{
        BinaryInterpolation, BoundedLinearInterpolation, BrentStrategy, ChandrupatlaStrategy,
        ConvexSearchStrategy, ExponentialProbing, InterpolationSearchStrategy, IqiStrategy,
        IqiV2Strategy, LinearInterpolation, LogAmountBinarySearch, LogPriceInterpolation,
        LogarithmicBisection, NewtonCentralStrategy, NewtonLogStrategy, PiecewiseLinearStrategy,
        QuadraticRegressionStrategy, SecantMethod, SqrtPriceInterpolation, TwoPhaseSearch,
        WeightedRegressionStrategy,
    },
    SwapToPriceStrategy, SWAP_TO_PRICE_MAX_ITERATIONS, SWAP_TO_PRICE_TOLERANCE,
};

// Price movements to test (as multipliers)
// Combined range covering both regular and stable pair scenarios
// DEBUG: Just +1bps for now
const PRICE_MOVEMENTS: &[f64] = &[
    1.00005, // +0.005% = +0.5 bps
    1.0001,  // +0.01% = +1 bps
    1.0005,  // +0.05% = +5 bps
    1.001,   // +0.1% = +10 bps
    1.005,   // +0.5% = +50 bps
    1.01,    // +1% = +100 bps
    1.05,    // +5% = +500 bps
    1.1,     // +10% = +1000 bps
    1.5,     // +50% = +5000 bps
    2.0,     // +100% = +10000 bps
];

#[derive(Serialize, Deserialize, Clone)]
pub struct BenchmarkResult {
    pub strategy: String,
    pub pool_id: String,
    pub protocol: String,
    pub token_in: String,
    pub token_out: String,
    pub spot_price: f64,
    pub target_price: f64,
    pub target_movement_bps: i32,
    pub actual_price: f64,
    pub amount_in: String,
    pub iterations: u32,
    pub gas: String,
    pub elapsed_micros: u64,
    pub convergence_error_bps: f64,
    pub status: String,
}

pub async fn run_benchmark(
    snapshot_path: &Path,
) -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
    // Load and process snapshot in one step
    let loaded = snapshot::load_and_process_snapshot(snapshot_path).await?;

    println!("Loaded snapshot with {} pools", loaded.states.len());
    println!("Block: {}", loaded.metadata.block_number);
    println!("Chain: {}", loaded.metadata.chain);

    let mut results = Vec::new();

    #[cfg(feature = "swap_to_price")]
    {
        // Filter for specific pool (None = all pools)
        const DEBUG_POOL_FILTER: Option<&str> = None;

        // Define all strategies to benchmark
        let strategies: Vec<(&str, Box<dyn SwapToPriceStrategy>)> = vec![
            (
                "log_amount",
                Box::new(InterpolationSearchStrategy::new(LogAmountBinarySearch)),
            ),
            ("iqi", Box::new(IqiStrategy)),
            ("iqi_v2", Box::new(IqiV2Strategy)),
            ("brent", Box::new(BrentStrategy)),
            ("chandrupatla", Box::new(ChandrupatlaStrategy)),
            ("newton_cd", Box::new(NewtonCentralStrategy)),
            ("newton_log", Box::new(NewtonLogStrategy)),
            ("convex", Box::new(ConvexSearchStrategy)),
            ("quad_regr", Box::new(QuadraticRegressionStrategy)),
        ];

        println!("Testing {} strategies: {}", strategies.len(),
            strategies.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(", "));

        for (pool_idx, (pool_id, state)) in loaded.states.iter().enumerate() {
            // DEBUG: Skip pools not matching filter
            if let Some(filter) = DEBUG_POOL_FILTER {
                // dbg!(pool_id);
                if !pool_id.contains(filter) {
                    continue;
                }
            }
            let component = match loaded.components.get(pool_id) {
                Some(c) => c,
                None => {
                    println!("  ⚠ Skipping {}: component not found", pool_id);
                    continue;
                }
            };

            println!(
                "\nBenchmarking pool {}/{}: {} ({})",
                pool_idx + 1,
                loaded.states.len(),
                pool_id,
                component.protocol_system
            );

            // Get tokens from the component (they already have all fields set)
            let tokens: Vec<Token> = component.tokens.clone();

            // Test all token pair directions
            for i in 0..tokens.len() {
                for j in 0..tokens.len() {
                    if i == j {
                        continue;
                    }

                    let token_in = &tokens[i];
                    let token_out = &tokens[j];

                    // Spot price of token_out in terms of token_in
                    let spot_price = match state.spot_price(token_out, token_in) {
                        Ok(price) => price,
                        Err(e) => {
                            println!(
                                "  ⚠ Could not get spot price for {} -> {}: {}",
                                token_in.symbol, token_out.symbol, e
                            );
                            continue;
                        }
                    };

                    // Skip if spot price is 0 (invalid/weird state)
                    if (spot_price - 0.0).abs() < f64::EPSILON {
                        println!(
                            "  ⚠ Spot price is approximately 0 for {} -> {}, skipping",
                            token_in.symbol, token_out.symbol
                        );
                        continue;
                    }

                    // Use scientific notation for very small or very large numbers
                    let spot_display = if spot_price.abs() < 0.0001 || spot_price.abs() > 1e9 {
                        format!("{:.6e}", spot_price)
                    } else {
                        format!("{:.6}", spot_price)
                    };

                    // Calculate limit price for this direction
                    let limit_price = match state.get_limits(token_in.address.clone(), token_out.address.clone()) {
                        Ok((max_amount_in, _)) => {
                            state.get_amount_out(max_amount_in, token_in, token_out)
                                .ok()
                                .and_then(|result| result.new_state.spot_price(token_out, token_in).ok())
                        }
                        Err(_) => None,
                    };

                    // Filter price movements to only include those within the limit
                    let valid_movements: Vec<f64> = if let Some(limit) = limit_price {
                        PRICE_MOVEMENTS
                            .iter()
                            .copied()
                            .filter(|&multiplier| spot_price * multiplier <= limit)
                            .collect()
                    } else {
                        // If we can't determine limit, skip all movements
                        Vec::new()
                    };

                    if valid_movements.is_empty() {
                        println!(
                            "  {} -> {} (spot: {}) - skipping: no valid price movements within limit",
                            token_in.symbol, token_out.symbol, spot_display
                        );
                        continue;
                    }

                    let limit_display = if let Some(limit) = limit_price {
                        if limit.abs() < 0.0001 || limit.abs() > 1e9 {
                            format!("{:.6e}", limit)
                        } else {
                            format!("{:.6}", limit)
                        }
                    } else {
                        "unknown".to_string()
                    };

                    println!(
                        "  {} -> {} (spot: {}, limit: {}, testing {} movements)",
                        token_in.symbol, token_out.symbol, spot_display, limit_display, valid_movements.len()
                    );

                    // Test each valid price movement with each strategy
                    for &multiplier in &valid_movements {
                        let target_price = spot_price * multiplier;
                        let bps_f64 = (multiplier - 1.0) * 10000.0;
                        let bps = bps_f64.round() as i32;

                        // Format bps display - use decimal places for very small movements
                        let bps_display = if bps_f64.abs() < 1.0 {
                            format!("{:.2}bps", bps_f64)
                        } else {
                            format!("{}bps", bps)
                        };

                        // Test each strategy
                        for (strategy_name, strategy) in &strategies {
                            let start = Instant::now();
                            let result = strategy.swap_to_price(
                                state.as_ref(),
                                target_price,
                                token_in,
                                token_out,
                            );
                            let elapsed = start.elapsed();

                            let bench_result = match result {
                                Ok(res) => {
                                    let convergence_error =
                                        ((res.actual_price - target_price).abs() / target_price)
                                            * 10000.0;

                                    BenchmarkResult {
                                        strategy: strategy_name.to_string(),
                                        pool_id: pool_id.clone(),
                                        protocol: component.protocol_system.clone(),
                                        token_in: token_in.symbol.clone(),
                                        token_out: token_out.symbol.clone(),
                                        spot_price,
                                        target_price,
                                        target_movement_bps: bps,
                                        actual_price: res.actual_price,
                                        amount_in: res.amount_in.to_string(),
                                        iterations: res.iterations,
                                        gas: res.gas.to_string(),
                                        elapsed_micros: elapsed.as_micros() as u64,
                                        convergence_error_bps: convergence_error,
                                        status: "success".to_string(),
                                    }
                                }
                                Err(e) => BenchmarkResult {
                                    strategy: strategy_name.to_string(),
                                    pool_id: pool_id.clone(),
                                    protocol: component.protocol_system.clone(),
                                    token_in: token_in.symbol.clone(),
                                    token_out: token_out.symbol.clone(),
                                    spot_price,
                                    target_price,
                                    target_movement_bps: bps,
                                    actual_price: 0.0,
                                    amount_in: "0".to_string(),
                                    iterations: 0,
                                    gas: "0".to_string(),
                                    elapsed_micros: elapsed.as_micros() as u64,
                                    convergence_error_bps: 0.0,
                                    status: format!("{:?}", e),
                                },
                            };

                            if bench_result.status == "success" {
                                // Check if converged within tolerance
                                let converged =
                                    bench_result.convergence_error_bps / 10000.0 <= SWAP_TO_PRICE_TOLERANCE;

                                let output = if bench_result.iterations == SWAP_TO_PRICE_MAX_ITERATIONS
                                    || !converged
                                {
                                    format!(
                                        "    +{:>7} [{:<14}]: {:>3} iters, {:.6} actual (diff: {:.4}%){}",
                                        bps_display,
                                        strategy_name,
                                        bench_result.iterations,
                                        bench_result.actual_price,
                                        bench_result.convergence_error_bps / 100.0,
                                        if !converged { " ⚠ NOT CONVERGED" } else { "" }
                                    )
                                } else {
                                    format!(
                                        "    +{:>7} [{:<14}]: {:>3} iters",
                                        bps_display, strategy_name, bench_result.iterations
                                    )
                                };
                                println!("{}", output);
                            } else {
                                println!(
                                    "    +{:>7} [{:<14}]: {}",
                                    bps_display, strategy_name, bench_result.status
                                );
                            }

                            results.push(bench_result);
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "swap_to_price"))]
    {
        return Err("This benchmark requires the 'swap_to_price' feature to be enabled".into());
    }

    Ok(results)
}
