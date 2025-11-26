use crate::benchmark::BenchmarkResult;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub timestamp: String,
    pub results: Vec<BenchmarkResult>,
    pub summary: Summary,
}

#[derive(Serialize, Deserialize)]
pub struct Summary {
    pub total_scenarios: usize,
    pub converged: usize,
    pub not_converged: usize,
    pub failed: usize,
    pub convergence_rate: f64,
    pub by_strategy: HashMap<String, StrategyStats>,
    pub by_protocol: HashMap<String, ProtocolStats>,
    pub by_movement: HashMap<i32, MovementStats>,
}

#[derive(Serialize, Deserialize)]
pub struct ProtocolStats {
    pub total: usize,
    pub converged: usize,
    pub not_converged: usize,
    pub iterations: IterationStats,
}

#[derive(Serialize, Deserialize)]
pub struct MovementStats {
    pub total: usize,
    pub converged: usize,
    pub not_converged: usize,
    pub iterations: IterationStats,
}

#[derive(Serialize, Deserialize)]
pub struct StrategyStats {
    pub total: usize,
    pub converged: usize,
    pub not_converged: usize,
    pub iterations: IterationStats,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct IterationStats {
    pub min: u32,
    pub max: u32,
    pub mean: f64,
    pub median: u32,
    pub p95: u32,
    pub p99: u32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TimeStats {
    pub min_us: u64,
    pub max_us: u64,
    pub mean_us: f64,
    pub median_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

pub fn save_results(
    results: &[BenchmarkResult],
    output_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let summary = calculate_summary(results);

    let report = BenchmarkReport {
        timestamp: Utc::now().to_rfc3339(),
        results: results.to_vec(),
        summary,
    };

    let filename = format!("run_{}.json", Utc::now().format("%Y%m%d_%H%M%S"));
    let output_path = output_dir.join(&filename);

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&output_path, json)?;

    Ok(output_path)
}

pub fn print_summary(results: &[BenchmarkResult]) {
    let summary = calculate_summary(results);

    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(80));

    #[cfg(feature = "swap_to_price")]
    {
        use tycho_simulation::swap_to_price::{SWAP_TO_PRICE_MAX_ITERATIONS, SWAP_TO_PRICE_TOLERANCE};
        println!("\nConfiguration:");
        println!("  Tolerance: {:.8}%", SWAP_TO_PRICE_TOLERANCE * 100.0);
        println!("  Max iterations: {}", SWAP_TO_PRICE_MAX_ITERATIONS);
    }

    println!("\nOverall:");
    println!("  Total scenarios: {}", summary.total_scenarios);
    println!("  Converged: {}", summary.converged);
    println!("  Not converged: {}", summary.not_converged);
    println!("  Failed (errors): {}", summary.failed);
    println!("  Convergence rate: {:.1}%", summary.convergence_rate * 100.0);

    println!("\nBy Strategy (Iterations):");
    println!(
        "  {:<16} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Strategy", "Total", "Converged", "Not Conv", "Min", "Mean", "Median", "P95", "P99"
    );
    println!("  {}", "-".repeat(100));

    let mut strategies: Vec<_> = summary.by_strategy.iter().collect();
    strategies.sort_by_key(|(name, _)| name.to_string());

    for (strategy, stats) in &strategies {
        if stats.converged > 0 {
            println!(
                "  {:<16} {:>8} {:>10} {:>12} {:>8} {:>8.1} {:>8} {:>8} {:>8}",
                strategy,
                stats.total,
                stats.converged,
                stats.not_converged,
                stats.iterations.min,
                stats.iterations.mean,
                stats.iterations.median,
                stats.iterations.p95,
                stats.iterations.p99
            );
        } else {
            println!(
                "  {:<16} {:>8} {:>10} {:>12} {:>8}",
                strategy, stats.total, stats.converged, stats.not_converged, "N/A"
            );
        }
    }

    // Print elapsed time statistics by strategy
    println!("\nBy Strategy (Elapsed Time in microseconds):");
    println!(
        "  {:<16} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Strategy", "Min", "Mean", "Median", "P95", "P99"
    );
    println!("  {}", "-".repeat(70));

    // Calculate time stats per strategy
    let time_stats = calculate_time_stats_by_strategy(results);
    let mut time_strategies: Vec<_> = time_stats.iter().collect();
    time_strategies.sort_by_key(|(name, _)| name.to_string());

    for (strategy, stats) in time_strategies {
        println!(
            "  {:<16} {:>10} {:>10.0} {:>10} {:>10} {:>10}",
            strategy,
            stats.min_us,
            stats.mean_us,
            stats.median_us,
            stats.p95_us,
            stats.p99_us
        );
    }

    println!("\nBy Protocol:");
    println!("  {:<20} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Protocol", "Total", "Converged", "Not Conv", "Min", "Mean", "Median", "P95", "P99");
    println!("  {}", "-".repeat(100));

    let mut protocols: Vec<_> = summary.by_protocol.iter().collect();
    protocols.sort_by_key(|(name, _)| name.to_string());

    for (protocol, stats) in protocols {
        if stats.converged > 0 {
            println!(
                "  {:<20} {:>8} {:>10} {:>12} {:>8} {:>8.1} {:>8} {:>8} {:>8}",
                protocol,
                stats.total,
                stats.converged,
                stats.not_converged,
                stats.iterations.min,
                stats.iterations.mean,
                stats.iterations.median,
                stats.iterations.p95,
                stats.iterations.p99
            );
        } else {
            println!(
                "  {:<20} {:>8} {:>10} {:>12} {:>8}",
                protocol, stats.total, stats.converged, stats.not_converged, "N/A"
            );
        }
    }

    println!("\nBy Price Movement:");
    println!("  {:>8} {:>8} {:>10} {:>12} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Bps", "Total", "Converged", "Not Conv", "Min", "Mean", "Median", "P95", "P99");
    println!("  {}", "-".repeat(100));

    let mut movements: Vec<_> = summary.by_movement.iter().collect();
    movements.sort_by_key(|(bps, _)| **bps);

    for (bps, stats) in movements {
        if stats.converged > 0 {
            println!(
                "  {:>8} {:>8} {:>10} {:>12} {:>8} {:>8.1} {:>8} {:>8} {:>8}",
                bps,
                stats.total,
                stats.converged,
                stats.not_converged,
                stats.iterations.min,
                stats.iterations.mean,
                stats.iterations.median,
                stats.iterations.p95,
                stats.iterations.p99
            );
        } else {
            println!(
                "  {:>8} {:>8} {:>10} {:>12} {:>8}",
                bps, stats.total, stats.converged, stats.not_converged, "N/A"
            );
        }
    }

    // Show top scenarios by iteration count
    let mut successful: Vec<_> = results
        .iter()
        .filter(|r| r.status == "success")
        .collect();
    successful.sort_by_key(|r| std::cmp::Reverse(r.iterations));

    if !successful.is_empty() {
        println!("\nWorst 10 Scenarios by Iteration Count:");
        println!(
            "  {:>6} {:<14} {:<15} {:<8} {:<8} {:>8}",
            "Iters", "Strategy", "Protocol", "In", "Out", "Price(bps)"
        );
        println!("  {}", "-".repeat(80));

        for result in successful.iter().take(10) {
            println!(
                "  {:>6} {:<14} {:<15} {:<8} {:<8} {:>8}",
                result.iterations,
                result.strategy,
                result.protocol,
                result.token_in,
                result.token_out,
                result.target_movement_bps
            );
        }
    }

    // Show failed scenarios
    let failed: Vec<_> = results
        .iter()
        .filter(|r| r.status != "success")
        .collect();

    if !failed.is_empty() {
        println!("\nFailed Scenarios ({}):", failed.len());
        for result in failed.iter().take(10) {
            println!(
                "  [{}] {} ({}, {} -> {}, +{}bps): {}",
                result.strategy,
                &result.pool_id[..8.min(result.pool_id.len())],
                result.protocol,
                result.token_in,
                result.token_out,
                result.target_movement_bps,
                result.status
            );
        }
        if failed.len() > 10 {
            println!("  ... and {} more", failed.len() - 10);
        }

        // Analyze failure patterns - group by (pool, token_in, token_out, bps)
        let mut failure_patterns: HashMap<(String, String, String, i32), Vec<(&str, &str)>> =
            HashMap::new();
        for result in &failed {
            let key = (
                result.pool_id.clone(),
                result.token_in.clone(),
                result.token_out.clone(),
                result.target_movement_bps,
            );
            failure_patterns
                .entry(key)
                .or_default()
                .push((&result.strategy, &result.status));
        }

        // Count scenarios that fail for ALL strategies
        let num_strategies = results.iter().map(|r| &r.strategy).collect::<std::collections::HashSet<_>>().len();
        let universal_failures: Vec<_> = failure_patterns
            .iter()
            .filter(|(_, strategies)| strategies.len() == num_strategies)
            .collect();

        println!(
            "\nFailure Analysis: {} unique scenarios fail, {} fail for ALL {} strategies",
            failure_patterns.len(),
            universal_failures.len(),
            num_strategies
        );

        // Show sample of universal failures by protocol
        let mut by_protocol: HashMap<String, usize> = HashMap::new();
        for ((pool_id, _, _, _), _) in &universal_failures {
            if let Some(result) = results.iter().find(|r| &r.pool_id == pool_id) {
                *by_protocol.entry(result.protocol.clone()).or_default() += 1;
            }
        }

        println!("\n  Universal failures by protocol:");
        for (protocol, count) in by_protocol.iter() {
            println!("    {}: {} scenarios", protocol, count);
        }

        // Show sample universal failures with error types
        println!("\n  Sample universal failures:");
        for ((pool_id, token_in, token_out, bps), strategies) in universal_failures.iter().take(5) {
            let protocol = results.iter().find(|r| &r.pool_id == pool_id).map(|r| r.protocol.as_str()).unwrap_or("?");
            let errors: Vec<_> = strategies.iter().map(|(_, err)| *err).collect::<std::collections::HashSet<_>>().into_iter().collect();
            println!(
                "    {} ({}) {} -> {} +{}bps: {:?}",
                &pool_id[..8.min(pool_id.len())],
                protocol,
                token_in,
                token_out,
                bps,
                errors
            );
        }
    }

    println!("\n{}", "=".repeat(80));
}

fn calculate_summary(results: &[BenchmarkResult]) -> Summary {
    #[cfg(feature = "swap_to_price")]
    use tycho_simulation::swap_to_price::SWAP_TO_PRICE_TOLERANCE;
    #[cfg(not(feature = "swap_to_price"))]
    const SWAP_TO_PRICE_TOLERANCE: f64 = 0.001;

    let total_scenarios = results.len();

    // Count converged (status==success AND within tolerance)
    let converged_results: Vec<_> = results
        .iter()
        .filter(|r| {
            r.status == "success" &&
            (r.convergence_error_bps / 10000.0) <= SWAP_TO_PRICE_TOLERANCE
        })
        .collect();
    let converged = converged_results.len();

    // Count not converged (status==success BUT outside tolerance)
    let not_converged = results
        .iter()
        .filter(|r| {
            r.status == "success" &&
            (r.convergence_error_bps / 10000.0) > SWAP_TO_PRICE_TOLERANCE
        })
        .count();

    // Count failed (status != "success")
    let failed = results
        .iter()
        .filter(|r| r.status != "success")
        .count();

    let convergence_rate = if total_scenarios > 0 {
        converged as f64 / total_scenarios as f64
    } else {
        0.0
    };

    // By strategy
    let mut by_strategy: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_strategy
            .entry(result.strategy.clone())
            .or_default()
            .push(result);
    }

    let strategy_stats: HashMap<String, StrategyStats> = by_strategy
        .into_iter()
        .map(|(strategy, results)| {
            let total = results.len();
            let converged_results: Vec<_> = results
                .iter()
                .filter(|r| {
                    r.status == "success"
                        && (r.convergence_error_bps / 10000.0) <= SWAP_TO_PRICE_TOLERANCE
                })
                .copied()
                .collect();
            let converged = converged_results.len();
            let not_converged = total - converged;

            let iterations = if !converged_results.is_empty() {
                calculate_iteration_stats(&converged_results)
            } else {
                IterationStats {
                    min: 0,
                    max: 0,
                    mean: 0.0,
                    median: 0,
                    p95: 0,
                    p99: 0,
                }
            };

            (
                strategy,
                StrategyStats {
                    total,
                    converged,
                    not_converged,
                    iterations,
                },
            )
        })
        .collect();

    // By protocol
    let mut by_protocol: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_protocol
            .entry(result.protocol.clone())
            .or_default()
            .push(result);
    }

    let protocol_stats: HashMap<String, ProtocolStats> = by_protocol
        .into_iter()
        .map(|(protocol, results)| {
            let total = results.len();
            let converged_results: Vec<_> = results
                .iter()
                .filter(|r| {
                    r.status == "success" &&
                    (r.convergence_error_bps / 10000.0) <= SWAP_TO_PRICE_TOLERANCE
                })
                .copied()
                .collect();
            let converged = converged_results.len();
            // Not converged includes both: scenarios that succeeded but didn't converge,
            // and scenarios that failed with errors
            let not_converged = total - converged;

            let iterations = if !converged_results.is_empty() {
                calculate_iteration_stats(&converged_results)
            } else {
                IterationStats {
                    min: 0,
                    max: 0,
                    mean: 0.0,
                    median: 0,
                    p95: 0,
                    p99: 0,
                }
            };

            (
                protocol,
                ProtocolStats {
                    total,
                    converged,
                    not_converged,
                    iterations,
                },
            )
        })
        .collect();

    // By movement
    let mut by_movement: HashMap<i32, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        by_movement
            .entry(result.target_movement_bps)
            .or_default()
            .push(result);
    }

    let movement_stats: HashMap<i32, MovementStats> = by_movement
        .into_iter()
        .map(|(bps, results)| {
            let total = results.len();
            let converged_results: Vec<_> = results
                .iter()
                .filter(|r| {
                    r.status == "success" &&
                    (r.convergence_error_bps / 10000.0) <= SWAP_TO_PRICE_TOLERANCE
                })
                .copied()
                .collect();
            let converged = converged_results.len();
            // Not converged includes both: scenarios that succeeded but didn't converge,
            // and scenarios that failed with errors
            let not_converged = total - converged;

            let iterations = if !converged_results.is_empty() {
                calculate_iteration_stats(&converged_results)
            } else {
                IterationStats {
                    min: 0,
                    max: 0,
                    mean: 0.0,
                    median: 0,
                    p95: 0,
                    p99: 0,
                }
            };

            (
                bps,
                MovementStats {
                    total,
                    converged,
                    not_converged,
                    iterations,
                },
            )
        })
        .collect();

    Summary {
        total_scenarios,
        converged,
        not_converged,
        failed,
        convergence_rate,
        by_strategy: strategy_stats,
        by_protocol: protocol_stats,
        by_movement: movement_stats,
    }
}

fn calculate_iteration_stats(results: &[&BenchmarkResult]) -> IterationStats {
    if results.is_empty() {
        return IterationStats {
            min: 0,
            max: 0,
            mean: 0.0,
            median: 0,
            p95: 0,
            p99: 0,
        };
    }

    let mut iterations: Vec<u32> = results.iter().map(|r| r.iterations).collect();
    iterations.sort_unstable();

    let min = iterations[0];
    let max = iterations[iterations.len() - 1];
    let mean = iterations.iter().sum::<u32>() as f64 / iterations.len() as f64;
    let median = iterations[iterations.len() / 2];
    let p95 = iterations[(iterations.len() as f64 * 0.95) as usize];
    let p99 = iterations[(iterations.len() as f64 * 0.99) as usize];

    IterationStats {
        min,
        max,
        mean,
        median,
        p95,
        p99,
    }
}

fn calculate_time_stats_by_strategy(results: &[BenchmarkResult]) -> HashMap<String, TimeStats> {
    let mut by_strategy: HashMap<String, Vec<u64>> = HashMap::new();

    for result in results {
        // Only include successful converged results
        if result.status == "success" {
            by_strategy
                .entry(result.strategy.clone())
                .or_default()
                .push(result.elapsed_micros);
        }
    }

    by_strategy
        .into_iter()
        .map(|(strategy, mut times)| {
            times.sort_unstable();

            let stats = if times.is_empty() {
                TimeStats {
                    min_us: 0,
                    max_us: 0,
                    mean_us: 0.0,
                    median_us: 0,
                    p95_us: 0,
                    p99_us: 0,
                }
            } else {
                let min_us = times[0];
                let max_us = times[times.len() - 1];
                let mean_us = times.iter().sum::<u64>() as f64 / times.len() as f64;
                let median_us = times[times.len() / 2];
                let p95_idx = ((times.len() as f64 * 0.95) as usize).min(times.len() - 1);
                let p99_idx = ((times.len() as f64 * 0.99) as usize).min(times.len() - 1);
                let p95_us = times[p95_idx];
                let p99_us = times[p99_idx];

                TimeStats {
                    min_us,
                    max_us,
                    mean_us,
                    median_us,
                    p95_us,
                    p99_us,
                }
            };

            (strategy, stats)
        })
        .collect()
}
