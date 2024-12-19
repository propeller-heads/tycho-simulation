use std::{collections::HashMap, env, time::Instant};

use clap::Parser;
use futures::{stream::BoxStream, StreamExt};
use num_bigint::BigUint;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tycho_simulation::{
    evm::{
        decoder::StreamDecodeError,
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            filters::{balancer_pool_filter, curve_pool_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    models::Token,
    protocol::models::BlockUpdate,
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_core::dto::Chain,
    utils::load_all_tokens,
};

#[derive(Parser, Debug, Clone, PartialEq)]
struct Cli {
    /// The exchanges to benchmark
    #[clap(long, number_of_values = 1)]
    exchange: Vec<String>,

    /// The number of swaps to benchmark
    #[clap(long, default_value = "100")]
    n_swaps: usize,

    /// The tvl threshold to filter the pools by
    #[clap(long, default_value = "1000.0")]
    tvl_threshold: f64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    let tycho_url =
        env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    let tvl_filter = ComponentFilter::with_tvl_range(cli.tvl_threshold, cli.tvl_threshold);

    let all_tokens = load_all_tokens(tycho_url.as_str(), false, Some(tycho_api_key.as_str())).await;

    let mut results = HashMap::new();

    for protocol in cli.exchange {
        {
            let stream = match protocol.as_str() {
                "uniswap_v2" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .build()
                    .await
                    .expect("Failed building Uniswap V2 protocol stream")
                    .boxed(),
                "uniswap_v3" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .build()
                    .await
                    .expect("Failed building Uniswap V3 protocol stream")
                    .boxed(),
                "balancer_v2" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<EVMPoolState<PreCachedDB>>(
                        "vm:balancer_v2",
                        tvl_filter.clone(),
                        Some(balancer_pool_filter),
                    )
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .build()
                    .await
                    .expect("Failed building Balancer V2 protocol stream")
                    .boxed(),
                "curve" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<EVMPoolState<PreCachedDB>>(
                        "vm:curve",
                        tvl_filter.clone(),
                        Some(curve_pool_filter),
                    )
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .build()
                    .await
                    .expect("Failed building Curve protocol stream")
                    .boxed(),
                _ => {
                    eprintln!("Unknown protocol: {}", protocol);
                    continue;
                }
            };

            info!("BENCHMARKING {} protocol on {} swaps", protocol, cli.n_swaps);
            let times = benchmark_swaps(stream, cli.n_swaps).await;
            results.insert(protocol, times);
        }
        // Add a small delay to ensure the WebSocket disconnection completes
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    analyze_results(&results, cli.n_swaps);
}

async fn benchmark_swaps(
    mut protocol_stream: BoxStream<'_, Result<BlockUpdate, StreamDecodeError>>,
    n: usize,
) -> Vec<u128> {
    let mut times = Vec::new();
    let mut pairs: HashMap<String, Vec<Token>> = HashMap::new();

    if let Some(Ok(message)) = protocol_stream.next().await {
        for (id, comp) in message.new_pairs.iter() {
            pairs
                .entry(id.clone())
                .or_insert_with(|| comp.tokens.clone());
        }

        if message.states.is_empty() {
            return times;
        }

        for (id, tokens) in pairs.iter().cycle() {
            if let Some(state) = message.states.get(id) {
                let amount_in =
                    BigUint::from(1u32) * BigUint::from(10u32).pow(tokens[0].decimals as u32);

                let start = Instant::now();
                let _ = state.get_amount_out(amount_in.clone(), &tokens[0], &tokens[1]);
                let duration = start.elapsed().as_nanos();

                times.push(duration);

                info!("Swap {} -> {} took {} ns", tokens[0].symbol, tokens[1].symbol, duration);

                if times.len() >= n {
                    break;
                }
            }
        }
    }

    times
}

fn calculate_std_dev(times: &[u128], avg: f64) -> f64 {
    let variance = times
        .iter()
        .map(|&time| (time as f64 - avg).powi(2))
        .sum::<f64>() /
        times.len() as f64;
    variance.sqrt()
}

/// Calculate the trimmed mean of a dataset.
///
/// The trimmed mean is calculated by removing outliers from the dataset. Outliers are defined as
/// values that are below the first quartile minus 1.5 times the interquartile range, or above the
/// third quartile plus 1.5 times the interquartile range. This gives us a more robust estimate of
/// the central tendency of the data.
fn calculate_trimmed_mean(times: &[u128]) -> Option<f64> {
    if times.is_empty() {
        return None;
    }

    // Sort the data
    let mut sorted_times = times.to_vec();
    sorted_times.sort_unstable();

    // Calculate quartiles
    let q1_index = sorted_times.len() / 4;
    let q3_index = 3 * sorted_times.len() / 4;

    let q1 = sorted_times[q1_index];
    let q3 = sorted_times[q3_index];
    let iqr = q3 - q1;

    // Convert to floating point for multiplication
    let lower_bound = (q1 as f64 - 1.5 * iqr as f64).max(0.0) as u128;
    let upper_bound = (q3 as f64 + 1.5 * iqr as f64) as u128;

    // Filter out outliers
    let filtered_times: Vec<&u128> = sorted_times
        .iter()
        .filter(|&&t| t >= lower_bound && t <= upper_bound)
        .collect();

    if filtered_times.is_empty() {
        None
    } else {
        // Calculate the trimmed mean
        Some(
            filtered_times
                .iter()
                .map(|&&t| t as f64)
                .sum::<f64>() /
                filtered_times.len() as f64,
        )
    }
}

fn analyze_results(results: &HashMap<String, Vec<u128>>, n_swaps: usize) {
    println!("\n========== Benchmark Results on {} swaps ==========", n_swaps);

    for (protocol, times) in results {
        let avg = times.iter().sum::<u128>() as f64 / times.len() as f64;
        let max = times.iter().max().unwrap_or(&0);
        let min = times.iter().min().unwrap_or(&0);
        let std_dev = calculate_std_dev(times, avg);
        let trimmed_mean = calculate_trimmed_mean(times).unwrap_or(f64::NAN);

        println!(
            "\n{} - Mean Time: {:.2} ns, Trimmed Mean Time: {:.2} ns, Max Time: {} ns, Min Time: {} ns, Std Dev: {:.2} ns",
            protocol, avg, trimmed_mean, max, min, std_dev
        );

        generate_histogram(times, 10);

        println!("\n---------------------------------------");
    }
}

fn generate_histogram(data: &[u128], num_bins: usize) {
    if data.is_empty() {
        println!("No data to display in histogram.");
        return;
    }

    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();
    let range = max - min;
    let bin_width = (range as f64 / num_bins as f64).ceil() as u128;

    // Initialize bins
    let mut bins = vec![0; num_bins];

    // Count data into bins
    for &value in data {
        let bin_index = ((value - min) / bin_width).min(num_bins as u128 - 1) as usize;
        bins[bin_index] += 1;
    }

    // Display the histogram
    println!("\nHistogram:");
    for (i, &count) in bins.iter().enumerate() {
        let lower_bound = min + (i as u128 * bin_width);
        let upper_bound = lower_bound + bin_width - 1;
        println!("{:>8} - {:<8} | {}", lower_bound, upper_bound, "*".repeat(count));
    }
}
