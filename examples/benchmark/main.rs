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
            ekubo::state::EkuboState,
            filters::{balancer_pool_filter, curve_pool_filter, uniswap_v4_pool_with_hook_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    models::Token,
    protocol::models::BlockUpdate,
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_common::models::Chain,
    utils::load_all_tokens,
};

#[derive(Parser, Debug, Clone, PartialEq)]
struct Cli {
    /// The exchanges to benchmark
    #[clap(long, number_of_values = 1, value_parser = validate_exchange)]
    exchange: Vec<String>,

    /// The number of swaps to benchmark
    #[clap(long, default_value = "100")]
    n_swaps: usize,

    /// The tvl threshold to filter the pools by
    #[clap(long, default_value = "1000.0")]
    tvl_threshold: f64,
}

fn validate_exchange(exchange: &str) -> Result<String, String> {
    const SUPPORTED_EXCHANGES: &[&str] =
        &["uniswap_v2", "uniswap_v3", "balancer_v2", "curve", "uniswap_v4", "ekubo_v2"];
    if SUPPORTED_EXCHANGES.contains(&exchange) {
        Ok(exchange.to_string())
    } else {
        Err(format!(
            "Unsupported exchange '{}'. Supported exchanges are: {:?}",
            exchange, SUPPORTED_EXCHANGES
        ))
    }
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

    let all_tokens = load_all_tokens(
        tycho_url.as_str(),
        false,
        Some(tycho_api_key.as_str()),
        Chain::Ethereum,
        None,
        None,
    )
    .await;

    let mut results = HashMap::new();

    for protocol in cli.exchange {
        {
            let stream = match protocol.as_str() {
                "uniswap_v2" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .skip_state_decode_failures(true)
                    .build()
                    .await
                    .expect("Failed building Uniswap V2 protocol stream")
                    .boxed(),
                "uniswap_v3" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .skip_state_decode_failures(true)
                    .build()
                    .await
                    .expect("Failed building Uniswap V3 protocol stream")
                    .boxed(),
                "uniswap_v4" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<UniswapV4State>(
                        "uniswap_v4",
                        tvl_filter.clone(),
                        Some(uniswap_v4_pool_with_hook_filter),
                    )
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .skip_state_decode_failures(true)
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
                    .skip_state_decode_failures(true)
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
                    .skip_state_decode_failures(true)
                    .build()
                    .await
                    .expect("Failed building Curve protocol stream")
                    .boxed(),
                "ekubo_v2" => ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
                    .exchange::<EkuboState>("ekubo_v2", tvl_filter.clone(), None)
                    .auth_key(Some(tycho_api_key.clone()))
                    .set_tokens(all_tokens.clone())
                    .await
                    .skip_state_decode_failures(true)
                    .build()
                    .await
                    .expect("Failed building Ekubo protocol stream")
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

        info!("Got {} pairs", pairs.len());

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

fn analyze_results(results: &HashMap<String, Vec<u128>>, n_swaps: usize) {
    println!("\n========== Benchmark Results on {} swaps ==========", n_swaps);

    for (protocol, times) in results {
        let avg = times.iter().sum::<u128>() as f64 / times.len() as f64;
        let max = times.iter().max().unwrap_or(&0);
        let min = times.iter().min().unwrap_or(&0);
        let std_dev = calculate_std_dev(times, avg);
        let median = calculate_median(times).unwrap_or(f64::NAN);

        println!(
            "\n{} - Mean Time: {:.2} ns, Median Time: {:.2} ns, Max Time: {} ns, Min Time: {} ns, Std Dev: {:.2} ns",
            protocol, avg, median, max, min, std_dev
        );

        generate_histogram(times, 10);

        println!("\n---------------------------------------");
    }
}

fn calculate_median(times: &[u128]) -> Option<f64> {
    if times.is_empty() {
        return None;
    }

    let mut sorted_times = times.to_vec();
    sorted_times.sort_unstable();

    // Calculate quartiles
    let q1_index = sorted_times.len() / 4;
    let q3_index = 3 * sorted_times.len() / 4;

    let q1 = sorted_times[q1_index];
    let q3 = sorted_times[q3_index];
    let iqr = q3 - q1;

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

fn generate_histogram(data: &[u128], num_bins: usize) {
    if data.is_empty() {
        println!("No data to display in histogram.");
        return;
    }

    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();
    let range = max - min;
    let bin_width = (range as f64 / num_bins as f64).ceil() as u128;

    let mut bins = vec![0; num_bins];

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
