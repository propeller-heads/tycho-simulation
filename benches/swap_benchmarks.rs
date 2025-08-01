use std::{collections::HashMap, env, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use futures::StreamExt;
use num_bigint::BigUint;
use rand::Rng;
use tokio::runtime::Runtime;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            ekubo::state::EkuboState,
            filters::{
                balancer_v2_pool_filter_after_dci_update, curve_pool_filter,
                uniswap_v4_pool_with_hook_filter,
            },
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    protocol::models::ProtocolComponent,
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_common::models::Chain,
    utils::load_all_tokens,
};

// Configuration constants and environment variables
const DEFAULT_N_SWAPS: usize = 100;
const DEFAULT_TVL_THRESHOLD: f64 = 1000.0;

fn get_config() -> (usize, f64) {
    let n_swaps = env::var("BENCH_N_SWAPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_SWAPS);

    let tvl_threshold = env::var("BENCH_TVL_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TVL_THRESHOLD);

    (n_swaps, tvl_threshold)
}

// Shared benchmark data structure organized by protocol
#[derive(Clone)]
struct ProtocolBenchmarkData {
    pools: HashMap<String, Vec<Token>>,
    components: HashMap<String, ProtocolComponent>,
    states: HashMap<String, Box<dyn ProtocolSim>>,
}

type BenchmarkDataByProtocol = HashMap<String, ProtocolBenchmarkData>;

async fn load_all_benchmark_data() -> BenchmarkDataByProtocol {
    let (_, tvl_threshold) = get_config();

    let tycho_url =
        env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
    let tycho_api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    let tvl_filter = ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold);

    let all_tokens =
        load_all_tokens(&tycho_url, false, Some(&tycho_api_key), Chain::Ethereum, None, None).await;

    // Build a single stream with all protocols chained
    let mut stream = ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
        .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
        .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
        .exchange::<UniswapV4State>(
            "uniswap_v4",
            tvl_filter.clone(),
            Some(uniswap_v4_pool_with_hook_filter),
        )
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:balancer_v2",
            tvl_filter.clone(),
            Some(balancer_v2_pool_filter_after_dci_update),
        )
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:curve",
            tvl_filter.clone(),
            Some(curve_pool_filter),
        )
        .exchange::<EkuboState>("ekubo_v2", tvl_filter.clone(), None)
        .auth_key(Some(tycho_api_key))
        .set_tokens(all_tokens)
        .await
        .skip_state_decode_failures(true)
        .build()
        .await
        .map_err(|e| {
            warn!("Failed to build protocol stream: {:?}", e);
            e
        })
        .expect("Failed to build protocol stream")
        .boxed();

    // Load initial data and organize by protocol
    let mut protocol_data: BenchmarkDataByProtocol = HashMap::new();

    match stream.next().await {
        Some(Ok(message)) => {
            info!(
                "Successfully loaded {} pairs and {} states",
                message.new_pairs.len(),
                message.states.len()
            );
            // Initialize protocol data structures
            let protocols =
                ["uniswap_v2", "uniswap_v3", "uniswap_v4", "balancer_v2", "curve", "ekubo_v2"];
            for protocol in protocols {
                protocol_data.insert(
                    protocol.to_string(),
                    ProtocolBenchmarkData {
                        pools: HashMap::new(),
                        components: HashMap::new(),
                        states: HashMap::new(),
                    },
                );
            }

            // Organize pools and components by protocol
            for (id, component) in message.new_pairs.iter() {
                let protocol_name = map_protocol_system_to_protocol(&component.protocol_system);
                if let Some(protocol_data) = protocol_data.get_mut(&protocol_name) {
                    protocol_data
                        .pools
                        .insert(id.clone(), component.tokens.clone());
                    protocol_data
                        .components
                        .insert(id.clone(), component.clone());
                }
            }

            // Organize states by protocol using the component information
            for (id, state) in message.states.into_iter() {
                // Find which protocol this state belongs to by looking up the component
                for (_protocol_name, data) in protocol_data.iter_mut() {
                    if data.components.contains_key(&id) {
                        data.states.insert(id.clone(), state);
                        break;
                    }
                }
            }
        }
        Some(Err(e)) => {
            warn!("Error loading protocol data: {:?}", e);
        }
        None => {
            warn!("No data received from protocol stream");
        }
    }

    protocol_data
}

fn map_protocol_system_to_protocol(protocol_system: &str) -> String {
    match protocol_system {
        "uniswap_v2" => "uniswap_v2".to_string(),
        "uniswap_v3" => "uniswap_v3".to_string(),
        "uniswap_v4" => "uniswap_v4".to_string(),
        "vm:balancer_v2" => "balancer_v2".to_string(),
        "vm:curve" => "curve".to_string(),
        "ekubo_v2" => "ekubo_v2".to_string(),
        _ => protocol_system.to_string(),
    }
}

fn benchmark_protocol_swaps(c: &mut Criterion, protocol: &str, data: &ProtocolBenchmarkData) {
    let (n_swaps, _) = get_config();

    let mut group = c.benchmark_group(format!("{protocol}_swaps"));
    group.measurement_time(Duration::from_secs(10)).sample_size(n_swaps);

    // Add pool metadata to the group
    let total_pools = data.pools.len();
    let working_states = data.states.len();
    info!("Protocol {protocol}: {total_pools} pools, {working_states} working states");

    // Create benchmark data for swaps
    let mut swap_scenarios = Vec::new();
    let mut rng = rand::rng();
    for (pool_id, tokens) in data.pools.iter().cycle().take(n_swaps) {
        if let Some(state) = data.states.get(pool_id) {
            if tokens.len() >= 2 {
                let (upper, _) = state.get_limits(tokens[0].address.clone(), tokens[1].address.clone()).expect("limits failed");
                let p: u32 = rng.random_range(1..=85);
                let amount_in = upper * BigUint::from(p as u32) / BigUint::from(100u32);
                swap_scenarios.push((
                    pool_id.clone(),
                    amount_in,
                    tokens[0].clone(),
                    tokens[1].clone(),
                    state,
                ));
            }
        }
    }

    if swap_scenarios.is_empty() {
        return; // Skip protocols with no available pools
    }

    // Verify first swap works and show some example swaps
    debug!("Sample swap scenarios for {protocol}:");
    for (i, (pool_id, amount_in, token_in, token_out, state)) in swap_scenarios.iter().take(5).enumerate() {
        match state.get_amount_out(amount_in.clone(), token_in, token_out) {
            Ok(result) => {
                debug!("  [{i}] Pool {}: {} {} -> {} {} (amount: {})", 
                    &pool_id[..8], amount_in, token_in.symbol, 
                    result.amount, token_out.symbol, result.amount);
            },
            Err(e) => {
                warn!("  [{i}] Pool {} FAILED: {} {} -> {} error: {}",
                    &pool_id[..8], amount_in, token_in.symbol, token_out.symbol, e);
                if i == 0 { // Only panic on first swap failure
                    panic!("Benchmark setup failed: {} swap {} -> {} should work",
                        protocol, token_in.symbol, token_out.symbol);
                }
            }
        }
    }

    group.bench_with_input(
        BenchmarkId::new(
            "get_amount_out",
            format!("{}_swaps_from_{}_pools", swap_scenarios.len(), data.pools.len()),
        ),
        &swap_scenarios,
        |b, scenarios| {
            let mut scenario_iter = scenarios.iter().cycle();
            b.iter(|| {
                let (_, amount_in, token_in, token_out, state) = scenario_iter.next().unwrap();
                state.get_amount_out(amount_in.clone(), token_in, token_out).expect("swap failed!");
            });
        },
    );

    group.finish();
}

fn swap_benchmarks(c: &mut Criterion) {
    // Initialize tracing for benchmarks
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let rt = Runtime::new().unwrap();
    let benchmark_data = rt.block_on(load_all_benchmark_data());
    rt.shutdown_background();

    // Benchmark each protocol (let benchmark_protocol_swaps handle empty data)
    for (protocol, data) in &benchmark_data {
        benchmark_protocol_swaps(c, protocol, data);
    }
}

criterion_group!(benches, swap_benchmarks);
criterion_main!(benches);
