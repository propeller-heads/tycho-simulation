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
            filters::curve_pool_filter,
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

// Curve benchmark data organized by factory
#[derive(Clone)]
struct CurveFactoryData {
    pools: HashMap<String, Vec<Token>>,
    components: HashMap<String, ProtocolComponent>,
    states: HashMap<String, Box<dyn ProtocolSim>>,
}

type CurveDataByFactory = HashMap<String, CurveFactoryData>;

async fn load_curve_benchmark_data() -> CurveDataByFactory {
    let (_, tvl_threshold) = get_config();
    
    let tycho_url =
        env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
    let tycho_api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    let tvl_filter = ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold);

    let all_tokens =
        load_all_tokens(&tycho_url, false, Some(&tycho_api_key), Chain::Ethereum, None, None).await;

    // Build stream with only Curve protocol
    let mut stream = ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:curve",
            tvl_filter.clone(),
            Some(curve_pool_filter),
        )
        .auth_key(Some(tycho_api_key))
        .set_tokens(all_tokens)
        .await
        .skip_state_decode_failures(true)
        .build()
        .await
        .map_err(|e| {
            warn!("Failed to build Curve protocol stream: {:?}", e);
            e
        })
        .expect("Failed to build Curve protocol stream")
        .boxed();

    // Load initial data and organize by factory
    let mut factory_data: CurveDataByFactory = HashMap::new();
    
    match stream.next().await {
        Some(Ok(message)) => {
            info!(
                "Successfully loaded {} Curve pairs and {} states",
                message.new_pairs.len(),
                message.states.len()
            );

            // Organize pools and components by factory
            for (id, component) in message.new_pairs.iter() {
                // Extract factory from static_attributes
                let factory_name = get_factory_name(component);
                
                let factory_data = factory_data
                    .entry(factory_name)
                    .or_insert_with(|| CurveFactoryData {
                        pools: HashMap::new(),
                        components: HashMap::new(),
                        states: HashMap::new(),
                    });
                    
                factory_data.pools.insert(id.clone(), component.tokens.clone());
                factory_data.components.insert(id.clone(), component.clone());
            }

            // Organize states by factory using the component information
            for (id, state) in message.states.into_iter() {
                // Find which factory this state belongs to by looking up the component
                for (_factory_name, data) in factory_data.iter_mut() {
                    if data.components.contains_key(&id) {
                        data.states.insert(id.clone(), state);
                        break;
                    }
                }
            }
        }
        Some(Err(e)) => {
            warn!("Error loading Curve protocol data: {:?}", e);
        }
        None => {
            warn!("No data received from Curve protocol stream");
        }
    }

    factory_data
}

fn get_factory_name(component: &ProtocolComponent) -> String {
    // Try to get factory from static_attributes
    if let Some(factory_bytes) = component.static_attributes.get("factory") {
        // Convert Bytes to string, taking first 8 chars for readability
        let factory_hex = String::from_utf8(factory_bytes.to_vec()).expect("failed parsing factory address");
        format!("factory_{}", &factory_hex)
    } else {
        "unknown_factory".to_string()
    }
}

fn benchmark_curve_factory_swaps(c: &mut Criterion, factory: &str, data: &CurveFactoryData) {
    let (n_swaps, _) = get_config();

    let mut group = c.benchmark_group(format!("curve_{}_swaps", factory));
    group.measurement_time(Duration::from_secs(10)).sample_size(10);

    // Add pool metadata to the group
    let total_pools = data.pools.len();
    let working_states = data.states.len();
    info!("Curve factory {factory}: {total_pools} pools, {working_states} working states");

    // Create benchmark data for swaps
    let mut swap_scenarios = Vec::new();
    let mut rng = rand::rng();
    for (pool_id, tokens) in data.pools.iter().cycle().take(n_swaps) {
        if let Some(state) = data.states.get(pool_id) {
            if tokens.len() >= 2 {
                let (upper, _) = state.get_limits(tokens[0].address.clone(), tokens[1].address.clone()).expect("limits failed");
                // take between 1% and 85% of the upper limit for swaps
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
        return; // Skip factories with no available pools
    }

    // Verify first swap works and show some example swaps
    debug!("Sample swap scenarios for Curve factory {factory}:");
    for (i, (pool_id, amount_in, token_in, token_out, state)) in swap_scenarios.iter().take(3).enumerate() {
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
                    panic!("Benchmark setup failed: Curve factory {} swap {} -> {} should work",
                        factory, token_in.symbol, token_out.symbol);
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

fn curve_factory_benchmarks(c: &mut Criterion) {
    // Initialize tracing for benchmarks
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let rt = Runtime::new().unwrap();
    let factory_data = rt.block_on(load_curve_benchmark_data());
    rt.shutdown_background();

    info!("Found {} Curve factories", factory_data.len());
    for (factory, data) in &factory_data {
        debug!("Factory {}: {} pools, {} states", factory, data.pools.len(), data.states.len());
    }

    // Benchmark each factory
    for (factory, data) in &factory_data {
        benchmark_curve_factory_swaps(c, factory, data);
    }
}

criterion_group!(benches, curve_factory_benchmarks);
criterion_main!(benches);