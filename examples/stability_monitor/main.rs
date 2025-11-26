extern crate tycho_simulation;

use std::{collections::HashMap, env, str::FromStr, time::Instant};

use chrono::{DateTime, Utc};
use clap::Parser;
use futures::StreamExt;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use serde_json::json;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};
use tycho_client::feed::component_tracker::ComponentFilter;
use tycho_common::models::{token::Token, Chain};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            ekubo::state::EkuboState,
            filters::{
                balancer_v2_pool_filter, curve_pool_filter, uniswap_v4_pool_with_euler_hook_filter,
                uniswap_v4_pool_with_hook_filter,
            },
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    protocol::models::{ProtocolComponent, Update},
    utils::{get_default_url, load_all_tokens},
};

#[derive(Parser)]
struct Cli {
    /// The tvl threshold to filter the graph by
    #[arg(long, default_value_t = 10.0)]
    tvl_threshold: f64,
    /// The target blockchain
    #[clap(long, default_value = "ethereum")]
    pub chain: String,
    /// Interval in seconds for block summary logging
    #[arg(long, default_value_t = 30)]
    summary_interval: u64,
}

#[derive(Debug)]
struct PoolData {
    component: ProtocolComponent,
    state: Box<dyn tycho_common::simulation::protocol_sim::ProtocolSim>,
    last_updated: DateTime<Utc>,
}

impl Clone for PoolData {
    fn clone(&self) -> Self {
        Self {
            component: self.component.clone(),
            state: self.state.clone(),
            last_updated: self.last_updated,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct ProtocolErrors {
    spot_price_errors: usize,
    swap_errors: usize,
    successful_swaps: usize,
    get_limits_errors: usize,
}

#[derive(Debug, Default)]
struct BlockStats {
    pools_updated: usize,
    pools_added: usize,
    pools_removed: usize,
    spot_price_errors: usize,
    swap_errors: usize,
    successful_swaps: usize,
    get_limits_errors: usize,
    protocol_counts: HashMap<String, usize>,
    protocol_errors: HashMap<String, ProtocolErrors>,
    processing_time_ms: u128,
}

struct StabilityMonitor {
    pools: HashMap<String, PoolData>,
    block_stats: BlockStats,
    last_block: Option<u64>,
}

impl StabilityMonitor {
    fn new() -> Self {
        Self { pools: HashMap::new(), block_stats: BlockStats::default(), last_block: None }
    }

    fn get_small_amount(
        &mut self,
        pool_state: &dyn tycho_common::simulation::protocol_sim::ProtocolSim,
        token_in: &Token,
        token_out: &Token,
        protocol_system: &str,
    ) -> Option<BigUint> {
        // Try to get limits from the pool state
        match pool_state.get_limits(token_in.address.clone(), token_out.address.clone()) {
            Ok((max_in, _max_out)) => {
                // Use 0.1% (1/1000) of the max amount as test amount
                let small_amount = max_in / BigUint::from(1000u64);
                if small_amount > BigUint::zero() {
                    Some(small_amount)
                } else {
                    // If result is zero, return minimum of 1
                    Some(BigUint::one())
                }
            }
            Err(_) => {
                // Track getLimits failure
                self.block_stats.get_limits_errors += 1;

                // Track per-protocol error
                self.block_stats
                    .protocol_errors
                    .entry(protocol_system.to_string())
                    .or_default()
                    .get_limits_errors += 1;

                None
            }
        }
    }

    fn test_pool_calculations(&mut self, pool_id: &str, pool_data: &PoolData) {
        let component = &pool_data.component;
        let state = &pool_data.state;

        if component.tokens.len() < 2 {
            return;
        }

        let token0 = &component.tokens[0];
        let token1 = &component.tokens[1];
        let protocol_system = &component.protocol_system;

        // Test spot price calculation
        match state.spot_price(token0, token1) {
            Ok(_price) => {
                // Spot price calculation successful, now test swaps
                self.test_swap_calculations(
                    pool_id,
                    state.as_ref(),
                    token0,
                    token1,
                    protocol_system,
                );
            }
            Err(e) => {
                self.block_stats.spot_price_errors += 1;

                // Track per-protocol error
                self.block_stats
                    .protocol_errors
                    .entry(protocol_system.clone())
                    .or_default()
                    .spot_price_errors += 1;

                warn!(
                    pool_id = pool_id,
                    protocol = component.protocol_system,
                    error = %e,
                    "Spot price calculation failed"
                );
            }
        }
    }

    fn test_swap_calculations(
        &mut self,
        pool_id: &str,
        state: &dyn tycho_common::simulation::protocol_sim::ProtocolSim,
        token0: &Token,
        token1: &Token,
        protocol_system: &str,
    ) {
        // Try to get test amounts, skip if getLimits fails
        let amount0 = self.get_small_amount(state, token0, token1, protocol_system);
        let amount1 = self.get_small_amount(state, token1, token0, protocol_system);

        // Test token0 -> token1 (only if we got a valid amount)
        if let Some(amount0) = amount0 {
            match state.get_amount_out(amount0, token0, token1) {
                Ok(_result) => {
                    self.block_stats.successful_swaps += 1;

                    // Track per-protocol success
                    self.block_stats
                        .protocol_errors
                        .entry(protocol_system.to_string())
                        .or_default()
                        .successful_swaps += 1;
                }
                Err(e) => {
                    self.block_stats.swap_errors += 1;

                    // Track per-protocol error
                    self.block_stats
                        .protocol_errors
                        .entry(protocol_system.to_string())
                        .or_default()
                        .swap_errors += 1;

                    warn!(
                        pool_id = pool_id,
                        protocol = protocol_system,
                        direction = "token0_to_token1",
                        error = %e,
                        "Swap calculation failed"
                    );
                }
            }
        }

        // Test token1 -> token0 (only if we got a valid amount)
        if let Some(amount1) = amount1 {
            match state.get_amount_out(amount1, token1, token0) {
                Ok(_result) => {
                    self.block_stats.successful_swaps += 1;

                    // Track per-protocol success
                    self.block_stats
                        .protocol_errors
                        .entry(protocol_system.to_string())
                        .or_default()
                        .successful_swaps += 1;
                }
                Err(e) => {
                    self.block_stats.swap_errors += 1;

                    // Track per-protocol error
                    self.block_stats
                        .protocol_errors
                        .entry(protocol_system.to_string())
                        .or_default()
                        .swap_errors += 1;

                    warn!(
                        pool_id = pool_id,
                        protocol = protocol_system,
                        direction = "token1_to_token0",
                        error = %e,
                        "Swap calculation failed"
                    );
                }
            }
        }
    }

    fn process_update(&mut self, update: Update) {
        let start_time = Instant::now();
        self.block_stats = BlockStats::default();

        // Track block progression
        if let Some(last_block) = self.last_block {
            if update.block_number_or_timestamp > last_block + 1 {
                warn!(
                    last_block = last_block,
                    current_block = update.block_number_or_timestamp,
                    "Block gap detected - possible stream interruption"
                );
            }
        }
        self.last_block = Some(update.block_number_or_timestamp);

        // Process new pools
        for (pool_id, component) in &update.new_pairs {
            if let Some(state) = update.states.get(pool_id) {
                let pool_data = PoolData {
                    component: component.clone(),
                    state: state.clone(),
                    last_updated: Utc::now(),
                };

                self.test_pool_calculations(pool_id, &pool_data);
                self.pools
                    .insert(pool_id.clone(), pool_data);
                self.block_stats.pools_added += 1;

                // Count by protocol
                *self
                    .block_stats
                    .protocol_counts
                    .entry(component.protocol_system.clone())
                    .or_insert(0) += 1;
            }
        }

        // Process updated pools
        for (pool_id, state) in &update.states {
            if let Some(pool_data) = self.pools.get_mut(pool_id) {
                pool_data.state = state.clone();
                pool_data.last_updated = Utc::now();
                self.block_stats.pools_updated += 1;

                // Count by protocol
                let protocol_system = pool_data
                    .component
                    .protocol_system
                    .clone();
                *self
                    .block_stats
                    .protocol_counts
                    .entry(protocol_system)
                    .or_insert(0) += 1;
            }
        }

        // Collect pool data for testing to avoid borrow issues
        let pools_to_test: Vec<(String, PoolData)> = update
            .states
            .keys()
            .filter_map(|pool_id| {
                self.pools
                    .get(pool_id)
                    .map(|pool_data| (pool_id.clone(), pool_data.clone()))
            })
            .collect();

        // Test calculations for updated pools
        for (pool_id, pool_data) in pools_to_test {
            self.test_pool_calculations(&pool_id, &pool_data);
        }

        // Process removed pools
        for pool_id in update.removed_pairs.keys() {
            if self.pools.remove(pool_id).is_some() {
                self.block_stats.pools_removed += 1;
            }
        }

        self.block_stats.processing_time_ms = start_time.elapsed().as_millis();

        // Log block summary if there was activity
        if self.block_stats.pools_updated > 0 ||
            self.block_stats.pools_added > 0 ||
            self.block_stats.pools_removed > 0
        {
            // Calculate aggregated protocol statistics for cleaner JSON output
            let mut protocol_summary = json!({});

            for (protocol, stats) in &self.block_stats.protocol_errors {
                let total_operations = stats.successful_swaps + stats.swap_errors;
                let success_rate = if total_operations > 0 {
                    stats.successful_swaps as f64 / total_operations as f64
                } else {
                    0.0
                };

                protocol_summary[protocol] = json!({
                    "pools": self.block_stats.protocol_counts.get(protocol).unwrap_or(&0),
                    "spot_price_errors": stats.spot_price_errors,
                    "swap_errors": stats.swap_errors,
                    "successful_swaps": stats.successful_swaps,
                    "get_limits_errors": stats.get_limits_errors,
                    "success_rate": (success_rate * 100.0).round() / 100.0
                });
            }

            // Also include protocols that had pool updates but no errors
            for (protocol, count) in &self.block_stats.protocol_counts {
                if !self
                    .block_stats
                    .protocol_errors
                    .contains_key(protocol)
                {
                    protocol_summary[protocol] = json!({
                        "pools": count,
                        "spot_price_errors": 0,
                        "swap_errors": 0,
                        "successful_swaps": 0,
                        "get_limits_errors": 0,
                        "success_rate": 0.0
                    });
                }
            }

            info!(
                target: "block_summary",
                "{}",
                json!({
                    "block": update.block_number_or_timestamp,
                    "timestamp": Utc::now().to_rfc3339(),
                    "pools_total": self.pools.len(),
                    "pools_updated": self.block_stats.pools_updated,
                    "pools_added": self.block_stats.pools_added,
                    "pools_removed": self.block_stats.pools_removed,
                    "total_spot_price_errors": self.block_stats.spot_price_errors,
                    "total_swap_errors": self.block_stats.swap_errors,
                    "total_successful_swaps": self.block_stats.successful_swaps,
                    "total_get_limits_errors": self.block_stats.get_limits_errors,
                    "protocol_details": protocol_summary,
                    "processing_time_ms": self.block_stats.processing_time_ms
                })
            );
        }
    }

    fn log_periodic_summary(&self) {
        let active_protocols: HashMap<String, usize> = self
            .pools
            .values()
            .map(|pool| &pool.component.protocol_system)
            .fold(HashMap::new(), |mut acc, protocol| {
                *acc.entry(protocol.clone()).or_insert(0) += 1;
                acc
            });

        info!(
            target: "periodic_summary",
            "{}",
            json!({
                "timestamp": Utc::now().to_rfc3339(),
                "total_pools": self.pools.len(),
                "active_protocols": active_protocols,
                "last_block": self.last_block
            })
        );
    }
}

fn register_exchanges(
    mut builder: ProtocolStreamBuilder,
    chain: &Chain,
    tvl_filter: ComponentFilter,
) -> ProtocolStreamBuilder {
    match chain {
        Chain::Ethereum => {
            builder = builder
                .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
                .exchange::<EVMPoolState<PreCachedDB>>(
                    "vm:balancer_v2",
                    tvl_filter.clone(),
                    Some(balancer_v2_pool_filter),
                )
                .exchange::<EVMPoolState<PreCachedDB>>(
                    "vm:curve",
                    tvl_filter.clone(),
                    Some(curve_pool_filter),
                )
                .exchange::<EkuboState>("ekubo_v2", tvl_filter.clone(), None)
                .exchange::<UniswapV4State>(
                    "uniswap_v4",
                    tvl_filter.clone(),
                    Some(uniswap_v4_pool_with_hook_filter),
                )
                .exchange::<UniswapV4State>(
                    "uniswap_v4_hooks",
                    tvl_filter.clone(),
                    Some(uniswap_v4_pool_with_euler_hook_filter),
                );
        }
        Chain::Base => {
            builder = builder.exchange::<UniswapV4State>(
                "uniswap_v4",
                tvl_filter.clone(),
                Some(uniswap_v4_pool_with_hook_filter),
            );
        }
        Chain::Unichain => {
            builder = builder.exchange::<UniswapV4State>(
                "uniswap_v4",
                tvl_filter.clone(),
                Some(uniswap_v4_pool_with_hook_filter),
            );
        }
        _ => {}
    }
    builder
}

fn setup_logging() {
    use tracing_subscriber::{fmt, EnvFilter};

    // JSON formatter for structured logging to stdout
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_logging();
    let cli = Cli::parse();

    let chain =
        Chain::from_str(&cli.chain).unwrap_or_else(|_| panic!("Unknown chain {}", cli.chain));

    let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| {
        get_default_url(&chain).unwrap_or_else(|| panic!("Unknown URL for chain {}", cli.chain))
    });

    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    // Early check for RPC_URL
    env::var("RPC_URL").expect("RPC_URL env variable should be set");

    info!(
        chain = %chain,
        tvl_threshold = cli.tvl_threshold,
        "Starting Tycho stability monitor"
    );

    let mut monitor = StabilityMonitor::new();

    // Set up periodic summary logging
    let mut summary_timer = interval(Duration::from_secs(cli.summary_interval));

    // Create the protocol stream
    let all_tokens =
        load_all_tokens(tycho_url.as_str(), false, Some(tycho_api_key.as_str()), chain, None, None)
            .await;

    let tvl_filter = ComponentFilter::with_tvl_range(cli.tvl_threshold, cli.tvl_threshold);
    let mut protocol_stream =
        register_exchanges(ProtocolStreamBuilder::new(&tycho_url, chain), &chain, tvl_filter)
            .auth_key(Some(tycho_api_key))
            .skip_state_decode_failures(true)
            .timeout(200)
            .set_tokens(all_tokens)
            .await
            .build()
            .await
            .expect("Failed building protocol stream");

    info!("Protocol stream initialized, starting monitoring...");

    // Main monitoring loop
    loop {
        tokio::select! {
            msg = protocol_stream.next() => {
                match msg {
                    Some(Ok(update)) => {
                        monitor.process_update(update);
                    }
                    Some(Err(e)) => {
                        error!(error = %e, "Protocol stream error");
                    }
                    None => {
                        error!("Protocol stream ended unexpectedly");
                        break;
                    }
                }
            }
            _ = summary_timer.tick() => {
                monitor.log_periodic_summary();
            }
        }
    }

    error!("Monitor shutting down");
    Ok(())
}
