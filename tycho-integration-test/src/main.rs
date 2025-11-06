mod execution;
mod metrics;
mod stream_processor;

use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

use alloy::{
    eips::BlockNumberOrTag,
    network::Ethereum,
    primitives::Address,
    providers::{Provider, ProviderBuilder, RootProvider},
    rpc::types::Block,
};
use alloy_chains::NamedChain;
use clap::Parser;
use dotenv::dotenv;
use itertools::Itertools;
use miette::{miette, IntoDiagnostic, NarratableReportHandler, WrapErr};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use rand::prelude::IndexedRandom;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use tycho_common::simulation::protocol_sim::ProtocolSim;
use plotters::prelude::*;
use tycho_ethereum::entrypoint_tracer::{
    allowance_slot_detector::{AllowanceSlotDetectorConfig, EVMAllowanceSlotDetector},
    balance_slot_detector::{BalanceSlotDetectorConfig, EVMBalanceSlotDetector},
};
use tycho_simulation::{
    evm::protocol::uniswap_v3::state::UniswapV3State,
    protocol::models::ProtocolComponent,
    rfq::protocols::hashflow::{client::HashflowClient, state::HashflowState},
    tycho_common::models::Chain,
    utils::load_all_tokens,
};

use crate::{
    execution::{
        encoding::encode_swap,
        models::{TychoExecutionInput, TychoExecutionResult},
        simulate_swap_transaction, tenderly,
    },
    stream_processor::{
        protocol_stream_processor::ProtocolStreamProcessor,
        rfq_stream_processor::RFQStreamProcessor, StreamUpdate, UpdateType,
    },
};

// Global storage for tick/duration measurements with percentage labels
// Store tuples of (num_ticks, duration_ms)
// Also track per-pool durations: component_id -> Vec<duration_ms>
lazy_static::lazy_static! {
    static ref GET_AMOUNT_OUT_DATA: Mutex<HashMap<String, Vec<(f64, f64)>>> = Mutex::new(HashMap::new());
    static ref GET_AMOUNT_OUT_DATA_UNIQUE: Mutex<HashMap<String, Vec<(f64, f64)>>> = Mutex::new(HashMap::new());
    static ref SEEN_COMPONENT_IDS: Mutex<HashMap<String, HashSet<String>>> = Mutex::new(HashMap::new());
    static ref POOL_DURATIONS: Mutex<HashMap<String, Vec<f64>>> = Mutex::new(HashMap::new());
    static ref CURRENT_PERCENTAGE_INDEX: Mutex<usize> = Mutex::new(0);
}

const MAX_GET_AMOUNT_OUT_SAMPLES: usize = 400;
const PERCENTAGES: [f64; 1] = [0.001];

#[derive(Parser, Clone)]
struct Cli {
    /// The tvl threshold in ETH/native token units to filter the graph by
    #[arg(long, default_value_t = 100.0)]
    tvl_threshold: f64,

    #[arg(long, default_value = "ethereum")]
    chain: Chain,

    #[arg(
        long,
        env = "TYCHO_API_KEY",
        hide_env_values = true,
        default_value = "sampletoken",
        hide_default_value = true
    )]
    tycho_api_key: String,

    #[arg(long, env = "TYCHO_URL")]
    tycho_url: String,

    #[arg(long, env = "RPC_URL")]
    rpc_url: String,

    /// Disable on-chain protocols
    #[arg(long, default_value_t = false)]
    disable_onchain: bool,

    /// Disable RFQ protocols
    #[arg(long, default_value_t = false)]
    disable_rfq: bool,

    /// Port for the Prometheus metrics server
    #[arg(long, default_value_t = 9898)]
    metrics_port: u16,

    /// Maximum number of updates to process in parallel.
    /// Set to 1 to process sequentially.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u8).range(1..))]
    parallel_updates: u8,

    /// Maximum number of simulations to run in parallel
    /// Set to 1 to process sequentially.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u8).range(1..))]
    parallel_simulations: u8,

    /// Maximum number of simulations (of updated states) to run per update
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u16).range(1..))]
    max_simulations: u16,

    /// Maximum number of simulations (of stale states) to run per update per protocol
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u16).range(1..))]
    max_simulations_stale: u16,

    /// The RFQ stream will skip messages for this duration (in seconds) after processing a message
    #[arg(long, default_value_t = 600)]
    skip_messages_duration: u64,

    /// Time to wait (in seconds) for block N+1 to exist before executing debug_traceCall
    #[arg(long, default_value_t = 12)]
    block_wait_time: u64,
}

impl Debug for Cli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cli")
            .field("tvl_threshold", &self.tvl_threshold)
            .field("chain", &self.chain)
            .field("tycho_api_key", &"****")
            .field("tycho_url", &self.tycho_url)
            .field("rpc_url", &self.rpc_url)
            .field("metrics_port", &self.metrics_port)
            .finish()
    }
}

#[derive(Default)]
struct TychoState {
    states: HashMap<String, Box<dyn ProtocolSim>>,
    components: HashMap<String, ProtocolComponent>,
    component_ids_by_protocol: HashMap<String, HashSet<String>>,
}

fn create_histogram_with_range(
    data: &[f64],
    title: &str,
    output_path: &str,
    x_label: &str,
    min_val: f64,
    max_val: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() {
        warn!("No data to plot for {}", title);
        return Ok(());
    }

    info!("Creating histogram for {} with {} data points", title, data.len());

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create bins
    let num_bins = 50;
    let bin_width = (max_val - min_val) / num_bins as f64;
    let mut bins = vec![0u32; num_bins];

    for &value in data {
        if value >= min_val && value <= max_val {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(num_bins - 1);
            bins[bin_index] += 1;
        }
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_val..max_val, 0u32..max_count)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc("Count")
        .draw()?;

    chart.draw_series(
        bins.iter().enumerate().map(|(i, &count)| {
            let x0 = min_val + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            Rectangle::new([(x0, 0), (x1, count)], BLUE.mix(0.5).filled())
        }),
    )?;

    root.present()?;
    info!("Histogram saved to {}", output_path);
    Ok(())
}

fn generate_histograms() -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating histograms...");

    // Get all data (num_ticks, duration) tuples - both regular and unique
    let get_amount_out_data = GET_AMOUNT_OUT_DATA
        .lock()
        .map_err(|e| format!("Failed to lock GET_AMOUNT_OUT_DATA: {}", e))?
        .clone();

    let get_amount_out_data_unique = GET_AMOUNT_OUT_DATA_UNIQUE
        .lock()
        .map_err(|e| format!("Failed to lock GET_AMOUNT_OUT_DATA_UNIQUE: {}", e))?
        .clone();

    // Find global min/max ticks for consistent x-axis across all histograms
    let all_tick_values: Vec<f64> = get_amount_out_data
        .values()
        .flatten()
        .map(|(ticks, _)| *ticks)
        .collect();
    let ticks_min = all_tick_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let ticks_max = all_tick_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    info!("Ticks range: {:.0} - {:.0}", ticks_min, ticks_max);

    // Create histograms for each percentage
    let output_dir = "/Users/tamaralipowski/Code/tycho-simulation";
    info!("Saving histograms to: {}", output_dir);

    for percentage in PERCENTAGES {
        let key = format!("{}", percentage);

        // Regular histograms (with duplicates)
        if let Some(data) = get_amount_out_data.get(&key) {
            info!("Creating regular histograms for percentage {}", percentage);

            create_count_histogram(
                data,
                &format!("Number of Pools by Tick Count (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_count_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Count",
                ticks_min,
                ticks_max,
            )?;

            create_duration_histogram(
                data,
                &format!("Average Duration by Tick Count (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_average_duration_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Average Duration (ms)",
                ticks_min,
                ticks_max,
                DurationAggregation::Average,
            )?;

            create_duration_histogram(
                data,
                &format!("Median Duration by Tick Count (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_median_duration_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Median Duration (ms)",
                ticks_min,
                ticks_max,
                DurationAggregation::Median,
            )?;
        }

        // Unique histograms (only unique component IDs)
        if let Some(data_unique) = get_amount_out_data_unique.get(&key) {
            info!("Creating unique histograms for percentage {} ({} unique pools)", percentage, data_unique.len());

            create_count_histogram(
                data_unique,
                &format!("Number of Unique Pools by Tick Count (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_count_unique_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Count",
                ticks_min,
                ticks_max,
            )?;

            create_duration_histogram(
                data_unique,
                &format!("Average Duration by Tick Count - Unique Pools (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_average_duration_unique_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Average Duration (ms)",
                ticks_min,
                ticks_max,
                DurationAggregation::Average,
            )?;

            create_duration_histogram(
                data_unique,
                &format!("Median Duration by Tick Count - Unique Pools (percentage={:.6})", percentage),
                &format!("{}/get_amount_out_histogram_ticks_median_duration_unique_{}.png", output_dir, percentage),
                "Number of Ticks",
                "Median Duration (ms)",
                ticks_min,
                ticks_max,
                DurationAggregation::Median,
            )?;
        }
    }

    info!("Histograms generated successfully");

    // Log pools with average duration > 15ms
    let pool_durations = POOL_DURATIONS
        .lock()
        .map_err(|e| format!("Failed to lock POOL_DURATIONS: {}", e))?
        .clone();

    let mut slow_pools: Vec<(String, f64, usize)> = pool_durations
        .iter()
        .filter_map(|(pool_id, durations)| {
            if durations.is_empty() {
                return None;
            }
            let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
            if avg_duration > 15.0 {
                Some((pool_id.clone(), avg_duration, durations.len()))
            } else {
                None
            }
        })
        .collect();

    // Sort by average duration descending
    slow_pools.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if slow_pools.is_empty() {
        warn!("No pools with average duration > 15ms");
    } else {
        warn!("Found {} pools with average duration > 15ms:", slow_pools.len());
        for (pool_id, avg_duration, sample_count) in slow_pools {
            warn!(
                "Pool ID: {} | Average: {:.2}ms | Samples: {}",
                pool_id, avg_duration, sample_count
            );
        }
    }

    Ok(())
}

enum DurationAggregation {
    Average,
    Median,
}

fn create_count_histogram(
    data: &[(f64, f64)],
    title: &str,
    output_path: &str,
    x_label: &str,
    y_label: &str,
    min_ticks: f64,
    max_ticks: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() {
        warn!("No data to plot for {}", title);
        return Ok(());
    }

    info!("Creating histogram for {} with {} data points", title, data.len());

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create bins
    let num_bins = 50;
    let bin_width = (max_ticks - min_ticks) / num_bins as f64;
    let mut bins: Vec<u32> = vec![0; num_bins];

    // Count data points in each bin
    for &(ticks, _) in data {
        if ticks >= min_ticks && ticks <= max_ticks {
            let bin_index = ((ticks - min_ticks) / bin_width).floor() as usize;
            let bin_index = bin_index.min(num_bins - 1);
            bins[bin_index] += 1;
        }
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_ticks..max_ticks, 0u32..max_count)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
        let x0 = min_ticks + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0), (x1, count)], BLUE.mix(0.5).filled())
    }))?;

    root.present()?;
    info!("Histogram saved to {}", output_path);
    Ok(())
}

fn create_duration_histogram(
    data: &[(f64, f64)],
    title: &str,
    output_path: &str,
    x_label: &str,
    y_label: &str,
    min_ticks: f64,
    max_ticks: f64,
    aggregation: DurationAggregation,
) -> Result<(), Box<dyn std::error::Error>> {
    if data.is_empty() {
        warn!("No data to plot for {}", title);
        return Ok(());
    }

    info!("Creating histogram for {} with {} data points", title, data.len());

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create bins
    let num_bins = 50;
    let bin_width = (max_ticks - min_ticks) / num_bins as f64;
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); num_bins];

    // Assign data points to bins
    for &(ticks, duration) in data {
        if ticks >= min_ticks && ticks <= max_ticks {
            let bin_index = ((ticks - min_ticks) / bin_width).floor() as usize;
            let bin_index = bin_index.min(num_bins - 1);
            bins[bin_index].push(duration);
        }
    }

    // Calculate average or median for each bin
    let bin_values: Vec<f64> = bins
        .iter()
        .map(|bin_durations| {
            if bin_durations.is_empty() {
                0.0
            } else {
                match aggregation {
                    DurationAggregation::Average => {
                        bin_durations.iter().sum::<f64>() / bin_durations.len() as f64
                    }
                    DurationAggregation::Median => {
                        let mut sorted = bin_durations.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                }
            }
        })
        .collect();

    let max_duration = bin_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0); // Ensure at least 1.0 for empty bins

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_ticks..max_ticks, 0.0..max_duration)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    chart.draw_series(bin_values.iter().enumerate().map(|(i, &duration)| {
        let x0 = min_ticks + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0.0), (x1, duration)], BLUE.mix(0.5).filled())
    }))?;

    root.present()?;
    info!("Histogram saved to {}", output_path);
    Ok(())
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    miette::set_hook(Box::new(|_| Box::new(NarratableReportHandler::new())))?;
    dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();

    // Initialize and start Prometheus metrics
    metrics::initialize_metrics();
    let metrics_task = metrics::create_metrics_exporter(cli.metrics_port).await?;

    // Run the main application logic and metrics server in parallel
    // If either fails, the other will be cancelled
    tokio::select! {
        result = run(cli) => {
            result?;
        }
        result = metrics_task => {
            result
                .into_diagnostic()
                .wrap_err("Metrics server task panicked")??;
        }
    }

    Ok(())
}

async fn run(cli: Cli) -> miette::Result<()> {
    info!("Starting integration test");

    let cli = Arc::new(cli);
    let chain = cli.chain;

    let rpc_tools = RPCTools::new(&cli.rpc_url, &chain).await?;

    // Load tokens from Tycho
    info!(%cli.tycho_url, "Loading tokens...");
    let all_tokens =
        load_all_tokens(&cli.tycho_url, false, Some(cli.tycho_api_key.as_str()), chain, None, None)
            .await
            .map_err(|e| miette!("Failed to load tokens: {e:?}"))?;
    info!(%cli.tycho_url, "Loaded tokens");

    // Run streams in background tasks
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    if !cli.disable_onchain {
        if let Ok(protocol_stream_processor) = ProtocolStreamProcessor::new(
            chain,
            cli.tycho_url.clone(),
            cli.tycho_api_key.clone(),
            cli.tvl_threshold,
        ) {
            protocol_stream_processor
                .run_stream(&all_tokens, tx.clone())
                .await?;
        }
    }
    if !cli.disable_rfq {
        if let Ok(rfq_stream_processor) = RFQStreamProcessor::new(
            chain,
            cli.tvl_threshold,
            cli.max_simulations as usize,
            Duration::from_secs(cli.skip_messages_duration),
        ) {
            rfq_stream_processor
                .run_stream(&all_tokens, tx)
                .await?;
        }
    }

    let tycho_state = Arc::new(RwLock::new(TychoState::default()));

    // Process streams updates
    info!("Waiting for first protocol update...");
    let semaphore = Arc::new(Semaphore::new(cli.parallel_updates as usize));
    while let Some(update) = rx.recv().await {
        let update = match update {
            Ok(u) => Arc::new(u),
            Err(e) => {
                warn!("{}", format_error_chain(&e));
                continue;
            }
        };

        let cli = cli.clone();
        let rpc_tools = rpc_tools.clone();
        let tycho_state = tycho_state.clone();
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .into_diagnostic()
            .wrap_err("Failed to acquire permit")?;
        tokio::spawn(async move {
            if let Err(e) = process_update(cli, chain, rpc_tools, tycho_state, &update).await {
                warn!("{}", format_error_chain(&e));
            }
            drop(permit);
        });
    }

    Ok(())
}

async fn process_update(
    cli: Arc<Cli>,
    chain: Chain,
    rpc_tools: RPCTools,
    tycho_state: Arc<RwLock<TychoState>>,
    update: &StreamUpdate,
) -> miette::Result<()> {
    info!(
        "Got protocol update with block/timestamp {}, {} new pairs, and {} states",
        update.update.block_number_or_timestamp,
        update.update.new_pairs.len(),
        update.update.states.len()
    );

    let block = match rpc_tools
        .provider
        .get_block_by_number(BlockNumberOrTag::Latest)
        .await
        .into_diagnostic()
        .wrap_err("Failed to fetch latest block")
        .ok()
        .flatten()
    {
        Some(b) => Arc::new(b),
        None => {
            warn!("Failed to retrieve last block, continuing to next message...");
            return Ok(());
        }
    };

    if let UpdateType::Protocol = update.update_type {
        {
            let mut current_state = tycho_state
                .write()
                .map_err(|e| miette!("Failed to acquire write lock on Tycho state: {e}"))?;
            for (id, comp) in update.update.new_pairs.iter() {
                current_state
                    .components
                    .insert(id.clone(), comp.clone());
                current_state
                    .component_ids_by_protocol
                    .entry(comp.protocol_system.clone())
                    .or_insert_with(HashSet::new)
                    .insert(id.clone());
            }
            for (id, state) in update.update.states.iter() {
                // this overwrites existing entries
                current_state
                    .states
                    .insert(id.clone(), state.clone());
            }
        }
        // Record block processing latency
        let latency_seconds =
            (update.received_at.as_secs_f64() - block.header.timestamp as f64).abs();
        metrics::record_block_processing_duration(latency_seconds);

        let block_delay = block
            .header
            .number
            .abs_diff(update.update.block_number_or_timestamp);
        metrics::record_protocol_update_block_delay(block_delay);
        // Consume messages that are older than the current block, to give the stream a chance
        // to catch up
        if block_delay > 0 {
            warn!(
                "Update block ({}) is behind the current block ({}), skipping to catch up.",
                update.update.block_number_or_timestamp, block.header.number
            );
            metrics::record_protocol_update_skipped();
            return Ok(());
        }

        if update.is_first_update {
            info!("Skipping simulation on first protocol update...");
            return Ok(());
        }
    }

    for (protocol, sync_state) in update.update.sync_states.iter() {
        metrics::record_protocol_sync_state(protocol, sync_state);
    }

    // Process updated states in parallel
    let semaphore = Arc::new(Semaphore::new(cli.parallel_simulations as usize));
    let mut tasks = Vec::new();

    for (id, state) in update
        .update
        .states
        .iter()
        .take(cli.max_simulations as usize)
    {
        let component = match update.update_type {
            UpdateType::Protocol => {
                let states = &tycho_state
                    .read()
                    .map_err(|e| miette!("Failed to acquire read lock on Tycho state: {e}"))?
                    .components;
                match states.get(id) {
                    Some(comp) => comp.clone(),
                    None => {
                        warn!(id=%id, "Component not found in cached protocol pairs. Potential causes: \
                        there was an error decoding the component, the component was evicted from the cache, \
                        or the component was never added to the cache. Skipping...");
                        continue;
                    }
                }
            }
            UpdateType::Rfq => match update.update.new_pairs.get(id) {
                Some(comp) => comp.clone(),
                None => {
                    warn!(id=%id, "Component not found in update's new pairs. Potential cause: \
                    the `states` and `new_pairs` lists don't contain the same items. Skipping...");
                    continue;
                }
            },
        };

        // Filter to only test uniswap_v3
        if component.protocol_system != "uniswap_v3" {
            continue;
        }
        let block = block.clone();
        let state_id = id.clone();
        let state = state.clone_box();
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .into_diagnostic()
            .wrap_err("Failed to acquire permit")?;

        let task = tokio::spawn(async move {
            let simulation_id = generate_simulation_id(&component.protocol_system, &state_id);
            let result =
                process_state(&simulation_id, chain, component, &block, state_id, state).await;
            drop(permit);
            result
        });
        tasks.push(task);
    }

    // Select states that were not updated in this block to test simulation and execution
    let selected_ids = {
        let current_state = tycho_state
            .read()
            .map_err(|e| miette!("Failed to acquire write lock on Tycho state: {e}"))?;

        let mut all_selected_ids = Vec::new();
        for component_ids in current_state
            .component_ids_by_protocol
            .values()
        {
            // Filter out IDs that are in the current update
            let available_ids: Vec<_> = component_ids
                .iter()
                .filter(|id| !update.update.states.keys().contains(id))
                .cloned()
                .collect();

            let protocol_selected_ids: Vec<_> = available_ids
                .choose_multiple(
                    &mut rand::rng(),
                    (cli.max_simulations_stale as usize).min(available_ids.len()),
                )
                .cloned()
                .collect();

            all_selected_ids.extend(protocol_selected_ids);
        }
        all_selected_ids
    };

    for id in selected_ids {
        let (component, state) = {
            let current_state = tycho_state
                .read()
                .map_err(|e| miette!("Failed to acquire read lock on Tycho state: {e}"))?;

            match (current_state.components.get(&id), current_state.states.get(&id)) {
                (Some(comp), Some(state)) => (comp.clone(), state.clone()),
                (None, _) => {
                    error!(id=%id, "Component not found in saved protocol components.");
                    continue;
                }
                (_, None) => {
                    error!(id=%id, "State not found in saved protocol states");
                    continue;
                }
            }
        };

        // Filter to only test uniswap_v3
        if component.protocol_system != "uniswap_v3" {
            continue;
        }

        let block = block.clone();
        let state_id = id.clone();
        let state = state.clone_box();
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .into_diagnostic()
            .wrap_err("Failed to acquire permit")?;

        let task = tokio::spawn(async move {
            let simulation_id = generate_simulation_id(&component.protocol_system, &state_id);
            let result =
                process_state(&simulation_id, chain, component, &block, state_id, state).await;
            drop(permit);
            result
        });
        tasks.push(task);
    }

    let mut block_execution_info = HashMap::new();

    for task in tasks {
        match task.await {
            Ok(execution_data) => {
                block_execution_info.extend(execution_data);
            }
            Err(e) => {
                warn!("Task failed: {:?}", e);
            }
        }
    }

    info!("Collected {} execution data for block {}", block_execution_info.len(), block.number());

    if block_execution_info.is_empty() {
        warn!("No simulations were gathered for block {}", block.number());
        return Ok(())
    }

    let execution_start_time = std::time::Instant::now();
    let results = match simulate_swap_transaction(
        &rpc_tools,
        block_execution_info.clone(),
        &block,
        cli.block_wait_time,
    )
    .await
    {
        Ok(results) => results,
        Err((e, _, _)) => return Err(e),
    };
    let execution_duration_seconds = execution_start_time.elapsed().as_secs_f64();

    warn!(
        event_type = "simulate_execution_duration",
        duration_seconds = execution_duration_seconds,
        num_simulations = block_execution_info.len(),
        "Simulate execution completed in {:.3}ms for {} simulations",
        execution_duration_seconds * 1000.0,
        block_execution_info.len()
    );

    // Check if we have enough samples for the current percentage
    let percentage = {
        let idx = CURRENT_PERCENTAGE_INDEX.lock().unwrap();
        PERCENTAGES[*idx]
    };
    let percentage_key = format!("{}", percentage);

    let get_amount_out_count = GET_AMOUNT_OUT_DATA
        .lock()
        .ok()
        .and_then(|d| d.get(&percentage_key).map(|v| v.len()))
        .unwrap_or(0);

    if get_amount_out_count >= MAX_GET_AMOUNT_OUT_SAMPLES {
        info!("Collected {} get_amount_out samples for percentage {}. Moving to next percentage or generating histograms...",
              MAX_GET_AMOUNT_OUT_SAMPLES, percentage);

        // Move to next percentage
        let mut idx = CURRENT_PERCENTAGE_INDEX.lock().unwrap();
        *idx += 1;

        if *idx >= PERCENTAGES.len() {
            // We've completed all percentages, generate histograms and exit
            info!("Completed all percentages. Generating histograms and exiting...");
            drop(idx); // Release lock before generating histograms
            if let Err(e) = generate_histograms() {
                error!("Failed to generate histograms: {}", e);
            }
            std::process::exit(0);
        } else {
            info!("Starting collection for percentage {}", PERCENTAGES[*idx]);
        }
    }

    let mut n_reverts = 0;
    let mut n_failures = 0;
    let total_simulations = results.len();
    for (simulation_id, result) in &results {
        let execution_info = match block_execution_info.get(simulation_id) {
            Some(info) => info,
            None => {
                error!("Simulation ID {simulation_id} not found in execution_info HashMap");
                continue;
            }
        }
        .clone();
        (n_reverts, n_failures) = process_execution_result(
            simulation_id,
            result,
            execution_info,
            (*block).clone(),
            chain.id().to_string(),
            n_reverts,
            n_failures,
        );
    }
    if n_reverts > 0 || n_failures > 0 {
        warn!(
            "Simulations reverted: {n_reverts}/{}. Simulations failed: {n_failures}/{}",
            total_simulations, total_simulations
        )
    }

    Ok(())
}

#[tracing::instrument(
    skip_all,
    fields(
        simulation_id = %simulation_id,
        protocol = %component.protocol_system,
        component_id = %state_id,
        block_number = %block.header.number,
    )
)]
async fn process_state(
    simulation_id: &str,
    chain: Chain,
    component: ProtocolComponent,
    block: &Block,
    state_id: String,
    state: Box<dyn ProtocolSim>,
) -> HashMap<String, TychoExecutionInput> {
    let tokens_len = component.tokens.len();
    if tokens_len < 2 {
        error!("Component has less than 2 tokens, skipping...");
        return HashMap::new();
    }
    // Get all the possible swap directions
    let swap_directions = match component.protocol_system.as_str() {
        HashflowClient::PROTOCOL_SYSTEM => {
            // Hashflow only supports swaps between the requested base and quote tokens
            // WARN: we read from state because the component.tokens original order
            // is modified here: src/protocol/models.rs: ProtocolComponent::from_with_tokens
            let state = match state
                .as_any()
                .downcast_ref::<HashflowState>()
            {
                Some(s) => s.clone(),
                None => {
                    warn!("Failed to downcast state to HashflowState");
                    return HashMap::new();
                }
            };
            vec![(state.base_token, state.quote_token)]
        }
        _ => component
            .tokens
            .iter()
            .permutations(2)
            .map(|perm| (perm[0].clone(), perm[1].clone()))
            .collect(),
    };
    let mut execution_infos = HashMap::new();
    for (i, (token_in, token_out)) in swap_directions.iter().enumerate() {
        info!(
            "Processing {} pool {state_id}, from {} to {}",
            component.protocol_system, token_in.symbol, token_out.symbol
        );

        // Get max input/output limits
        let (max_input, max_output) = match state
            .get_limits(token_in.address.clone(), token_out.address.clone())
            .into_diagnostic()
            .wrap_err(format!(
                "Error getting limits for token_in: {}, and token_out: {}",
                token_in.symbol, token_out.symbol
            )) {
            Ok(limits) => {
                metrics::record_get_limits_success(&component.protocol_system);
                limits
            }
            Err(e) => {
                error!(
                    event_type = "get_limits_failure",
                    token_in = %token_in.address,
                    token_out = %token_out.address,
                    error = %format_error_chain(&e),
                    state = ?state,
                    "Get limits operation failed: {}", format_error_chain(&e)
                );
                metrics::record_get_limits_failure(&component.protocol_system);
                continue;
            }
        };
        info!(
            "Retrieved limits: max input {max_input} {}; max output {max_output} {}",
            token_in.symbol, token_out.symbol
        );

        // Calculate amount_in based on current percentage
        let percentage = {
            let idx = CURRENT_PERCENTAGE_INDEX.lock().unwrap();
            PERCENTAGES[*idx]
        };
        let percentage_biguint = BigUint::from((percentage * 1000000.0) as u32);
        let ten_thousand = BigUint::from(1000000u32);
        let amount_in = (&max_input * &percentage_biguint) / &ten_thousand;
        if amount_in.is_zero() {
            warn!("Calculated amount_in is zero, skipping...");
            continue;
        }

        // Get expected amount out using tycho-simulation and measure duration
        let start_time = std::time::Instant::now();
        let amount_out_result = match state
            .get_amount_out(amount_in.clone(), token_in, token_out)
            .into_diagnostic()
            .wrap_err(format!(
                "Error calculating amount out at {:.1}% with input of {amount_in} {}.",
                percentage * 100.0,
                token_in.symbol,
            )) {
            Ok(res) => {
                metrics::record_get_amount_out_success(&component.protocol_system);
                res
            }
            Err(e) => {
                error!(
                    event_type = "get_amount_out_failure",
                    token_in = %token_in.address,
                    token_out = %token_out.address,
                    amount_in = %amount_in,
                    error = %format_error_chain(&e),
                    state = ?state,
                    "Get amount out operation failed: {}", format_error_chain(&e)
                );
                metrics::record_get_amount_out_failure(&component.protocol_system);
                continue;
            }
        };
        let duration_seconds = start_time.elapsed().as_secs_f64();
        let expected_amount_out = amount_out_result.amount;

        metrics::record_get_amount_out_duration(&component.protocol_system, duration_seconds);

        // Get the number of ticks from the state
        let num_ticks = if component.protocol_system == "uniswap_v3" {
            if let Some(uniswap_state) = state.as_any().downcast_ref::<UniswapV3State>() {
                uniswap_state.tick_count() as f64
            } else {
                warn!("Failed to downcast to UniswapV3State for component {}", state_id);
                0.0
            }
        } else {
            // For non-UniswapV3 pools, we could add support for other protocols here
            0.0
        };

        // Store tick count and duration for histogram and get sample count
        let percentage_key = format!("{}", percentage);
        let mut current_sample_count = 0;
        let duration_ms = duration_seconds * 1000.0;

        // Store all samples (including duplicates)
        if let Ok(mut data) = GET_AMOUNT_OUT_DATA.lock() {
            let vec = data.entry(percentage_key.clone()).or_insert_with(Vec::new);
            if vec.len() < MAX_GET_AMOUNT_OUT_SAMPLES {
                vec.push((num_ticks, duration_ms));
                current_sample_count = vec.len();
            } else {
                current_sample_count = vec.len();
            }
        }

        // Store unique samples only (check if component_id was seen for this percentage)
        let component_id = state_id.clone();
        let is_new_component = if let Ok(mut seen) = SEEN_COMPONENT_IDS.lock() {
            let seen_set = seen.entry(percentage_key.clone()).or_insert_with(HashSet::new);
            seen_set.insert(component_id.clone())
        } else {
            false
        };

        if is_new_component {
            if let Ok(mut data_unique) = GET_AMOUNT_OUT_DATA_UNIQUE.lock() {
                let vec = data_unique.entry(percentage_key.clone()).or_insert_with(Vec::new);
                vec.push((num_ticks, duration_ms));
            }
        }

        // Track duration per pool for logging slow pools
        if let Ok(mut pool_durations) = POOL_DURATIONS.lock() {
            let durations = pool_durations.entry(component_id).or_insert_with(Vec::new);
            durations.push(duration_ms);
        }

        warn!(
            event_type = "get_amount_out_duration",
            token_in = %token_in.address,
            token_out = %token_out.address,
            amount_in = %amount_in,
            amount_out = %expected_amount_out,
            duration_seconds = duration_seconds,
            num_ticks = num_ticks,
            percentage = percentage,
            sample_count = current_sample_count,
            max_samples = MAX_GET_AMOUNT_OUT_SAMPLES,
            "Get amount out operation completed in {:.3}ms (ticks: {}) [Sample {}/{} for percentage {}]",
            duration_seconds * 1000.0,
            num_ticks,
            current_sample_count,
            MAX_GET_AMOUNT_OUT_SAMPLES,
            percentage
        );

        // Simulate execution amount out against the RPC
        let (solution, transaction) = match encode_swap(
            &component,
            Arc::from(state.clone_box()),
            token_in,
            token_out,
            amount_in.clone(),
            expected_amount_out.clone(),
            chain,
        ) {
            Ok(res) => res,
            Err(e) => {
                warn!("{}", format_error_chain(&e));
                continue;
            }
        };
        execution_infos.insert(
            format!("{}-{:?}", simulation_id, i),
            TychoExecutionInput {
                solution,
                transaction,
                expected_amount_out,
                protocol_system: component.protocol_system.clone(),
                component_id: component.id.to_string(),
            },
        );
    }
    execution_infos
}

/// Processes the result of a Tycho simulation execution and emits metrics.
///
/// Handles success, revert, and failure cases by logging appropriate events and recording
/// metrics. Calculates slippage for successful executions and updates counters for
/// reverts and failures.
///
/// Returns updated counters for reverts and failures.
#[tracing::instrument(
    skip_all,
    fields(
        simulation_id = %simulation_id,
        protocol = %execution_info.protocol_system,
        block_number = %block.header.number,
        component_id = %execution_info.component_id,
    )
)]
fn process_execution_result(
    simulation_id: &String,
    result: &TychoExecutionResult,
    execution_info: TychoExecutionInput,
    block: Block,
    chain_id: String,
    mut n_reverts: i32,
    mut n_failures: i32,
) -> (i32, i32) {
    match result {
        TychoExecutionResult::Success { gas_used, amount_out } => {
            info!(
                event_type = "simulation_execution_success",
                amount_out = amount_out.to_string(),
                gas_used = gas_used,
                "Simulation execution succeeded"
            );

            metrics::record_simulation_execution_success(&execution_info.protocol_system);

            // Calculate slippage: positive = simulated > expected, negative = simulated <
            // expected
            let slippage = if amount_out >= &execution_info.expected_amount_out {
                let diff = amount_out - &execution_info.expected_amount_out;
                ((diff.clone() * BigUint::from(10000u32)) / &execution_info.expected_amount_out)
                    .to_f64()
                    .unwrap_or(0.0) /
                    100.0
            } else {
                let diff = &execution_info.expected_amount_out - amount_out;
                -((diff.clone() * BigUint::from(10000u32)) / &execution_info.expected_amount_out)
                    .to_f64()
                    .unwrap_or(0.0) /
                    100.0
            };

            info!(
                event_type = "execution_slippage",
                slippage_ratio = slippage,
                "Execution slippage: {:.2}%",
                slippage
            );
            metrics::record_execution_slippage(&execution_info.protocol_system, slippage);
        }
        TychoExecutionResult::Revert { reason, state_overwrites, overwrite_metadata } => {
            n_reverts += 1;
            let error_msg = reason.to_string();

            // Extract revert reason from error message
            // Error format is typically "Transaction reverted: <reason>"
            let revert_reason =
                if let Some(reason) = error_msg.strip_prefix("Transaction reverted: ") {
                    reason
                } else {
                    &error_msg
                };

            // Extract error name (first word or function signature)
            let error_name = extract_error_name(revert_reason);

            // Generate Tenderly URL for debugging without state overrides
            let overrides =
                tenderly::TenderlySimParams { network: Some(chain_id), ..Default::default() };
            let tenderly_url = tenderly::build_tenderly_url(
                &overrides,
                Some(&execution_info.transaction),
                Some(&block),
                Address::from_slice(&execution_info.solution.sender[..20]),
            );

            let overwrites_string = if let Some(overwrites) = state_overwrites.as_ref() {
                tenderly::get_overwrites_string(overwrites, overwrite_metadata)
            } else {
                String::new()
            };
            metrics::record_simulation_execution_failure(
                &execution_info.protocol_system,
                &error_name,
            );
        }
        TychoExecutionResult::Failed { error_msg } => {
            n_failures += 1;

            error!(
                event_type = "simulation_execution_failure",
                error_message = %error_msg,
                "Failed to simulate swap: {error_msg}"
            );
            metrics::record_simulation_execution_failure(
                &execution_info.protocol_system,
                error_msg,
            );
        }
    }
    (n_reverts, n_failures)
}

/// Extract the error name from a revert reason string
/// Examples:
/// - "TychoRouter__NegativeSlippage(1000, 990)" -> "TychoRouter__NegativeSlippage"
/// - "arithmetic underflow or overflow" -> "arithmetic underflow or overflow"
/// - "Error(string): insufficient balance" -> "Error"
fn extract_error_name(revert_reason: &str) -> String {
    // Check if it's a function-style error (e.g., "ErrorName(...)")
    if let Some(paren_pos) = revert_reason.find('(') {
        revert_reason[..paren_pos]
            .trim()
            .to_string()
    } else if let Some(colon_pos) = revert_reason.find(':') {
        // Handle "Error: message" format
        revert_reason[..colon_pos]
            .trim()
            .to_string()
    } else {
        // Return the whole message for simple errors
        revert_reason.trim().to_string()
    }
}

#[derive(Clone)]
struct RPCTools {
    rpc_url: String,
    provider: RootProvider<Ethereum>,
    evm_balance_slot_detector: Arc<EVMBalanceSlotDetector>,
    evm_allowance_slot_detector: Arc<EVMAllowanceSlotDetector>,
}

impl RPCTools {
    pub async fn new(rpc_url: &str, chain: &Chain) -> miette::Result<Self> {
        let provider: RootProvider<Ethereum> = ProviderBuilder::default()
            .with_chain(
                NamedChain::try_from(chain.id())
                    .into_diagnostic()
                    .wrap_err("Invalid chain")?,
            )
            .connect(rpc_url)
            .await
            .into_diagnostic()
            .wrap_err("Failed to connect to provider")?;
        let evm_balance_slot_detector = Arc::new(
            EVMBalanceSlotDetector::new(BalanceSlotDetectorConfig {
                rpc_url: rpc_url.to_string(),
                ..Default::default()
            })
            .into_diagnostic()?,
        );
        let evm_allowance_slot_detector = Arc::new(
            EVMAllowanceSlotDetector::new(AllowanceSlotDetectorConfig {
                rpc_url: rpc_url.to_string(),
                ..Default::default()
            })
            .into_diagnostic()?,
        );
        Ok(Self {
            rpc_url: rpc_url.to_string(),
            provider,
            evm_balance_slot_detector,
            evm_allowance_slot_detector,
        })
    }
}

/// Generate a unique simulation ID based on protocol system and state ID
fn generate_simulation_id(protocol_system: &str, state_id: &str) -> String {
    let random_number: u32 = rand::random::<u32>() % 90000 + 10000; // Range 10000-99999
    let component_prefix = state_id
        .chars()
        .take(8)
        .collect::<String>();
    format!("{}_{}_{}", protocol_system, component_prefix, random_number)
}

/// Format the full error chain into a single string, without newlines
fn format_error_chain(e: &miette::Error) -> String {
    let mut chain = vec![];
    for cause in e.chain() {
        chain.push(format!("{cause}"));
    }
    chain.join(" -> ")
}
