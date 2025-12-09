//! Pool Depth Analysis CLI
//!
//! Calculates historical liquidity depth for pools at various price impact levels.
//!
//! # Usage
//!
//! ```bash
//! TYCHO_URL="tycho-dev.propellerheads.xyz" \
//! TYCHO_API_KEY="sampletoken" \
//! RPC_URL="https://ethereum-mainnet..." \
//! cargo run --release --example depth_analysis -- \
//!     --pool-ids "0xc7bbec68d12a0d1830360f8ec58fa599ba1b0e9b" \
//!     --start-date 2025-11-01 \
//!     --end-date 2025-11-10 \
//!     --protocol uniswap-v3 \
//!     --output depth_report.csv
//! ```
//!
//! # Environment Variables
//!
//! - `TYCHO_URL` - Tycho RPC endpoint
//! - `TYCHO_API_KEY` - API authentication key
//! - `RPC_URL` - Ethereum RPC for timestamp->block conversion

#[path = "../common/mod.rs"]
mod common;

use std::collections::HashMap;

use alloy::providers::Provider;
use anyhow::{anyhow, Context, Result};
use chrono::{NaiveDate, TimeZone, Utc};
use clap::Parser;
use csv::Writer;
use num_bigint::BigUint;
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim};

use common::{
    convert_component, create_eth_provider, decode_states, fetch_components, fetch_snapshots,
    timestamp_to_block, ChainArg, ProtocolType, TychoConfig,
};

/// Price impact levels in basis points
const IMPACT_BPS_LEVELS: [u32; 4] = [0, 10, 50, 100];

#[derive(Parser, Debug)]
#[command(author, version, about = "Analyze historical liquidity depth for AMM pools")]
struct Args {
    /// Comma-separated list of pool IDs
    #[arg(long, value_delimiter = ',')]
    pool_ids: Vec<String>,

    /// Start date (YYYY-MM-DD format)
    #[arg(long)]
    start_date: String,

    /// End date (YYYY-MM-DD format)
    #[arg(long)]
    end_date: String,

    /// Protocol type
    #[arg(long, value_enum, default_value = "uniswap-v3")]
    protocol: ProtocolType,

    /// Output CSV file path
    #[arg(long, default_value = "depth_analysis.csv")]
    output: String,

    /// Chain
    #[arg(long, value_enum, default_value = "ethereum")]
    chain: ChainArg,
}

/// Single depth analysis record
#[derive(Debug)]
struct DepthRecord {
    date: NaiveDate,
    block: u64,
    pool_id: String,
    token0_symbol: String,
    token1_symbol: String,
    spot_price: f64,
    // Sell token0 (buy token1) depths
    depth_sell_t0_0bps: f64,
    depth_sell_t0_10bps: f64,
    depth_sell_t0_50bps: f64,
    depth_sell_t0_100bps: f64,
    // Sell token1 (buy token0) depths
    depth_sell_t1_0bps: f64,
    depth_sell_t1_10bps: f64,
    depth_sell_t1_50bps: f64,
    depth_sell_t1_100bps: f64,
}

/// Parse a date string (YYYY-MM-DD) to NaiveDate
fn parse_date(date_str: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
        .context(format!("Invalid date format '{}'. Expected YYYY-MM-DD", date_str))
}

/// Generate all dates in range (inclusive)
fn generate_date_range(start: NaiveDate, end: NaiveDate) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    let mut current = start;
    while current <= end {
        dates.push(current);
        current = current.succ_opt().unwrap_or(current);
    }
    dates
}

/// Convert date to midday (12:00 UTC) Unix timestamp
fn date_to_midday_timestamp(date: NaiveDate) -> u64 {
    Utc.from_utc_datetime(&date.and_hms_opt(12, 0, 0).unwrap())
        .timestamp() as u64
}

/// Convert BigUint to f64, adjusting for token decimals
fn biguint_to_f64(value: &BigUint, decimals: u32) -> f64 {
    let divisor = 10u128.pow(decimals);
    let value_u128 = value.to_string().parse::<u128>().unwrap_or(0);
    value_u128 as f64 / divisor as f64
}

/// Calculate depth at a target price impact in basis points
///
/// Returns the amount of `token_in` that can be sold before the price
/// drops by `target_bps` basis points.
fn calculate_depth_at_bps(
    state: &dyn ProtocolSim,
    token_in: &Token,
    token_out: &Token,
    target_bps: u32,
) -> Result<f64> {
    // For 0 bps, there's no depth (you can't trade without moving the price)
    if target_bps == 0 {
        return Ok(0.0);
    }

    let initial_price = state
        .spot_price(token_in, token_out)
        .map_err(|e| anyhow!("Failed to get initial spot price: {}", e))?;

    // Get max tradeable amount
    let (max_in, _) = state
        .get_limits(token_in.address.clone(), token_out.address.clone())
        .map_err(|e| anyhow!("Failed to get limits: {}", e))?;

    // Binary search for the amount that causes target price impact
    let mut low = BigUint::from(0u64);
    let mut high = max_in.clone();

    // Use a reasonable precision (100 iterations should be enough)
    for _ in 0..100 {
        if low >= high {
            break;
        }

        let mid = (&low + &high) / 2u64;
        if mid == low {
            break;
        }

        // Try to get amount out and new state
        match state.get_amount_out(mid.clone(), token_in, token_out) {
            Ok(result) => {
                // Get new price from the resulting state
                match result.new_state.spot_price(token_in, token_out) {
                    Ok(new_price) => {
                        let price_impact = (initial_price - new_price) / initial_price;
                        let impact_bps = price_impact * 10000.0;

                        if impact_bps < target_bps as f64 {
                            low = mid;
                        } else {
                            high = mid;
                        }
                    }
                    Err(_) => {
                        // Can't get price for this state, reduce search space
                        high = mid;
                    }
                }
            }
            Err(_) => {
                // Amount too large, reduce search space
                high = mid;
            }
        }
    }

    // Verify and log the final impact
    let final_amount = biguint_to_f64(&high, token_in.decimals);
    if let Ok(result) = state.get_amount_out(high.clone(), token_in, token_out) {
        if let Ok(new_price) = result.new_state.spot_price(token_in, token_out) {
            let actual_impact = (initial_price - new_price) / initial_price;
            let actual_bps = actual_impact * 10000.0;
            // println!(
            //     "    [DEBUG] Target: {} bps, Actual: {:.4} bps, Amount: {:.6} {}, New price: {:.6}",
            //     target_bps, actual_bps, final_amount, token_in.symbol, new_price
            // );
        }
    }

    Ok(final_amount)
}

/// Calculate depths for all target bps levels
fn calculate_all_depths(
    state: &dyn ProtocolSim,
    token_in: &Token,
    token_out: &Token,
) -> Result<[f64; 4]> {
    let mut depths = [0.0; 4];
    for (i, &bps) in IMPACT_BPS_LEVELS.iter().enumerate() {
        depths[i] = calculate_depth_at_bps(state, token_in, token_out, bps)?;
    }
    Ok(depths)
}

/// Print results to console
fn print_results(results: &[DepthRecord]) {
    println!("\n=== Depth Analysis Report ===\n");

    for record in results {
        println!("Date: {} | Block: {}", record.date, record.block);
        println!("Pool: {} ({}/{})", record.pool_id, record.token0_symbol, record.token1_symbol);
        println!(
            "Spot Price: {:.6} {}/{}",
            record.spot_price, record.token1_symbol, record.token0_symbol
        );
        println!();

        println!("  Sell {} (buy {}):", record.token0_symbol, record.token1_symbol);
        println!("    0 bps:   {:.4} {}", record.depth_sell_t0_0bps, record.token0_symbol);
        println!("    10 bps:  {:.4} {}", record.depth_sell_t0_10bps, record.token0_symbol);
        println!("    50 bps:  {:.4} {}", record.depth_sell_t0_50bps, record.token0_symbol);
        println!("    100 bps: {:.4} {}", record.depth_sell_t0_100bps, record.token0_symbol);
        println!();

        println!("  Sell {} (buy {}):", record.token1_symbol, record.token0_symbol);
        println!("    0 bps:   {:.4} {}", record.depth_sell_t1_0bps, record.token1_symbol);
        println!("    10 bps:  {:.4} {}", record.depth_sell_t1_10bps, record.token1_symbol);
        println!("    50 bps:  {:.4} {}", record.depth_sell_t1_50bps, record.token1_symbol);
        println!("    100 bps: {:.4} {}", record.depth_sell_t1_100bps, record.token1_symbol);
        println!();
        println!("{}", "-".repeat(60));
    }
}

/// Export results to CSV
fn export_csv(results: &[DepthRecord], output_path: &str) -> Result<()> {
    let mut wtr = Writer::from_path(output_path)
        .context(format!("Failed to create CSV file: {}", output_path))?;

    // Write header
    wtr.write_record([
        "date",
        "block",
        "pool_id",
        "token0",
        "token1",
        "spot_price",
        "depth_sell_t0_0bps",
        "depth_sell_t0_10bps",
        "depth_sell_t0_50bps",
        "depth_sell_t0_100bps",
        "depth_sell_t1_0bps",
        "depth_sell_t1_10bps",
        "depth_sell_t1_50bps",
        "depth_sell_t1_100bps",
    ])?;

    // Write records
    for record in results {
        wtr.write_record([
            record.date.to_string(),
            record.block.to_string(),
            record.pool_id.clone(),
            record.token0_symbol.clone(),
            record.token1_symbol.clone(),
            format!("{:.8}", record.spot_price),
            format!("{:.8}", record.depth_sell_t0_0bps),
            format!("{:.8}", record.depth_sell_t0_10bps),
            format!("{:.8}", record.depth_sell_t0_50bps),
            format!("{:.8}", record.depth_sell_t0_100bps),
            format!("{:.8}", record.depth_sell_t1_0bps),
            format!("{:.8}", record.depth_sell_t1_10bps),
            format!("{:.8}", record.depth_sell_t1_50bps),
            format!("{:.8}", record.depth_sell_t1_100bps),
        ])?;
    }

    wtr.flush()?;
    println!("\nExported to: {}", output_path);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Parse dates
    let start_date = parse_date(&args.start_date)?;
    let end_date = parse_date(&args.end_date)?;

    if start_date > end_date {
        return Err(anyhow!(
            "Start date ({}) must be before or equal to end date ({})",
            start_date,
            end_date
        ));
    }

    let dates = generate_date_range(start_date, end_date);
    println!("Analyzing {} days from {} to {}", dates.len(), start_date, end_date);

    if args.pool_ids.is_empty() {
        return Err(anyhow!("At least one --pool-ids must be provided"));
    }

    // Initialize Tycho config
    let config = TychoConfig::from_env(args.chain)?;
    let rpc_client = config.create_rpc_client()?;
    let eth_provider = create_eth_provider()?;

    println!("Connecting to Tycho at: {}", config.url);
    println!("Protocol: {:?}", args.protocol);
    println!("Pool IDs: {:?}", args.pool_ids);

    // Load tokens
    println!("Loading tokens...");
    let tokens = config.load_tokens().await?;
    println!("Loaded {} tokens", tokens.len());

    // Fetch components (static metadata)
    let protocol_system = args.protocol.as_system_name();
    println!("Fetching components...");
    let components =
        fetch_components(&rpc_client, protocol_system, &args.pool_ids, config.dto_chain).await?;
    println!("Fetched {} components", components.len());

    if components.is_empty() {
        return Err(anyhow!("No components found for the provided pool IDs"));
    }

    // Convert components for simulation
    let sim_components: HashMap<String, tycho_simulation::protocol::models::ProtocolComponent> =
        components
            .iter()
            .map(|(id, c)| (id.clone(), convert_component(c, &tokens)))
            .collect();

    // Process each date
    let mut results: Vec<DepthRecord> = Vec::new();

    for date in &dates {
        let timestamp = date_to_midday_timestamp(*date);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check if timestamp is in the future
        let block = if timestamp > now {
            // Future date: use latest state
            println!(
                "\nProcessing {} (timestamp: {} - FUTURE, using latest state)...",
                date, timestamp
            );
            eth_provider.get_block_number().await?
        } else {
            // Historical date: convert to block number
            println!("\nProcessing {} (timestamp: {})...", date, timestamp);
            timestamp_to_block(&eth_provider, timestamp).await?
        };

        // Fetch snapshots at this block
        let snapshots = fetch_snapshots(
            &rpc_client,
            protocol_system,
            &components,
            block,
            config.dto_chain,
        )
        .await?;

        if snapshots.get_states().is_empty() {
            eprintln!("  No snapshots found for block {}", block);
            continue;
        }

        // Decode states
        let states = decode_states(&args.protocol, snapshots.get_states(), &tokens).await?;

        // Calculate depths for each pool
        for (pool_id, state) in &states {
            if let Some(component) = sim_components.get(pool_id) {
                let pool_tokens: Vec<&Token> = component.tokens.iter().collect();

                if pool_tokens.len() < 2 {
                    eprintln!("  Pool {} has fewer than 2 tokens", pool_id);
                    continue;
                }

                let token0 = pool_tokens[0];
                let token1 = pool_tokens[1];

                // Get spot price
                let spot_price = match state.spot_price(token0, token1) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("  Failed to get spot price for {}: {}", pool_id, e);
                        continue;
                    }
                };

                // Calculate depths in both directions
                let depths_sell_t0 = match calculate_all_depths(state.as_ref(), token0, token1) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!(
                            "  Failed to calculate sell {} depths for {}: {}",
                            token0.symbol, pool_id, e
                        );
                        [0.0; 4]
                    }
                };

                let depths_sell_t1 = match calculate_all_depths(state.as_ref(), token1, token0) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!(
                            "  Failed to calculate sell {} depths for {}: {}",
                            token1.symbol, pool_id, e
                        );
                        [0.0; 4]
                    }
                };

                results.push(DepthRecord {
                    date: *date,
                    block,
                    pool_id: pool_id.clone(),
                    token0_symbol: token0.symbol.clone(),
                    token1_symbol: token1.symbol.clone(),
                    spot_price,
                    depth_sell_t0_0bps: depths_sell_t0[0],
                    depth_sell_t0_10bps: depths_sell_t0[1],
                    depth_sell_t0_50bps: depths_sell_t0[2],
                    depth_sell_t0_100bps: depths_sell_t0[3],
                    depth_sell_t1_0bps: depths_sell_t1[0],
                    depth_sell_t1_10bps: depths_sell_t1[1],
                    depth_sell_t1_50bps: depths_sell_t1[2],
                    depth_sell_t1_100bps: depths_sell_t1[3],
                });

                println!(
                    "  {} ({}/{}) - price: {:.4}",
                    pool_id, token0.symbol, token1.symbol, spot_price
                );
            }
        }
    }

    // Print and export results
    print_results(&results);
    export_csv(&results, &args.output)?;

    println!("\nDepth analysis complete!");
    Ok(())
}
