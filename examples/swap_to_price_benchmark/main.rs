mod benchmark;
mod reporting;
mod snapshot;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub const TARGET_TOKENS: &[&str] = &[
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", // USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7", // USDT
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", // WETH
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", // WBTC
    "0x6b175474e89094c44da98b954eedeac495271d0f", // DAI
    "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6", // WBTC PoS
    "0xae7ab96520de3a18e5e111b5eaab095312d7fe84", // stETH
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", // cbBTC
    "0x4c9edd5852cd905f086c759e8383e09bff1e68b3", // USDe
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0", // wstETH
    "0xdc035d45d973e3ec169d2276ddab16f1e407384f", // USDS
    "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee", // weETH
    "0x9d39a5de30e57443bff2a8307a4256c8797a3497", // sUSDe
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8", // PYUSD
    "0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d", // USD1
    "0x5ee5bf7ae06d1be5997a1a72006fe6c607ec6de8", // aEthWBTC
    "0xa3931d71877c0e7a3148cb7eb4463524fec27fbd", // sUSDS
    "0x18084fba666a33d37592fa2633fd49a74dd93a88", // tBTC
    "0xad55aebc9b8c03fc43cd9f62260391c13c23e7c0", // cUSDO
    "0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f", // GHO
    "0x23878914efe38d27c4d67ab83ed1b93a74d4086a", // aEthUSDT
    "0x8292bb45bf1ee4d140127049757c2e0ff06317ed", // RLUSD
    "0x4d5f47fa6a74757f35c14fd3a6ef8e3c9bc514e8", // aEthWETH
    "0x4956b52ae2ff65d74ca2d61207523288e4528f96", // RLP
    "0xda5e1988097297dcdc1f90d4dfe7909e847cbef6", // WLFI
    "0x98c23e9d8f34fefb1b7bd6a91b7ff122f4e16f5c", // aEthUSDC
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9", // AAVE
    "0x83f20f44975d03b1b09e64809b757c47f942beea", // sDAI
    "0xe07f9d810a48ab5c3c914ba3ca53af14e4491e8a", // GYD
    "0xba100000625a3754423978a60c9317c58a424e3d", // BAL
    "0x7cfadfd5645b50be87d546f42699d863648251ad", // waUSDCn
    "0xdf7837de1f2fa4631d716cf2502f8b230f1dcc32", // TEL
    "0x4e107a0000db66f0e9fd2039288bf811dd1f9c74", // VAL
];

pub const TARGET_PROTOCOLS: &[&str] = &[
    "uniswap_v2",
    "uniswap_v3",
    "sushiswap_v2",
    "pancakeswap_v2",
    "pancakeswap_v3",
    "uniswap_v4",
    "uniswap_v4_hooks",
    "ekubo_v2",
    // "balancer_v3", // Currently throwing Unknown extractor
    "vm:balancer_v2",
    "vm:curve",
    "vm:maverick_v2",
];

/// Finds the latest snapshot file in a directory
fn find_latest_snapshot(dir: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if !dir.exists() {
        return Err(format!("Snapshot directory does not exist: {}", dir.display()).into());
    }

    let mut snapshots: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "bin")
                .unwrap_or(false)
                && entry.file_name().to_str()
                    .map(|s| s.starts_with("snapshot_"))
                    .unwrap_or(false)
        })
        .collect();

    if snapshots.is_empty() {
        return Err(format!("No snapshot files found in {}", dir.display()).into());
    }

    // Sort by modification time, newest first
    snapshots.sort_by_key(|entry| {
        entry.metadata()
            .and_then(|m| m.modified())
            .ok()
    });
    snapshots.reverse();

    Ok(snapshots[0].path())
}

#[derive(Parser)]
#[command(name = "swap-to-price-benchmark")]
#[command(about = "Benchmark swap-to-price algorithms across protocols", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a FeedMessage snapshot
    Snapshot {
        /// Output folder for snapshot (will create snapshot_<block>.bin)
        #[arg(long, default_value = "examples/swap_to_price_benchmark/data")]
        output_folder: PathBuf,

        /// Tycho API URL
        #[arg(long, env = "TYCHO_URL", default_value = "tycho-beta.propellerheads.xyz")]
        tycho_url: String,

        /// Tycho API key
        #[arg(long, env = "TYCHO_AUTH_KEY")]
        api_key: Option<String>,

        /// Minimum TVL filter (ETH)
        #[arg(long, default_value = "40")]
        min_tvl: f64,

        /// Maximum number of pools per protocol to include in snapshot
        #[arg(long, default_value = "20")]
        pool_count_limit: usize,
    },

    /// Load and decode a FeedMessage snapshot (for testing)
    Run {
        /// Path to snapshot file (defaults to latest in examples/benchmark/data)
        #[arg(long)]
        snapshot: Option<PathBuf>,

        /// Benchmark query_supply instead of swap_to_price
        /// (query_supply tracks trade/execution price instead of spot price)
        #[arg(long, default_value = "false")]
        query_supply: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Snapshot {
            output_folder,
            tycho_url,
            api_key,
            min_tvl,
            pool_count_limit,
        } => {
            println!("Creating snapshot...");

            let protocols = TARGET_PROTOCOLS.to_vec();
            let chain = tycho_common::models::Chain::Ethereum;

            let (saved_snapshot, output_path) = snapshot::save_snapshot(
                &tycho_url,
                api_key.as_deref(),
                chain,
                &protocols,
                min_tvl,
                Some(TARGET_TOKENS),
                &output_folder,
                pool_count_limit,
            )
            .await?;

            println!("\n✅ Snapshot saved successfully!");
            println!("   File: {}", output_path.display());
            println!("   Block: {}", saved_snapshot.metadata.block_number);
            println!("   Chain: {}", saved_snapshot.metadata.chain);
            println!("   Protocols: {}", saved_snapshot.protocols.join(", "));
            println!("   Components: {}", saved_snapshot.metadata.total_components);
            println!("   Tokens: {}", saved_snapshot.tokens.len());
        }

        Commands::Run { snapshot, query_supply } => {
            // Find snapshot file (use provided or find latest)
            let snapshot_path = match snapshot {
                Some(path) => path,
                None => {
                    let default_dir = std::path::PathBuf::from("examples/swap_to_price_benchmark/data");
                    find_latest_snapshot(&default_dir)?
                }
            };

            #[cfg(feature = "swap_to_price")]
            {
                let mode = if query_supply { "query_supply" } else { "swap_to_price" };
                println!("Running benchmark on snapshot (mode: {})...", mode);
                println!("   File: {}", snapshot_path.display());

                // Run the benchmark
                let results = benchmark::run_benchmark(&snapshot_path, query_supply).await?;

                // Create output directory
                let output_dir = std::path::PathBuf::from("examples/swap_to_price_benchmark/runs");
                if !output_dir.exists() {
                    std::fs::create_dir_all(&output_dir)?;
                }

                // Save and print results
                let output_path = reporting::save_results(&results, &output_dir)?;
                reporting::print_summary(&results);

                println!("\nResults saved to: {}", output_path.display());
            }

            #[cfg(not(feature = "swap_to_price"))]
            {
                println!("Loading snapshot...");
                println!("   File: {}", snapshot_path.display());

                // Load and decode snapshot (everything is self-contained!)
                let loaded = snapshot::load_snapshot(&snapshot_path).await?;

                println!("\n✅ Snapshot loaded and decoded successfully!");
                println!("   Block: {}", loaded.metadata.block_number);
                println!("   Chain: {}", loaded.metadata.chain);
                println!("   Captured: {}", loaded.metadata.captured_at.format("%Y-%m-%d %H:%M:%S UTC"));
                println!("   States: {}", loaded.states.len());
                println!("   Components: {}", loaded.components.len());

                // Show some sample pools
                println!("\n📊 Sample pools:");
                for (pool_id, component) in loaded.components.iter().take(5) {
                    let tokens_str = component
                        .tokens
                        .iter()
                        .map(|t| t.symbol.as_str())
                        .collect::<Vec<_>>()
                        .join("/");
                    println!("   {} - {} ({})", component.protocol_system, tokens_str, pool_id);
                }

                if loaded.components.len() > 5 {
                    println!("   ... and {} more", loaded.components.len() - 5);
                }

                println!("\n💡 Tip: Add --features swap_to_price to run the full benchmark");
            }
        }
    }

    Ok(())
}
