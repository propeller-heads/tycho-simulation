use std::{env, fs::File, io::Write};
use std::collections::HashMap;

use clap::{Parser, Subcommand};
use futures::StreamExt;
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use tycho_client::feed::component_tracker::ComponentFilter;
use tycho_core::{dto::Chain, Bytes};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
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
    utils::load_all_tokens,
};

mod ui;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run in interactive mode
    Interactive,
    
    /// Run with command line arguments
    Swap {
        /// Token in symbol (e.g. "WETH")
        #[arg(short = 'i', long)]
        token_in: String,

        /// Token out symbol (e.g. "USDC")
        #[arg(short = 'o', long)]
        token_out: String,

        /// Amount of token_in to swap (in human readable form, e.g. "1.5")
        #[arg(short = 'a', long)]
        amount: f64,

        /// The minimum TVL threshold for pools
        #[arg(short = 't', long, default_value_t = 1000.0)]
        tvl_threshold: f64,
    },
}

#[derive(Debug)]
struct SwapSimulation {
    pool_name: String,
    pool_address: String,
    amount_out: BigUint,
    gas: u64,
    effective_price: f64,
}

fn get_canonical_address(symbol: &str) -> Option<&'static str> {
    match symbol.to_uppercase().as_str() {
        "USDC" => Some("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"),
        "WETH" => Some("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"),
        "DAI" => Some("0x6b175474e89094c44da98b954eedeac495271d0f"),
        "USDT" => Some("0xdac17f958d2ee523a2206206994597c13d831ec7"),
        "WBTC" => Some("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"),
        _ => None
    }
}

async fn get_initial_state(
    tycho_url: &str, 
    tycho_api_key: &str,
    tvl_threshold: f64,
    all_tokens: HashMap<Bytes, Token>,
) -> anyhow::Result<BlockUpdate> {
    println!("Fetching state from indexer...");
    let tvl_filter = ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold);
    let mut protocol_stream = ProtocolStreamBuilder::new(tycho_url, Chain::Ethereum)
        .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
        .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:balancer_v2",
            tvl_filter.clone(),
            Some(balancer_pool_filter),
        )
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:curve",
            tvl_filter.clone(),
            Some(curve_pool_filter),
        )
        .exchange::<UniswapV4State>(
            "uniswap_v4",
            tvl_filter.clone(),
            Some(uniswap_v4_pool_with_hook_filter),
        )
        .auth_key(Some(tycho_api_key.to_string()))
        .skip_state_decode_failures(true)
        .set_tokens(all_tokens.clone())
        .await
        .build()
        .await?;

    let state = protocol_stream.next().await.expect("Failed to get initial state")?;
    
    // Log pool information for debugging
    let mut pool_log = File::create("pools.log")?;
    for (_id, comp) in &state.new_pairs {
        writeln!(pool_log, "Pool {} with tokens:", comp.address)?;
        for token in &comp.tokens {
            writeln!(pool_log, "  {} ({})", token.symbol, token.address)?;
        }
    }

    Ok(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Interactive => {
            ui::run_interactive().await?
        }
        Commands::Swap { token_in, token_out, amount, tvl_threshold } => {
            let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
            let tycho_api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

            // Perform an early check to ensure `RPC_URL` is set.
            // This prevents errors from occurring later during execution.
            env::var("RPC_URL").expect("RPC_URL env variable should be set");

            // Load tokens and find our input/output tokens
            let all_tokens = load_all_tokens(tycho_url.as_str(), false, Some(tycho_api_key.as_str())).await;
            
            // Log all tokens before finding the specific ones we need
            let mut token_log = File::create("all_tokens.log")?;
            writeln!(token_log, "Total tokens received: {}", all_tokens.len())?;
            writeln!(token_log, "\nDetailed token list:")?;
            for (_, token) in &all_tokens {
                writeln!(
                    token_log,
                    "Symbol: {:<10} Address: {:<42} Decimals: {}",
                    token.symbol, token.address, token.decimals
                )?;
            }
            
            // Try to get canonical addresses first
            let token_in_addr = get_canonical_address(&token_in);
            let token_out_addr = get_canonical_address(&token_out);

            let (_, token_in) = if let Some(addr) = token_in_addr {
                all_tokens
                    .iter()
                    .find(|(_, t)| t.address.to_string().eq_ignore_ascii_case(addr))
                    .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_in))?
            } else {
                all_tokens
                    .iter()
                    .find(|(_, t)| t.symbol.eq_ignore_ascii_case(&token_in))
                    .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_in))?
            };

            let (_, token_out) = if let Some(addr) = token_out_addr {
                all_tokens
                    .iter()
                    .find(|(_, t)| t.address.to_string().eq_ignore_ascii_case(addr))
                    .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_out))?
            } else {
                all_tokens
                    .iter()
                    .find(|(_, t)| t.symbol.eq_ignore_ascii_case(&token_out))
                    .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_out))?
            };

            println!("Looking for tokens:");
            println!("  {} ({})", token_in.symbol, token_in.address);
            println!("  {} ({})", token_out.symbol, token_out.address);

            // Also write to log file
            let mut log_file = File::create("swap_finder.log")?;
            writeln!(log_file, "Looking for tokens:")?;
            writeln!(log_file, "  {} ({})", token_in.symbol, token_in.address)?;
            writeln!(log_file, "  {} ({})", token_out.symbol, token_out.address)?;

            // Calculate input amount in token decimals
            let amount_in = BigUint::from(
                (amount * 10f64.powi(token_in.decimals as i32)) as u64
            );

            println!("Finding pools for swap {} {} -> {} ...", amount, token_in.symbol, token_out.symbol);

            // Get initial state (cached or fresh)
            let initial_state = get_initial_state(
                &tycho_url,
                &tycho_api_key,
                tvl_threshold,
                all_tokens.clone(),
            ).await?;

            let mut simulations = Vec::new();

            // Find pools that have both tokens and simulate swaps
            for (id, comp) in initial_state.new_pairs.iter() {
                writeln!(log_file, "Checking pool {} with tokens:", comp.address)?;
                for t in &comp.tokens {
                    writeln!(log_file, "  - {}: {}", t.symbol, t.address)?;
                }
                
                // Check both token orderings
                let has_tokens = (comp.tokens.iter().any(|t| t.address == token_in.address) 
                                 && comp.tokens.iter().any(|t| t.address == token_out.address))
                                || (comp.tokens.iter().any(|t| t.address == token_out.address) 
                                    && comp.tokens.iter().any(|t| t.address == token_in.address));

                if has_tokens {
                    writeln!(log_file, "Found matching pool!")?;
                    if let Some(state) = initial_state.states.get(id) {
                        // Try both directions
                        let result = state.get_amount_out(amount_in.clone(), token_in, token_out)
                            .or_else(|_| state.get_amount_out(amount_in.clone(), token_out, token_in));
                        
                        if let Ok(result) = result {
                            let effective_price = result.amount.to_f64().unwrap() 
                                / (amount_in.to_f64().unwrap() * 10f64.powi(token_out.decimals as i32 - token_in.decimals as i32));
                            
                            simulations.push(SwapSimulation {
                                pool_name: format!("{:?}", state),
                                pool_address: format!("{:#x}", comp.address),
                                amount_out: result.amount,
                                gas: result.gas.to_u64().unwrap(),
                                effective_price,
                            });
                        }
                    }
                }
            }

            // Sort by amount out (best to worst)
            simulations.sort_by(|a, b| b.amount_out.cmp(&a.amount_out));

            println!("\nFound {} pools with this pair. Results sorted by best price:", simulations.len());
            println!("{:<40} {:<15} {:<10} {:<10}", "Pool", "Amount Out", "Price", "Gas");
            println!("{}", "-".repeat(75));

            for sim in simulations {
                // Truncate the pool name to make it more readable
                let pool_name = if sim.pool_name.len() > 40 {
                    format!("{}...", &sim.pool_name[..37])
                } else {
                    sim.pool_name
                };

                println!(
                    "{:<40} {:<15.6} {:<10.6} {:<10}",
                    format!("{} ({})", pool_name, sim.pool_address),
                    sim.amount_out.to_f64().unwrap() / 10f64.powi(token_out.decimals as i32),
                    sim.effective_price,
                    sim.gas
                );
            }
        }
    }

    Ok(())
} 