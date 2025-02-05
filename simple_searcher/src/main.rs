mod graph;

use clap::Parser;
use futures::StreamExt;
use num_bigint::BigUint;
use std::env;
use std::time::Instant;
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, EnvFilter};
use tycho_client::feed::component_tracker::ComponentFilter;
use tycho_core::{dto::Chain, Bytes};
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            filters::{balancer_pool_filter, uniswap_v4_pool_with_hook_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    utils::load_all_tokens,
};

use crate::graph::graph::PoolGraph;

/// Graph based solver
#[derive(Parser)]
struct Cli {
    /// The tvl threshold to filter the graph by
    #[arg(short, long, default_value_t = 10.0)]
    tvl_threshold: f64,
    /// The token address to start the search from
    /// Default is WETH
    #[arg(short, long, default_value = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2")]
    token_address: String,
    /// The sell amount to start the search from (currently only accepts integer values that exclude decimals)
    /// Default is 1 ETH
    #[arg(short, long, default_value_t = 1u32)]
    sell_amount: u32,
    /// The graph depth to search. Default is 3
    #[arg(short, long, default_value_t = 3)]
    depth: u32,
}

pub async fn start_app() {
    // Parse command-line arguments into a Cli struct
    let cli = Cli::parse();

    let tycho_url =
        env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    let tvl_filter = ComponentFilter::with_tvl_range(cli.tvl_threshold, cli.tvl_threshold);

    let all_tokens = load_all_tokens(tycho_url.as_str(), false, Some(tycho_api_key.as_str())).await;

    let mut graph: PoolGraph = PoolGraph::new();

    let mut protocol_stream = ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
        .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
        .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
        .exchange::<EVMPoolState<PreCachedDB>>(
            "vm:balancer_v2",
            tvl_filter.clone(),
            Some(balancer_pool_filter),
        )
        .exchange::<UniswapV4State>(
            "uniswap_v4",
            tvl_filter.clone(),
            Some(uniswap_v4_pool_with_hook_filter),
        )
        .skip_state_decode_failures(true)
        .auth_key(Some(tycho_api_key.clone()))
        .set_tokens(all_tokens.clone())
        .await
        .build()
        .await
        .expect("Failed building protocol stream");

    // WETH
    let token_addreess: Bytes = cli.token_address.as_str().into();
    let token = all_tokens.get(&token_addreess).unwrap();

    // Loop through block updates
    #[allow(clippy::never_loop)]
    while let Some(message) = protocol_stream.next().await {
        let message = message.expect("Could not receive message");

        info!("Received block: {:?}", message.block_number);
        info!("Adding {:?} pairs to the graph", message.new_pairs.len());
        for (address, tokens) in message.new_pairs {
            let state = message.states.get(&address);
            if state.is_none() {
                error!("State not found for new pair: {:?}", address);
                continue;
            }
            graph.add_pool(
                tokens.tokens[0].clone(),
                tokens.tokens[1].clone(),
                address,
                state.unwrap().clone(),
            );
        }

        let circular_routes = graph.find_circular_routes(token, 3);
        println!("Found {} circular routes", circular_routes.len());

        // calculate how long does it take to process the loop
        let start = Instant::now();

        let mut n_success = 0_usize;
        let mut n_profitable = 0_usize;
        
        for route in circular_routes {
            // Start by selling 1K USDC
            let mut sell_token = token.clone();
            let target_amount = BigUint::from(1u32) * BigUint::from(10u32).pow(sell_token.decimals as u32);

            let mut sell_amount = target_amount.clone();

            let mut error = false;

            for (id, path) in route.clone() {
                let (token0, token1) = graph.get_pool(id).unwrap();

                let (current_sell_token, buy_token) = if token0.address == sell_token.address {
                    (token0, token1)
                } else {
                    (token1, token0)
                };
                let amount_out = path
                    .get_amount_out(sell_amount.clone(), &current_sell_token, &buy_token);
                
                match amount_out {
                    Ok(amount_out) => {
                        debug!(
                            "Selling {:?} {:?} for {:?} {:?}",
                            amount_out.amount, current_sell_token.symbol, amount_out.amount, buy_token.symbol
                        );

                        sell_amount = amount_out.amount;
                        sell_token = buy_token;
                        
                    }
                    Err(_) => {
                        error = true;
                        break;
                    }
                }

            }
            if error {
                continue;
            }
            n_success += 1;
            debug!("Final amount: {:?}", sell_amount);
            if sell_amount > BigUint::from(target_amount) {
                let route_ids: Vec<String> = route.iter().map(|(id, _)| id.to_string()).collect();
                println!("Found possibly profitable route: {:?}. AmountOut {:?}", route_ids, sell_amount);
                n_profitable += 1;
            }
        }
        
        let duration = start.elapsed();
        info!("Time elapsed in processing block: {:?}", duration);
        info!("Number of successful routes: {:?}", n_success);
        info!("Number of profitable routes: {:?}", n_profitable);

        // TODO: Implement updates for every block
        break
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let format = fmt::format()
        .with_level(true) // Show log levels
        .with_target(false) // Hide module paths
        .compact(); // Use a compact format

    fmt()
        .event_format(format)
        .with_env_filter(EnvFilter::from_default_env()) // Use RUST_LOG for log levels
        .init();

    info!("Starting application...");

    start_app().await;
    Ok(())
}
