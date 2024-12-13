mod ui;
pub mod utils;

extern crate tycho_simulation;

use crate::utils::load_all_tokens;
use clap::Parser;
use futures::{future::select_all, StreamExt};
use std::env;
use tokio::{sync::mpsc, task::JoinHandle};
use tycho_client::feed::component_tracker::ComponentFilter;
use tycho_core::dto::Chain;
use tycho_simulation::{
    evm::{
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            filters::{balancer_pool_filter, curve_pool_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    protocol::models::BlockUpdate,
};

#[derive(Parser)]
struct Cli {
    /// The tvl threshold to filter the graph by
    #[arg(short, long, default_value_t = 1000.0)]
    tvl_threshold: f64,
}

#[tokio::main]
async fn main() {
    utils::setup_tracing();
    // Parse command-line arguments into a Cli struct
    let cli = Cli::parse();

    let tycho_url =
        env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-dev.propellerheads.xyz".to_string());
    let tycho_api_key: String =
        env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

    // Create communication channels for inter-thread communication
    let (tick_tx, tick_rx) = mpsc::channel::<BlockUpdate>(12);

    let tycho_message_processor: JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        let all_tokens = load_all_tokens(tycho_url.as_str(), Some(tycho_api_key.as_str())).await;
        let tvl_filter = ComponentFilter::with_tvl_range(cli.tvl_threshold, cli.tvl_threshold);
        let mut protocol_stream = ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
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
            .auth_key(Some(tycho_api_key.clone()))
            .set_tokens(all_tokens)
            .await
            .build()
            .await
            .expect("Failed building protocol stream");

        // Loop through block updates
        while let Some(msg) = protocol_stream.next().await {
            tick_tx
                .send(msg.unwrap())
                .await
                .expect("Sending tick failed!")
        }
        anyhow::Result::Ok(())
    });

    // If testing without the UI - spawn a consumer task to consume the messages (uncomment below)
    // let (tick_tx, mut tick_rx) = mpsc::channel::<BlockState>(12);

    // let consumer_handle = task::spawn(async move {
    //     while let Some(message) = tick_rx.recv().await {
    //         println!("got message: {:?}", message);
    //     }
    //     Ok(())
    // });
    // let tasks = [tycho_message_processor, consumer_handle];

    let terminal = ratatui::init();
    let terminal_app = tokio::spawn(async move {
        ui::App::new(tick_rx)
            .run(terminal)
            .await
    });
    let tasks = [tycho_message_processor, terminal_app];
    let _ = select_all(tasks).await;
    ratatui::restore();
}
