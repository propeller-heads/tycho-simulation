use std::collections::HashMap;
use futures::StreamExt;
use tokio::sync::mpsc::Sender;
use tracing::info;

use tycho_client::{
    feed::component_tracker::ComponentFilter, rpc::RPCClient,
    HttpRPCClient,
};
use tycho_core::{dto::Chain, Bytes};

use tycho_simulation::{
    evm::{
        engine_db::{
            simulation_db::BlockHeader, tycho_db::PreCachedDB, update_engine, SHARED_TYCHO_DB,
        },
        protocol::{
            uniswap_v2::state::UniswapV2State, uniswap_v3::state::UniswapV3State,
            vm::state::EVMPoolState,
        },
        tycho_models::{AccountUpdate, ResponseAccount},
    },
    models::ERC20Token,
    protocol::{
        stream_decoder::{BlockUpdate},
        uniswap_v2::state::UniswapV2State,
        uniswap_v3::state::UniswapV3State,
    },
};
use tycho_simulation::protocol::stream_decoder::{ProtocolStreamBuilder};

const ZERO_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

fn balancer_pool_filter(component: &ComponentWithState) -> bool {
    // Check for rate_providers in static_attributes
    info!("Checking Balancer pool {}", component.component.id);
    if component.component.protocol_system != "vm:balancer" {
        return true;
    }
    if let Some(rate_providers_data) = component
        .component
        .static_attributes
        .get("rate_providers")
    {
        let rate_providers_str = str::from_utf8(rate_providers_data).expect("Invalid UTF-8 data");
        let parsed_rate_providers =
            serde_json::from_str::<Vec<String>>(rate_providers_str).expect("Invalid JSON format");

        info!("Parsed rate providers: {:?}", parsed_rate_providers);
        let has_dynamic_rate_provider = parsed_rate_providers
            .iter()
            .any(|provider| provider != ZERO_ADDRESS);

        info!("Has dynamic rate provider: {:?}", has_dynamic_rate_provider);
        if has_dynamic_rate_provider {
            info!(
                "Filtering out Balancer pool {} because it has dynamic rate_providers",
                component.component.id
            );
            return false;
        }
    } else {
        info!("Balancer pool does not have `rate_providers` attribute");
    }
    let unsupported_pool_types: HashSet<&str> = [
        "ERC4626LinearPoolFactory",
        "EulerLinearPoolFactory",
        "SiloLinearPoolFactory",
        "YearnLinearPoolFactory",
        "ComposableStablePoolFactory",
    ]
    .iter()
    .cloned()
    .collect();

    // Check pool_type in static_attributes
    if let Some(pool_type_data) = component
        .component
        .static_attributes
        .get("pool_type")
    {
        // Convert the decoded bytes to a UTF-8 string
        let pool_type = str::from_utf8(pool_type_data).expect("Invalid UTF-8 data");
        if unsupported_pool_types.contains(pool_type) {
            info!(
                "Filtering out Balancer pool {} because it has type {}",
                component.component.id, pool_type
            );
            return false;
        } else {
            info!("Balancer pool with type {} will not be filtered out.", pool_type);
        }
    }
    info!(
        "Balancer pool with static attributes {:?} will not be filtered out.",
        component.component.static_attributes
    );
    info!("Balancer pool will not be filtered out.");
    true
}

// TODO: Make extractors configurable
pub async fn process_messages(
    tycho_url: String,
    auth_key: Option<String>,
    state_tx: Sender<BlockUpdate>,
    tvl_threshold: f64,
) {
    let all_tokens = load_all_tokens(tycho_url.as_str(), auth_key.as_deref()).await;
    let mut protocol_stream = ProtocolStreamBuilder::new(&tycho_url, Chain::Ethereum)
        .exchange::<UniswapV2State>("uniswap_v2", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .exchange::<UniswapV3State>("uniswap_v3", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .auth_key(auth_key.clone())
        .set_tokens(all_tokens)
        .build()
        .await;

    // Loop through tycho messages
    while let Some(msg) = protocol_stream.next().await {
        state_tx
            .send(msg.unwrap())
            .await
            .expect("Sending tick failed!");
    }
}

pub async fn load_all_tokens(
    tycho_url: &str,
    auth_key: Option<&str>,
) -> HashMap<Bytes, ERC20Token> {
    let rpc_url = format!("https://{tycho_url}");
    let rpc_client = HttpRPCClient::new(rpc_url.as_str(), auth_key).unwrap();

    #[allow(clippy::mutable_key_type)]
    rpc_client
        .get_all_tokens(Chain::Ethereum, Some(100), Some(42), 3_000)
        .await
        .expect("Unable to load tokens")
        .into_iter()
        .map(|token| {
            let token_clone = token.clone();
            (
                token.address.clone(),
                token.try_into().unwrap_or_else(|_| {
                    panic!("Couldn't convert {:?} into ERC20 token.", token_clone)
                }),
            )
        })
        .collect::<HashMap<_, ERC20Token>>()
}

pub fn start(
    tycho_url: String,
    auth_key: Option<String>,
    state_tx: Sender<BlockUpdate>,
    tvl_threshold: f64,
) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    info!("Starting tycho data feed...");

    rt.block_on(async {
        tokio::spawn(async move {
            process_messages(tycho_url, auth_key, state_tx, tvl_threshold).await;
        })
        .await
        .unwrap();
    });
}
