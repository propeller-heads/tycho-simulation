use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    str,
    str::FromStr,
};
use futures::StreamExt;

use alloy_primitives::{Address, B256};
use chrono::Utc;
use num_bigint::BigInt;
use tokio::sync::mpsc::Sender;
use tokio_stream::{wrappers::ReceiverStream};
use tracing::info;

use tycho_client::{
    feed::{component_tracker::ComponentFilter, synchronizer::ComponentWithState},
    rpc::RPCClient,
    stream::TychoStreamBuilder,
    HttpRPCClient,
};
use tycho_core::{dto::Chain, Bytes};
use tycho_simulation::{
    evm::{
        engine_db::{
            simulation_db::BlockHeader, tycho_db::PreCachedDB, update_engine, SHARED_TYCHO_DB,
        },
        protocol::{
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            vm::{state::EVMPoolState, utils::json_deserialize_be_bigint_list},
        },
        tycho_models::{AccountUpdate, ResponseAccount},
    },
    models::Token,
    protocol::{
        stream_decoder::{BlockUpdate, TychoStreamDecoder},
        uniswap_v2::state::UniswapV2State,
        uniswap_v3::state::UniswapV3State,
    },
};
use tycho_simulation::protocol::stream_decoder::tycho_stream;

const ZERO_ADDRESS: &str = "0x0000000000000000000000000000000000000000";

fn balancer_pool_filter(component: &ComponentWithState) -> bool {
    // Check for rate_providers in static_attributes
    debug!("Checking Balancer pool {}", component.component.id);
    if component.component.protocol_system != "vm:balancer_v2" {
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

        debug!("Parsed rate providers: {:?}", parsed_rate_providers);
        let has_dynamic_rate_provider = parsed_rate_providers
            .iter()
            .any(|provider| provider != ZERO_ADDRESS);

        debug!("Has dynamic rate provider: {:?}", has_dynamic_rate_provider);
        if has_dynamic_rate_provider {
            debug!(
                "Filtering out Balancer pool {} because it has dynamic rate_providers",
                component.component.id
            );
            return false;
        }
    } else {
        debug!("Balancer pool does not have `rate_providers` attribute");
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
            debug!(
                "Filtering out Balancer pool {} because it has type {}",
                component.component.id, pool_type
            );
            return false;
        } else {
            debug!("Balancer pool with type {} will not be filtered out.", pool_type);
        }
    }
    debug!(
        "Balancer pool with static attributes {:?} will not be filtered out.",
        component.component.static_attributes
    );
    debug!("Balancer pool will not be filtered out.");
    true
}

fn curve_pool_filter(component: &ComponentWithState) -> bool {
    if let Some(asset_types) = component
        .component
        .static_attributes
        .get("asset_types")
    {
        if json_deserialize_be_bigint_list(asset_types)
            .unwrap()
            .iter()
            .any(|t| t != &BigInt::ZERO)
        {
            info!(
                "Filtering out Curve pool {} because it has unsupported token type",
                component.component.id
            );
            return false;
        }
    }

    if let Some(asset_type) = component
        .component
        .static_attributes
        .get("asset_type")
    {
        let types_str = str::from_utf8(asset_type).expect("Invalid UTF-8 data");
        if types_str != "0x00" {
            info!(
                "Filtering out Curve pool {} because it has unsupported token type",
                component.component.id
            );
            return false;
        }
    }

    if let Some(stateless_addrs) = component
        .state
        .attributes
        .get("stateless_contract_addr_0")
    {
        let impl_str = str::from_utf8(stateless_addrs).expect("Invalid UTF-8 data");
        // Uses oracles
        if impl_str == "0x847ee1227a9900b73aeeb3a47fac92c52fd54ed9" {
            info!(
                "Filtering out Curve pool {} because it has proxy implementation {}",
                component.component.id, impl_str
            );
            return false;
        }
    }
    true
}

pub async fn process_messages(
    tycho_url: String,
    auth_key: Option<String>,
    state_tx: Sender<BlockUpdate>,
    tvl_threshold: f64,
) {
    // Connect to Tycho
    let (jh, tycho_rx) = TychoStreamBuilder::new(&tycho_url, Chain::Ethereum)
        .exchange("uniswap_v2", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .exchange("uniswap_v3", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .exchange("vm:balancer_v2", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .exchange("vm:curve", ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold))
        .auth_key(auth_key.clone())
        .build()
        .await
        .expect("Failed to build tycho stream");

    let all_tokens = load_all_tokens(tycho_url.as_str(), auth_key.as_deref()).await;

    let mut protocol_stream = tycho_stream(all_tokens, tycho_rx).await;

    // Loop through tycho messages
    while let Some(msg) = protocol_stream.next().await {
        state_tx
            .send(msg.unwrap())
            .await
            .expect("Sending tick failed!");
    }

    jh.await.unwrap();
}

pub async fn load_all_tokens(
    tycho_url: &str,
    auth_key: Option<&str>,
) -> HashMap<Bytes, Token> {
    let rpc_url = format!("https://{tycho_url}");
    let rpc_client = HttpRPCClient::new(rpc_url.as_str(), auth_key).unwrap();

    #[allow(clippy::mutable_key_type)]
    rpc_client
        .get_all_tokens(Chain::Ethereum, Some(100), Some(42), 3_000)
        .await
        .expect("Unable to load tokens")
        .into_iter()
        .map(|token| {
            (
                token.address.clone(),
                token
                    .clone()
                    .try_into()
                    .unwrap_or_else(|_| {
                        panic!("Couldn't convert {:?} into ERC20 token.", token.clone())
                    }),
            )
        })
        .collect::<HashMap<_, Token>>()
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
