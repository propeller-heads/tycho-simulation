//! Shared utilities for historical examples.
//!
//! This module contains common functionality for working with Tycho historical data,
//! including timestamp-to-block conversion, component fetching, and state decoding.

use std::{collections::HashMap, env};

use alloy::{providers::Provider, transports::http::reqwest::Url};
use anyhow::{anyhow, Context, Result};
use clap::ValueEnum;
use tycho_client::{
    feed::{
        synchronizer::{ComponentWithState, Snapshot},
        BlockHeader,
    },
    rpc::{HttpRPCClientOptions, RPCClient, SnapshotParameters, RPC_CLIENT_CONCURRENCY},
    HttpRPCClient,
};
use tycho_common::{
    dto::{Chain as DtoChain, ProtocolComponent, ProtocolComponentsRequestBody},
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};
use tycho_simulation::{
    evm::protocol::{
        uniswap_v2::state::UniswapV2State, uniswap_v3::state::UniswapV3State,
        uniswap_v4::state::UniswapV4State,
    },
    protocol::models::{DecoderContext, TryFromWithBlock},
    utils::{get_default_url, load_all_tokens},
};

/// Supported protocol types for simulation
#[derive(Debug, Clone, ValueEnum)]
pub enum ProtocolType {
    UniswapV2,
    UniswapV3,
    UniswapV4,
}

impl ProtocolType {
    pub fn as_system_name(&self) -> &'static str {
        match self {
            ProtocolType::UniswapV2 => "uniswap_v2",
            ProtocolType::UniswapV3 => "uniswap_v3",
            ProtocolType::UniswapV4 => "uniswap_v4",
        }
    }
}

/// Supported chains
#[derive(Debug, Clone, ValueEnum, Copy)]
pub enum ChainArg {
    Ethereum,
    Base,
    Unichain,
}

impl From<ChainArg> for Chain {
    fn from(value: ChainArg) -> Self {
        match value {
            ChainArg::Ethereum => Chain::Ethereum,
            ChainArg::Base => Chain::Base,
            ChainArg::Unichain => Chain::Unichain,
        }
    }
}

impl From<ChainArg> for DtoChain {
    fn from(value: ChainArg) -> Self {
        match value {
            ChainArg::Ethereum => DtoChain::Ethereum,
            ChainArg::Base => DtoChain::Base,
            ChainArg::Unichain => DtoChain::Unichain,
        }
    }
}

/// Configuration for Tycho client
pub struct TychoConfig {
    pub url: String,
    pub api_key: String,
    pub chain: Chain,
    pub dto_chain: DtoChain,
}

impl TychoConfig {
    /// Create config from environment variables and chain argument
    pub fn from_env(chain_arg: ChainArg) -> Result<Self> {
        let chain: Chain = chain_arg.into();
        let dto_chain: DtoChain = chain_arg.into();

        let url = env::var("TYCHO_URL")
            .ok()
            .or_else(|| get_default_url(&chain))
            .ok_or_else(|| anyhow!("TYCHO_URL not set and no default for chain {:?}", chain))?;

        let api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());

        Ok(Self { url, api_key, chain, dto_chain })
    }

    /// Create an HTTP RPC client
    pub fn create_rpc_client(&self) -> Result<HttpRPCClient> {
        let rpc_url = format!("https://{}", self.url);
        let rpc_options = HttpRPCClientOptions::new().with_auth_key(Some(self.api_key.clone()));
        HttpRPCClient::new(&rpc_url, rpc_options).context("Failed to create Tycho RPC client")
    }

    /// Load all tokens from Tycho
    pub async fn load_tokens(&self) -> Result<HashMap<Bytes, Token>> {
        load_all_tokens(
            &self.url,
            false,
            Some(&self.api_key),
            true,
            self.chain,
            None,
            None,
        )
        .await
        .context("Failed to load tokens")
    }
}

/// Convert Unix timestamp to Ethereum block number via binary search
pub async fn timestamp_to_block<P: Provider>(provider: &P, target_timestamp: u64) -> Result<u64> {
    let latest = provider
        .get_block_number()
        .await
        .context("Failed to get latest block number")?;

    // Binary search for the block
    let mut low = 1u64;
    let mut high = latest;

    while low < high {
        let mid = (low + high) / 2;
        let block = provider
            .get_block_by_number(mid.into())
            .await
            .context("Failed to get block")?
            .ok_or_else(|| anyhow!("Block {} not found", mid))?;

        let block_ts = block.header.timestamp;

        if block_ts < target_timestamp {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // Verify we found a reasonable block
    let block = provider
        .get_block_by_number(low.into())
        .await
        .context("Failed to get block")?
        .ok_or_else(|| anyhow!("Block {} not found", low))?;

    println!(
        "Found block {} with timestamp {} (target: {})",
        low, block.header.timestamp, target_timestamp
    );

    Ok(low)
}

/// Fetch ProtocolComponents by IDs from Tycho
pub async fn fetch_components(
    rpc_client: &HttpRPCClient,
    protocol_system: &str,
    component_ids: &[String],
    chain: DtoChain,
) -> Result<HashMap<String, ProtocolComponent>> {
    let body =
        ProtocolComponentsRequestBody::id_filtered(protocol_system, component_ids.to_vec(), chain);

    let response = rpc_client
        .get_protocol_components_paginated(&body, None, RPC_CLIENT_CONCURRENCY)
        .await
        .context("Failed to fetch protocol components")?;

    Ok(response
        .protocol_components
        .into_iter()
        .map(|c| (c.id.clone(), c))
        .collect())
}

/// Fetch snapshots at specific block using get_snapshots
pub async fn fetch_snapshots(
    rpc_client: &HttpRPCClient,
    protocol_system: &str,
    components: &HashMap<String, ProtocolComponent>,
    block_number: u64,
    chain: DtoChain,
) -> Result<Snapshot> {
    let params = SnapshotParameters::new(chain, protocol_system, components, &[], block_number)
        .include_balances(true)
        .include_tvl(false);

    rpc_client
        .get_snapshots(&params, None, RPC_CLIENT_CONCURRENCY)
        .await
        .context("Failed to fetch snapshots")
}

/// Decode UniswapV2 snapshots into pool states
pub async fn decode_v2_states(
    snapshots: &HashMap<String, ComponentWithState>,
    tokens: &HashMap<Bytes, Token>,
) -> Result<HashMap<String, Box<dyn ProtocolSim>>> {
    let mut states: HashMap<String, Box<dyn ProtocolSim>> = HashMap::new();
    let decoder_context = DecoderContext::new();
    let block_header = BlockHeader {
        number: 0,
        hash: Bytes::from(vec![0; 32]),
        parent_hash: Bytes::from(vec![0; 32]),
        revert: false,
        timestamp: 0,
    };

    for (id, snapshot) in snapshots {
        match UniswapV2State::try_from_with_header(
            snapshot.clone(),
            block_header.clone(),
            &HashMap::new(),
            tokens,
            &decoder_context,
        )
        .await
        {
            Ok(decoded_state) => {
                states.insert(id.clone(), Box::new(decoded_state));
            }
            Err(e) => {
                eprintln!("Failed to decode component {}: {:?}", id, e);
            }
        }
    }

    Ok(states)
}

/// Decode UniswapV3 snapshots into pool states
pub async fn decode_v3_states(
    snapshots: &HashMap<String, ComponentWithState>,
    tokens: &HashMap<Bytes, Token>,
) -> Result<HashMap<String, Box<dyn ProtocolSim>>> {
    let mut states: HashMap<String, Box<dyn ProtocolSim>> = HashMap::new();
    let decoder_context = DecoderContext::new();
    let block_header = BlockHeader {
        number: 0,
        hash: Bytes::from(vec![0; 32]),
        parent_hash: Bytes::from(vec![0; 32]),
        revert: false,
        timestamp: 0,
    };

    for (id, snapshot) in snapshots {
        match UniswapV3State::try_from_with_header(
            snapshot.clone(),
            block_header.clone(),
            &HashMap::new(),
            tokens,
            &decoder_context,
        )
        .await
        {
            Ok(decoded_state) => {
                states.insert(id.clone(), Box::new(decoded_state));
            }
            Err(e) => {
                eprintln!("Failed to decode component {}: {:?}", id, e);
            }
        }
    }

    Ok(states)
}

/// Decode UniswapV4 snapshots into pool states
pub async fn decode_v4_states(
    snapshots: &HashMap<String, ComponentWithState>,
    tokens: &HashMap<Bytes, Token>,
) -> Result<HashMap<String, Box<dyn ProtocolSim>>> {
    let mut states: HashMap<String, Box<dyn ProtocolSim>> = HashMap::new();
    let decoder_context = DecoderContext::new();
    let block_header = BlockHeader {
        number: 0,
        hash: Bytes::from(vec![0; 32]),
        parent_hash: Bytes::from(vec![0; 32]),
        revert: false,
        timestamp: 0,
    };

    for (id, snapshot) in snapshots {
        match UniswapV4State::try_from_with_header(
            snapshot.clone(),
            block_header.clone(),
            &HashMap::new(),
            tokens,
            &decoder_context,
        )
        .await
        {
            Ok(decoded_state) => {
                states.insert(id.clone(), Box::new(decoded_state));
            }
            Err(e) => {
                eprintln!("Failed to decode component {}: {:?}", id, e);
            }
        }
    }

    Ok(states)
}

/// Decode snapshots based on protocol type
pub async fn decode_states(
    protocol: &ProtocolType,
    snapshots: &HashMap<String, ComponentWithState>,
    tokens: &HashMap<Bytes, Token>,
) -> Result<HashMap<String, Box<dyn ProtocolSim>>> {
    match protocol {
        ProtocolType::UniswapV2 => decode_v2_states(snapshots, tokens).await,
        ProtocolType::UniswapV3 => decode_v3_states(snapshots, tokens).await,
        ProtocolType::UniswapV4 => decode_v4_states(snapshots, tokens).await,
    }
}

/// Convert ProtocolComponent from tycho_common::dto to tycho_simulation::protocol::models
pub fn convert_component(
    comp: &ProtocolComponent,
    tokens: &HashMap<Bytes, Token>,
) -> tycho_simulation::protocol::models::ProtocolComponent {
    let token_vec: Vec<Token> = comp
        .tokens
        .iter()
        .filter_map(|addr| tokens.get(addr).cloned())
        .collect();

    tycho_simulation::protocol::models::ProtocolComponent::from_with_tokens(comp.clone(), token_vec)
}

/// Create an Ethereum provider from RPC_URL environment variable
pub fn create_eth_provider() -> Result<impl Provider> {
    let eth_rpc_url = env::var("RPC_URL")
        .context("RPC_URL environment variable required for timestamp conversion")?;

    use alloy::providers::ProviderBuilder;
    use std::str::FromStr;

    let url = Url::from_str(&eth_rpc_url).context("Invalid RPC URL")?;
    Ok(ProviderBuilder::new().connect_http(url))
}
