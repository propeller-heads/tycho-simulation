//! Snapshot persistence for protocol states via FeedMessage serialization.
//!
//! This module provides utilities to capture and restore protocol states by
//! serializing/deserializing `FeedMessage` objects from tycho-client into a
//! self-contained `Snapshot` structure.
//!
//! ## Approach
//! - **Capture**: Use `TychoStreamBuilder` to get raw `FeedMessage`, wrap in `Snapshot`, serialize to JSON
//! - **Restore**: Deserialize `Snapshot`, decode using embedded tokens and protocols
//!
//! ## Example
//! ```no_run
//! use swap_to_price::snapshot::{save_snapshot, load_and_process_snapshot};
//! use tycho_common::models::Chain;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Capture snapshot
//!     let protocols = vec!["uniswap_v2", "uniswap_v3"];
//!     let (_snapshot, path) = save_snapshot(
//!         "tycho-beta.propellerheads.xyz",
//!         Some("your_key"),
//!         Chain::Ethereum,
//!         &protocols,
//!         100.0, // min TVL
//!         None,  // no token filter
//!         Path::new("./snapshots"),
//!         10,    // max pools per protocol
//!     ).await?;
//!
//!     // Load and process snapshot in one step
//!     let loaded = load_and_process_snapshot(&path).await?;
//!
//!     // Use decoded states directly
//!     for (pool_id, state) in &loaded.states {
//!         println!("Pool: {}", pool_id);
//!     }
//!     Ok(())
//! }
//! ```

use std::{collections::{HashMap, HashSet}, fs::File, path::Path, str::FromStr};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::time::{timeout, Duration};
use tracing::{info, warn};
use tycho_client::{
    feed::{component_tracker::ComponentFilter, BlockHeader, FeedMessage, HeaderLike},
    stream::TychoStreamBuilder,
};
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};
use tycho_simulation::{
    evm::{
        decoder::TychoStreamDecoder,
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            filters::{balancer_v2_pool_filter, balancer_v3_pool_filter, curve_pool_filter},
            pancakeswap_v2::state::PancakeswapV2State,
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            uniswap_v4::state::UniswapV4State,
            vm::state::EVMPoolState,
        },
    },
    protocol::models::{ProtocolComponent, Update},
    utils::load_all_tokens,
};

/// A self-contained snapshot of protocol states.
///
/// Contains everything needed to restore protocol simulations:
/// - Raw FeedMessage with all component data
/// - List of protocols included
/// - Token information extracted from the feed
/// - Metadata about when and how the snapshot was created
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Snapshot {
    /// The raw FeedMessage from Tycho
    pub feed_message: FeedMessage<BlockHeader>,

    /// List of protocol names included in this snapshot
    pub protocols: Vec<String>,

    /// All tokens referenced in the snapshot
    pub tokens: HashMap<Bytes, Token>,

    /// Metadata about the snapshot
    pub metadata: SnapshotMetadata,
}

/// Metadata about when and how a snapshot was created
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SnapshotMetadata {
    /// When the snapshot was captured
    pub captured_at: DateTime<Utc>,

    /// Block number of the snapshot
    pub block_number: u64,

    /// Blockchain network
    pub chain: String,

    /// Minimum TVL filter used
    pub min_tvl: f64,

    /// Total number of components in snapshot
    pub total_components: usize,
}

/// Result of loading and decoding a snapshot
pub struct LoadedSnapshot {
    /// Decoded protocol states ready to use
    pub states: HashMap<String, Box<dyn ProtocolSim>>,

    /// Protocol components (pool information)
    pub components: HashMap<String, ProtocolComponent>,

    /// The underlying Update from the decoder (kept for potential future use)
    #[allow(dead_code)]
    pub update: Update,

    /// Original snapshot metadata
    pub metadata: SnapshotMetadata,
}

/// Filters a FeedMessage to only include components where ALL tokens are target tokens.
///
/// This ensures we only get pools like USDC/USDT, WETH/DAI, etc. where both tokens
/// are in our target list, not pools like USDC/RandomToken.
///
/// # Arguments
/// * `feed_msg` - The FeedMessage to filter
/// * `target_tokens` - List of token address strings (hex) to filter by
///
/// # Returns
/// Filtered FeedMessage with only components where all tokens are in the target list
fn filter_feed_message(
    mut feed_msg: FeedMessage<BlockHeader>,
    target_tokens: &[&str],
) -> FeedMessage<BlockHeader> {
    let target_token_set: HashSet<Bytes> = target_tokens
        .iter()
        .filter_map(|addr_str| Bytes::from_str(addr_str).ok())
        .collect();

    info!("Filtering for {} target tokens", target_token_set.len());

    // Filter each protocol's state messages to only include pools where ALL tokens are in target set
    for protocol_msg in feed_msg.state_msgs.values_mut() {
        protocol_msg.snapshots.states.retain(|_id, component_with_state| {
            component_with_state.component.tokens.iter().all(|token_addr| {
                target_token_set.contains(token_addr)
            })
        });
    }

    feed_msg
}

/// Limits the number of pools per protocol in a FeedMessage.
///
/// This function keeps only the first `pool_count_limit` pools for each protocol.
/// Pools are kept in the order they appear in the FeedMessage.
///
/// # Arguments
/// * `feed_msg` - The FeedMessage to limit
/// * `pool_count_limit` - Maximum number of pools to keep per protocol
fn limit_pools_per_protocol(feed_msg: &mut FeedMessage<BlockHeader>, pool_count_limit: usize) {
    info!("Limiting pools per protocol to {}", pool_count_limit);

    for (protocol, protocol_msg) in feed_msg.state_msgs.iter_mut() {
        let original_count = protocol_msg.snapshots.states.len();

        if original_count > pool_count_limit {
            // Keep only the first pool_count_limit pools
            // Convert to vec, take first N, convert back to map
            let limited_states: std::collections::HashMap<_, _> = protocol_msg
                .snapshots
                .states
                .iter()
                .take(pool_count_limit)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            protocol_msg.snapshots.states = limited_states;

            info!(
                "  {}: {} -> {} pools",
                protocol,
                original_count,
                pool_count_limit
            );
        } else {
            info!("  {}: {} pools", protocol, original_count);
        }
    }
}

/// Fetches a FeedMessage from Tycho stream with custom filters per protocol.
///
/// This allows different filter strategies for each protocol:
/// - For snapshots: `ComponentFilter::with_tvl_range(min_tvl * 0.9, min_tvl)`
/// - For tests: `ComponentFilter::Ids(vec!["0x..."])`
///
/// # Arguments
/// * `tycho_url` - URL of the Tycho API (e.g., "tycho-beta.propellerheads.xyz")
/// * `api_key` - Optional API key for authentication
/// * `chain` - The blockchain to query (e.g., Chain::Ethereum)
/// * `protocol_filters` - List of (protocol_name, ComponentFilter) pairs
///
/// # Returns
/// The first FeedMessage from the stream
pub async fn fetch_feed_message(
    tycho_url: &str,
    api_key: Option<&str>,
    chain: Chain,
    protocol_filters: &[(&str, ComponentFilter)],
) -> Result<FeedMessage<BlockHeader>, Box<dyn std::error::Error>> {
    info!("Connecting to Tycho at {}...", tycho_url);

    // Create TychoStreamBuilder
    let mut stream_builder = TychoStreamBuilder::new(tycho_url, chain.into());

    // Add protocols with their respective filters
    for (protocol, filter) in protocol_filters {
        info!("Adding protocol: {} with filter: {:?}", protocol, filter);
        stream_builder = stream_builder.exchange(protocol, filter.clone());
    }

    // Add auth key if provided
    if let Some(key) = api_key {
        stream_builder = stream_builder.auth_key(Some(key.to_string()));
    }

    // Build the stream
    info!("Building stream...");
    let (handle, mut rx) = stream_builder.build().await?;

    // Wait for first message with timeout
    info!("Waiting for first message...");
    let feed_msg_result = timeout(Duration::from_secs(60), rx.recv())
        .await
        .map_err(|_| "Timeout waiting for first message")?
        .ok_or("Stream closed without sending a message")?;

    // Unwrap the Result from the channel
    let feed_msg = feed_msg_result.map_err(|e| format!("Stream error: {}", e))?;

    // Shutdown the stream
    handle.abort();

    Ok(feed_msg)
}

/// Creates a Snapshot from a FeedMessage and saves it to disk.
///
/// # Arguments
/// * `feed_msg` - The FeedMessage to create the snapshot from
/// * `tycho_url` - URL of the Tycho API (for loading token data)
/// * `api_key` - Optional API key for authentication
/// * `chain` - The blockchain
/// * `protocols` - List of protocol names included
/// * `min_tvl` - Minimum TVL filter used (for metadata)
/// * `target_tokens` - Optional list of token address strings (hex) to filter components by
/// * `output_folder` - Directory where the snapshot will be saved
/// * `pool_count_limit` - Maximum number of pools per protocol to include
///
/// # Returns
/// Tuple of (Snapshot, PathBuf) - the snapshot structure and the actual file path
pub async fn create_and_save_snapshot(
    mut feed_msg: FeedMessage<BlockHeader>,
    tycho_url: &str,
    api_key: Option<&str>,
    chain: Chain,
    protocols: &[&str],
    min_tvl: f64,
    target_tokens: Option<&[&str]>,
    output_folder: &Path,
    pool_count_limit: usize,
) -> Result<(Snapshot, std::path::PathBuf), Box<dyn std::error::Error>> {
    // Apply token filter if provided
    if let Some(tokens) = target_tokens {
        feed_msg = filter_feed_message(feed_msg, tokens);
    }

    // Limit pools per protocol
    limit_pools_per_protocol(&mut feed_msg, pool_count_limit);

    // Extract block number for logging
    let block_number = feed_msg
        .state_msgs
        .values()
        .next()
        .and_then(|msg| msg.header.clone().block())
        .map(|block| block.number)
        .unwrap_or(0);

    info!("Processing FeedMessage for block {}", block_number);
    info!(
        "  Protocols: {}",
        feed_msg.state_msgs.keys().cloned().collect::<Vec<_>>().join(", ")
    );

    // Count total components
    let total_components: usize = feed_msg
        .state_msgs
        .values()
        .map(|msg| msg.snapshots.states.len())
        .sum();
    info!("  Total components: {}", total_components);

    // Extract tokens from the feed and filter components
    let tokens = extract_tokens_from_feed(&mut feed_msg, tycho_url, api_key, chain).await?;

    // Create the Snapshot structure
    let snapshot = Snapshot {
        feed_message: feed_msg,
        protocols: protocols.iter().map(|s| s.to_string()).collect(),
        tokens,
        metadata: SnapshotMetadata {
            captured_at: Utc::now(),
            block_number,
            chain: chain.to_string(),
            min_tvl,
            total_components,
        },
    };

    // Create output directory if it doesn't exist
    if !output_folder.exists() {
        std::fs::create_dir_all(output_folder)?;
    }

    // Build filename with block number
    let filename = format!("snapshot_{}.bin", block_number);
    let output_path = output_folder.join(&filename);

    // Serialize to MessagePack
    info!("Saving snapshot to {}...", output_path.display());
    let file = File::create(&output_path)?;
    rmp_serde::encode::write(&mut std::io::BufWriter::new(file), &snapshot)?;

    info!("Snapshot saved successfully!");

    Ok((snapshot, output_path))
}

/// Captures a snapshot by connecting to Tycho and saving it to a binary file.
///
/// This is a convenience function that combines `fetch_feed_message` and
/// `create_and_save_snapshot` using TVL-based filtering.
///
/// # Arguments
/// * `tycho_url` - URL of the Tycho API (e.g., "tycho-beta.propellerheads.xyz")
/// * `api_key` - Optional API key for authentication
/// * `chain` - The blockchain to snapshot (e.g., Chain::Ethereum)
/// * `protocols` - List of protocol names to include (e.g., ["uniswap_v2", "uniswap_v3"])
/// * `min_tvl` - Minimum TVL filter for components (in ETH)
/// * `target_tokens` - Optional list of token address strings (hex) to filter components by
/// * `output_folder` - Directory where the snapshot will be saved
/// * `pool_count_limit` - Maximum number of pools per protocol to include in snapshot
///
/// # Returns
/// Tuple of (Snapshot, PathBuf) - the snapshot structure and the actual file path
pub async fn save_snapshot(
    tycho_url: &str,
    api_key: Option<&str>,
    chain: Chain,
    protocols: &[&str],
    min_tvl: f64,
    target_tokens: Option<&[&str]>,
    output_folder: &Path,
    pool_count_limit: usize,
) -> Result<(Snapshot, std::path::PathBuf), Box<dyn std::error::Error>> {
    // Create TVL filter for each protocol
    let tvl_filter = ComponentFilter::with_tvl_range(min_tvl * 0.9, min_tvl);
    let protocol_filters: Vec<_> = protocols
        .iter()
        .map(|p| (*p, tvl_filter.clone()))
        .collect();

    // Fetch FeedMessage from stream
    let feed_msg = fetch_feed_message(tycho_url, api_key, chain, &protocol_filters).await?;

    // Create and save snapshot
    create_and_save_snapshot(
        feed_msg,
        tycho_url,
        api_key,
        chain,
        protocols,
        min_tvl,
        target_tokens,
        output_folder,
        pool_count_limit,
    )
    .await
}

/// Loads a snapshot from a binary file.
///
/// This function deserializes a `Snapshot` from MessagePack format.
/// Use `process_snapshot` to decode the snapshot into protocol states.
///
/// # Arguments
/// * `snapshot_path` - Path to the snapshot .bin file
///
/// # Returns
/// `Snapshot` containing the raw data
pub fn load_snapshot(snapshot_path: &Path) -> Result<Snapshot, Box<dyn std::error::Error>> {
    info!("Loading snapshot from {}...", snapshot_path.display());

    // Deserialize Snapshot from MessagePack
    let file = File::open(snapshot_path)?;
    let snapshot: Snapshot = rmp_serde::decode::from_read(std::io::BufReader::new(file))?;

    info!("Loaded snapshot:");
    info!("  Block: {}", snapshot.metadata.block_number);
    info!("  Protocols: {}", snapshot.protocols.join(", "));
    info!("  Tokens: {}", snapshot.tokens.len());
    info!("  Components: {}", snapshot.metadata.total_components);

    Ok(snapshot)
}

/// Processes a snapshot by decoding it into protocol states.
///
/// This function uses the embedded tokens and protocols in the snapshot
/// to configure a decoder, and returns all decoded states ready to use.
///
/// # Arguments
/// * `snapshot` - The snapshot to process
///
/// # Returns
/// `LoadedSnapshot` containing decoded states, components, and metadata
pub async fn process_snapshot(
    snapshot: &Snapshot,
) -> Result<LoadedSnapshot, Box<dyn std::error::Error>> {
    // Create decoder with proper configuration
    let decoder = create_decoder_for_protocols(&snapshot.protocols, snapshot.tokens.clone()).await;

    // Decode the FeedMessage
    info!("Decoding snapshot...");
    let update = decoder.decode(&snapshot.feed_message).await?;

    info!("Snapshot decoded successfully!");
    info!("  States: {}", update.states.len());
    info!("  New pairs: {}", update.new_pairs.len());
    info!("  Removed pairs: {}", update.removed_pairs.len());

    Ok(LoadedSnapshot {
        states: update.states.clone(),
        components: update.new_pairs.clone(),
        update,
        metadata: snapshot.metadata.clone(),
    })
}

/// Convenience function to load and process a snapshot in one step.
///
/// Combines `load_snapshot` and `process_snapshot`.
///
/// # Arguments
/// * `snapshot_path` - Path to the snapshot .bin file
///
/// # Returns
/// `LoadedSnapshot` containing decoded states, components, and metadata
pub async fn load_and_process_snapshot(
    snapshot_path: &Path,
) -> Result<LoadedSnapshot, Box<dyn std::error::Error>> {
    let snapshot = load_snapshot(snapshot_path)?;
    process_snapshot(&snapshot).await
}

/// Extracts tokens from a FeedMessage and loads full token data from Tycho.
///
/// Tokens are loaded from Tycho API to get accurate decimal information.
/// Components with tokens that can't be loaded are filtered out.
///
/// # Arguments
/// * `feed_msg` - The FeedMessage to extract tokens from (will be modified to filter components)
/// * `tycho_url` - Tycho API URL
/// * `api_key` - Optional API key
/// * `chain` - Blockchain chain
///
/// # Returns
/// HashMap of tokens indexed by their address
async fn extract_tokens_from_feed(
    feed_msg: &mut FeedMessage<BlockHeader>,
    tycho_url: &str,
    api_key: Option<&str>,
    chain: Chain,
) -> Result<HashMap<Bytes, Token>, Box<dyn std::error::Error>> {
    // Load all tokens from Tycho to get accurate decimals
    info!("Loading token data from Tycho...");
    let all_tokens = load_all_tokens(
        tycho_url,
        false, // use TLS
        api_key,
        false, // no compression
        chain,
        None,  // no min quality filter
        None,  // no max days filter
    )
    .await?;

    info!("Loaded {} tokens from Tycho", all_tokens.len());

    // Collect token addresses from components
    let mut token_addresses = HashSet::new();
    for protocol_msg in feed_msg.state_msgs.values() {
        for component_with_state in protocol_msg.snapshots.states.values() {
            for token_addr in &component_with_state.component.tokens {
                token_addresses.insert(token_addr.clone());
            }
        }
    }

    info!("Found {} unique token addresses in components", token_addresses.len());

    // Filter components to only those where all tokens are available
    let mut components_before = 0;
    let mut components_after = 0;

    for protocol_msg in feed_msg.state_msgs.values_mut() {
        components_before += protocol_msg.snapshots.states.len();

        protocol_msg.snapshots.states.retain(|_id, component_with_state| {
            component_with_state.component.tokens.iter().all(|token_addr| {
                all_tokens.contains_key(token_addr)
            })
        });

        components_after += protocol_msg.snapshots.states.len();
    }

    info!(
        "Filtered components: {} -> {} (removed {} with unknown tokens)",
        components_before,
        components_after,
        components_before - components_after
    );

    // Extract only the tokens we need
    let mut tokens = HashMap::new();
    for addr in token_addresses {
        if let Some(token) = all_tokens.get(&addr) {
            tokens.insert(addr.clone(), token.clone());
        }
    }

    info!("Using {} tokens with proper decimal information", tokens.len());

    Ok(tokens)
}

/// Creates and configures a TychoStreamDecoder for the specified protocols.
///
/// Currently supports:
/// - `uniswap_v2` → `UniswapV2State`
/// - `sushiswap_v2` → `UniswapV2State`
/// - `pancakeswap_v2` → `PancakeswapV2State`
/// - `uniswap_v3` → `UniswapV3State`
/// - `pancakeswap_v3` → `UniswapV3State`
/// - `uniswap_v4` → `UniswapV4State`
/// - `uniswap_v4_hooks` → `UniswapV4State`
/// - `balancer_v3` → `EVMPoolState` (with filter)
/// - `vm:balancer_v2` → `EVMPoolState` (with filter)
/// - `vm:curve` → `EVMPoolState` (with filter)
/// - `vm:maverick_v2` → `EVMPoolState`
///
/// # Arguments
/// * `protocols` - List of protocol names to register decoders for
/// * `tokens` - HashMap of tokens to set in the decoder
///
/// # Returns
/// A configured `TychoStreamDecoder<BlockHeader>` ready to decode FeedMessages
async fn create_decoder_for_protocols(
    protocols: &[String],
    tokens: HashMap<Bytes, Token>,
) -> TychoStreamDecoder<BlockHeader> {
    let mut decoder = TychoStreamDecoder::<BlockHeader>::new();

    // Register decoders for each protocol
    for protocol in protocols {
        match protocol.as_str() {
            "uniswap_v2" => {
                info!("Registering decoder for uniswap_v2");
                decoder.register_decoder::<UniswapV2State>("uniswap_v2");
            }
            "sushiswap_v2" => {
                info!("Registering decoder for sushiswap_v2");
                decoder.register_decoder::<UniswapV2State>("sushiswap_v2");
            }
            "pancakeswap_v2" => {
                info!("Registering decoder for pancakeswap_v2");
                decoder.register_decoder::<PancakeswapV2State>("pancakeswap_v2");
            }
            "uniswap_v3" => {
                info!("Registering decoder for uniswap_v3");
                decoder.register_decoder::<UniswapV3State>("uniswap_v3");
            }
            "pancakeswap_v3" => {
                info!("Registering decoder for pancakeswap_v3");
                decoder.register_decoder::<UniswapV3State>("pancakeswap_v3");
            }
            "uniswap_v4" => {
                info!("Registering decoder for uniswap_v4");
                decoder.register_decoder::<UniswapV4State>("uniswap_v4");
            }
            "uniswap_v4_hooks" => {
                info!("Registering decoder for uniswap_v4_hooks");
                decoder.register_decoder::<UniswapV4State>("uniswap_v4_hooks");
            }
            "ekubo_v2" => {
                info!("Registering decoder for ekubo_v2");
                decoder.register_decoder::<tycho_simulation::evm::protocol::ekubo::state::EkuboState>("ekubo_v2");
            }
            "balancer_v3" => {
                info!("Registering decoder for balancer_v3 with filter");
                decoder.register_decoder::<EVMPoolState<PreCachedDB>>("balancer_v3");
                decoder.register_filter("balancer_v3", balancer_v3_pool_filter);
            }
            "vm:balancer_v2" => {
                info!("Registering decoder for vm:balancer_v2 with filter");
                decoder.register_decoder::<EVMPoolState<PreCachedDB>>("vm:balancer_v2");
                decoder.register_filter("vm:balancer_v2", balancer_v2_pool_filter);
            }
            "vm:curve" => {
                info!("Registering decoder for vm:curve with filter");
                decoder.register_decoder::<EVMPoolState<PreCachedDB>>("vm:curve");
                decoder.register_filter("vm:curve", curve_pool_filter);
            }
            "vm:maverick_v2" => {
                info!("Registering decoder for vm:maverick_v2");
                decoder.register_decoder::<EVMPoolState<PreCachedDB>>("vm:maverick_v2");
            }
            protocol_name => {
                warn!("Unsupported protocol in snapshot: {}", protocol_name);
                warn!("  Supported protocols: uniswap_v2, sushiswap_v2, pancakeswap_v2,");
                warn!("  uniswap_v3, pancakeswap_v3, uniswap_v4, uniswap_v4_hooks,");
                warn!("  balancer_v3, vm:balancer_v2, vm:curve, vm:maverick_v2");
            }
        }
    }

    // Set tokens
    decoder.set_tokens(tokens).await;

    // Skip decode failures to be resilient
    decoder.skip_state_decode_failures(true);

    decoder
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[tokio::test]
    #[ignore] // Run with: cargo test --example swap_to_price_benchmark -- --ignored --nocapture test_snapshot_roundtrip_by_component_ids
    async fn test_snapshot_roundtrip_by_component_ids() {
        // Initialize tracing for test output
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        // Get Tycho credentials from environment
        let tycho_url = env::var("TYCHO_URL")
            .unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
        let api_key = env::var("TYCHO_AUTH_KEY").ok();

        if api_key.is_none() {
            eprintln!("⚠️  Warning: TYCHO_AUTH_KEY not set. Test may fail due to rate limiting.");
        }

        let chain = Chain::Ethereum;

        // Use specific component IDs for each protocol (1 pool per protocol)
        let protocol_filters = vec![
            ("ekubo_v2", ComponentFilter::Ids(vec!["0xca5b3ef9770bb95940bd4e0bff5ead70a5973d904a8b370b52147820e61a2ff6".to_string()])),
            ("vm:maverick_v2", ComponentFilter::Ids(vec!["0x31373595f40ea48a7aab6cbcb0d377c6066e2dca".to_string()])),
            ("vm:curve", ComponentFilter::Ids(vec!["0xd51a44d3fae010294c616388b506acda1bfaae46".to_string()])),
            ("pancakeswap_v3", ComponentFilter::Ids(vec!["0x04c8577958ccc170eb3d2cca76f9d51bc6e42d8f".to_string()])),
            ("uniswap_v2", ComponentFilter::Ids(vec!["0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc".to_string()])),
            ("uniswap_v3", ComponentFilter::Ids(vec!["0xc5af84701f98fa483ece78af83f11b6c38aca71d".to_string()])),
            ("vm:balancer_v2", ComponentFilter::Ids(vec!["0x96646936b91d6b9d7d0c47c496afbf3d6ec7b6f8000200000000000000000019".to_string()])),
            ("pancakeswap_v2", ComponentFilter::Ids(vec!["0x4ab6702b3ed3877e9b1f203f90cbef13d663b0e8".to_string()])),
            ("sushiswap_v2", ComponentFilter::Ids(vec!["0x397ff1542f962076d0bfe58ea045ffa2d347aca0".to_string()])),
            ("uniswap_v4_hooks", ComponentFilter::Ids(vec!["0xa60399112940a5870efa234b6a178c26b7fe996fbc23acdae 5a8d7575cd64865".to_string()])),
            ("uniswap_v4", ComponentFilter::Ids(vec!["0xf6e8088529094bc485561fa2a03e3d19c9a60f5d99a997e8fe16ab4ca2db277a".to_string()])),
        ];

        let protocols: Vec<&str> = protocol_filters.iter().map(|(p, _)| *p).collect();

        println!("\n📸 Fetching specific components from Tycho...");
        println!("   URL: {}", tycho_url);
        println!("   Protocols: {}", protocols.join(", "));

        // Step 1: Fetch FeedMessage with specific component IDs
        let feed_msg = fetch_feed_message(&tycho_url, api_key.as_deref(), chain, &protocol_filters)
            .await
            .expect("Failed to fetch feed message");

        // Create temp directory for test
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

        // Step 2: Create and save snapshot
        let (saved_snapshot, snapshot_path) = create_and_save_snapshot(
            feed_msg,
            &tycho_url,
            api_key.as_deref(),
            chain,
            &protocols,
            0.0, // min_tvl not used when filtering by IDs
            None,
            temp_dir.path(),
            100, // No limit needed since we're filtering by specific IDs
        )
        .await
        .expect("Failed to create snapshot");

        println!("\n✅ Snapshot created!");
        println!("   File: {}", snapshot_path.display());
        println!("   Block: {}", saved_snapshot.metadata.block_number);
        println!("   Components: {}", saved_snapshot.metadata.total_components);

        // Step 3: Load snapshot back
        println!("\n📂 Loading snapshot from file...");
        let loaded_snapshot = load_snapshot(&snapshot_path).expect("Failed to load snapshot");

        // Step 4: Verify round-trip equality
        println!("\n🔄 Verifying round-trip serialization...");

        assert!(saved_snapshot == loaded_snapshot, "Snapshot should be equal after round-trip");
        
        // // Assert metadata
        // assert_eq!(
        //     saved_snapshot.metadata, loaded_snapshot.metadata,
        //     "Metadata should be equal"
        // );

        // // Assert protocols
        // assert_eq!(
        //     saved_snapshot.protocols, loaded_snapshot.protocols,
        //     "Protocols should be equal"
        // );

        // // Assert tokens
        // assert_eq!(
        //     saved_snapshot.tokens, loaded_snapshot.tokens,
        //     "Tokens should be equal"
        // );

        // // Assert FeedMessage state_msgs
        // for (protocol, saved_state_msg) in &saved_snapshot.feed_message.state_msgs {
        //     let loaded_state_msg = loaded_snapshot
        //         .feed_message
        //         .state_msgs
        //         .get(protocol)
        //         .unwrap_or_else(|| panic!("Missing state message for protocol: {}", protocol));

        //     // Compare header
        //     assert_eq!(
        //         saved_state_msg.header, loaded_state_msg.header,
        //         "StateSyncMessage header for {} should be equal", protocol
        //     );

        //     // Compare deltas
        //     assert_eq!(
        //         saved_state_msg.deltas, loaded_state_msg.deltas,
        //         "StateSyncMessage deltas for {} should be equal", protocol
        //     );

        //     // Compare removed_components
        //     assert_eq!(
        //         saved_state_msg.removed_components, loaded_state_msg.removed_components,
        //         "StateSyncMessage removed_components for {} should be equal", protocol
        //     );

        //     // Compare snapshots.states
        //     let saved_states = &saved_state_msg.snapshots.states;
        //     let loaded_states = &loaded_state_msg.snapshots.states;
        //     assert_eq!(
        //         saved_states.len(), loaded_states.len(),
        //         "StateSyncMessage snapshots.states length for {} should be equal ({} vs {})",
        //         protocol, saved_states.len(), loaded_states.len()
        //     );
        //     let mut saved_states_vec: Vec<_> = saved_states.iter().collect();
        //     let mut loaded_states_vec: Vec<_> = loaded_states.iter().collect();
        //     saved_states_vec.sort_by_key(|(k, _)| *k);
        //     loaded_states_vec.sort_by_key(|(k, _)| *k);
        //     for ((saved_key, saved_cws), (loaded_key, loaded_cws)) in saved_states_vec.iter().zip(loaded_states_vec.iter()) {
        //         assert_eq!(saved_key, loaded_key, "State key mismatch for {}", protocol);

        //         // Compare ComponentWithState fields individually
        //         assert_eq!(
        //             saved_cws.component, loaded_cws.component,
        //             "ComponentWithState.component mismatch for {} key {}", protocol, saved_key
        //         );
        //         assert_eq!(
        //             saved_cws.component_tvl, loaded_cws.component_tvl,
        //             "ComponentWithState.component_tvl mismatch for {} key {}", protocol, saved_key
        //         );
        //         // Compare entrypoints (sorted by external_id)
        //         let mut saved_eps = saved_cws.entrypoints.clone();
        //         let mut loaded_eps = loaded_cws.entrypoints.clone();
        //         saved_eps.sort_by(|a, b| a.0.entry_point.external_id.cmp(&b.0.entry_point.external_id));
        //         loaded_eps.sort_by(|a, b| a.0.entry_point.external_id.cmp(&b.0.entry_point.external_id));
        //         assert_eq!(
        //             saved_eps.len(), loaded_eps.len(),
        //             "ComponentWithState.entrypoints length mismatch for {} key {}", protocol, saved_key
        //         );
        //         for (idx, (saved_ep, loaded_ep)) in saved_eps.iter().zip(loaded_eps.iter()).enumerate() {
        //             // Compare EntryPointWithTracingParams
        //             assert_eq!(
        //                 saved_ep.0, loaded_ep.0,
        //                 "EntryPointWithTracingParams mismatch for {} key {} entrypoint {}", protocol, saved_key, idx
        //             );
        //             // Compare TracingResult
        //             assert_eq!(
        //                 saved_ep.1.retriggers.len(), loaded_ep.1.retriggers.len(),
        //                 "TracingResult.retriggers length mismatch for {} key {} entrypoint {}: saved={}, loaded={}",
        //                 protocol, saved_key, idx, saved_ep.1.retriggers.len(), loaded_ep.1.retriggers.len()
        //             );
        //             assert_eq!(
        //                 saved_ep.1.accessed_slots.len(), loaded_ep.1.accessed_slots.len(),
        //                 "TracingResult.accessed_slots length mismatch for {} key {} entrypoint {}: saved={}, loaded={}",
        //                 protocol, saved_key, idx, saved_ep.1.accessed_slots.len(), loaded_ep.1.accessed_slots.len()
        //             );
        //             assert_eq!(
        //                 saved_ep.1, loaded_ep.1,
        //                 "TracingResult mismatch for {} key {} entrypoint {}", protocol, saved_key, idx
        //             );
        //         }

        //         // Compare ResponseProtocolState fields
        //         assert_eq!(
        //             saved_cws.state.component_id, loaded_cws.state.component_id,
        //             "ResponseProtocolState.component_id mismatch for {} key {}", protocol, saved_key
        //         );
        //         assert_eq!(
        //             saved_cws.state.balances, loaded_cws.state.balances,
        //             "ResponseProtocolState.balances mismatch for {} key {}", protocol, saved_key
        //         );

        //         // Compare attributes individually
        //         let saved_attrs = &saved_cws.state.attributes;
        //         let loaded_attrs = &loaded_cws.state.attributes;
        //         assert_eq!(
        //             saved_attrs.len(), loaded_attrs.len(),
        //             "Attributes length mismatch for {} key {}: saved has {}, loaded has {}",
        //             protocol, saved_key, saved_attrs.len(), loaded_attrs.len()
        //         );
        //         for (attr_key, saved_attr_val) in saved_attrs {
        //             let loaded_attr_val = loaded_attrs.get(attr_key).unwrap_or_else(|| {
        //                 panic!("Missing attribute {} for {} key {}", attr_key, protocol, saved_key)
        //             });
        //             assert_eq!(
        //                 saved_attr_val, loaded_attr_val,
        //                 "Attribute {} mismatch for {} key {}", attr_key, protocol, saved_key
        //             );
        //         }
        //     }

        //     // Compare snapshots.vm_storage
        //     let saved_vm = &saved_state_msg.snapshots.vm_storage;
        //     let loaded_vm = &loaded_state_msg.snapshots.vm_storage;
        //     assert_eq!(
        //         saved_vm.len(), loaded_vm.len(),
        //         "StateSyncMessage snapshots.vm_storage length for {} should be equal ({} vs {})",
        //         protocol, saved_vm.len(), loaded_vm.len()
        //     );
        //     let mut saved_vm_vec: Vec<_> = saved_vm.iter().collect();
        //     let mut loaded_vm_vec: Vec<_> = loaded_vm.iter().collect();
        //     saved_vm_vec.sort_by_key(|(k, _)| *k);
        //     loaded_vm_vec.sort_by_key(|(k, _)| *k);
        //     for ((saved_key, saved_val), (loaded_key, loaded_val)) in saved_vm_vec.iter().zip(loaded_vm_vec.iter()) {
        //         assert_eq!(saved_key, loaded_key, "VM storage key mismatch for {}", protocol);
        //         assert_eq!(saved_val, loaded_val, "VM storage value mismatch for {} key {:?}", protocol, saved_key);
        //     }
        // }

        // // Assert FeedMessage sync_states
        // for (key, saved_sync_state) in &saved_snapshot.feed_message.sync_states {
        //     let loaded_sync_state = loaded_snapshot
        //         .feed_message
        //         .sync_states
        //         .get(key)
        //         .unwrap_or_else(|| panic!("Missing sync state for key: {}", key));
        //     assert_eq!(
        //         saved_sync_state, loaded_sync_state,
        //         "SynchronizerState for {} should be equal", key
        //     );
        // }

        // println!("   ✅ Round-trip verification passed!");

        // // Step 5: Process snapshot into protocol states
        // println!("\n⚙️ Processing snapshot into protocol states...");
        // let loaded = process_snapshot(&loaded_snapshot)
        //     .await
        //     .expect("Failed to process snapshot");

        // println!("\n✅ Snapshot processed!");
        // println!("   States decoded: {}", loaded.states.len());
        // println!("   Components: {}", loaded.components.len());

        // assert!(
        //     !loaded.states.is_empty(),
        //     "Should have decoded at least one state"
        // );

        // println!("\n✅ All roundtrip tests passed!");
    }
}
