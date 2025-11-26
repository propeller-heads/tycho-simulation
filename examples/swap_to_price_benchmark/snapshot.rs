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
//! use swap_to_price::snapshot::{save_snapshot, load_snapshot};
//! use tycho_common::models::Chain;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Capture snapshot
//!     let protocols = vec!["uniswap_v2", "uniswap_v3"];
//!     save_snapshot(
//!         "tycho-beta.propellerheads.xyz",
//!         Some("your_key"),
//!         Chain::Ethereum,
//!         &protocols,
//!         100.0, // min TVL
//!         Path::new("snapshot.json")
//!     ).await?;
//!
//!     // Restore snapshot - everything is self-contained!
//!     let loaded = load_snapshot(Path::new("snapshot.json")).await?;
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
#[derive(Serialize, Deserialize, Clone)]
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
#[derive(Serialize, Deserialize, Clone)]
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
            info!("  {}: {} pools (no limit applied)", protocol, original_count);
        }
    }
}

/// Captures a snapshot by connecting to Tycho and saving it to a JSON file.
///
/// This function creates a `TychoStreamBuilder`, subscribes to the specified protocols,
/// receives the first message, wraps it in a `Snapshot` structure with all metadata,
/// and saves it as JSON with filename: `snapshot_<block_number>.json`
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
    info!("Connecting to Tycho at {}...", tycho_url);

    // Create TychoStreamBuilder
    let mut stream_builder = TychoStreamBuilder::new(tycho_url, chain.into());

    // Add protocols with TVL filter
    let tvl_filter = ComponentFilter::with_tvl_range(min_tvl * 0.9, min_tvl);
    for protocol in protocols {
        info!("Adding protocol: {}", protocol);
        stream_builder = stream_builder.exchange(protocol, tvl_filter.clone());
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
    let mut feed_msg = feed_msg_result.map_err(|e| format!("Stream error: {}", e))?;

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

    info!("Received FeedMessage for block {}", block_number);
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

    // Serialize to MessagePack (faster than JSON, good serde compatibility)
    info!("Saving snapshot to {}...", output_path.display());
    let file = File::create(&output_path)?;
    rmp_serde::encode::write(&mut std::io::BufWriter::new(file), &snapshot)?;

    info!("Snapshot saved successfully!");

    // Shutdown the stream
    handle.abort();

    Ok((snapshot, output_path))
}

/// Loads a snapshot from bincode and decodes it into protocol states.
///
/// This function deserializes a `Snapshot` from bincode, uses the embedded tokens and protocols
/// to configure a decoder, and returns all decoded states ready to use.
///
/// # Arguments
/// * `snapshot_path` - Path to the snapshot .bin file
///
/// # Returns
/// `LoadedSnapshot` containing decoded states, components, and metadata
pub async fn load_snapshot(
    snapshot_path: &Path,
) -> Result<LoadedSnapshot, Box<dyn std::error::Error>> {
    info!("Loading snapshot from {}...", snapshot_path.display());

    // Deserialize Snapshot from bincode (much faster than JSON)
    let file = File::open(snapshot_path)?;
    let snapshot: Snapshot = bincode::deserialize_from(file)?;

    info!("Loaded snapshot:");
    info!("  Block: {}", snapshot.metadata.block_number);
    info!("  Protocols: {}", snapshot.protocols.join(", "));
    info!("  Tokens: {}", snapshot.tokens.len());
    info!("  Components: {}", snapshot.metadata.total_components);

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
        metadata: snapshot.metadata,
    })
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
    #[ignore] // Run with: cargo test --example swap_to_price -- --ignored --nocapture
    async fn test_create_and_load_snapshot() {
        // Initialize tracing for test output
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        // Get Tycho credentials from environment
        let tycho_url = env::var("TYCHO_URL")
            .unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
        let api_key = env::var("TYCHO_AUTH_KEY")
            .ok();

        if api_key.is_none() {
            eprintln!("⚠️  Warning: TYCHO_AUTH_KEY not set. Test may fail due to rate limiting.");
            eprintln!("   Set it with: export TYCHO_AUTH_KEY=your_key");
        }

        let protocols = vec!["uniswap_v2", "uniswap_v3"];
        let chain = Chain::Ethereum;
        let min_tvl = 100.0;


        println!("\n📸 Creating snapshot from live Tycho stream...");
        println!("   URL: {}", tycho_url);
        println!("   Protocols: {}", protocols.join(", "));
        println!("   Min TVL: {} ETH", min_tvl);

        // Create temp directory for test
        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

        // Step 1: Create snapshot from live stream
        let (saved_snapshot, snapshot_path) = save_snapshot(
            &tycho_url,
            api_key.as_deref(),
            chain,
            &protocols,
            min_tvl,
            None, // No token filter for test
            temp_dir.path(),
            10,   // Default pool limit for test
        )
        .await
        .expect("Failed to create snapshot");

        println!("\n✅ Snapshot created successfully!");
        println!("   File: {}", snapshot_path.display());
        println!("   Block: {}", saved_snapshot.metadata.block_number);
        println!("   Components: {}", saved_snapshot.metadata.total_components);
        println!("   Tokens: {}", saved_snapshot.tokens.len());
        println!("   Protocols: {}", saved_snapshot.protocols.join(", "));

        // Verify snapshot structure
        assert!(!saved_snapshot.protocols.is_empty(), "Protocols should not be empty");
        assert!(
            !saved_snapshot.tokens.is_empty(),
            "Tokens should not be empty (found {})",
            saved_snapshot.tokens.len()
        );
        assert!(
            saved_snapshot.metadata.total_components > 0,
            "Should have at least one component (found {})",
            saved_snapshot.metadata.total_components
        );
        assert!(saved_snapshot.metadata.block_number > 0, "Block number should be non-zero");

        println!("\n📂 Loading snapshot from file...");

        // Step 2: Load snapshot from file
        let loaded = load_snapshot(&snapshot_path)
            .await
            .expect("Failed to load snapshot");

        println!("\n✅ Snapshot loaded successfully!");
        println!("   States decoded: {}", loaded.states.len());
        println!("   Components: {}", loaded.components.len());

        // Verify loaded snapshot
        assert!(!loaded.states.is_empty(), "Should have decoded at least one state");
        assert_eq!(
            loaded.metadata.block_number,
            saved_snapshot.metadata.block_number,
            "Block numbers should match"
        );

        // Verify we can use the states
        println!("\n🔍 Testing decoded states:");
        for (pool_id, state) in loaded.states.iter().take(3) {
            let component = loaded.components.get(pool_id)
                .expect("Component should exist for every state");

            let token_symbols: Vec<_> = component.tokens.iter()
                .map(|t| t.symbol.as_str())
                .collect();

            println!("   {} - {}", component.protocol_system, token_symbols.join("/"));

            // Verify we can call ProtocolSim methods
            let fee = state.fee();
            assert!(fee >= 0.0 && fee <= 1.0, "Fee should be between 0 and 1");

            // Try to get spot price if it's a 2-token pool
            if component.tokens.len() == 2 {
                let spot_result = state.spot_price(
                    &component.tokens[0],
                    &component.tokens[1],
                );

                match spot_result {
                    Ok(price) => {
                        println!("     Spot price: {} {}/{}", price, component.tokens[1].symbol, component.tokens[0].symbol);
                        assert!(price > 0.0, "Spot price should be positive");
                    }
                    Err(e) => {
                        println!("     Spot price unavailable: {:?}", e);
                    }
                }
            }
        }

        println!("\n✅ All tests passed! Snapshot system is working correctly.");
    }
}
