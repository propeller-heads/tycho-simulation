use std::{collections::HashMap, sync::Arc};

use alloy::primitives::B256;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tycho_client::feed::FeedMessage;

use crate::evm::stream::ProtocolStreamExtension;

/// Result type for external data source operations
pub type ExternalDataResult<T> = Result<T, ExternalDataError>;

/// Error types for external data source operations
#[derive(Debug, thiserror::Error)]
pub enum ExternalDataError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Data parsing failed: {0}")]
    DataParsingFailed(String),
    #[error("Block synchronization failed: {0}")]
    BlockSyncFailed(String),
    #[error("State management error: {0}")]
    StateManagementError(String),
    #[error("External source error: {0}")]
    ExternalSourceError(String),
}

/// Core interface for external data sources that provide FeedMessage updates.
///
/// This trait is the main integration point for external teams wanting to provide
/// off-chain data to the Tycho-Simulation. Sources can be either pull-based
/// (oracle-style, called once per block) or push-based (continuous updates with
/// revert capability).
///
/// ## Integration Flow
/// 1. Implement this trait for your data source
/// 2. Wrap in `ConfiguredExternalDataSource` or `ConfiguredPushDataSource`
/// 3. Convert to `ProtocolStreamExtension` via `.into()`
/// 4. Add to `ProtocolStreamBuilder` with `add_stream_extension()`
///
/// ## Usage Patterns
/// - **Pull-based**: Oracle feeds, governance data, periodic price updates
/// - **Push-based**: High-frequency speculative updates (implement `PushDataSource` too)
#[async_trait]
pub trait ExternalDataSource: Send + Sync {
    /// Get updates for a specific block, returning a complete FeedMessage.
    ///
    /// **For pull-based sources**: Generate updates based on block info and context.
    /// You have full control over the FeedMessage structure and can target specific
    /// protocol components. Return `None` if no updates are needed for this block.
    ///
    /// **For push-based sources**: This is called for revert checking only. Compare
    /// the context (actual on-chain changes) against your speculative state and
    /// return revert updates if your predictions were wrong.
    ///
    /// # Parameters
    /// - `block_number`: The block number to provide updates for
    /// - `block_hash`: The canonical block hash from Tycho's main stream
    /// - `context`: Current block's FeedMessage for context or revert checking
    ///
    /// # Returns
    /// - `Some(FeedMessage)`: Updates to emit downstream
    /// - `None`: No updates needed for this block
    ///
    /// # Important Notes
    /// - You must construct complete, valid FeedMessages
    /// - Target specific protocol components in your message
    /// - Handle errors gracefully - failures are logged but don't stop processing
    async fn get_updates_for_block(
        &mut self,
        block_number: u64,
        block_hash: B256,
        context: &FeedMessage,
    ) -> ExternalDataResult<Option<FeedMessage>>;

    /// Human-readable identifier for this data source (used in logs and metrics).
    ///
    /// Should be unique and descriptive, e.g., "chainlink_eth_usd", "angstrom_v4_hooks"
    fn source_identifier(&self) -> &str;

    /// Extract block number and hash for your target protocol from a FeedMessage.
    ///
    /// This method is critical for synchronization - it tells the system which block
    /// this FeedMessage represents so it can call `get_updates_for_block()` with the
    /// correct parameters and make buffering decisions for push-based sources.
    ///
    /// **Implementation depends on your target protocol:**
    /// - Uniswap V4 hook source: extract block number from uniswap v4 header
    /// - Multi-protocol source (e.g. Pyth): might want to use protocol with the latest block
    ///
    /// # Parameters
    /// - `feed_message`: The FeedMessage to extract block info from
    ///
    /// # Returns
    /// - `Ok((block_number, block_hash))`: Successfully extracted block info
    /// - `Err(ExternalDataError)`: Failed to extract - message will be skipped
    ///
    /// # Important Notes
    /// - Must be consistent with your target protocol's message structure
    /// - Errors here will cause the entire message to be skipped
    /// - This drives all synchronization logic for your source
    fn extract_block_info(
        &self,
        feed_message: &FeedMessage,
    ) -> Result<(u64, B256), ExternalDataError>;
}

/// Extension trait for push-based sources that emit continuous speculative updates.
///
/// Implement this trait in addition to `ExternalDataSource` for sources that emit
/// high-frequency, speculative updates that may arrive multiple times per block.
/// This is typically used for sources like Angstrom that predict state changes
/// before they happen on-chain.
///
/// ## How Push Sources Work
/// 1. `start_push_stream()` is called once to begin continuous emission
/// 2. Your source emits `SpeculativeUpdate`s via the provided channel
/// 3. Updates are buffered/emitted based on your configured synchronization strategy
/// 4. `get_updates_for_block()` is called for revert checking when blocks arrive
///
/// ## Synchronization Strategies
/// - **Conservative**: Keep all updates, emit only for exact next block
/// - **LatestOnly**: Keep only latest update per block, emit only for exact next block
///
/// ## Error Handling
/// - Errors from `start_push_stream()` stop the entire source
/// - Individual update failures should be handled gracefully within your implementation
/// - Connection failures should trigger automatic reconnection logic
#[async_trait]
pub trait PushDataSource: ExternalDataSource {
    /// Start continuous emission of speculative updates to the provided channel.
    ///
    /// This method should:
    /// 1. Establish connections (WebSockets, gRPC streams, etc.)
    /// 2. Spawn background tasks for continuous data emission
    /// 3. Handle reconnection logic and error recovery
    /// 4. Emit `SpeculativeUpdate`s via the `update_sender` channel
    ///
    /// The method should return immediately after starting background tasks.
    /// The actual data emission happens asynchronously in the background.
    ///
    /// # Parameters
    /// - `update_sender`: Channel to send `SpeculativeUpdate`s through
    ///
    /// # Returns
    /// - `Ok(())`: Successfully started background emission
    /// - `Err(ExternalDataError)`: Failed to start - entire source will be disabled
    ///
    /// # Important Notes
    /// - Must handle connection failures and reconnection automatically
    /// - Should emit updates with realistic `expected_block` values
    /// - Individual update failures should not stop the entire stream
    /// - Use `SPECULATIVE_BLOCK_HASH` for speculative updates
    async fn start_push_stream(
        &mut self,
        update_sender: mpsc::Sender<SpeculativeUpdate>,
    ) -> ExternalDataResult<()>;
}

/// Speculative update that may or may not materialize on-chain.
#[derive(Debug)]
pub struct SpeculativeUpdate {
    pub feed_message: FeedMessage,
    pub expected_block: u64,
}

/// Synchronization action for speculative updates.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncAction {
    Emit,
    Buffer,
    Drop,
}

/// Strategy for synchronizing speculative updates with Tycho's block progression.
pub trait BlockSynchronizationStrategy: Send + Sync {
    /// Determine action for a speculative update.
    fn should_emit(&mut self, current_tycho_block: Option<u64>, update_block: u64) -> SyncAction;

    /// Merge/filter multiple buffered updates for the same block.
    fn merge_buffered_updates(
        &mut self,
        block_number: u64,
        updates: Vec<SpeculativeUpdate>,
    ) -> Vec<SpeculativeUpdate>;

    /// Called when a new Tycho block arrives.
    fn on_tycho_block(&mut self, block_number: u64);

    /// Strategy identifier for logging.
    fn strategy_name(&self) -> &'static str;
}

/// Emit for exact next block only, keep all updates.
#[derive(Debug, Default)]
pub struct ConservativeSyncStrategy;

/// Emit for exact next block only, keep latest update per block.
#[derive(Debug, Default)]
pub struct LatestOnlyStrategy;

impl BlockSynchronizationStrategy for ConservativeSyncStrategy {
    fn should_emit(&mut self, current_tycho_block: Option<u64>, update_block: u64) -> SyncAction {
        match current_tycho_block {
            None => SyncAction::Buffer,
            Some(current) => {
                if update_block < current {
                    SyncAction::Drop
                } else if update_block == current + 1 {
                    SyncAction::Emit
                } else {
                    SyncAction::Buffer
                }
            }
        }
    }

    fn merge_buffered_updates(
        &mut self,
        _block_number: u64,
        updates: Vec<SpeculativeUpdate>,
    ) -> Vec<SpeculativeUpdate> {
        updates
    }

    fn on_tycho_block(&mut self, _block_number: u64) {}

    fn strategy_name(&self) -> &'static str {
        "Conservative"
    }
}

impl BlockSynchronizationStrategy for LatestOnlyStrategy {
    fn should_emit(&mut self, current_tycho_block: Option<u64>, update_block: u64) -> SyncAction {
        match current_tycho_block {
            None => SyncAction::Buffer,
            Some(current) => {
                if update_block < current {
                    SyncAction::Drop
                } else if update_block == current + 1 {
                    SyncAction::Emit
                } else {
                    SyncAction::Buffer
                }
            }
        }
    }

    fn merge_buffered_updates(
        &mut self,
        _block_number: u64,
        updates: Vec<SpeculativeUpdate>,
    ) -> Vec<SpeculativeUpdate> {
        if updates.is_empty() {
            return updates;
        }
        vec![updates.into_iter().next_back().unwrap()]
    }

    fn on_tycho_block(&mut self, _block_number: u64) {}

    fn strategy_name(&self) -> &'static str {
        "LatestOnly"
    }
}

/// Component that synchronizes speculative updates with Tycho's block progression
pub struct BlockSynchronizer {
    strategy: Box<dyn BlockSynchronizationStrategy>,
    current_tycho_block: Option<u64>,
    buffered_updates: HashMap<u64, Vec<SpeculativeUpdate>>,
    max_buffer_size: usize,
}

impl BlockSynchronizer {
    pub fn new(strategy: Box<dyn BlockSynchronizationStrategy>) -> Self {
        Self {
            strategy,
            current_tycho_block: None,
            buffered_updates: HashMap::new(),
            max_buffer_size: 1000, // Prevent unbounded memory growth
        }
    }

    pub fn with_max_buffer_size(mut self, max_size: usize) -> Self {
        self.max_buffer_size = max_size;
        self
    }

    /// Process a speculative update and return the action to take
    pub fn process_speculative_update(
        &mut self,
        update: SpeculativeUpdate,
    ) -> (SyncAction, Vec<SpeculativeUpdate>) {
        let action = self
            .strategy
            .should_emit(self.current_tycho_block, update.expected_block);

        match action {
            SyncAction::Emit => (SyncAction::Emit, vec![update]),
            SyncAction::Buffer => {
                // Check buffer size limits
                let total_buffered: usize = self
                    .buffered_updates
                    .values()
                    .map(|v| v.len())
                    .sum();
                if total_buffered >= self.max_buffer_size {
                    tracing::warn!(
                        "Buffer size limit reached ({}), dropping speculative update for block {}",
                        self.max_buffer_size,
                        update.expected_block
                    );
                    (SyncAction::Drop, vec![])
                } else {
                    // Add to buffer and let strategy merge if needed
                    let block_number = update.expected_block;
                    self.buffered_updates
                        .entry(block_number)
                        .or_default()
                        .push(update);

                    // Get all updates for this block and let strategy merge them
                    let block_updates = self
                        .buffered_updates
                        .remove(&block_number)
                        .unwrap_or_default();
                    let merged_updates = self
                        .strategy
                        .merge_buffered_updates(block_number, block_updates);

                    // Store the merged result back
                    if !merged_updates.is_empty() {
                        self.buffered_updates
                            .insert(block_number, merged_updates);
                    }

                    (SyncAction::Buffer, vec![])
                }
            }
            SyncAction::Drop => {
                tracing::debug!(
                    "Dropping stale speculative update for block {} (current Tycho block: {:?})",
                    update.expected_block,
                    self.current_tycho_block
                );
                (SyncAction::Drop, vec![])
            }
        }
    }

    /// Process a new Tycho block and return any buffered updates that should now be emitted
    pub fn process_tycho_block(&mut self, block_number: u64) -> Vec<SpeculativeUpdate> {
        self.current_tycho_block = Some(block_number);
        self.strategy
            .on_tycho_block(block_number);

        // Check buffered updates to see if any should now be emitted
        let mut to_emit = Vec::new();
        let mut blocks_to_remove = Vec::new();

        for (&buffered_block, updates) in &mut self.buffered_updates {
            // Take all updates for this block and let strategy process them
            let block_updates = std::mem::take(updates);
            let mut remaining_updates = Vec::new();
            let mut updates_to_emit = Vec::new();

            // Re-evaluate each update individually first
            for update in block_updates {
                let action = self
                    .strategy
                    .should_emit(self.current_tycho_block, update.expected_block);
                match action {
                    SyncAction::Emit => updates_to_emit.push(update),
                    SyncAction::Buffer => remaining_updates.push(update),
                    SyncAction::Drop => {
                        tracing::debug!(
                            "Dropping previously buffered update for block {}",
                            update.expected_block
                        );
                    }
                }
            }

            // Let strategy merge the updates that should be emitted
            if !updates_to_emit.is_empty() {
                let merged_to_emit = self
                    .strategy
                    .merge_buffered_updates(buffered_block, updates_to_emit);
                to_emit.extend(merged_to_emit);
            }

            // Let strategy merge the updates that should remain buffered
            if !remaining_updates.is_empty() {
                let merged_remaining = self
                    .strategy
                    .merge_buffered_updates(buffered_block, remaining_updates);
                if merged_remaining.is_empty() {
                    blocks_to_remove.push(buffered_block);
                } else {
                    *updates = merged_remaining;
                }
            } else {
                blocks_to_remove.push(buffered_block);
            }
        }

        // Clean up empty entries
        for block in blocks_to_remove {
            self.buffered_updates.remove(&block);
        }

        // Clean up very old buffered updates to prevent memory leaks
        if let Some(current) = self.current_tycho_block {
            let cutoff = current.saturating_sub(self.max_buffer_size as u64);
            self.buffered_updates
                .retain(|&block, _| block > cutoff);
        }

        to_emit
    }
}

/// Configuration for external data sources.
#[derive(Debug, Clone)]
pub struct ExternalDataSourceConfig {
    pub source_id: String,
    /// For push-based sources only
    pub sync_strategy: Option<SyncStrategyConfig>,
    /// For push-based sources only  
    pub max_buffer_size: Option<usize>,
}

/// Synchronization strategies for push-based sources.
#[derive(Debug, Clone)]
pub enum SyncStrategyConfig {
    Conservative,
    LatestOnly,
}

/// External data source with configuration.
pub struct ConfiguredExternalDataSource<T> {
    pub source: T,
    pub config: ExternalDataSourceConfig,
}

/// Push-based external data source with configuration.
pub struct ConfiguredPushDataSource<T> {
    pub source: T,
    pub config: ExternalDataSourceConfig,
}

impl SyncStrategyConfig {
    pub fn create_strategy(&self) -> Box<dyn BlockSynchronizationStrategy> {
        match self {
            SyncStrategyConfig::Conservative => Box::new(ConservativeSyncStrategy),
            SyncStrategyConfig::LatestOnly => Box::new(LatestOnlyStrategy),
        }
    }
}

impl<T> From<ConfiguredExternalDataSource<T>> for ProtocolStreamExtension
where
    T: ExternalDataSource + 'static,
{
    /// Convert external data source into a ProtocolStreamExtension.
    ///
    /// Creates channels and spawns a background task that:
    /// 1. Receives main stream messages via the tx channel
    /// 2. Calls the external data source to get updates for each block
    /// 3. Emits FeedMessages via the rx channel
    fn from(mut source: ConfiguredExternalDataSource<T>) -> Self {
        let (tx, mut main_rx) = mpsc::channel::<Arc<FeedMessage>>(100);
        let (ext_tx, ext_rx) = mpsc::channel::<FeedMessage>(100);

        tokio::spawn(async move {
            while let Some(feed_message) = main_rx.recv().await {
                if let Ok((block_number, block_hash)) = source
                    .source
                    .extract_block_info(&feed_message)
                {
                    match source
                        .source
                        .get_updates_for_block(block_number, block_hash, &feed_message)
                        .await
                    {
                        Ok(Some(update_message)) => {
                            let _ = ext_tx.send(update_message).await;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::warn!(
                                "External data source '{}' failed for block {}: {}",
                                source.config.source_id,
                                block_number,
                                e
                            );
                        }
                    }
                }
            }
        });

        ProtocolStreamExtension { tx: Some(tx), rx: ext_rx }
    }
}

impl<T> From<ConfiguredPushDataSource<T>> for ProtocolStreamExtension
where
    T: PushDataSource + 'static,
{
    /// Convert push-based data source into a ProtocolStreamExtension.
    ///
    /// Creates a background system that handles:
    /// 1. Continuous speculative updates from the external source
    /// 2. Block synchronization to ensure proper emission timing
    /// 3. Revert checking when new blocks arrive
    /// 4. Coordination between speculative updates and block progression
    fn from(mut source: ConfiguredPushDataSource<T>) -> Self {
        let (tx, mut main_rx) = mpsc::channel::<Arc<FeedMessage>>(100);
        let (ext_tx, ext_rx) = mpsc::channel::<FeedMessage>(100);

        let (spec_update_tx, mut spec_update_rx) = mpsc::channel::<SpeculativeUpdate>(1000);
        let source_id = source.config.source_id.clone();
        let strategy_config = source
            .config
            .sync_strategy
            .as_ref()
            .unwrap_or(&SyncStrategyConfig::Conservative);
        let strategy = strategy_config.create_strategy();
        let max_buffer_size = source
            .config
            .max_buffer_size
            .unwrap_or(1000);
        let mut synchronizer =
            BlockSynchronizer::new(strategy).with_max_buffer_size(max_buffer_size);

        tracing::info!(
            "Initializing push source '{}' with {:?} strategy (max buffer: {})",
            source_id,
            strategy_config,
            max_buffer_size
        );

        tokio::spawn(async move {
            if let Err(e) = source
                .source
                .start_push_stream(spec_update_tx)
                .await
            {
                tracing::error!("Failed to start push stream for source '{}': {}", source_id, e);
                return;
            }

            loop {
                tokio::select! {
                    Some(spec_update) = spec_update_rx.recv() => {
                        let (action, updates_to_emit) = synchronizer.process_speculative_update(spec_update);
                        if let SyncAction::Emit = action {
                            for update in updates_to_emit {
                                let _ = ext_tx.send(update.feed_message).await;
                            }
                        }
                    }

                    Some(feed_message) = main_rx.recv() => {
                        if let Ok((block_number, block_hash)) = source.source.extract_block_info(&feed_message) {
                            let buffered_updates_to_emit = synchronizer.process_tycho_block(block_number);
                            for buffered_update in buffered_updates_to_emit {
                                let _ = ext_tx.send(buffered_update.feed_message).await;
                            }

                            match source.source.get_updates_for_block(block_number, block_hash, &feed_message).await {
                                Ok(Some(revert_message)) => {
                                    let _ = ext_tx.send(revert_message).await;
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    tracing::warn!(
                                        "Push source '{}' revert check failed for block {}: {}",
                                        source_id, block_number, e
                                    );
                                }
                            }
                        }
                    }

                    else => break,
                }
            }
        });

        ProtocolStreamExtension { tx: Some(tx), rx: ext_rx }
    }
}

/// Placeholder hash used for speculative updates that don't have real block hashes yet
pub const SPECULATIVE_BLOCK_HASH: B256 = B256::ZERO;

/// Helper to check if a block hash indicates a speculative update
pub fn is_speculative_block_hash(block_hash: B256) -> bool {
    block_hash == SPECULATIVE_BLOCK_HASH
}
