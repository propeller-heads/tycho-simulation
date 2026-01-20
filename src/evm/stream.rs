//! Builder for configuring a multi-protocol stream.
//!
//! Provides a builder for creating a multi-protocol stream that produces
//! [`protocol::models::Update`] messages. It runs one synchronization worker per protocol
//! and a supervisor that aggregates updates, ensuring gap‑free streaming
//! and robust state tracking.
//!
//! ## Context
//!
//! This stream wraps [`tycho_client::stream::TychoStream`]. It decodes `FeedMessage`s
//! into [`protocol::models::Update`]s. Internally, each protocol runs in its own
//! synchronization worker, and a supervisor aggregates their messages per block.
//!
//! ### Protocol Synchronization Worker
//! A synchronization worker runs the snapshot + delta protocol from `tycho-indexer`.
//! - It first downloads components and their snapshots.
//! - It then streams deltas.
//! - It reacts to new or paused components by pulling snapshots or removing them from the active
//!   set.
//!
//! Each worker emits snapshots and deltas to the supervisor.
//!
//! ### Stream Supervisor
//! The supervisor aggregates worker messages by block and assigns sync status.
//! - It ensures workers produce gap-free messages.
//! - It flags late workers as `Delayed`, and marks them `Stale` if they exceed `max_missed_blocks`.
//! - It marks workers with terminal errors as `Ended`.
//!
//! Aggregating by block adds small latency, since the supervisor waits briefly for
//! all workers to emit. This latency only applies to workers in `Ready` or `Delayed`.
//!
//! The stream ends only when **all** workers are `Stale` or `Ended`.
//!
//! ## Configuration
//!
//! The builder lets you customize:
//!
//! ### Protocols
//! Select which protocols to synchronize.
//!
//! ### Tokens & Minimum Token Quality
//! Provide an initial set of tokens of interest. The first message includes only
//! components whose tokens match this set. The stream adds new tokens automatically
//! when a component is deployed and its quality exceeds `min_token_quality`.
//!
//! ### StreamEndPolicy
//! Control when the stream ends based on worker states. By default, it ends when all
//! workers are `Stale` or `Ended`.
//!
//! ## Stream
//! The stream emits one [`protocol::models::Update`] every `block_time`. Each update
//! reports protocol synchronization states and any changes.
//!
//! The `new_components` field lists newly deployed components and their tokens.
//!
//! The stream aims to run indefinitely. Internal retry and reconnect logic handle
//! most errors, so users should rarely need to restart it manually.
//!
//! ## Example
//! ```no_run
//! use tycho_common::models::Chain;
//! use tycho_simulation::evm::stream::ProtocolStreamBuilder;
//! use tycho_simulation::utils::load_all_tokens;
//! use futures::StreamExt;
//! use tycho_client::feed::component_tracker::ComponentFilter;
//! use tycho_simulation::evm::protocol::uniswap_v2::state::UniswapV2State;
//!
//! #[tokio::main]
//! async fn main() {
//!     let all_tokens = load_all_tokens(
//!         "tycho-beta.propellerheads.xyz",
//!         false,
//!         Some("sampletoken"),
//!         true,
//!         Chain::Ethereum,
//!         None,
//!         None,
//!     )
//!     .await
//!     .expect("Failed loading tokens");
//!
//!     let mut protocol_stream =
//!         ProtocolStreamBuilder::new("tycho-beta.propellerheads.xyz", Chain::Ethereum)
//!             .auth_key(Some("sampletoken".to_string()))
//!             .skip_state_decode_failures(true)
//!             .exchange::<UniswapV2State>(
//!                 "uniswap_v2", ComponentFilter::with_tvl_range(5.0, 10.0), None
//!             )
//!             .set_tokens(all_tokens)
//!             .await
//!             .build()
//!             .await
//!             .expect("Failed building protocol stream");
//!
//!     // Loop through block updates
//!     while let Some(msg) = protocol_stream.next().await {
//!         dbg!(msg).expect("failed decoding");
//!     }
//! }
//! ```
use std::{collections::HashMap, sync::Arc, time};

use futures::{Stream, StreamExt};
use tokio::sync::mpsc::Receiver;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, warn};
use tycho_client::{
    feed::{
        component_tracker::ComponentFilter, synchronizer::ComponentWithState, BlockHeader,
        BlockSynchronizerError, FeedMessage, SynchronizerState,
    },
    stream::{RetryConfiguration, StreamError, TychoStreamBuilder},
};
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};

use crate::{
    evm::{
        decoder::{StreamDecodeError, TychoStreamDecoder},
        protocol::uniswap_v4::hooks::hook_handler_creator::initialize_hook_handlers,
    },
    protocol::{
        errors::InvalidSnapshotError,
        models::{DecoderContext, TryFromWithBlock, Update},
    },
};

const EXCHANGES_REQUIRING_FILTER: [&str; 2] = ["vm:balancer_v2", "vm:curve"];

#[derive(Default, Debug, Clone, Copy)]
pub enum StreamEndPolicy {
    /// End stream if all states are Stale or Ended (default)
    #[default]
    AllEndedOrStale,
    /// End stream if any protocol ended
    AnyEnded,
    /// End stream if any protocol ended or is stale
    AnyEndedOrStale,
    /// End stream if any protocol is stale
    AnyStale,
}

impl StreamEndPolicy {
    fn should_end<'a>(&self, states: impl IntoIterator<Item = &'a SynchronizerState>) -> bool {
        let mut it = states.into_iter();
        match self {
            StreamEndPolicy::AllEndedOrStale => false,
            StreamEndPolicy::AnyEnded => it.any(|s| matches!(s, SynchronizerState::Ended(_))),
            StreamEndPolicy::AnyStale => it.any(|s| matches!(s, SynchronizerState::Stale(_))),
            StreamEndPolicy::AnyEndedOrStale => {
                it.any(|s| matches!(s, SynchronizerState::Stale(_) | SynchronizerState::Ended(_)))
            }
        }
    }
}

/// The source of feed messages for the protocol stream.
enum StreamSource {
    /// Live connection to Tycho server
    Tycho(TychoStreamBuilder),
    /// Pre-built receiver (for testing or custom sources)
    Receiver(Receiver<Result<FeedMessage<BlockHeader>, BlockSynchronizerError>>),
}

/// Builds and configures the multi protocol stream described in the [module-level docs](self).
///
/// See the module documentation for details on protocols, configuration options, and
/// stream behavior.
pub struct ProtocolStreamBuilder {
    decoder: TychoStreamDecoder<BlockHeader>,
    source: StreamSource,
    stream_end_policy: StreamEndPolicy,
}

impl ProtocolStreamBuilder {
    /// Creates a new builder for a multi-protocol stream.
    ///
    /// See the [module-level docs](self) for full details on stream behavior and configuration.
    pub fn new(tycho_url: &str, chain: Chain) -> Self {
        Self {
            decoder: TychoStreamDecoder::new(),
            source: StreamSource::Tycho(TychoStreamBuilder::new(tycho_url, chain.into())),
            stream_end_policy: StreamEndPolicy::default(),
        }
    }

    /// Creates a new builder from a pre-built receiver.
    ///
    /// This is useful for testing or custom feed sources. Tycho-client specific configuration
    /// methods (like `block_time`, `auth_key`, etc.) will have no effect when using this
    /// constructor.
    ///
    /// # Example
    /// ```ignore
    /// use tokio::sync::mpsc;
    /// use tycho_simulation::evm::stream::ProtocolStreamBuilder;
    /// use tycho_simulation::evm::protocol::uniswap_v2::state::UniswapV2State;
    /// use tycho_client::feed::component_tracker::ComponentFilter;
    ///
    /// let (tx, rx) = mpsc::channel(100);
    /// // Send test messages via tx...
    ///
    /// let stream = ProtocolStreamBuilder::from_receiver(rx)
    ///     .exchange::<UniswapV2State>("uniswap_v2", ComponentFilter::default(), None)
    ///     .build()
    ///     .await?;
    /// ```
    pub fn from_receiver(
        rx: Receiver<Result<FeedMessage<BlockHeader>, BlockSynchronizerError>>,
    ) -> Self {
        Self {
            decoder: TychoStreamDecoder::new(),
            source: StreamSource::Receiver(rx),
            stream_end_policy: StreamEndPolicy::default(),
        }
    }

    /// Helper to apply configuration only when source is Tycho.
    /// Returns self unchanged if source is a Receiver.
    fn with_tycho(mut self, f: impl FnOnce(TychoStreamBuilder) -> TychoStreamBuilder) -> Self {
        if let StreamSource::Tycho(builder) = self.source {
            self.source = StreamSource::Tycho(f(builder));
        } else {
            warn!("with_tycho() called but source is not Tycho. Ignoring configuration.");
        }
        self
    }

    /// Adds a specific exchange to the stream.
    ///
    /// This configures the builder to include a new protocol synchronizer for `name`,
    /// filtering its components according to `filter` and optionally `filter_fn`.
    ///
    /// The type parameter `T` specifies the decoder type for this exchange. All
    /// component states for this exchange will be decoded into instances of `T`.
    ///
    /// # Parameters
    ///
    /// - `name`: The protocol or exchange name (e.g., `"uniswap_v4"`, `"vm:balancer_v2"`).
    /// - `filter`: Defines the set of components to include in the stream. Ignored when using
    ///   `from_receiver()`.
    /// - `filter_fn`: Optional custom filter function for client-side filtering of components not
    ///   expressible in `filter`.
    ///
    /// # Notes
    ///
    /// For certain protocols (e.g., `"uniswap_v4"`, `"vm:balancer_v2"`, `"vm:curve"`), omitting
    /// `filter_fn` may cause decoding errors or incorrect results. In these cases, a proper
    /// filter function is required to ensure correct decoding and quoting logic.
    pub fn exchange<T>(
        mut self,
        name: &str,
        filter: ComponentFilter,
        filter_fn: Option<fn(&ComponentWithState) -> bool>,
    ) -> Self
    where
        T: ProtocolSim
            + TryFromWithBlock<ComponentWithState, BlockHeader, Error = InvalidSnapshotError>
            + Send
            + 'static,
    {
        // Register with TychoStreamBuilder only for live connections
        if let StreamSource::Tycho(builder) = self.source {
            self.source = StreamSource::Tycho(builder.exchange(name, filter));
        }

        // Decoder registration happens regardless of source
        self.decoder.register_decoder::<T>(name);
        if let Some(predicate) = filter_fn {
            self.decoder
                .register_filter(name, predicate);
        }

        if EXCHANGES_REQUIRING_FILTER.contains(&name) && filter_fn.is_none() {
            warn!("Warning: For exchange type '{}', it is necessary to set a filter function because not all pools are supported. See all filters at src/evm/protocol/filters.rs", name);
        }

        self
    }

    /// Adds a specific exchange to the stream with decoder context.
    ///
    /// This configures the builder to include a new protocol synchronizer for `name`,
    /// filtering its components according to `filter` and optionally `filter_fn`. It also registers
    /// the DecoderContext (this is useful to test protocols that are not live yet)
    ///
    /// The type parameter `T` specifies the decoder type for this exchange. All
    /// component states for this exchange will be decoded into instances of `T`.
    ///
    /// # Parameters
    ///
    /// - `name`: The protocol or exchange name (e.g., `"uniswap_v4"`, `"vm:balancer_v2"`).
    /// - `filter`: Defines the set of components to include in the stream. Ignored when using
    ///   `from_receiver()`.
    /// - `filter_fn`: Optional custom filter function for client-side filtering of components not
    ///   expressible in `filter`.
    /// - `decoder_context`: The decoder context for this exchange
    ///
    /// # Notes
    ///
    /// For certain protocols (e.g., `"uniswap_v4"`, `"vm:balancer_v2"`, `"vm:curve"`), omitting
    /// `filter_fn` may cause decoding errors or incorrect results. In these cases, a proper
    /// filter function is required to ensure correct decoding and quoting logic.
    pub fn exchange_with_decoder_context<T>(
        mut self,
        name: &str,
        filter: ComponentFilter,
        filter_fn: Option<fn(&ComponentWithState) -> bool>,
        decoder_context: DecoderContext,
    ) -> Self
    where
        T: ProtocolSim
            + TryFromWithBlock<ComponentWithState, BlockHeader, Error = InvalidSnapshotError>
            + Send
            + 'static,
    {
        // Register with TychoStreamBuilder only for live connections
        if let StreamSource::Tycho(builder) = self.source {
            self.source = StreamSource::Tycho(builder.exchange(name, filter));
        }

        // Decoder registration happens regardless of source
        self.decoder
            .register_decoder_with_context::<T>(name, decoder_context);
        if let Some(predicate) = filter_fn {
            self.decoder
                .register_filter(name, predicate);
        }

        if EXCHANGES_REQUIRING_FILTER.contains(&name) && filter_fn.is_none() {
            warn!("Warning: For exchange type '{}', it is necessary to set a filter function because not all pools are supported. See all filters at src/evm/protocol/filters.rs", name);
        }

        self
    }

    /// Sets the block time interval for the stream.
    ///
    /// This controls how often the stream produces updates.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn block_time(self, block_time: u64) -> Self {
        self.with_tycho(|b| b.block_time(block_time))
    }

    /// Sets the network operation timeout (deprecated).
    ///
    /// Use [`latency_buffer`] instead for controlling latency.
    /// This method is retained for backwards compatibility.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    #[deprecated = "Use latency_buffer instead"]
    pub fn timeout(self, timeout: u64) -> Self {
        self.with_tycho(|b| b.timeout(timeout))
    }

    /// Sets the latency buffer to aggregate same-block messages.
    ///
    /// This allows the supervisor to wait a short interval for all synchronizers to emit
    /// before aggregating.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn latency_buffer(self, timeout: u64) -> Self {
        self.with_tycho(|b| b.timeout(timeout))
    }

    /// Sets the maximum number of blocks a synchronizer may miss before being marked as `Stale`.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn max_missed_blocks(self, n: u64) -> Self {
        self.with_tycho(|b| b.max_missed_blocks(n))
    }

    /// Sets how long a synchronizer may take to process the initial message.
    ///
    /// Useful for data-intensive protocols where startup decoding takes longer.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn startup_timeout(self, timeout: time::Duration) -> Self {
        self.with_tycho(|b| b.startup_timeout(timeout))
    }

    /// Configures the stream to exclude state updates.
    ///
    /// This reduces bandwidth and decoding workload if protocol state is not of
    /// interest (e.g. only process new tokens).
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn no_state(self, no_state: bool) -> Self {
        self.with_tycho(|b| b.no_state(no_state))
    }

    /// Sets the API key for authenticating with the Tycho server.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn auth_key(self, auth_key: Option<String>) -> Self {
        self.with_tycho(|b| b.auth_key(auth_key))
    }

    /// Disables TLS/ SSL for the connection, using http and ws protocols.
    ///
    /// This is not recommended for production use.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn no_tls(self, no_tls: bool) -> Self {
        self.with_tycho(|b| b.no_tls(no_tls))
    }

    /// Disable compression for the connection.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn disable_compression(self) -> Self {
        self.with_tycho(|b| b.disable_compression())
    }

    /// Sets the stream end policy.
    ///
    /// Controls when the stream should stop based on synchronizer states.
    ///
    /// ## Note
    /// The stream always ends latest if all protocols are stale or ended independent of
    /// this configuration. This allows you to end the stream earlier than that.
    ///
    /// See [self::StreamEndPolicy] for possible configuration options.
    pub fn stream_end_policy(mut self, stream_end_policy: StreamEndPolicy) -> Self {
        self.stream_end_policy = stream_end_policy;
        self
    }

    /// Sets the initial tokens to consider during decoding.
    ///
    /// Only components containing these tokens will be decoded initially.
    /// New tokens may be added automatically if they meet the quality threshold.
    pub async fn set_tokens(self, tokens: HashMap<Bytes, Token>) -> Self {
        self.decoder.set_tokens(tokens).await;
        self
    }

    /// Skips decoding errors for component state updates.
    ///
    /// Allows the stream to continue processing even if some states fail to decode,
    /// logging a warning instead of panicking.
    pub fn skip_state_decode_failures(mut self, skip: bool) -> Self {
        self.decoder
            .skip_state_decode_failures(skip);
        self
    }

    /// Configures the retry policy for websocket reconnects.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn websocket_retry_config(self, config: &RetryConfiguration) -> Self {
        self.with_tycho(|b| b.websockets_retry_config(config))
    }

    /// Configures the retry policy for state synchronization.
    ///
    /// Note: Has no effect when using `from_receiver()`.
    pub fn state_synchronizer_retry_config(self, config: &RetryConfiguration) -> Self {
        self.with_tycho(|b| b.state_synchronizer_retry_config(config))
    }

    pub fn get_decoder(&self) -> &TychoStreamDecoder<BlockHeader> {
        &self.decoder
    }

    /// Builds and returns the configured protocol stream.
    ///
    /// See the module-level docs for details on stream behavior and emitted messages.
    /// This method applies all builder settings and starts the stream.
    pub async fn build(
        self,
    ) -> Result<impl Stream<Item = Result<Update, StreamDecodeError>>, StreamError> {
        initialize_hook_handlers().map_err(|e| {
            StreamError::SetUpError(format!("Error initializing hook handlers: {e:?}"))
        })?;

        let rx = match self.source {
            StreamSource::Tycho(builder) => {
                let (_, rx) = builder.build().await?;
                rx
            }
            StreamSource::Receiver(rx) => rx,
        };

        let decoder = Arc::new(self.decoder);

        let stream = Box::pin(
            ReceiverStream::new(rx)
                .take_while(move |msg| match msg {
                    Ok(msg) => {
                        let states = msg.sync_states.values();
                        if self
                            .stream_end_policy
                            .should_end(states)
                        {
                            error!(
                                "Block stream ended due to {:?}: {:?}",
                                self.stream_end_policy, msg.sync_states
                            );
                            futures::future::ready(false)
                        } else {
                            futures::future::ready(true)
                        }
                    }
                    Err(e) => {
                        error!("Block stream ended with terminal error: {e}");
                        futures::future::ready(false)
                    }
                })
                .then({
                    let decoder = decoder.clone(); // Clone the decoder for the closure
                    move |msg| {
                        let decoder = decoder.clone(); // Clone again for the async block
                        async move {
                            let msg = msg.expect("Safe since stream ends if we receive an error");
                            decoder.decode(&msg).await.map_err(|e| {
                                debug!(msg=?msg, "Decode error: {}", e);
                                e
                            })
                        }
                    }
                }),
        );
        Ok(stream)
    }
}

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    use std::{path::Path, time::Duration};

    use tokio::sync::mpsc;

    use super::*;

    /// Creates a mock receiver that emits messages at a fixed interval.
    pub fn mock_receiver(
        messages: Vec<FeedMessage<BlockHeader>>,
        interval: Duration,
    ) -> Receiver<Result<FeedMessage<BlockHeader>, BlockSynchronizerError>> {
        let (tx, rx) = mpsc::channel(messages.len().max(1));

        tokio::spawn(async move {
            for msg in messages {
                tokio::time::sleep(interval).await;
                if tx.send(Ok(msg)).await.is_err() {
                    break;
                }
            }
        });

        rx
    }

    /// Creates a mock receiver from a JSON fixture file.
    pub fn mock_receiver_from_file(
        path: impl AsRef<Path>,
        interval: Duration,
    ) -> Result<Receiver<Result<FeedMessage<BlockHeader>, BlockSynchronizerError>>, std::io::Error>
    {
        let data = std::fs::read(path)?;
        let messages: Vec<FeedMessage<BlockHeader>> = serde_json::from_slice(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(mock_receiver(messages, interval))
    }
}
