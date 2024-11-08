use crate::{
    models::ERC20Token,
    protocol::{
        errors::InvalidSnapshotError,
        models::{ProtocolComponent, TryFromWithBlock},
        state::ProtocolSim,
    },
};
use ethers::prelude::H160;
use futures::{Stream, StreamExt};
use std::{
    collections::{hash_map::Entry, HashMap},
    str::FromStr,
    sync::Arc,
};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use futures::future::BoxFuture;
use thiserror::Error;
use tokio::sync::mpsc::Receiver;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, warn};
use tycho_client::feed::{synchronizer::ComponentWithState, FeedMessage, Header};
use tycho_core::Bytes;
use crate::evm::tycho_models::Block;
use crate::protocol::uniswap_v2::state::UniswapV2State;
use crate::protocol::uniswap_v3::state::UniswapV3State;

#[derive(Error, Debug)]
pub enum StreamDecodeError {
    #[error("{0}")]
    Fatal(String),
}

#[derive(Debug)]
pub struct BlockUpdate {
    pub block_number: u64,
    /// The current state of all pools
    pub states: HashMap<String, Box<dyn ProtocolSim>>,
    /// The new pairs that were added in this block
    pub new_pairs: HashMap<String, ProtocolComponent>,
    /// The pairs that were removed in this block
    pub removed_pairs: HashMap<String, ProtocolComponent>,
}

impl BlockUpdate {
    pub fn new(
        block_number: u64,
        states: HashMap<String, Box<dyn ProtocolSim>>,
        new_pairs: HashMap<String, ProtocolComponent>,
    ) -> Self {
        BlockUpdate { block_number, states, new_pairs, removed_pairs: HashMap::new() }
    }

    pub fn set_removed_pairs(mut self, pairs: HashMap<String, ProtocolComponent>) -> Self {
        self.removed_pairs = pairs;
        self
    }
}

#[derive(Default)]
struct DecoderState {
    // TODO: it would be nicer to allow system to supply token data in a more
    //  flexible way, e.g. via a function call.
    tokens: HashMap<Bytes, ERC20Token>,
    // TODO: Do we really need Box here or would Arc be better?
    states: HashMap<String, Box<dyn ProtocolSim>>,
}


type DecodeFut = Pin<Box<dyn Future<Output = Result<Box<dyn ProtocolSim>, InvalidSnapshotError>> + Send + Sync>>;

type RegistryFn = dyn Fn(
    ComponentWithState,
    Header,
) -> DecodeFut + Send + Sync;


pub struct TychoStreamDecoder {
    state: Arc<RwLock<DecoderState>>,
    skip_state_decode_failures: bool,
    min_token_quality: u32,
    registry: HashMap<String, Box<RegistryFn>>,
}

impl TychoStreamDecoder
{
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DecoderState::default())),
            skip_state_decode_failures: false,
            min_token_quality: 51,
            registry: HashMap::new(),
        }
    }

    pub async fn set_tokens(&self, tokens: HashMap<Bytes, ERC20Token>) {
        let mut guard = self.state.write().await;
        guard.tokens = tokens;
    }

    // Method to register a decoder
    pub fn register_decoder<T>(&mut self, exchange: &str)
    where
        T: ProtocolSim + TryFromWithBlock<ComponentWithState, Error=InvalidSnapshotError> + Send + 'static,
    {
        let decoder = Box::new(move |component: ComponentWithState, header: Header| {
            Box::pin(async move {
                T::try_from_with_block(component, header).await.map(|c| Box::new(c) as Box<dyn ProtocolSim>)
            }) as DecodeFut
        });
        self.registry.insert(exchange.to_string(), decoder);
    }

    pub async fn decode(&self, msg: FeedMessage) -> Result<BlockUpdate, StreamDecodeError> {
        // stores all states updated in this tick/msg
        // TODO: this is box cause we need to modify it, but later we clone it so it seems inefficient...
        let mut updated_states = HashMap::new();
        let mut new_pairs = HashMap::new();
        let mut removed_pairs = HashMap::new();

        let block = msg
            .state_msgs
            .values()
            .next()
            .ok_or_else(|| StreamDecodeError::Fatal("Missing block!".into()))?
            .header
            .clone();

        for (protocol, protocol_msg) in msg.state_msgs.iter() {
            // Add any new tokens
            if let Some(deltas) = protocol_msg.deltas.as_ref() {
                let mut state_guard = self.state.write().await;
                let res = deltas
                    .new_tokens
                    .iter()
                    .filter_map(|(addr, t)| {
                        if t.quality < self.min_token_quality ||
                            !state_guard.tokens.contains_key(addr)
                        {
                            return None;
                        }

                        let token: Result<ERC20Token, std::num::TryFromIntError> = t.clone().try_into();
                        let result = match token {
                            Ok(t) => Ok((addr.clone(), t)),
                            Err(e) => Err(StreamDecodeError::Fatal(format!(
                                "Failed decoding token {e} {addr:#044x}"
                            ))),
                        };
                        Some(result)
                    })
                    .collect::<Result<HashMap<Bytes, ERC20Token>, StreamDecodeError>>()?;

                if !res.is_empty() {
                    debug!(n = res.len(), "NewTokens");
                    state_guard.tokens.extend(res);
                }
            }

            let state_guard = self.state.read().await;
            removed_pairs.extend(
                protocol_msg
                    .removed_components
                    .iter()
                    .flat_map(|(id, comp)| {
                        let tokens = comp
                            .tokens
                            .iter()
                            .flat_map(|addr| state_guard.tokens.get(addr).cloned())
                            .collect::<Vec<_>>();

                        if tokens.len() == comp.tokens.len() {
                            // TODO: propagate the error outside here instead of ignoring it
                            let address = string_to_h160(id).ok()?;
                            Some((id.clone(), ProtocolComponent::new(address, tokens)))
                        } else {
                            // We may reach this point if the removed component
                            //  contained low quality tokens, in this case the component
                            //  was never added so we can skip emitting it.
                            None
                        }
                    }),
            );

            let mut new_components = HashMap::new();

            // PROCESS SNAPSHOTS
            for (id, snapshot) in protocol_msg
                .snapshots
                .get_states()
                .clone()
            {
                let addr = match string_to_h160(id.as_str()) {
                    Ok(res) => res,
                    Err(e) => {
                        warn!(pool = id, error = %e, "StateDecodingFailure");
                        if self.skip_state_decode_failures {
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                };

                // Construct component from snapshot
                let mut component_tokens = Vec::new();
                for token in snapshot.component.tokens.clone() {
                    match state_guard.tokens.get(&token) {
                        Some(token) => component_tokens.push(token.clone()),
                        None => {
                            debug!("Token not found {}, ignoring pool {:x?}", token, id);
                            continue;
                        }
                    }
                }
                new_pairs.insert(id.clone(), ProtocolComponent::new(addr, component_tokens));

                // Construct state from snapshot
                if let Some(state_decode_f) = self.registry.get(protocol.as_str()) {
                    match state_decode_f(snapshot, block.clone()).await {
                        Ok(state) => {
                            new_components.insert(id.clone(), state);
                        }
                        Err(e) => {
                            if self.skip_state_decode_failures {
                                warn!(pool = id, error = %e, "StateDecodingFailure");
                                continue;
                            } else {
                                return Err(StreamDecodeError::Fatal(format!("{e}")));
                            }
                        }
                    }
                } else if self.skip_state_decode_failures {
                    warn!(pool = id, "MissingDecoderRegistration");
                    continue;
                } else {
                    return Err(StreamDecodeError::Fatal(format!(
                        "Missing decoder registration for: {id}"
                    )));
                }
            }

            if !new_components.is_empty() {
                debug!("Decoded {} snapshots for protocol {}", new_components.len(), protocol);
            }
            updated_states.extend(new_components);

            // PROCESS DELTAS
            if let Some(deltas) = protocol_msg.deltas.clone() {
                for (id, update) in deltas.state_updates {
                    match updated_states.entry(id.clone()) {
                        Entry::Occupied(mut entry) => {
                            // if state exists in updated_states, apply the delta to it
                            let state: &mut Box<dyn ProtocolSim> = entry.get_mut();
                            state
                                .delta_transition(update)
                                .map_err(|e| {
                                    StreamDecodeError::Fatal(format!("TransitionFailure: {e:?}"))
                                })?;
                        }
                        Entry::Vacant(_) => {
                            match state_guard.states.get(&id) {
                                // if state does not exist in updated_states, apply the delta to the
                                // stored state
                                Some(stored_state) => {
                                    let mut state = stored_state.clone();
                                    state
                                        .delta_transition(update)
                                        .map_err(|e| {
                                            StreamDecodeError::Fatal(format!(
                                                "TransitionFailure: {e:?}"
                                            ))
                                        })?;
                                    updated_states.insert(id, state);
                                }
                                None => warn!(
                                    pool = id,
                                    reason = "MissingState",
                                    "DeltaTransitionError"
                                ),
                            }
                        }
                    }
                }
            };
        }

        // Persist the newly added/updated states
        let mut state_guard = self.state.write().await;
        state_guard
            .states
            .extend(updated_states.clone().into_iter());

        // Send the tick with all updated states
        Ok(BlockUpdate::new(block.number, updated_states, new_pairs)
            .set_removed_pairs(removed_pairs))
    }

}


pub async fn tycho_stream(tokens: HashMap<Bytes, ERC20Token>, tycho_rx: Receiver<FeedMessage>) -> impl Stream<Item=Result<BlockUpdate, StreamDecodeError>> {
    let mut decoder = TychoStreamDecoder::new();

    decoder
        .set_tokens(tokens)
        .await;
    decoder.register_decoder::<UniswapV2State>("uniswap_v2");
    decoder.register_decoder::<UniswapV3State>("uniswap_v3");

    let decoder = Arc::new(decoder);

    Box::pin(ReceiverStream::new(tycho_rx).then(
        {
            let decoder = decoder.clone(); // Clone the decoder for the closure
            move |msg| {
                let decoder = decoder.clone(); // Clone again for the async block
                async move {
                    decoder.decode(msg).await
                }
            }
        }
    ))
}

fn string_to_h160(s: &str) -> Result<H160, StreamDecodeError> {
    let trimmed = if let Some(stripped) = s.strip_prefix("0x") { stripped } else { s };

    // Ensure the string has at least 20 characters (40 hex digits)
    if trimmed.len() < 40 {
        return Err(StreamDecodeError::Fatal(format!("Failed to decode {s} as H160")));
    }

    // Slice off the first 40 characters (20 bytes as hex)
    let sliced = &trimmed[..40];

    H160::from_str(sliced)
        .map_err(|e| StreamDecodeError::Fatal(format!("Failed to decode {trimmed} as H160: {e}")))
}
