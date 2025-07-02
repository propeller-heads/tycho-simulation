use std::{collections::HashMap, pin::Pin, sync::Arc};

use chrono::Utc;
use futures::{StreamExt, stream::select_all};
use tycho_simulation::{
    protocol::{errors::InvalidSnapshotError, models::Update, state::ProtocolSim},
    tycho_client::feed::{FeedMessage, synchronizer::StateSyncMessage},
};

use crate::client::RFQClient;

type DecodeFut =
    Pin<Box<dyn Future<Output = Result<Box<dyn ProtocolSim>, InvalidSnapshotError>> + Send + Sync>>;
type RegistryFn = dyn Fn(StateSyncMessage) -> DecodeFut + Send + Sync;
pub struct RFQDecoder {
    registry: HashMap<String, Box<RegistryFn>>,
}

impl RFQDecoder {
    pub async fn decode(&self, msg: FeedMessage) -> Update {
        // Something like:
        let mut states = HashMap::new();
        for (provider, state_msg) in msg.state_msgs {
            // Create ProtocolStates for all pairs
            // we will never have a partial update, so we need to construct the states over and over
            // again here
            if let Some(decode_fn) = self.registry.get(&provider) {
                match decode_fn(state_msg).await {
                    Ok(sim) => {
                        states.insert(provider.clone(), sim);
                    }
                    Err(err) => {
                        eprintln!("Error decoding {}: {:?}", provider, err);
                    }
                }
            } else {
                eprintln!("No decoder registered for provider: {}", provider);
            }
            // gather all deleted pairs
        }

        let timestamp = Utc::now().timestamp() as u64;

        Update {
            block_number: timestamp,
            states,
            new_pairs: Default::default(),
            removed_pairs: Default::default(),
        }
    }
}

pub struct RFQStreamBuilder {
    providers: Vec<Arc<dyn RFQClient>>,
    decoder: RFQDecoder,
}

impl RFQStreamBuilder {
    pub fn new(decoder: RFQDecoder) -> Self {
        Self { providers: Vec::new(), decoder }
    }

    pub fn add_provider(&mut self, provider: Arc<dyn RFQClient>) {
        self.providers.push(provider);
    }

    pub async fn build(self, tx: tokio::sync::mpsc::Sender<Update>) {
        let streams: Vec<_> = self
            .providers
            .iter()
            .map(|provider| provider.stream())
            .collect();

        let mut merged = select_all(streams);

        while let Some((provider, msg)) = merged.next().await {
            let update = self
                .decoder
                .decode(FeedMessage {
                    state_msgs: HashMap::from([(provider.clone(), msg)]),
                    sync_states: HashMap::new(),
                })
                .await;

            tx.send(update).await.unwrap();
        }
    }
}
