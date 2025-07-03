use std::{collections::HashMap, pin::Pin, sync::Arc};

use chrono::Utc;
use futures::{StreamExt, stream::select_all};
use tycho_simulation::{
    evm::decoder::TychoStreamDecoder,
    protocol::{
        errors::InvalidSnapshotError,
        models::{TryFromWithBlock, Update},
        state::ProtocolSim,
    },
    tycho_client::feed::{
        FeedMessage,
        synchronizer::{ComponentWithState, StateSyncMessage},
    },
};

use crate::client::RFQClient;

pub struct RFQStreamBuilder {
    providers: Vec<Arc<dyn RFQClient>>,
    decoder: TychoStreamDecoder,
}

impl RFQStreamBuilder {
    pub fn new() -> Self {
        Self { providers: Vec::new(), decoder: TychoStreamDecoder::new() }
    }

    pub fn add_provider<T>(mut self, name: &str, provider: Arc<dyn RFQClient>) -> Self
    where
        T: ProtocolSim
            + TryFromWithBlock<ComponentWithState, Error = InvalidSnapshotError>
            + Send
            + 'static,
    {
        self.providers.push(provider);
        self.decoder.register_decoder::<T>(name);
        self
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
                .await
                .unwrap();

            tx.send(update).await.unwrap();
        }
    }
}
