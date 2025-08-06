use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use alloy::primitives::{aliases::I24, FixedBytes, U256};
use futures::StreamExt;
use jsonrpsee::{proc_macros::rpc, ws_client::WsClientBuilder};
use tokio::sync::mpsc::{channel, Sender};
use tokio_stream::wrappers::ReceiverStream;
use tycho_client::{
    feed::{synchronizer::StateSyncMessage, FeedMessage, Header},
    stream::StreamError,
};
use tycho_common::{
    dto::{BlockChanges, ProtocolStateDelta},
    Bytes,
};

use super::{slot0_stream::Slot0Update, unlock_stream::ConsensusDataWithBlock};
use crate::evm::{
    protocol::uniswap_v4::angstrom::{slot0_stream::Slot0Client, unlock_stream::UnlockClient},
    stream::ProtocolStreamExtension,
};

pub(super) type PoolId = FixedBytes<32>;

#[rpc(client, namespace = "angstrom")]
#[async_trait::async_trait]
pub trait SubApi {
    #[subscription(
        name = "subscribeAmm",
        unsubscribe = "unsubscribeAmm",
        item = Slot0Update
    )]
    async fn subscribe_amm(&self, pools: HashSet<PoolId>) -> jsonrpsee::core::SubscriptionResult;
}

#[rpc(client, namespace = "consensus")]
#[async_trait::async_trait]
pub trait ConsensusApi {
    #[subscription(
        name = "subscribeEmptyBlockAttestations",
        unsubscribe = "unsubscribeEmptyBlockAttestations",
        item = ConsensusDataWithBlock<Bytes>
    )]
    async fn subscribe_empty_block_attestations(&self) -> jsonrpsee::core::SubscriptionResult;
}

/// Extends into the protocol stream to allow for us to inject critical data for unlocking +
/// keeping slot0 up to date.
pub struct AngstromStreamExt {
    slot0_client: Slot0Client,
    unlock_client: UnlockClient,
    /// Given that slot0 stream will send out data, even if there are no orders.
    // We don't need to worry about trying to roll-back state. This is because the
    // Slot0 stream will give us the updated pool state for these blocks. All we want
    // to ensure is that we only send out updates that are aligned with tycho's block
    // number.
    tycho_header: Header,

    update_receiver: ReceiverStream<Arc<FeedMessage>>,
    update_sender: Sender<FeedMessage>,
}

impl AngstromStreamExt {
    pub async fn create_angstrom_stream_ext(
        angstrom_url: &str,
    ) -> Result<ProtocolStreamExtension, StreamError> {
        if !(angstrom_url.starts_with("ws") && angstrom_url.starts_with("wss")) {
            return Err(StreamError::SetUpError("Angstrom url requires websocket.".to_string()));
        }

        let (receiving_tx, receiving_rx) = channel(100);
        let (update_sender, sending_rx) = channel(100);
        let update_receiver = ReceiverStream::new(receiving_rx);

        let channels = ProtocolStreamExtension { tx: Some(receiving_tx), rx: sending_rx };

        let client = WsClientBuilder::default()
            .build(angstrom_url)
            .await
            .map_err(|e| StreamError::WebSocketConnectionError(e.to_string()))?;
        let arced_client = Arc::new(client);
        // connect to angstrom streams.
        let slot0_client = Slot0Client::new(arced_client.clone());
        let unlock_client = UnlockClient::new(arced_client);

        let this = Self {
            slot0_client,
            unlock_client,
            tycho_header: Header::default(),
            update_receiver,
            update_sender,
        };
        tokio::spawn(this.run());

        Ok(channels)
    }

    pub async fn send_slot0_update(&mut self, update: Slot0Update) {
        if update.current_block != self.tycho_header.number {
            return;
        }

        let mut attributes = HashMap::new();

        attributes.insert("liquidity".to_string(), Bytes::from(update.liquidity.to_be_bytes()));
        attributes.insert(
            "sqrt_price_x96".to_string(),
            Bytes::from(U256::from(update.sqrt_price_x96).to_be_bytes::<32>()),
        );
        attributes.insert(
            "tick".to_string(),
            Bytes::from(I24::unchecked_from(update.tick).to_be_bytes::<3>()),
        );

        let state_update = ProtocolStateDelta {
            component_id: update.uni_pool_id.to_string(),
            updated_attributes: attributes,
            ..Default::default()
        };

        let mut state_updates = HashMap::new();
        state_updates.insert("angstrom".to_string(), state_update);

        let deltas = BlockChanges { state_updates, ..Default::default() };
        let message = StateSyncMessage {
            header: self.tycho_header.clone(),
            deltas: Some(deltas),
            ..Default::default()
        };

        let state_msgs = HashMap::from_iter([("angstrom".to_string(), message)]);

        let message = FeedMessage { state_msgs, sync_states: Default::default() };
        self.update_sender
            .send(message)
            .await
            .expect("feed is no longer being serviced");
    }
    pub async fn send_unlock_update(&mut self, update: ConsensusDataWithBlock<Bytes>) {
        todo!()
    }

    pub async fn on_tycho_message(&mut self, feed_message: Arc<FeedMessage>) {
        if let Some(state_message) = feed_message.state_msgs.get("angstrom") {
            self.tycho_header = state_message.header.clone();
        }
    }

    pub async fn run(mut self) {
        loop {
            tokio::select! {
                Some(feed_message) = self.update_receiver.next() => {
                    self.on_tycho_message(feed_message).await;

                }
                Some(update) = self.slot0_client.next() => {
                    self.send_slot0_update(update).await;
                }
                Some(update) = self.unlock_client.next() => {
                    self.send_unlock_update(update).await;
                }
            }
        }
    }
}
