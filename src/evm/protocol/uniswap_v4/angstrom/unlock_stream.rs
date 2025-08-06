use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{FutureExt, Stream, StreamExt};
use jsonrpsee::{
    core::{client::Subscription, ClientError},
    ws_client::WsClient,
};
use serde::{Deserialize, Serialize};
use tycho_common::Bytes;

use super::streams::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsensusDataWithBlock<T> {
    pub data: T,
    pub block: u64,
}

type ReconnectionFut = Box<
    dyn Future<Output = Result<Subscription<ConsensusDataWithBlock<Bytes>>, ClientError>> + Send,
>;

pub struct UnlockClient {
    client: Arc<WsClient>,
    subscription: Option<Subscription<ConsensusDataWithBlock<Bytes>>>,
    pending_subscription: Option<Pin<ReconnectionFut>>,
}

impl UnlockClient {
    /// Create a new Slot0Client from a jsonrpsee WebSocket client
    pub fn new(client: Arc<WsClient>) -> Self {
        Self { client, subscription: None, pending_subscription: None }
    }

    fn reconnect(&mut self) {
        let client = self.client.clone();

        let connection_future = Box::pin(async move {
            client
                .subscribe_empty_block_attestations()
                .await
        });
        self.pending_subscription = Some(connection_future);
    }
}

impl Stream for UnlockClient {
    type Item = ConsensusDataWithBlock<Bytes>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(mut new_stream_future) = self.pending_subscription.take() {
            match new_stream_future.poll_unpin(cx) {
                Poll::Ready(stream) => match stream {
                    Ok(stream) => {
                        self.subscription = Some(stream);
                    }
                    Err(_) => {
                        cx.waker().wake_by_ref();
                        self.reconnect();
                    }
                },
                Poll::Pending => {
                    self.pending_subscription = Some(new_stream_future);
                }
            }
        }

        if let Some(cur_sub) = self.subscription.as_mut() {
            if let Poll::Ready(update) = cur_sub.poll_next_unpin(cx) {
                match update {
                    Some(Ok(update)) => return Poll::Ready(Some(update)),
                    Some(Err(_)) | None => {
                        cx.waker().wake_by_ref();
                        self.reconnect();
                    }
                }
            }
        }

        Poll::Pending
    }
}
