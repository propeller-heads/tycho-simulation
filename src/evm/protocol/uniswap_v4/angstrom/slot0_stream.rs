use std::{
    collections::HashSet,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use alloy::primitives::U160;
use futures::{FutureExt, Stream, StreamExt};
use jsonrpsee::{
    core::{client::Subscription, ClientError},
    ws_client::WsClient,
};
use serde::{Deserialize, Serialize};

use super::streams::*;

type ReconnectionFut =
    Box<dyn Future<Output = Result<Subscription<Slot0Update>, ClientError>> + Send>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct Slot0Update {
    /// there will be 120 updates per block or per 100ms
    pub seq_id: u16,
    /// in case of block lag on node
    pub current_block: u64,
    pub angstrom_pool_id: PoolId,
    pub uni_pool_id: PoolId,

    pub sqrt_price_x96: U160,
    pub liquidity: u128,
    pub tick: i32,
}

/// Trait for streams that provide slot0 updates with dynamic pool subscription
/// management
pub trait Slot0Stream: Stream<Item = Slot0Update> + Unpin + Send {
    /// Subscribe to updates for a set of pools
    fn subscribe_pools(&mut self, pools: HashSet<PoolId>);

    /// Unsubscribe from updates for a set of pools
    fn unsubscribe_pools(&mut self, pools: HashSet<PoolId>);

    /// Get the current set of subscribed pools
    fn subscribed_pools(&self) -> &HashSet<PoolId>;
}

/// Client for subscribing to slot0 updates via jsonrpsee
pub struct Slot0Client {
    client: Arc<WsClient>,
    subscription: Option<Subscription<Slot0Update>>,
    subscribed_pools: HashSet<PoolId>,
    pending_subscription: Option<Pin<ReconnectionFut>>,
    waker: Option<Waker>,
}

impl Slot0Client {
    /// Create a new Slot0Client from a jsonrpsee WebSocket client
    pub fn new(client: Arc<WsClient>) -> Self {
        Self {
            client,
            subscription: None,
            subscribed_pools: HashSet::new(),
            pending_subscription: None,
            waker: None,
        }
    }

    fn reconnect(&mut self) {
        let client = self.client.clone();
        let pools = self.subscribed_pools.clone();

        let connection_future = Box::pin(async move { client.subscribe_amm(pools).await });
        self.pending_subscription = Some(connection_future);
    }
}

impl Stream for Slot0Client {
    type Item = Slot0Update;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.waker.is_none() {
            self.waker = Some(cx.waker().clone());
        }

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

impl Slot0Stream for Slot0Client {
    fn subscribe_pools(&mut self, pools: HashSet<PoolId>) {
        self.subscribed_pools.extend(pools);

        let client = self.client.clone();
        let pools = self.subscribed_pools.clone();

        let connection_future = Box::pin(async move { client.subscribe_amm(pools).await });
        self.pending_subscription = Some(connection_future);

        if let Some(waker) = self.waker.as_ref() {
            waker.wake_by_ref();
        }
    }

    fn unsubscribe_pools(&mut self, pools: HashSet<PoolId>) {
        for pool in pools {
            self.subscribed_pools.remove(&pool);
        }

        let client = self.client.clone();
        let pools = self.subscribed_pools.clone();

        let connection_future = Box::pin(async move { client.subscribe_amm(pools).await });
        self.pending_subscription = Some(connection_future);

        if let Some(waker) = self.waker.as_ref() {
            waker.wake_by_ref();
        }
    }

    fn subscribed_pools(&self) -> &HashSet<PoolId> {
        &self.subscribed_pools
    }
}
