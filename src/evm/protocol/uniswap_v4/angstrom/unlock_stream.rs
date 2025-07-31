use super::streams::*;
use alloy::primitives::Bytes;
use futures::FutureExt;
use futures::StreamExt;
use jsonrpsee::core::client::Subscription;
use jsonrpsee::core::ClientError;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::task::Context;
use std::task::{Poll, Waker};

use alloy::primitives::U160;
use futures::Stream;
use jsonrpsee::ws_client::WsClient;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsensusDataWithBlock<T> {
    pub data: T,
    pub block: u64,
}

pub struct PoolUnlockStream {
    client: Arc<WsClient>,
    subscription: Option<Subscription<ConsensusDataWithBlock<Bytes>>>,
    pending_subscription: Option<
        Pin<
            Box<
                dyn Future<
                        Output = Result<Subscription<ConsensusDataWithBlock<Bytes>>, ClientError>,
                    > + Send,
            >,
        >,
    >,
    waker: Option<Waker>,
}
