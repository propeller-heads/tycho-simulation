use std::{
    collections::HashSet,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use super::slot0_stream::Slot0Update;
use super::unlock_stream::ConsensusDataWithBlock;
use alloy::primitives::{FixedBytes, U160};
use futures::{FutureExt, Stream, StreamExt};
use jsonrpsee::{
    core::{client::Subscription, ClientError, Serialize},
    proc_macros::rpc,
    ws_client::WsClient,
};
use serde::Deserialize;

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
        item = ConsensusDataWithBlock<alloy::primitives::Bytes>
    )]
    async fn subscribe_empty_block_attestations(&self) -> jsonrpsee::core::SubscriptionResult;
}
