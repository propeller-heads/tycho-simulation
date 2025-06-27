use std::collections::HashMap;

use tycho_simulation::{
    models::Token,
    protocol::{errors::InvalidSnapshotError, models::TryFromWithBlock},
    tycho_client::feed::{Header, synchronizer::ComponentWithState},
    tycho_common::Bytes,
};

use crate::protocols::bebop::state::BebopState;

impl TryFromWithBlock<ComponentWithState> for BebopState {
    type Error = InvalidSnapshotError;

    async fn try_from_with_block(
        snapshot: ComponentWithState,
        block: Header,
        account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        all_tokens: &HashMap<Bytes, Token>,
    ) -> Result<Self, Self::Error> {
        todo!()
    }
}
