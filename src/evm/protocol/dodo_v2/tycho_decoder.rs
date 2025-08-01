use crate::evm::protocol::dodo_v2::state::DodoV2State;
use crate::protocol::errors::InvalidSnapshotError;
use crate::protocol::models::TryFromWithBlock;
use alloy::primitives::U256;
use std::collections::HashMap;
use tycho_client::feed::synchronizer::ComponentWithState;
use tycho_client::feed::Header;
use tycho_common::models::token::Token;
use tycho_common::Bytes;

impl TryFromWithBlock<ComponentWithState> for DodoV2State {
    type Error = InvalidSnapshotError;

    async fn try_from_with_block(
        snapshot: ComponentWithState,
        _block: Header,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        _all_tokens: &HashMap<Bytes, Token>,
    ) -> Result<Self, Self::Error> {
        let b = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("B")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("B".to_string()))?,
        );

        let q = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("Q")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("Q".to_string()))?,
        );

        let b0 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("B0")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("B0".to_string()))?,
        );

        let q0 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("Q0")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("Q0".to_string()))?,
        );

        let r = u8::from(
            snapshot
                .state
                .attributes
                .get("R")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("R".to_string()))?,
        );

        let k = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("K")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("K".to_string()))?,
        );

        let i = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("I")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("I".to_string()))?,
        );

        let base_token = snapshot.component.tokens[0].clone();
        let quote_token = snapshot.component.tokens[1].clone();

        Ok(DodoV2State::new(i, k, b, q, b0, q0, r, base_token, quote_token))
    }
}
