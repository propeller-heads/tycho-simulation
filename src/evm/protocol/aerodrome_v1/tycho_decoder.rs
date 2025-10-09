use std::collections::HashMap;

use alloy::primitives::U256;
use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
use tycho_common::{models::token::Token, Bytes};

use crate::{
    evm::protocol::aerodrome_v1::state::AerodromeV1PoolState,
    protocol::{
        errors::InvalidSnapshotError,
        models::{DecoderContext, TryFromWithBlock},
    },
};

impl TryFromWithBlock<ComponentWithState, BlockHeader> for AerodromeV1PoolState {
    type Error = InvalidSnapshotError;

    /// Decodes a `ComponentWithState` into a `AerodromeV1PoolState`. Errors with a
    /// `InvalidSnapshotError` if either reserve0 or reserve1 attributes are missing.
    async fn try_from_with_header(
        snapshot: ComponentWithState,
        _block: BlockHeader,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        all_tokens: &HashMap<Bytes, Token>,
        _decoder_context: &DecoderContext,
    ) -> Result<Self, Self::Error> {
        let reserve0 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("reserve0")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve0".to_string()))?,
        );

        let reserve1 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("reserve1")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve1".to_string()))?,
        );

        let stable = snapshot
            .state
            .attributes
            .get("stable")
            .ok_or(InvalidSnapshotError::MissingAttribute("stable".to_string()))?
            .first() ==
            Some(&1);

        let base_token_address = &snapshot.component.tokens[0];
        let quote_token_address = &snapshot.component.tokens[1];

        let base_token = all_tokens
            .get(base_token_address)
            .ok_or_else(|| {
                InvalidSnapshotError::ValueError(format!(
                    "Base token not found: {base_token_address}"
                ))
            })?
            .clone();

        let quote_token = all_tokens
            .get(quote_token_address)
            .ok_or_else(|| {
                InvalidSnapshotError::ValueError(format!(
                    "Quote token not found: {quote_token_address}"
                ))
            })?
            .clone();

        Ok(Self::new(reserve0, reserve1, stable, base_token, quote_token))
    }
}
