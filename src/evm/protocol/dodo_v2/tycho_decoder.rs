use std::collections::HashMap;

use alloy::primitives::{Address, U256};
use tycho_client::feed::{synchronizer::ComponentWithState, Header};
use tycho_common::{models::token::Token, Bytes};

use crate::{
    evm::protocol::dodo_v2::state::{DodoV2State, RState},
    protocol::{errors::InvalidSnapshotError, models::TryFromWithBlock},
};

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

        let r = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("R")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("R".to_string()))?,
        );

        let r_state = RState::try_from(r)
            .map_err(|_| InvalidSnapshotError::ValueError("Unsupported R value".to_string()))?;

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

        let mt_fee_rate = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("MT_FEE_RATE")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("MT_FEE_RATE".to_string()))?,
        );

        let lp_fee_rate = snapshot
            .state
            .attributes
            .get("LP_FEE_RATE")
            .map(|v| U256::from_be_slice(v))
            .unwrap_or_else(|| U256::ZERO);

        let mt_fee_base = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("MT_FEE_BASE")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("MT_FEE_BASE".to_string()))?,
        );

        let mt_fee_quote = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("MT_FEE_QUOTE")
                .ok_or_else(|| {
                    InvalidSnapshotError::MissingAttribute("MT_FEE_QUOTE".to_string())
                })?,
        );

        let base_token = snapshot
            .component
            .tokens
            .first()
            .cloned()
            .ok_or_else(|| {
                InvalidSnapshotError::ValueError(
                    "Missing base token in snapshot.component.tokens".to_string(),
                )
            })?;

        let quote_token = snapshot
            .component
            .tokens
            .get(1)
            .cloned()
            .ok_or_else(|| {
                InvalidSnapshotError::ValueError(
                    "Missing quote token in snapshot.component.tokens".to_string(),
                )
            })?;

        let base_token_addr = Address::from_slice(&base_token);
        let quote_token_addr = Address::from_slice(&quote_token);

        let component_balances = snapshot
            .state
            .balances
            .iter()
            .map(|(k, v)| (Address::from_slice(k), U256::from_be_slice(v)))
            .collect::<HashMap<_, _>>();

        if !component_balances.contains_key(&base_token_addr) ||
            !component_balances.contains_key(&quote_token_addr)
        {
            return Err(InvalidSnapshotError::ValueError(
                "Component balances do not contain base or quote token".to_string(),
            ));
        }

        Ok(DodoV2State::new(
            i,
            k,
            b,
            q,
            b0,
            q0,
            r_state,
            lp_fee_rate,
            mt_fee_rate,
            mt_fee_quote,
            mt_fee_base,
            base_token,
            quote_token,
            component_balances,
            snapshot.component.id,
        )?)
    }
}
