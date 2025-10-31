use std::collections::HashMap;

use crate::{
    evm::protocol::cowamm::{bmath, state::CowAMMState},
    protocol::{errors::InvalidSnapshotError, models::TryFromWithBlock},
};
use alloy::primitives::U256;
use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
use tycho_common::{
    dto::{ProtocolComponent, ResponseProtocolState},
    models::token::Token,
    simulation::errors::{SimulationError, TransitionError},
    Bytes,
};

const BYTES: usize = 32;

fn header() -> BlockHeader {
    BlockHeader {
        number: 1,
        hash: Bytes::from(vec![0; 32]),
        parent_hash: Bytes::from(vec![0; 32]),
        revert: false,
        timestamp: 0,
    }
}

impl TryFromWithBlock<ComponentWithState, BlockHeader> for CowAMMState {
    type Error = InvalidSnapshotError;

    async fn try_from_with_block(
        snapshot: ComponentWithState,
        _block: BlockHeader,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        _all_tokens: &HashMap<Bytes, Token>,
    ) -> Result<Self, Self::Error> {
        let address = snapshot
            .component
            .static_attributes
            .get("address")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("address".to_string()))?
            .clone();

        let token_a = snapshot
            .component
            .static_attributes
            .get("token_a")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("token_a".to_string()))?
            .clone();

        let liquidity_a = U256::from_be_bytes::<BYTES>(
            snapshot
                .state
                .attributes
                .get("liquidity_a")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("liquidity_a".to_string()))?
                .as_ref()
                .try_into()
                .map_err(|err| {
                    InvalidSnapshotError::ValueError(format!(
                        "liquidity_a length mismatch: {err:?}"
                    ))
                })?,
        );

        let liquidity_b = U256::from_be_bytes::<BYTES>(
            snapshot
                .state
                .attributes
                .get("liquidity_b")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("liquidity_b".to_string()))?
                .as_ref()
                .try_into()
                .map_err(|err| {
                    InvalidSnapshotError::ValueError(format!(
                        "liquidity_b length mismatch: {err:?}"
                    ))
                })?,
        );

        let token_b = snapshot
            .component
            .static_attributes
            .get("token_b")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("token_b".to_string()))?
            .clone();

        let lp_token = snapshot
            .component
            .static_attributes
            .get("lp_token")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("lp_token".to_string()))?
            .clone();

        let lp_token_supply = U256::from_be_bytes::<BYTES>(
            snapshot
                .state
                .attributes
                .get("lp_token_supply")
                .ok_or_else(|| {
                    InvalidSnapshotError::MissingAttribute("lp_token_supply".to_string())
                })?
                .as_ref()
                .try_into()
                .map_err(|err| {
                    InvalidSnapshotError::ValueError(format!(
                        "lp_token_supply length mismatch: {err:?}"
                    ))
                })?,
        );

        let fee = 0u64;

        //weight_a and weight_b are left padded big endian numbers of 32 bytes
        //we want a U256 number from the hex representation
        let weight_a = U256::from_be_bytes::<BYTES>(
            snapshot
                .component
                .static_attributes
                .get("weight_a")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("weight_a".to_string()))?
                .as_ref()
                .try_into()
                .map_err(|err| {
                    InvalidSnapshotError::ValueError(format!("weight_a length mismatch: {err:?}"))
                })?,
        );

        let weight_b = U256::from_be_bytes::<BYTES>(
            snapshot
                .component
                .static_attributes
                .get("weight_b")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("weight_b".to_string()))?
                .as_ref()
                .try_into()
                .map_err(|err| {
                    InvalidSnapshotError::ValueError(format!("weight_b length mismatch: {err:?}"))
                })?,
        );

        Ok(Self::new(
            address,
            token_a,
            token_b,
            liquidity_a,
            liquidity_b,
            lp_token,
            lp_token_supply,
            weight_a,
            weight_b,
            fee,
        ))
    }
}

pub fn attributes() -> HashMap<String, Bytes> {
    HashMap::from([
        ("liquidity_a".to_string(), Bytes::from(vec![0; 32])),
        ("liquidity_b".to_string(), Bytes::from(vec![0; 32])),
        ("lp_token_supply".to_string(), Bytes::from(vec![0; 32])),
    ])
}
pub fn static_attributes() -> HashMap<String, Bytes> {
    HashMap::from([
        ("address".to_string(), Bytes::from(vec![0; 32])),
        ("weight_a".to_string(), Bytes::from(vec![0; 32])),
        ("weight_b".to_string(), Bytes::from(vec![0; 32])),
        ("token_a".to_string(), Bytes::from(vec![0; 32])),
        ("token_b".to_string(), Bytes::from(vec![0; 32])),
        ("lp_token".to_string(), Bytes::from(vec![0; 32])),
        ("fee".to_string(), 0u64.into()),
    ])
}

pub fn component() -> ProtocolComponent {
    ProtocolComponent { static_attributes: static_attributes(), ..Default::default() }
}

pub fn state() -> CowAMMState {
    CowAMMState::new(
        Bytes::from(vec![0; 32]),
        Bytes::from(vec![0; 32]),
        Bytes::from(vec![0; 32]),
        U256::from(0),
        U256::from(0),
        Bytes::from(vec![0; 32]),
        U256::from(0),
        U256::from(0),
        U256::from(0),
        0u64,
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use tycho_common::dto::ResponseProtocolState;

    use super::*;

    #[tokio::test]
    async fn test_cowamm_try_from_with_block() {
        let snapshot = ComponentWithState {
            state: ResponseProtocolState { attributes: attributes(), ..Default::default() },
            component: component(),
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = CowAMMState::try_from_with_block(
            snapshot,
            header(),
            &HashMap::default(),
            &HashMap::default(),
        )
        .await
        .unwrap();

        assert_eq!(state(), result);
    }

    #[tokio::test]
    #[rstest]
    #[case::missing_weight_a("address")]
    #[case::missing_weight_a("weight_a")]
    #[case::missing_weight_b("weight_b")]
    #[case::missing_token_a("token_a")]
    #[case::missing_token_b("token_b")]
    #[case::missing_liquidity_a("liquidity_a")]
    #[case::missing_liquidity_b("liquidity_b")]
    #[case::missing_lp_token("lp_token")]
    #[case::missing_lp_token_supply("lp_token_supply")]

    async fn test_cowamm_try_from_missing_attribute(#[case] missing_attribute: &str) {
        let mut component = component();
        let mut attributes = attributes();

        let _ = match missing_attribute {
            "liquidity_a" | "liquidity_b" | "lp_token_supply" => {
                attributes.remove(missing_attribute)
            }
            "address" | "weight_a" | "weight_b" | "token_a" | "token_b" | "lp_token" => component
                .static_attributes
                .remove(missing_attribute),
            &_ => None,
        };

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes,
                balances: HashMap::new(),
            },
            component,
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = CowAMMState::try_from_with_block(
            snapshot,
            header(),
            &HashMap::default(),
            &HashMap::default(),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InvalidSnapshotError::MissingAttribute(ref attr) if attr == missing_attribute
        ));
    }
}
