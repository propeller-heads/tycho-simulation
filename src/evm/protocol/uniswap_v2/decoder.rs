use std::{collections::HashMap, sync::Arc};

use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
use tycho_common::{
    models::{protocol::ProtocolComponent, token::Token, ChangeType},
    Bytes,
};

use crate::{
    evm::protocol::{cpmm::protocol::cpmm_try_from_with_header, uniswap_v2::state::UniswapV2State},
    protocol::{
        errors::InvalidSnapshotError,
        models::{DecoderContext, TryFromWithBlock},
    },
};

impl TryFromWithBlock<ComponentWithState, BlockHeader> for UniswapV2State {
    type Error = InvalidSnapshotError;

    /// Decodes a `ComponentWithState` into a `UniswapV2State`. Errors with a `InvalidSnapshotError`
    /// if either reserve0 or reserve1 attributes are missing.
    async fn try_from_with_header(
        snapshot: ComponentWithState,
        _block: BlockHeader,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        tokens: &HashMap<Bytes, Token>,
        _decoder_context: &DecoderContext,
    ) -> Result<Self, Self::Error> {
        let component = ProtocolComponent::<Arc<Token>>::new(
            snapshot.component.id.as_str(),
            snapshot
                .component
                .protocol_system
                .as_str(),
            snapshot
                .component
                .protocol_type_name
                .as_str(),
            snapshot.component.chain.into(),
            snapshot
                .component
                .tokens
                .iter()
                .map(|t| Arc::new(tokens.get(t).unwrap().clone()))
                .collect(),
            snapshot.component.contract_ids.clone(),
            snapshot
                .component
                .static_attributes
                .clone(),
            ChangeType::Creation,
            snapshot.component.creation_tx.clone(),
            snapshot.component.created_at,
        );
        let (reserve0, reserve1) = cpmm_try_from_with_header(snapshot)?;

        Ok(Self::new(reserve0, reserve1, Arc::new(component)))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, str::FromStr, sync::Arc};

    use alloy::primitives::U256;
    use chrono::NaiveDateTime;
    use rstest::rstest;
    use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
    use tycho_common::{
        dto::ResponseProtocolState,
        models::{protocol::ProtocolComponent, token::Token, Chain, ChangeType},
        Bytes,
    };

    use super::super::state::UniswapV2State;
    use crate::protocol::{
        errors::InvalidSnapshotError,
        models::{DecoderContext, TryFromWithBlock},
    };

    fn header() -> BlockHeader {
        BlockHeader {
            number: 1,
            hash: Bytes::from(vec![0; 32]),
            parent_hash: Bytes::from(vec![0; 32]),
            revert: false,
            timestamp: 1,
        }
    }

    fn component(t0_decimals: u32, token_1_decimals: u32) -> Arc<ProtocolComponent<Arc<Token>>> {
        let t0 = Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000000").unwrap(),
            "T0",
            t0_decimals,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let t1 = Token::new(
            &Bytes::from_str("0x0000000000000000000000000000000000000001").unwrap(),
            "T0",
            token_1_decimals,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        Arc::new(ProtocolComponent::<_>::new(
            "0xtest",
            "uniswap_v2",
            "uniswap_v2_pool",
            Chain::Ethereum,
            vec![Arc::new(t0), Arc::new(t1)],
            vec![],
            HashMap::new(),
            ChangeType::Creation,
            Bytes::default(),
            NaiveDateTime::default(),
        ))
    }

    #[tokio::test]
    async fn test_usv2_try_from() {
        let c = component(18, 18);
        let tokens = c
            .tokens
            .clone()
            .into_iter()
            .map(|t| (t.address.clone(), t.as_ref().clone()))
            .collect();
        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component: Arc::try_unwrap(c).unwrap().into(),
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let decoder_context = DecoderContext::new();
        let result = UniswapV2State::try_from_with_header(
            snapshot,
            header(),
            &HashMap::new(),
            &tokens,
            &decoder_context,
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            UniswapV2State::new(U256::from(0u64), U256::from(0u64), component(18, 18))
        );
    }

    #[tokio::test]
    #[rstest]
    #[case::missing_reserve0("reserve0")]
    #[case::missing_reserve1("reserve1")]
    async fn test_usv2_try_from_missing_attribute(#[case] missing_attribute: &str) {
        let mut attributes = HashMap::from([
            ("reserve0".to_string(), Bytes::from(vec![0; 32])),
            ("reserve1".to_string(), Bytes::from(vec![0; 32])),
        ]);
        attributes.remove(missing_attribute);

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes,
                balances: HashMap::new(),
            },
            component: Default::default(),
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let decoder_context = DecoderContext::new();
        let result = UniswapV2State::try_from_with_header(
            snapshot,
            header(),
            &HashMap::new(),
            &HashMap::new(),
            &decoder_context,
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InvalidSnapshotError::MissingAttribute(ref x) if x == missing_attribute
        ));
    }
}
