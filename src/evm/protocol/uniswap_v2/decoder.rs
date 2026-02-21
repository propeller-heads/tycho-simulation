use std::collections::HashMap;

use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
use tycho_common::{models::token::Token, Bytes};

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
        _all_tokens: &HashMap<Bytes, Token>,
        _decoder_context: &DecoderContext,
    ) -> Result<Self, Self::Error> {
        let fee_bps = snapshot
            .component
            .static_attributes
            .get("fee")
            .map(|b| {
                let bytes = b.as_ref();
                let mut buf = [0u8; 4];
                let start = 4usize.saturating_sub(bytes.len());
                let src_start = bytes.len().saturating_sub(4);
                buf[start..].copy_from_slice(&bytes[src_start..]);
                u32::from_be_bytes(buf)
            })
            .unwrap_or(30);
        if fee_bps > 10000 {
            return Err(InvalidSnapshotError::ValueError(format!(
                "fee_bps must be <= 10000, got {fee_bps}"
            )));
        }
        let (reserve0, reserve1) = cpmm_try_from_with_header(snapshot)?;
        Ok(Self::new_with_fee(reserve0, reserve1, fee_bps))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::primitives::U256;
    use rstest::rstest;
    use tycho_client::feed::synchronizer::ComponentWithState;
    use tycho_common::{dto::ResponseProtocolState, Bytes};

    use super::super::state::UniswapV2State;
    use crate::{
        evm::protocol::test_utils::try_decode_snapshot_with_defaults,
        protocol::errors::InvalidSnapshotError,
    };

    #[tokio::test]
    async fn test_usv2_try_from() {
        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component: Default::default(),
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), UniswapV2State::new(U256::from(0u64), U256::from(0u64)));
    }

    #[tokio::test]
    async fn test_usv2_try_from_with_custom_fee() {
        let mut component: tycho_common::dto::ProtocolComponent = Default::default();
        // BigInt(25).to_signed_bytes_be() produces [25]
        component
            .static_attributes
            .insert("fee".to_string(), Bytes::from(vec![25u8]));

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component,
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.fee_bps, 25);
        assert_eq!(state.reserve0, U256::from(0u64));
        assert_eq!(state.reserve1, U256::from(0u64));
    }

    #[tokio::test]
    async fn test_usv2_try_from_missing_fee_defaults_to_30() {
        // No "fee" in static_attributes => defaults to 30 bps
        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component: Default::default(),
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().fee_bps, 30);
    }

    #[tokio::test]
    async fn test_usv2_try_from_zero_fee() {
        let mut component: tycho_common::dto::ProtocolComponent = Default::default();
        component
            .static_attributes
            .insert("fee".to_string(), Bytes::from(vec![0u8]));

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component,
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().fee_bps, 0);
    }

    #[tokio::test]
    async fn test_usv2_try_from_fee_exceeds_max() {
        let mut component: tycho_common::dto::ProtocolComponent = Default::default();
        // 10001 in big-endian = [0x27, 0x11]
        component
            .static_attributes
            .insert("fee".to_string(), Bytes::from(vec![0x27, 0x11]));

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component,
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::ValueError(_)));
    }

    #[tokio::test]
    async fn test_usv2_try_from_empty_fee_bytes() {
        let mut component: tycho_common::dto::ProtocolComponent = Default::default();
        component
            .static_attributes
            .insert("fee".to_string(), Bytes::from(vec![]));

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                component_id: "State1".to_owned(),
                attributes: HashMap::from([
                    ("reserve0".to_string(), Bytes::from(vec![0; 32])),
                    ("reserve1".to_string(), Bytes::from(vec![0; 32])),
                ]),
                balances: HashMap::new(),
            },
            component,
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        // Empty bytes => all zeros => fee_bps=0, which is valid
        assert!(result.is_ok());
        assert_eq!(result.unwrap().fee_bps, 0);
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

        let result = try_decode_snapshot_with_defaults::<UniswapV2State>(snapshot).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InvalidSnapshotError::MissingAttribute(ref x) if x == missing_attribute
        ));
    }
}
