use std::collections::HashMap;

use alloy::primitives::U256;
use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
use tycho_common::{models::token::Token, Bytes};

use super::{math, state::CurveStableSwapState};
use crate::protocol::{
    errors::InvalidSnapshotError,
    models::{DecoderContext, TryFromWithBlock},
};

impl TryFromWithBlock<ComponentWithState, BlockHeader> for CurveStableSwapState {
    type Error = InvalidSnapshotError;

    async fn try_from_with_header(
        snapshot: ComponentWithState,
        _block: BlockHeader,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        all_tokens: &HashMap<Bytes, Token>,
        _decoder_context: &DecoderContext,
    ) -> Result<Self, Self::Error> {
        let attributes = &snapshot.state.attributes;

        let reserve0 = U256::from_be_slice(
            attributes
                .get("reserve0")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve0".to_string()))?,
        );

        let reserve1 = U256::from_be_slice(
            attributes
                .get("reserve1")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve1".to_string()))?,
        );

        // On-chain A() returns raw A without precision. Internal math requires A * A_PRECISION.
        let raw_a = U256::from_be_slice(
            attributes
                .get("A")
                .ok_or(InvalidSnapshotError::MissingAttribute("A".to_string()))?,
        );
        let amplification = raw_a * math::A_PRECISION;

        let fee = U256::from_be_slice(
            attributes
                .get("fee")
                .ok_or(InvalidSnapshotError::MissingAttribute("fee".to_string()))?,
        );

        // Token decimals in canonical order (lower address = token0).
        // This matches how reserve0/reserve1 map to tokens across Tycho.
        let tokens = &snapshot.component.tokens;
        if tokens.len() != 2 {
            return Err(InvalidSnapshotError::MissingAttribute(format!(
                "expected 2 tokens, got {}",
                tokens.len()
            )));
        }

        let (addr0, addr1) =
            if tokens[0] < tokens[1] { (&tokens[0], &tokens[1]) } else { (&tokens[1], &tokens[0]) };

        let token0_decimals = all_tokens
            .get(addr0)
            .map(|t| t.decimals)
            .ok_or(InvalidSnapshotError::MissingAttribute("token0 decimals".to_string()))?;

        let token1_decimals = all_tokens
            .get(addr1)
            .map(|t| t.decimals)
            .ok_or(InvalidSnapshotError::MissingAttribute("token1 decimals".to_string()))?;

        let rate0 = math::rate_from_decimals(token0_decimals)
            .map_err(|e| InvalidSnapshotError::ValueError(e.to_string()))?;
        let rate1 = math::rate_from_decimals(token1_decimals)
            .map_err(|e| InvalidSnapshotError::ValueError(e.to_string()))?;

        Ok(CurveStableSwapState::new(reserve0, reserve1, amplification, fee, rate0, rate1))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::primitives::U256;
    use tycho_client::feed::{synchronizer::ComponentWithState, BlockHeader};
    use tycho_common::{
        dto::{ProtocolComponent as DtoProtocolComponent, ResponseProtocolState},
        models::{token::Token, Chain},
        Bytes,
    };

    use super::super::{math, state::CurveStableSwapState};
    use crate::protocol::{errors::InvalidSnapshotError, models::TryFromWithBlock};

    fn make_snapshot(
        attributes: HashMap<String, Bytes>,
        token_addresses: Vec<Bytes>,
    ) -> ComponentWithState {
        ComponentWithState {
            state: ResponseProtocolState {
                component_id: "CurvePool".to_owned(),
                attributes,
                balances: HashMap::new(),
            },
            component: DtoProtocolComponent { tokens: token_addresses, ..Default::default() },
            component_tvl: None,
            entrypoints: Vec::new(),
        }
    }

    fn dai_address() -> Bytes {
        Bytes::from(vec![0u8; 20])
    }

    fn usdc_address() -> Bytes {
        Bytes::from(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }

    fn all_tokens() -> HashMap<Bytes, Token> {
        let mut tokens = HashMap::new();
        tokens.insert(
            dai_address(),
            Token::new(&dai_address(), "DAI", 18, 0, &[Some(10_000)], Chain::Ethereum, 100),
        );
        tokens.insert(
            usdc_address(),
            Token::new(&usdc_address(), "USDC", 6, 0, &[Some(10_000)], Chain::Ethereum, 100),
        );
        tokens
    }

    #[tokio::test]
    async fn test_decode_snapshot() {
        let snapshot = make_snapshot(
            HashMap::from([
                ("reserve0".to_string(), Bytes::from(U256::from(1_000_000u64).to_be_bytes_vec())),
                ("reserve1".to_string(), Bytes::from(U256::from(2_000_000u64).to_be_bytes_vec())),
                ("A".to_string(), Bytes::from(U256::from(50u64).to_be_bytes_vec())),
                ("fee".to_string(), Bytes::from(U256::from(4_000_000u64).to_be_bytes_vec())),
            ]),
            vec![dai_address(), usdc_address()],
        );

        let result = CurveStableSwapState::try_from_with_header(
            snapshot,
            BlockHeader::default(),
            &HashMap::default(),
            &all_tokens(),
            &Default::default(),
        )
        .await;

        assert!(result.is_ok());
        let expected = CurveStableSwapState::new(
            U256::from(1_000_000u64),
            U256::from(2_000_000u64),
            U256::from(50u64) * math::A_PRECISION,
            U256::from(4_000_000u64),
            math::rate_from_decimals(18).unwrap(),
            math::rate_from_decimals(6).unwrap(),
        );
        assert_eq!(result.unwrap(), expected);
    }

    #[tokio::test]
    async fn test_decode_missing_reserve0() {
        let snapshot = make_snapshot(
            HashMap::from([(
                "reserve1".to_string(),
                Bytes::from(U256::from(100u64).to_be_bytes_vec()),
            )]),
            vec![dai_address(), usdc_address()],
        );

        let result = CurveStableSwapState::try_from_with_header(
            snapshot,
            BlockHeader::default(),
            &HashMap::default(),
            &all_tokens(),
            &Default::default(),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InvalidSnapshotError::MissingAttribute(ref x) if x == "reserve0"
        ));
    }

    #[tokio::test]
    async fn test_decode_missing_amplification() {
        let snapshot = make_snapshot(
            HashMap::from([
                ("reserve0".to_string(), Bytes::from(U256::from(500u64).to_be_bytes_vec())),
                ("reserve1".to_string(), Bytes::from(U256::from(600u64).to_be_bytes_vec())),
                ("fee".to_string(), Bytes::from(U256::from(4_000_000u64).to_be_bytes_vec())),
            ]),
            vec![dai_address(), usdc_address()],
        );

        let result = CurveStableSwapState::try_from_with_header(
            snapshot,
            BlockHeader::default(),
            &HashMap::default(),
            &all_tokens(),
            &Default::default(),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            InvalidSnapshotError::MissingAttribute(ref x) if x == "A"
        ));
    }
}
