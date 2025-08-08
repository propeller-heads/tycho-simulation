use std::collections::{HashMap, HashSet};

use tycho_client::feed::synchronizer::ComponentWithState;
use tycho_common::{models::token::Token, Bytes};

use super::{models::HashflowMarketMakerLevels, state::HashflowState};
use crate::{
    protocol::{errors::InvalidSnapshotError, models::TryFromWithBlock},
    rfq::{models::TimestampHeader, protocols::hashflow::client::HashflowClient},
};

impl TryFromWithBlock<ComponentWithState, TimestampHeader> for HashflowState {
    type Error = InvalidSnapshotError;

    async fn try_from_with_header(
        snapshot: ComponentWithState,
        _timestamp_header: TimestampHeader,
        _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>,
        all_tokens: &HashMap<Bytes, Token>,
    ) -> Result<Self, Self::Error> {
        let state_attrs = snapshot.state.attributes;

        if snapshot.component.tokens.len() != 2 {
            return Err(InvalidSnapshotError::ValueError(
                "Component must have 2 tokens (base and quote)".to_string(),
            ));
        }

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

        // Parse the HashFlow market maker levels from the component attributes
        let levels_data = state_attrs
            .get("levels")
            .ok_or_else(|| {
                InvalidSnapshotError::MissingAttribute("levels attribute not found".to_string())
            })?;

        let levels: HashflowMarketMakerLevels = serde_json::from_slice(levels_data)
            .map_err(|e| {
                InvalidSnapshotError::ValueError(format!("Invalid levels JSON: {e}"))
            })?;

        // Validate that the levels correspond to the correct token pair
        if levels.pair.base_token != *base_token_address ||
            levels.pair.quote_token != *quote_token_address
        {
            return Err(InvalidSnapshotError::ValueError(format!(
                "Token pair mismatch: expected {base_token_address}/{quote_token_address}, got {}/{}",
                levels.pair.base_token, levels.pair.quote_token
            )));
        }

        // Create HashFlow client with minimal configuration for decoding
        let auth_user = "".to_string();
        let auth_key = "".to_string();

        let tvl_threshold = state_attrs
            .get("tvl_threshold")
            .and_then(|b| String::from_utf8_lossy(b).parse::<f64>().ok())
            .unwrap_or(0.0);

        let poll_time = state_attrs
            .get("poll_time")
            .and_then(|b| String::from_utf8_lossy(b).parse::<u64>().ok())
            .unwrap_or(60);

        let client = HashflowClient::new(
            snapshot.component.chain.into(),
            HashSet::from([base_token_address.clone(), quote_token_address.clone()]),
            tvl_threshold,
            HashSet::new(), // quote_tokens for TVL normalization
            auth_user,
            auth_key,
            poll_time,
        )
        .map_err(|e| {
            InvalidSnapshotError::MissingAttribute(format!("Couldn't create HashflowClient: {e}"))
        })?;

        Ok(HashflowState::new(base_token, quote_token, levels, client))
    }
}

#[cfg(test)]
mod tests {
    use tycho_common::{
        dto::{Chain, ChangeType, ProtocolComponent, ResponseProtocolState},
        models::Chain as ModelChain,
    };

    use super::*;

    fn wbtc() -> Token {
        Token::new(
            &hex::decode("2260fac5e5542a773aa44fbcfedf7c193bc2c599")
                .unwrap()
                .into(),
            "WBTC",
            8,
            0,
            &[Some(10_000)],
            ModelChain::Ethereum,
            100,
        )
    }

    fn usdc() -> Token {
        Token::new(
            &hex::decode("a0b86991c6218a76c1d19d4a2e9eb0ce3606eb48")
                .unwrap()
                .into(),
            "USDC",
            6,
            0,
            &[Some(10_000)],
            ModelChain::Ethereum,
            100,
        )
    }

    fn create_test_levels() -> serde_json::Value {
        let wbtc_token = wbtc();
        let usdc_token = usdc();

        serde_json::json!({
            "pair": {
                "baseToken": format!("0x{}", hex::encode(&wbtc_token.address)),
                "quoteToken": format!("0x{}", hex::encode(&usdc_token.address))
            },
            "levels": [
                {
                    "q": "1.5",
                    "p": "65000.0"
                },
                {
                    "q": "2.0", 
                    "p": "64950.0"
                },
                {
                    "q": "0.5",
                    "p": "65100.0"
                }
            ]
        })
    }

    fn create_test_snapshot() -> (ComponentWithState, HashMap<Bytes, Token>) {
        let wbtc_token = wbtc();
        let usdc_token = usdc();
        let levels = create_test_levels();

        let mut tokens = HashMap::new();
        tokens.insert(wbtc_token.address.clone(), wbtc_token.clone());
        tokens.insert(usdc_token.address.clone(), usdc_token.clone());

        let mut state_attributes = HashMap::new();
        
        // Serialize the levels to JSON
        let levels_json = serde_json::to_vec(&levels).expect("Failed to serialize levels");
        state_attributes.insert("levels".to_string(), levels_json.into());
        
        // Optional attributes
        state_attributes.insert(
            "tvl_threshold".to_string(),
            "1000.0".as_bytes().to_vec().into(),
        );
        state_attributes.insert(
            "poll_time".to_string(),
            "30".as_bytes().to_vec().into(),
        );

        let snapshot = ComponentWithState {
            state: ResponseProtocolState {
                attributes: state_attributes,
                component_id: "hashflow_wbtc_usdc".to_string(),
                balances: HashMap::new(),
            },
            component: ProtocolComponent {
                id: "hashflow_wbtc_usdc".to_string(),
                protocol_system: "hashflow".to_string(),
                protocol_type_name: "hashflow".to_string(),
                chain: Chain::Ethereum,
                tokens: vec![wbtc_token.address.clone(), usdc_token.address.clone()],
                contract_ids: Vec::new(),
                static_attributes: HashMap::new(),
                change: ChangeType::Creation,
                creation_tx: Bytes::default(),
                created_at: chrono::NaiveDateTime::default(),
            },
            component_tvl: None,
            entrypoints: Vec::new(),
        };

        (snapshot, tokens)
    }

    #[tokio::test]
    async fn test_try_from_with_header() {
        let (snapshot, tokens) = create_test_snapshot();

        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader { timestamp: 1703097600u64 },
            &HashMap::new(),
            &tokens,
        )
        .await
        .expect("create state from snapshot");

        assert_eq!(result.base_token.symbol, "WBTC");
        assert_eq!(result.quote_token.symbol, "USDC");
        assert_eq!(result.levels.levels.len(), 3);
        assert_eq!(result.levels.levels[0].quantity, 1.5);
        assert_eq!(result.levels.levels[0].price, 65000.0);
        assert_eq!(result.levels.levels[1].quantity, 2.0);
        assert_eq!(result.levels.levels[1].price, 64950.0);
        assert_eq!(result.levels.levels[2].quantity, 0.5);
        assert_eq!(result.levels.levels[2].price, 65100.0);
    }

    #[tokio::test]
    async fn test_try_from_missing_levels() {
        let (mut snapshot, tokens) = create_test_snapshot();
        snapshot.state.attributes.remove("levels");
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader::default(),
            &HashMap::new(),
            &tokens,
        )
        .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::MissingAttribute(_)));
    }

    #[tokio::test]
    async fn test_try_from_missing_token() {
        // Test missing second token (only one token in array)
        let (mut snapshot, tokens) = create_test_snapshot();
        snapshot.component.tokens.pop(); // Remove the second token
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader::default(),
            &HashMap::new(),
            &tokens,
        )
        .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::ValueError(_)));
    }

    #[tokio::test]
    async fn test_try_from_too_many_tokens() {
        // Test with three tokens instead of two
        let (mut snapshot, mut tokens) = create_test_snapshot();
        
        let dai_token = Token::new(
            &hex::decode("6b175474e89094c44da98b954eedeac495271d0f")
                .unwrap()
                .into(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            ModelChain::Ethereum,
            100,
        );
        
        tokens.insert(dai_token.address.clone(), dai_token.clone());
        snapshot.component.tokens.push(dai_token.address);
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader::default(),
            &HashMap::new(),
            &tokens,
        )
        .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::ValueError(_)));
    }

    #[tokio::test]
    async fn test_try_from_invalid_levels_json() {
        let (mut snapshot, tokens) = create_test_snapshot();

        // Insert invalid JSON for levels
        snapshot.state.attributes.insert(
            "levels".to_string(),
            "invalid json".as_bytes().to_vec().into(),
        );
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader::default(),
            &HashMap::new(),
            &tokens,
        )
        .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::ValueError(_)));
    }

    #[tokio::test]
    async fn test_try_from_token_pair_mismatch() {
        let (mut snapshot, tokens) = create_test_snapshot();
        
        // Create levels with different token pair
        let dai_token = Token::new(
            &hex::decode("6b175474e89094c44da98b954eedeac495271d0f")
                .unwrap()
                .into(),
            "DAI",
            18,
            0,
            &[Some(10_000)],
            ModelChain::Ethereum,
            100,
        );
        
        let mismatched_levels = serde_json::json!({
            "pair": {
                "baseToken": format!("0x{}", hex::encode(&dai_token.address)), // Different from snapshot
                "quoteToken": format!("0x{}", hex::encode(&usdc().address))
            },
            "levels": [
                {
                    "q": "1000.0",
                    "p": "1.0"
                }
            ]
        });
        
        let levels_json = serde_json::to_vec(&mismatched_levels).expect("Failed to serialize levels");
        snapshot.state.attributes.insert("levels".to_string(), levels_json.into());
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader::default(),
            &HashMap::new(),
            &tokens,
        )
        .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), InvalidSnapshotError::ValueError(_)));
    }

    #[tokio::test]
    async fn test_try_from_minimal_attributes() {
        // Test with only required attributes (levels)
        let (mut snapshot, tokens) = create_test_snapshot();
        
        // Remove all optional attributes, keep only levels
        let levels_data = snapshot.state.attributes.get("levels").unwrap().clone();
        snapshot.state.attributes.clear();
        snapshot.state.attributes.insert("levels".to_string(), levels_data);
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader { timestamp: 1703097600u64 },
            &HashMap::new(),
            &tokens,
        )
        .await
        .expect("create state from minimal snapshot");

        assert_eq!(result.base_token.symbol, "WBTC");
        assert_eq!(result.quote_token.symbol, "USDC");
        assert_eq!(result.levels.levels.len(), 3);
    }

    #[tokio::test]
    async fn test_try_from_empty_levels() {
        let (mut snapshot, tokens) = create_test_snapshot();
        
        // Create levels with empty price levels array
        let empty_levels = serde_json::json!({
            "pair": {
                "baseToken": format!("0x{}", hex::encode(&wbtc().address)),
                "quoteToken": format!("0x{}", hex::encode(&usdc().address))
            },
            "levels": []
        });
        
        let levels_json = serde_json::to_vec(&empty_levels).expect("Failed to serialize levels");
        snapshot.state.attributes.insert("levels".to_string(), levels_json.into());
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader { timestamp: 1703097600u64 },
            &HashMap::new(),
            &tokens,
        )
        .await
        .expect("create state with empty levels");

        assert_eq!(result.base_token.symbol, "WBTC");
        assert_eq!(result.quote_token.symbol, "USDC");
        assert_eq!(result.levels.levels.len(), 0);
    }

    #[tokio::test]
    async fn test_try_from_invalid_tvl_threshold() {
        let (mut snapshot, tokens) = create_test_snapshot();
        
        // Insert invalid TVL threshold (should fall back to default)
        snapshot.state.attributes.insert(
            "tvl_threshold".to_string(),
            "not_a_number".as_bytes().to_vec().into(),
        );
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader { timestamp: 1703097600u64 },
            &HashMap::new(),
            &tokens,
        )
        .await
        .expect("create state with invalid tvl threshold");

        // Should succeed with default values
        assert_eq!(result.base_token.symbol, "WBTC");
        assert_eq!(result.quote_token.symbol, "USDC");
    }

    #[tokio::test]
    async fn test_try_from_invalid_poll_time() {
        let (mut snapshot, tokens) = create_test_snapshot();
        
        // Insert invalid poll time (should fall back to default)
        snapshot.state.attributes.insert(
            "poll_time".to_string(),
            "not_a_number".as_bytes().to_vec().into(),
        );
        
        let result = HashflowState::try_from_with_header(
            snapshot,
            TimestampHeader { timestamp: 1703097600u64 },
            &HashMap::new(),
            &tokens,
        )
        .await
        .expect("create state with invalid poll time");

        // Should succeed with default values
        assert_eq!(result.base_token.symbol, "WBTC");
        assert_eq!(result.quote_token.symbol, "USDC");
    }
} 