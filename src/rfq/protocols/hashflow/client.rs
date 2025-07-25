#![allow(dead_code)] // TODO remove this
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    time::SystemTime,
};

use alloy::primitives::{utils::keccak256, Address};
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use tracing::{error, info};
use tycho_common::{
    models::protocol::GetAmountOutParams, simulation::indicatively_priced::SignedQuote, Bytes,
};
use tokio::time::{interval, Duration};

use crate::{
    rfq::{
        client::RFQClient,
        errors::RFQError,
        models::TimestampHeader,
        protocols::hashflow::models::{
            HashflowPriceLevelsResponse, 
            HashflowMarketMakersResponse, HashflowMarketMakerLevel
        },
    },
    tycho_client::feed::synchronizer::{ComponentWithState, Snapshot, StateSyncMessage},
    tycho_common::dto::{ProtocolComponent, ResponseProtocolState},
    tycho_core::dto::Chain,
};

fn checksum(address: &String) -> Result<String, RFQError> {
    let checksum = Address::from_str(address)
        .map_err(|_| RFQError::ParsingError(format!("Invalid token address: {address}.")))?;
    Ok(checksum.to_checksum(None))
}

fn pair_to_hashflow_format(pair: &(String, String)) -> Result<String, RFQError> {
    // Checksum addresses to match API output
    let token0 = checksum(&pair.0)?;
    let token1 = checksum(&pair.1)?;
    Ok(format!("{token0}/{token1}"))
}

fn bytes_to_address(address: &Bytes) -> Result<Address, RFQError> {
    if address.len() == 20 {
        Ok(Address::from_slice(address))
    } else {
        Err(RFQError::InvalidInput(format!("Invalid ERC20 token address: {address:?}")))
    }
}

/// Maps a Chain to its corresponding Hashflow chain identifier
fn chain_to_hashflow_id(chain: Chain) -> Result<u32, RFQError> {
    match chain {
        Chain::Ethereum => Ok(1),
        Chain::Base => Ok(8453),
        _ => Err(RFQError::FatalError(format!("Unsupported chain: {chain:?}"))),
    }
}

#[derive(Clone)]
pub struct HashflowClient {
    chain: Chain,
    chain_id: u32,
    price_levels_endpoint: String,
    market_makers_endpoint: String,
    quote_endpoint: String,
    // Pairs that we want prices for
    pairs: HashSet<String>,
    // Min tvl value in the quote token.
    tvl: f64,
    // HTTP client for API requests
    http_client: Client,
    // Authorization key from environment
    auth_key: String,
    // Source identifier for API requests
    source: String,
    // quote tokens to normalize to for TVL purposes. Should have the same prices.
    quote_tokens: HashSet<String>,
    // Cached market makers
    market_makers: Vec<String>,
}

impl HashflowClient {
    pub fn new(
        chain: Chain,
        pairs: HashSet<(String, String)>,
        tvl: f64,
        quote_tokens: HashSet<String>,
        auth_key: String,
    ) -> Result<Self, RFQError> {
        let chain_id = chain_to_hashflow_id(chain)?;

        let mut pair_names: HashSet<String> = HashSet::new();
        for pair in pairs {
            pair_names.insert(pair_to_hashflow_format(&pair)?);
        }
        let mut quote_tokens_checksummed: HashSet<String> = HashSet::new();

        for token in quote_tokens.iter() {
            quote_tokens_checksummed.insert(checksum(token)?.to_string());
        }

        Ok(Self {
            chain,
            chain_id,
            price_levels_endpoint: "https://api.hashflow.com/taker/v3/price-levels".to_string(),
            market_makers_endpoint: "https://api.hashflow.com/taker/v3/market-makers".to_string(),
            quote_endpoint: "https://api.hashflow.com/taker/v3/rfq".to_string(),
            pairs: pair_names,
            tvl,
            http_client: Client::new(),
            auth_key,
            source: "propellerheads".to_string(),
            quote_tokens: quote_tokens_checksummed,
            market_makers: Vec::new(), // Will be populated during first fetch
        })
    }

    fn create_component_with_state(
        &self,
        component_id: String,
        tokens: Vec<tycho_common::Bytes>,
        mm_level: &HashflowMarketMakerLevel,
        tvl: f64,
    ) -> ComponentWithState {
        let protocol_component = ProtocolComponent {
            id: component_id.clone(),
            protocol_system: "rfq:hashflow".to_string(),
            protocol_type_name: "hashflow_pool".to_string(),
            chain: self.chain,
            tokens,
            contract_ids: vec![], // empty for RFQ
            static_attributes: Default::default(),
            change: Default::default(),
            creation_tx: Default::default(),
            created_at: Default::default(),
        };

        let mut attributes = HashMap::new();

        // Store price levels as JSON string
        if !mm_level.levels.is_empty() {
            let levels_json = serde_json::to_string(&mm_level.levels).unwrap_or_default();
            attributes.insert("levels".to_string(), levels_json.as_bytes().to_vec().into());
        }

        // Store pair information
        attributes.insert("base_token".to_string(), mm_level.pair.base_token.as_bytes().to_vec().into());
        attributes.insert("quote_token".to_string(), mm_level.pair.quote_token.as_bytes().to_vec().into());
        attributes.insert("base_token_name".to_string(), mm_level.pair.base_token_name.as_bytes().to_vec().into());
        attributes.insert("quote_token_name".to_string(), mm_level.pair.quote_token_name.as_bytes().to_vec().into());

        ComponentWithState {
            state: ResponseProtocolState {
                component_id: component_id.clone(),
                attributes,
                balances: HashMap::new(),
            },
            component: protocol_component,
            component_tvl: Some(tvl),
            entrypoints: vec![],
        }
    }

    async fn fetch_market_makers(&mut self) -> Result<(), RFQError> {
        let query_params = vec![
            ("source", self.source.clone()),
            ("baseChainType", "evm".to_string()),
            ("baseChainId", self.chain_id.to_string()),
        ];

        let request = self
            .http_client
            .get(&self.market_makers_endpoint)
            .query(&query_params)
            .header("accept", "application/json")
            .header("Authorization", &self.auth_key);

        let response = request
            .send()
            .await
            .map_err(|e| RFQError::ConnectionError(format!("Failed to fetch market makers: {e}")))?;

        if !response.status().is_success() {
            return Err(RFQError::ConnectionError(format!(
                "HTTP error {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let mm_response: HashflowMarketMakersResponse = response
            .json()
            .await
            .map_err(|e| RFQError::ParsingError(format!("Failed to parse market makers response: {e}")))?;

        self.market_makers = mm_response.market_makers;

        info!("Fetched {} market makers: {:?}", self.market_makers.len(), self.market_makers);

        Ok(())
    }

    async fn fetch_price_levels(&self) -> Result<HashMap<String, Vec<HashflowMarketMakerLevel>>, RFQError> {
        let mut query_params = vec![
            ("source", self.source.clone()),
            ("baseChainType", "evm".to_string()),
            ("baseChainId", self.chain_id.to_string()),
        ];

        // Add market makers as array parameters
        for mm in &self.market_makers {
            query_params.push(("marketMakers[]", mm.clone()));
        }

        let request = self
            .http_client
            .get(&self.price_levels_endpoint)
            .query(&query_params)
            .header("accept", "application/json")
            .header("Authorization", &self.auth_key);

        let response = request
            .send()
            .await
            .map_err(|e| RFQError::ConnectionError(format!("Failed to fetch price levels: {e}")))?;

        if !response.status().is_success() {
            return Err(RFQError::ConnectionError(format!(
                "HTTP error {}: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            )));
        }

        let price_response: HashflowPriceLevelsResponse = response
            .json()
            .await
            .map_err(|e| RFQError::ParsingError(format!("Failed to parse price levels response: {e}")))?;

        if price_response.status != "success" {
            return Err(RFQError::ConnectionError(format!(
                "API returned error status: {}",
                price_response.error.unwrap_or_default()
            )));
        }

        info!("Fetched price levels: {:?}", price_response.levels.clone());

        price_response.levels.ok_or_else(|| {
            RFQError::ParsingError("API response missing levels".to_string())
        })
    }
}

#[async_trait]
impl RFQClient for HashflowClient {

    fn stream(
        &self,
    ) -> BoxStream<'static, Result<(String, StateSyncMessage<TimestampHeader>), RFQError>> {
        let mut client = self.clone();

        Box::pin(async_stream::stream! {
            let mut current_components: HashMap<String, ComponentWithState> = HashMap::new();
            let mut ticker = interval(Duration::from_secs(1));
            let mut market_makers_fetched = false;

            info!("Starting Hashflow price levels polling every 10 seconds");

            loop {
                ticker.tick().await;
                // Fetch market makers on first iteration
                if !market_makers_fetched {
                    match client.fetch_market_makers().await {
                        Ok(()) => {
                            market_makers_fetched = true;
                            info!("Successfully fetched market makers");
                        }
                        Err(e) => {
                            println!("Failed to fetch market makers: {}", e);
                            error!("Failed to fetch market makers: {}", e);
                            continue;
                        }
                    }
                }

                match client.fetch_price_levels().await {
                    Ok(levels_by_mm) => {
                        let mut new_components = HashMap::new();

                        // Process all market maker levels
                        for (_market_maker, mm_levels) in levels_by_mm.iter() {
                            for mm_level in mm_levels {
                                let pair_key = mm_level.pair_key();

                                if client.pairs.contains(&pair_key) {
                                    // Hash the pair key for component id
                                    let component_id = format!("{}", keccak256(&pair_key));

                                    let token_0_bytes = Bytes::from_str(&mm_level.pair.base_token)
                                        .map_err(|_| RFQError::ParsingError(format!("String cannot be converted to bytes {}", mm_level.pair.base_token)))?;
                                    let token_1_bytes = Bytes::from_str(&mm_level.pair.quote_token)
                                        .map_err(|_| RFQError::ParsingError(format!("String cannot be converted to bytes {}", mm_level.pair.quote_token)))?;

                                    let tokens = vec![token_0_bytes, token_1_bytes];

                                    let tvl = mm_level.calculate_tvl();

                                    // Apply TVL normalization if needed
                                    let normalized_tvl = if !client.quote_tokens.contains(&mm_level.pair.quote_token) {
                                        // Try to find a quote price for normalization
                                        let mut quote_tvl = None;
                                        for quote_token in &client.quote_tokens {
                                            let quote_pair_key = format!("{}/{}", mm_level.pair.quote_token, quote_token);
                                            for (_mm, mm_levels_inner) in levels_by_mm.iter() {
                                                for quote_mm_level in mm_levels_inner {
                                                    if quote_mm_level.pair_key() == quote_pair_key {
                                                        if let Some(price) = quote_mm_level.get_estimated_price(tvl) {
                                                            quote_tvl = Some(tvl * price);
                                                            break;
                                                        }
                                                    }
                                                }
                                                if quote_tvl.is_some() {
                                                    break;
                                                }
                                            }
                                            if quote_tvl.is_some() {
                                                break;
                                            }
                                        }
                                        quote_tvl.unwrap_or(tvl)
                                    } else {
                                        tvl
                                    };

                                    if normalized_tvl < client.tvl {
                                        continue;
                                    }

                                    let component_with_state = client.create_component_with_state(
                                        component_id.clone(),
                                        tokens,
                                        mm_level,
                                        normalized_tvl
                                    );
                                    new_components.insert(component_id, component_with_state);
                                }
                            }
                        }

                        // Find components that were removed
                        let removed_components: HashMap<String, ProtocolComponent> = current_components
                            .iter()
                            .filter(|&(id, _)| !new_components.contains_key(id))
                            .map(|(k, v)| (k.clone(), v.component.clone()))
                            .collect();

                        // Update current state
                        current_components = new_components.clone();

                        let snapshot = Snapshot {
                            states: new_components,
                            vm_storage: HashMap::new(),
                        };
                        let timestamp = SystemTime::now().duration_since(
                            SystemTime::UNIX_EPOCH
                        ).map_err(
                            |_| RFQError::ParsingError("SystemTime before UNIX EPOCH!".into())
                        )?.as_secs();

                        let msg = StateSyncMessage::<TimestampHeader> {
                            header: TimestampHeader { timestamp },
                            snapshots: snapshot,
                            deltas: None,
                            removed_components,
                        };

                        yield Ok(("hashflow".to_string(), msg));
                    },
                    Err(e) => {
                        println!("Failed to fetch price levels from Hashflow API: {}", e);
                        error!("Failed to fetch price levels from Hashflow API: {}", e);
                        continue;
                    }
                }
            }
        })
    }
    
    async fn request_binding_quote(&self, _params: &GetAmountOutParams) -> Result<SignedQuote, RFQError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use tokio::time::timeout;
    use futures::StreamExt;
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access and HASHFLOW_KEY environment variable
    async fn test_hashflow_api_polling() {
        use std::env;
        let hashflow_key = env::var("HASHFLOW_KEY").unwrap();

        let lpt = "0x58b6a8a3302369daec383334672404ee733ab239".to_string();
        let usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".to_string();

        let quote_tokens = HashSet::from([
            String::from("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"), // USDC
            String::from("0xdac17f958d2ee523a2206206994597c13d831ec7"), // USDT
        ]);

        let client = HashflowClient::new(
            Chain::Ethereum,
            HashSet::from_iter(vec![(lpt.to_string(), usdc.to_string())]),
            10.0, // $10 minimum TVL
            quote_tokens,
            hashflow_key,
        )
        .unwrap();

        let mut stream = client.stream();

        // Test API polling and message reception with timeout
        let result = timeout(Duration::from_secs(10), async {
            let mut message_count = 0;
            let max_messages = 3;

            while let Some(result) = stream.next().await {
                match result {
                    Ok((component_id, msg)) => {
                        println!("Received message with ID: {component_id}");

                        assert!(!component_id.is_empty());
                        assert_eq!(component_id, "hashflow");
                        assert!(msg.header.timestamp > 0);

                        let snapshot = &msg.snapshots;

                        println!("Received {} components in this message", snapshot.states.len());
                        for (id, component_with_state) in &snapshot.states {
                            assert_eq!(
                                component_with_state.component.protocol_system,
                                "rfq:hashflow"
                            );
                            assert_eq!(
                                component_with_state.component.protocol_type_name,
                                "hashflow_pool"
                            );
                            assert_eq!(component_with_state.component.chain, Chain::Ethereum);

                            let attributes = &component_with_state.state.attributes;

                            // Check that levels exist
                            if attributes.contains_key("levels") {
                                assert!(!attributes["levels"].is_empty());
                            }

                            if let Some(tvl) = component_with_state.component_tvl {
                                assert!(tvl >= 0.0);
                                println!("Component {id} TVL: ${tvl:.2}");
                            }
                        }

                        message_count += 1;
                        if message_count >= max_messages {
                            break;
                        }
                    }
                    Err(e) => {
                        panic!("Stream error: {e}");
                    }
                }
            }

            assert!(message_count > 0, "Should have received at least one message");
            println!("Successfully received {message_count} messages");
        })
        .await;

        match result {
            Ok(_) => println!("Test completed successfully"),
            Err(_) => panic!("Test timed out - no messages received within 10 seconds"),
        }
    }
}
