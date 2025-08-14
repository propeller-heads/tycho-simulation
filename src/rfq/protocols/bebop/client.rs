use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    time::SystemTime,
};

use alloy::primitives::{utils::keccak256, Address};
use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use http::Request;
use num_bigint::BigUint;
use prost::Message as ProstMessage;
use reqwest::Client;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{handshake::client::generate_key, Message},
};
use tracing::{error, info, warn};
use tycho_common::{
    models::{protocol::GetAmountOutParams, Chain},
    simulation::indicatively_priced::SignedQuote,
    Bytes,
};

use crate::{
    rfq::{
        client::RFQClient,
        errors::RFQError,
        models::TimestampHeader,
        protocols::bebop::models::{
            BebopOrderToSign, BebopPriceData, BebopPricingUpdate, BebopQuoteResponse,
        },
    },
    tycho_client::feed::synchronizer::{ComponentWithState, Snapshot, StateSyncMessage},
    tycho_common::dto::{ProtocolComponent, ResponseProtocolState},
};

fn bytes_to_address(address: &Bytes) -> Result<Address, RFQError> {
    if address.len() == 20 {
        Ok(Address::from_slice(address))
    } else {
        Err(RFQError::InvalidInput(format!("Invalid ERC20 token address: {address:?}")))
    }
}

/// Maps a Chain to its corresponding Bebop WebSocket URL
fn chain_to_bebop_url(chain: Chain) -> Result<String, RFQError> {
    let chain_path = match chain {
        Chain::Ethereum => "ethereum",
        Chain::Base => "base",
        _ => return Err(RFQError::FatalError(format!("Unsupported chain: {chain:?}"))),
    };
    let url = format!("api.bebop.xyz/pmm/{chain_path}/v3");
    Ok(url)
}

#[derive(Clone, Debug)]
pub struct BebopClient {
    chain: Chain,
    price_ws: String,
    quote_endpoint: String,
    // Tokens that we want prices for
    tokens: HashSet<Bytes>,
    // Min tvl value in the quote token.
    tvl: f64,
    // name header for authentication
    ws_user: String,
    // key header for authentication
    ws_key: String,
    // quote tokens to normalize to for TVL purposes. Should have the same prices.
    quote_tokens: HashSet<Bytes>,
}

impl BebopClient {
    pub fn new(
        chain: Chain,
        tokens: HashSet<Bytes>,
        tvl: f64,
        ws_user: String,
        ws_key: String,
        quote_tokens: HashSet<Bytes>,
    ) -> Result<Self, RFQError> {
        let url = chain_to_bebop_url(chain)?;
        Ok(Self {
            price_ws: "wss://".to_string() + &url + "/pricing?format=protobuf",
            quote_endpoint: "https://".to_string() + &url + "/quote",
            tokens,
            chain,
            tvl,
            ws_user,
            ws_key,
            quote_tokens,
        })
    }

    fn create_component_with_state(
        &self,
        component_id: String,
        tokens: Vec<tycho_common::Bytes>,
        price_data: &BebopPriceData,
        tvl: f64,
    ) -> ComponentWithState {
        let protocol_component = ProtocolComponent {
            id: component_id.clone(),
            protocol_system: "rfq:bebop".to_string(),
            protocol_type_name: "bebop_pool".to_string(),
            chain: self.chain.into(),
            tokens,
            contract_ids: vec![], // empty for RFQ
            static_attributes: Default::default(),
            change: Default::default(),
            creation_tx: Default::default(),
            created_at: Default::default(),
        };

        let mut attributes = HashMap::new();

        // Store all bids and asks as JSON strings, since we cannot store arrays
        // Convert flat arrays [price1, size1, price2, size2, ...] to pairs [(price1, size1),
        // (price2, size2), ...]
        if !price_data.bids.is_empty() {
            let bids_pairs: Vec<(f32, f32)> = price_data
                .bids
                .chunks_exact(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect();
            let bids_json = serde_json::to_string(&bids_pairs).unwrap_or_default();
            attributes.insert("bids".to_string(), bids_json.as_bytes().to_vec().into());
        }
        if !price_data.asks.is_empty() {
            let asks_pairs: Vec<(f32, f32)> = price_data
                .asks
                .chunks_exact(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect();
            let asks_json = serde_json::to_string(&asks_pairs).unwrap_or_default();
            attributes.insert("asks".to_string(), asks_json.as_bytes().to_vec().into());
        }

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
}

#[async_trait]
impl RFQClient for BebopClient {
    fn stream(
        &self,
    ) -> BoxStream<'static, Result<(String, StateSyncMessage<TimestampHeader>), RFQError>> {
        let tokens = self.tokens.clone();
        let url = self.price_ws.clone();
        let tvl_threshold = self.tvl;
        let name = self.ws_user.clone();
        let authorization = self.ws_key.clone();
        let client = self.clone();

        Box::pin(async_stream::stream! {
            let mut current_components: HashMap<String, ComponentWithState> = HashMap::new();
            let mut reconnect_attempts = 0;
            const MAX_RECONNECT_ATTEMPTS: u32 = 10;

            loop {
                let request = Request::builder()
                    .method("GET")
                    .uri(&url)
                    .header("Host", "api.bebop.xyz")
                    .header("Upgrade", "websocket")
                    .header("Connection", "Upgrade")
                    .header("Sec-WebSocket-Key", generate_key())
                    .header("Sec-WebSocket-Version", "13")
                    .header("name", &name)
                    .header("Authorization", &authorization)
                    .body(())
                    .map_err(|_| RFQError::FatalError("Failed to build request".into()))?;

                // Connect to Bebop WebSocket with custom headers
                let (ws_stream, _) = match connect_async_with_config(request, None, false).await {
                    Ok(connection) => {
                        info!("Successfully connected to Bebop WebSocket");
                        reconnect_attempts = 0; // Reset counter on successful connection
                        connection
                    },
                    Err(e) => {
                        reconnect_attempts += 1;
                        error!("Failed to connect to Bebop WebSocket (attempt {}): {}", reconnect_attempts, e);

                        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS {
                            yield Err(RFQError::ConnectionError(format!("Failed to connect after {MAX_RECONNECT_ATTEMPTS} attempts: {e}")));
                            return;
                        }

                        let backoff_duration = Duration::from_secs(2_u64.pow(reconnect_attempts.min(5)));
                        info!("Retrying connection in {} seconds...", backoff_duration.as_secs());
                        sleep(backoff_duration).await;
                        continue;
                    }
                };

                let (_, mut ws_receiver) = ws_stream.split();

                // Message processing loop
                while let Some(msg) = ws_receiver.next().await {
                    match msg {
                        Ok(Message::Binary(data)) => {
                            match BebopPricingUpdate::decode(&data[..]) {
                                Ok(protobuf_update) => {
                                    let mut new_components = HashMap::new();

                                    // Process all pairs directly from protobuf
                                    for price_data in &protobuf_update.pairs {
                                        let base_bytes = Bytes::from(price_data.base.clone());
                                        let quote_bytes = Bytes::from(price_data.quote.clone());
                                        if tokens.contains(&base_bytes) && tokens.contains(&quote_bytes) {
                                            let pair_tokens = vec![
                                                base_bytes.clone(), quote_bytes.clone()
                                            ];

                                            let mut quote_price_data: Option<BebopPriceData> = None;
                                            // The quote token is not one of the approved quote tokens
                                            // Get the price, so we can normalize our TVL calculation
                                            if !client.quote_tokens.contains(&quote_bytes) {
                                                for approved_quote_token in &client.quote_tokens {
                                                    // Look for the quote pair in the same protobuf update
                                                    if let Some(quote_data) = protobuf_update.pairs.iter()
                                                        .find(|p| p.base == quote_bytes.as_ref() && p.quote == approved_quote_token.as_ref()) {
                                                        quote_price_data = Some(quote_data.clone());
                                                        break;
                                                    }
                                                }

                                                // Quote token doesn't have price levels in approved quote tokens.
                                                // Skip.
                                                if quote_price_data.is_none() {
                                                    warn!("Quote token does not have price levels in approved quote token. Skipping.");
                                                    continue;
                                                }
                                            }

                                            let tvl = price_data.calculate_tvl(quote_price_data);
                                            if tvl < tvl_threshold {
                                                continue;
                                            }

                                            let pair_str = format!("{}/{}", hex::encode(&base_bytes), hex::encode(&quote_bytes));
                                            let component_id = format!("{}", keccak256(pair_str.as_bytes()));
                                            let component_with_state = client.create_component_with_state(
                                                component_id.clone(),
                                                pair_tokens,
                                                price_data,
                                                tvl
                                            );
                                            new_components.insert(component_id, component_with_state);
                                        }
                                    }

                                    // Find components that were removed (existed before but not in this update)
                                    // This includes components with no bids or asks, since they are filtered
                                    // out by the tvl threshold.
                                    let removed_components: HashMap<String, ProtocolComponent> = current_components
                                        .iter()
                                        .filter(|&(id, _)| !new_components.contains_key(id))
                                        .map(|(k, v)| (k.clone(), v.component.clone()))
                                        .collect();

                                    // Update our current state
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
                                        deltas: None, // Deltas are always None - all the changes are absolute
                                        removed_components,
                                    };

                                    // Yield one message containing all updated pairs
                                    yield Ok(("bebop".to_string(), msg));
                                },
                                Err(e) => {
                                    error!("Failed to parse protobuf message: {}", e);
                                    break;
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            info!("WebSocket connection closed by server");
                            break;
                        }
                        Err(e) => {
                            error!("WebSocket error: {}", e);
                            break;
                        }
                        _ => {} // Ignore other message types
                    }
                }

                // If we're here, the message loop exited - always attempt to reconnect
                reconnect_attempts += 1;
                if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS {
                    yield Err(RFQError::ConnectionError(format!("Connection failed after {MAX_RECONNECT_ATTEMPTS} attempts")));
                    return;
                }

                let backoff_duration = Duration::from_secs(2_u64.pow(reconnect_attempts.min(5)));
                info!("Reconnecting in {} seconds (attempt {})...", backoff_duration.as_secs(), reconnect_attempts);
                sleep(backoff_duration).await;
                // Continue to the next iteration of the main loop
            }
        })
    }

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<SignedQuote, RFQError> {
        let sell_token = bytes_to_address(&params.token_in)?.to_string();
        let buy_token = bytes_to_address(&params.token_out)?.to_string();
        let sell_amount = params.amount_in.to_string();
        let sender = bytes_to_address(&params.sender)?.to_string();
        let receiver = bytes_to_address(&params.receiver)?.to_string();

        let url = self.quote_endpoint.clone();

        let client = Client::new();

        let request = client
            .get(&url)
            .query(&[
                ("sell_tokens", sell_token),
                ("buy_tokens", buy_token),
                ("sell_amounts", sell_amount),
                ("taker_address", sender),
                ("receiver_address", receiver),
                ("approval_type", "Standard".into()),
                ("skip_validation", "true".into()),
                ("skip_taker_checks", "true".into()),
                ("gasless", "false".into()),
                ("expiry_type", "standard".into()),
                ("fee", "0".into()),
                ("is_ui", "false".into()),
                ("source", self.ws_user.clone()),
            ])
            .header("accept", "application/json")
            .header("name", &self.ws_user)
            .header("source-auth", &self.ws_key)
            .header("Authorization", &self.ws_key);

        let response = request.send().await.map_err(|e| {
            RFQError::ConnectionError(format!("Failed to send Bebop quote request: {e}"))
        })?;

        let quote_response = response
            .json::<BebopQuoteResponse>()
            .await
            .map_err(|e| {
                RFQError::ParsingError(format!("Failed to parse Bebop quote response: {e}"))
            })?;

        match quote_response {
            BebopQuoteResponse::Success(quote) => {
                let mut quote_attributes: HashMap<String, Bytes> = HashMap::new();
                quote_attributes.insert("calldata".into(), quote.tx.data);
                quote_attributes.insert(
                    "partial_fill_offset".into(),
                    Bytes::from(
                        quote
                            .partial_fill_offset
                            .to_be_bytes()
                            .to_vec(),
                    ),
                );
                let signed_quote = match quote.to_sign {
                    BebopOrderToSign::Single(ref single) => SignedQuote {
                        base_token: params.token_in.clone(),
                        quote_token: params.token_out.clone(),
                        amount_in: BigUint::from_str(&single.taker_amount).map_err(|_| {
                            RFQError::ParsingError(format!(
                                "Failed to parse amount in string: {}",
                                single.taker_amount
                            ))
                        })?,
                        amount_out: BigUint::from_str(&single.maker_amount).map_err(|_| {
                            RFQError::ParsingError(format!(
                                "Failed to parse amount out string: {}",
                                single.maker_amount
                            ))
                        })?,
                        quote_attributes,
                    },
                    BebopOrderToSign::Aggregate(aggregate) => {
                        let taker_amounts: Vec<BigUint> = aggregate
                            .taker_amounts
                            .into_iter()
                            .flatten()
                            .map(|amount| {
                                BigUint::from_str(&amount).map_err(|_| {
                                    RFQError::ParsingError(format!(
                                        "Failed to parse amount in string: {amount}",
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let maker_amounts: Vec<BigUint> = aggregate
                            .maker_amounts
                            .into_iter()
                            .flatten()
                            .map(|amount| {
                                BigUint::from_str(&amount).map_err(|_| {
                                    RFQError::ParsingError(format!(
                                        "Failed to parse amount in string: {amount}",
                                    ))
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        SignedQuote {
                            base_token: params.token_in.clone(),
                            quote_token: params.token_out.clone(),
                            amount_in: taker_amounts.into_iter().sum(),
                            amount_out: maker_amounts.into_iter().sum(),
                            quote_attributes,
                        }
                    }
                };

                Ok(signed_quote)
            }
            BebopQuoteResponse::Error(err) => {
                return Err(RFQError::FatalError(format!(
                    "Bebop API error: code {} - {} (requestId: {})",
                    err.error.error_code, err.error.message, err.error.request_id
                )));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use dotenv::dotenv;
    use futures::SinkExt;
    use tokio::{net::TcpListener, time::timeout};
    use tokio_tungstenite::accept_async;

    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access and setting proper env vars
    async fn test_bebop_websocket_connection() {
        // We test with quote tokens that are not USDC in order to ensure our normalization works
        // fine
        let wbtc = Bytes::from_str("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599").unwrap();
        let weth = Bytes::from_str("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2").unwrap();

        let ws_user = String::from("tycho");
        dotenv().expect("Missing .env file");
        let ws_key = env::var("BEBOP_KEY").expect("BEBOP_KEY environment variable is required");

        let quote_tokens = HashSet::from([
            // Use addresses we forgot to checksum (to test checksumming)
            Bytes::from_str("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap(), // USDC
            Bytes::from_str("0xdac17f958d2ee523a2206206994597c13d831ec7").unwrap(), // USDT
        ]);

        let client = BebopClient::new(
            Chain::Ethereum,
            HashSet::from_iter(vec![weth.clone(), wbtc.clone()]),
            10.0, // $10 minimum TVL
            ws_user,
            ws_key,
            quote_tokens,
        )
        .unwrap();

        let mut stream = client.stream();

        // Test connection and message reception with timeout
        let result = timeout(Duration::from_secs(10), async {
            let mut message_count = 0;
            let max_messages = 5;

            while let Some(result) = stream.next().await {
                match result {
                    Ok((component_id, msg)) => {
                        println!("Received message with ID: {component_id}");

                        assert!(!component_id.is_empty());
                        assert_eq!(component_id, "bebop");
                        assert!(msg.header.timestamp > 0);
                        assert!(!msg.snapshots.states.is_empty());

                        let snapshot = &msg.snapshots;

                        // We got at least one component
                        assert!(!snapshot.states.is_empty());

                        println!("Received {} components in this message", snapshot.states.len());
                        for (id, component_with_state) in &snapshot.states {
                            assert_eq!(
                                component_with_state
                                    .component
                                    .protocol_system,
                                "rfq:bebop"
                            );
                            assert_eq!(
                                component_with_state
                                    .component
                                    .protocol_type_name,
                                "bebop_pool"
                            );
                            assert_eq!(
                                component_with_state.component.chain,
                                Chain::Ethereum.into()
                            );

                            let attributes = &component_with_state.state.attributes;

                            // Check that bids and asks exist and have non-empty byte strings
                            assert!(attributes.contains_key("bids"));
                            assert!(attributes.contains_key("asks"));
                            assert!(!attributes["bids"].is_empty());
                            assert!(!attributes["asks"].is_empty());

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

    #[tokio::test]
    async fn test_websocket_reconnection() {
        // Start a mock WebSocket server that will drop connections intermittently
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();

        // Creates a thread-safe counter.
        let connection_count = Arc::new(Mutex::new(0u32));

        // We must clone - since we want to read the original value at the end of the test.
        let connection_count_clone = connection_count.clone();

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                *connection_count_clone.lock().unwrap() += 1;
                let count = *connection_count_clone.lock().unwrap();
                println!("Mock server: Connection #{count} established");

                tokio::spawn(async move {
                    if let Ok(ws_stream) = accept_async(stream).await {
                        let (mut ws_sender, _ws_receiver) = ws_stream.split();

                        // Create test protobuf message
                        let weth_addr =
                            hex::decode("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2").unwrap();
                        let usdc_addr =
                            hex::decode("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap();

                        let test_price_data = BebopPriceData {
                            base: weth_addr,
                            quote: usdc_addr,
                            last_update_ts: 1752617378,
                            bids: vec![3070.05f32, 0.325717f32],
                            asks: vec![3070.527f32, 0.325717f32],
                        };

                        let pricing_update = BebopPricingUpdate { pairs: vec![test_price_data] };

                        let test_message = pricing_update.encode_to_vec();

                        if count == 1 {
                            // First connection: Send message successfully, then drop
                            println!("Mock server: Connection #1 - sending message then dropping.");
                            let _ = ws_sender
                                .send(Message::Binary(test_message.clone().into()))
                                .await;

                            // Give time for message to be processed, then drop the connection.
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            println!("Mock server: Dropping connection #1");
                            let _ = ws_sender.close().await;
                        } else if count == 2 {
                            // Second connection: Send message successfully and maintain connection
                            println!("Mock server: Connection #2 - maintaining stable connection.");
                            let _ = ws_sender
                                .send(Message::Binary(test_message.clone().into()))
                                .await;
                        }
                    }
                });
            }
        });

        // Wait a moment for the server to start
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut test_quote_tokens = HashSet::new();
        test_quote_tokens
            .insert(Bytes::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap());

        let tokens_formatted = vec![
            Bytes::from_str("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2").unwrap(),
            Bytes::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap(),
        ];

        // Bypass the new() constructor to mock the URL to point to our mock server.
        let client = BebopClient {
            chain: Chain::Ethereum,
            price_ws: format!("ws://127.0.0.1:{}", addr.port()),
            tokens: tokens_formatted.into_iter().collect(),
            tvl: 1000.0,
            ws_user: "test_user".to_string(),
            ws_key: "test_key".to_string(),
            quote_tokens: test_quote_tokens,
            quote_endpoint: "".to_string(),
        };

        let start_time = std::time::Instant::now();
        let mut successful_messages = 0;
        let mut connection_errors = 0;
        let mut first_message_received = false;
        let mut second_message_received = false;

        // Expected flow:
        // 1. Receive first message successfully
        // 2. Connection drops
        // 3. Client reconnects
        // 4. Receive second message successfully
        // Timeout if two messages are not received within 5 seconds.
        while start_time.elapsed() < Duration::from_secs(5) && successful_messages < 2 {
            match timeout(Duration::from_millis(1000), client.stream().next()).await {
                Ok(Some(result)) => match result {
                    Ok((_component_id, _message)) => {
                        successful_messages += 1;
                        println!("Received successful message {successful_messages}");

                        if successful_messages == 1 {
                            first_message_received = true;
                            println!("First message received - connection should drop after this.");
                        } else if successful_messages == 2 {
                            second_message_received = true;
                            println!("Second message received after reconnection.");
                        }
                    }
                    Err(e) => {
                        connection_errors += 1;
                        println!("Connection error during reconnection: {e:?}");
                    }
                },
                Ok(None) => {
                    panic!("Stream ended unexpectedly");
                }
                Err(_) => {
                    println!("Timeout waiting for message (normal during reconnections)");
                    continue;
                }
            }
        }

        let final_connection_count = *connection_count.lock().unwrap();

        // 1. Exactly 2 connection attempts (initial + reconnect)
        // 2. Exactly 2 successful messages (one before drop, one after reconnect)

        assert_eq!(final_connection_count, 2);
        assert!(first_message_received);
        assert!(second_message_received);
        assert_eq!(connection_errors, 0);
        assert_eq!(successful_messages, 2);
    }

    #[tokio::test]
    #[ignore] // Requires network access and setting proper env vars
    async fn test_bebop_quote_single_order() {
        let token_in = Bytes::from_str("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2").unwrap();
        let token_out = Bytes::from_str("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599").unwrap();
        let ws_user = String::from("tycho");
        dotenv().expect("Missing .env file");
        let ws_key =
            env::var("BEBOP_WS_KEY").expect("BEBOP_WS_KEY environment variable is required");

        let client = BebopClient::new(
            Chain::Ethereum,
            HashSet::from_iter(vec![token_in.clone(), token_out.clone()]),
            10.0, // $10 minimum TVL
            ws_user,
            ws_key,
            HashSet::new(),
        )
        .unwrap();

        let router = Bytes::from_str("0xfD0b31d2E955fA55e3fa641Fe90e08b677188d35").unwrap();

        let params = GetAmountOutParams {
            amount_in: BigUint::from(1_000000000000000000u64),
            token_in: token_in.clone(),
            token_out: token_out.clone(),
            sender: router.clone(),
            receiver: router,
        };
        let quote = client
            .request_binding_quote(&params)
            .await
            .unwrap();

        assert_eq!(quote.base_token, token_in);
        assert_eq!(quote.quote_token, token_out);
        assert_eq!(quote.amount_in, BigUint::from(1_000000000000000000u64));

        // Assuming the BTC - WETH price doesn't change too much at the time of running this
        assert!(quote.amount_out > BigUint::from(3000000u64));

        // SWAP_SINGLE_SELECTOR = 0x4dcebcba;
        assert_eq!(
            quote
                .quote_attributes
                .get("calldata")
                .unwrap()[..4],
            Bytes::from_str("0x4dcebcba")
                .unwrap()
                .to_vec()
        );
        let partial_fill_offset_slice = quote
            .quote_attributes
            .get("partial_fill_offset")
            .unwrap()
            .as_ref();
        let mut partial_fill_offset_array = [0u8; 8];
        partial_fill_offset_array.copy_from_slice(partial_fill_offset_slice);

        assert_eq!(u64::from_be_bytes(partial_fill_offset_array), 12);
    }

    #[tokio::test]
    #[ignore] // Requires network access and setting proper env vars
    async fn test_bebop_quote_aggregate_order() {
        // This will make a quote request similar to the previous test but with a very big amount
        // We expect the Bebop Quote to have an aggregate order (split between different mms)
        let token_in = Bytes::from_str("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48").unwrap();
        let token_out = Bytes::from_str("0xfAbA6f8e4a5E8Ab82F62fe7C39859FA577269BE3").unwrap();
        let ws_user = String::from("tycho");
        dotenv().expect("Missing .env file");
        let ws_key =
            env::var("BEBOP_WS_KEY").expect("BEBOP_WS_KEY environment variable is required");

        let client = BebopClient::new(
            Chain::Ethereum,
            HashSet::from_iter(vec![token_in.clone(), token_out.clone()]),
            10.0, // $10 minimum TVL
            ws_user,
            ws_key,
            HashSet::new(),
        )
        .unwrap();

        let router = Bytes::from_str("0xfD0b31d2E955fA55e3fa641Fe90e08b677188d35").unwrap();

        let amount_in = BigUint::from_str("20_000_000_000").unwrap(); // 20k USDC
        let params = GetAmountOutParams {
            amount_in: amount_in.clone(),
            token_in: token_in.clone(),
            token_out: token_out.clone(),
            sender: router.clone(),
            receiver: router,
        };
        let quote = client
            .request_binding_quote(&params)
            .await
            .unwrap();

        assert_eq!(quote.base_token, token_in);
        assert_eq!(quote.quote_token, token_out);
        assert_eq!(quote.amount_in, amount_in);

        // Assuming the USDC - ONDO price doesn't change too much at the time of running this
        assert!(quote.amount_out > BigUint::from_str("18000000000000000000000").unwrap()); // ~19k ONDO

        // SWAP_AGGREGATE_SELECTOR = 0xa2f74893;
        assert_eq!(
            quote
                .quote_attributes
                .get("calldata")
                .unwrap()[..4],
            Bytes::from_str("0xa2f74893")
                .unwrap()
                .to_vec()
        );
        let partial_fill_offset_slice = quote
            .quote_attributes
            .get("partial_fill_offset")
            .unwrap()
            .as_ref();
        let mut partial_fill_offset_array = [0u8; 8];
        partial_fill_offset_array.copy_from_slice(partial_fill_offset_slice);

        // This is the only attribute that is significantly different for the Single and Aggregate
        // Order
        assert_eq!(u64::from_be_bytes(partial_fill_offset_array), 2);
    }
}
