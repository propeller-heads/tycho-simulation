use futures::{Stream, StreamExt};
use neon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use tycho_simulation::{
    evm::{
        decoder::StreamDecodeError,
        engine_db::tycho_db::PreCachedDB,
        protocol::{
            filters::{balancer_pool_filter, curve_pool_filter},
            uniswap_v2::state::UniswapV2State,
            uniswap_v3::state::UniswapV3State,
            vm::state::EVMPoolState,
        },
        stream::ProtocolStreamBuilder,
    },
    models::Token,
    protocol::{
        models::{BlockUpdate, ProtocolComponent},
        state::ProtocolSim,
    },
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_core::dto::Chain,
    utils::load_all_tokens,
};
use std::io::Write;
use num_traits::cast::ToPrimitive;
use tokio::sync::{mpsc, Mutex as TokioMutex};
use std::any::Any;

// First, let's add a type alias at the top of the file to make things cleaner
type StreamType = Box<dyn Stream<Item = Result<BlockUpdate, StreamDecodeError>> + Send + Unpin>;

struct SimulationClient {
    tokens: Vec<Token>,
    states: Arc<TokioMutex<HashMap<String, (Box<dyn ProtocolSim>, ProtocolComponent)>>>,
    protocol_stream: Arc<TokioMutex<StreamType>>,
}

impl Finalize for SimulationClient {}

impl SimulationClient {
    async fn new(tycho_url: &str, api_key: Option<&str>, tvl_threshold: Option<f64>) -> anyhow::Result<Self> {
        log("Initializing SimulationClient...");
        
        // Use provided TVL threshold or default to 10,000
        let tvl_threshold = tvl_threshold.unwrap_or(10_000.0);
        log(&format!("Using TVL threshold: {}", tvl_threshold));
        let tvl_filter = ComponentFilter::with_tvl_range(tvl_threshold, tvl_threshold);
        
        let all_tokens = load_all_tokens(tycho_url, false, api_key).await;
        log(&format!("Loaded {} tokens", all_tokens.len()));

        // Match the protocol stream configuration from the example
        let protocol_stream = ProtocolStreamBuilder::new(tycho_url, Chain::Ethereum)
            .exchange::<UniswapV2State>("uniswap_v2", tvl_filter.clone(), None)
            .exchange::<UniswapV3State>("uniswap_v3", tvl_filter.clone(), None)
            .exchange::<EVMPoolState<PreCachedDB>>("vm:curve", tvl_filter.clone(), Some(curve_pool_filter))
            .exchange::<EVMPoolState<PreCachedDB>>(
                "vm:balancer_v2",
                tvl_filter.clone(),
                Some(balancer_pool_filter),
            )
            .auth_key(api_key.map(|s| s.to_string()))
            .set_tokens(all_tokens.clone())
            .await
            .build()
            .await?;

        log("Protocol stream built successfully");

        let protocol_stream = Arc::new(TokioMutex::new(
            Box::new(protocol_stream) as StreamType
        ));
        
        // Store pairs like in the example
        let states = Arc::new(TokioMutex::new(HashMap::new()));
        
        // Get initial state
        {  // Add scope to limit the borrow
            let mut stream = protocol_stream.lock().await;
            if let Some(Ok(update)) = stream.next().await {
                let mut states_map = states.lock().await;
                log(&format!("Received block {}", update.block_number));
                log(&format!("Received update with {} new pairs", update.new_pairs.len()));
                
                // Store pairs and states
                for (id, comp) in update.new_pairs {
                    if let Some(state) = update.states.get(&id) {
                        states_map.insert(id.clone(), (state.clone(), comp.clone()));
                    }
                }
                
                log(&format!("Total pools now: {}", states_map.len()));
            }
        }  // stream is dropped here, releasing the lock

        // Create a channel for stream updates
        let (tx, mut rx) = mpsc::channel(100);
        
        // Start background task to continuously update states
        let stream_ref = Arc::clone(&protocol_stream);
        let states_ref = Arc::clone(&states);
        
        tokio::spawn(async move {
            loop {
                let mut stream = stream_ref.lock().await;
                let next = stream.next().await;
                drop(stream);

                match next {
                    Some(Ok(update)) => {
                        if tx.send(update).await.is_err() {
                            log("Channel closed, stopping stream task");
                            break;
                        }
                    }
                    Some(Err(e)) => {
                        log(&format!("Error receiving update: {}", e));
                    }
                    None => {
                        log("Stream ended");
                        break;
                    }
                }
            }
        });

        // Start update processing task
        tokio::spawn(async move {
            while let Some(update) = rx.recv().await {
                let mut states_map = states_ref.lock().await;
                log(&format!("Received block {}", update.block_number));
                
                // Update existing states
                for (id, state) in &update.states {
                    if let Some((existing_state, _)) = states_map.get_mut(id) {
                        *existing_state = state.clone();
                    }
                }

                // Add new pairs
                let new_pairs_count = update.new_pairs.len();
                for (id, comp) in update.new_pairs {
                    if let Some(state) = update.states.get(&id) {
                        states_map.insert(id.clone(), (state.clone(), comp.clone()));
                    }
                }
                if new_pairs_count > 0 {
                    log(&format!("Added {} new pools", new_pairs_count));
                }
            }
            log("Update processor stopped");
        });

        // Convert HashMap<Bytes, Token> to Vec<Token>
        let tokens = all_tokens.into_values().collect::<Vec<_>>();
        log(&format!("Converted {} tokens to vector", tokens.len()));

        Ok(Self {
            tokens,
            states,
            protocol_stream: Arc::clone(&protocol_stream),  // Clone the Arc instead of moving
        })
    }

    async fn find_pool_for_pair(&self, token0: &Token, token1: &Token) -> Option<(Box<dyn ProtocolSim>, ProtocolComponent)> {
        log(&format!("Looking for pool with tokens {} and {}", token0.address, token1.address));
        let states = self.states.lock().await;
        log(&format!("Total pools available: {}", states.len()));
        
        let result = states
            .values()
            .find(|(_, comp)| {
                let pool_tokens = &comp.tokens;
                (pool_tokens[0].address == token0.address && pool_tokens[1].address == token1.address)
                    || (pool_tokens[0].address == token1.address && pool_tokens[1].address == token0.address)
            })
            .map(|(state, comp)| (state.clone(), comp.clone()));

        if result.is_some() {
            log("Found matching pool");
        } else {
            log("No matching pool found");
        }

        result
    }
}

pub struct JsSimulationClient(Arc<SimulationClient>);

impl Finalize for JsSimulationClient {}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("createClient", create_client)?;
    cx.export_function("getSpotPrice", get_spot_price)?;
    cx.export_function("getAmountOut", get_amount_out)?;
    Ok(())
}

fn create_client(mut cx: FunctionContext) -> JsResult<JsPromise> {
    let tycho_url = cx.argument::<JsString>(0)?.value(&mut cx);
    let api_key = cx.argument_opt(1).map(|arg| {
        arg.downcast::<JsString, FunctionContext>(&mut cx)
            .unwrap()
            .value(&mut cx)
    });
    let tvl_threshold = cx.argument_opt(2).map(|arg| {
        arg.downcast::<JsNumber, FunctionContext>(&mut cx)
            .unwrap()
            .value(&mut cx)
    });

    let channel = cx.channel();
    let (deferred, promise) = cx.promise();

    let rt = tokio::runtime::Runtime::new().unwrap();
    std::thread::spawn(move || {
        rt.block_on(async {
            match SimulationClient::new(&tycho_url, api_key.as_deref(), tvl_threshold).await {
                Ok(client) => {
                    deferred.settle_with(&channel, move |mut cx| {
                        Ok(cx.boxed(JsSimulationClient(Arc::new(client))))
                    });
                }
                Err(e) => {
                    let error_msg = format!("Failed to create client: {}", e);
                    deferred.settle_with(&channel, move |mut cx| {
                        Ok(cx.error(error_msg)?)
                    });
                }
            }
        });
    });

    Ok(promise)
}

fn normalize_address(address: &str) -> String {
    address.trim_start_matches("0x").to_lowercase()
}

fn get_protocol_name(state: &Box<dyn ProtocolSim>) -> String {
    // Get full type name for logging
    let type_name = std::any::type_name_of_val(&**state);
    log(&format!("Full type name: {}", type_name));
    
    // Extract protocol name from type
    let protocol = if type_name.contains("UniswapV2State") {
        "uniswap_v2"
    } else if type_name.contains("UniswapV3State") {
        "uniswap_v3"
    } else if type_name.contains("EVMPoolState") && type_name.contains("curve") {
        "curve"
    } else if type_name.contains("EVMPoolState") && type_name.contains("balancer") {
        "balancer_v2"
    } else {
        log(&format!("Unknown protocol type: {}", type_name));
        "unknown"
    };

    log(&format!("Detected protocol: {}", protocol));
    protocol.to_string()
}

fn get_spot_price(mut cx: FunctionContext) -> JsResult<JsPromise> {
    log("get_spot_price called");
    let client = cx.argument::<JsBox<JsSimulationClient>>(0)?;
    let token0_address = cx.argument::<JsString>(1)?.value(&mut cx);
    let token1_address = cx.argument::<JsString>(2)?.value(&mut cx);
    
    let normalized_token0 = normalize_address(&token0_address);
    let normalized_token1 = normalize_address(&token1_address);
    
    log(&format!("Looking for token0: {}", token0_address));
    log(&format!("Normalized token0 address: {}", normalized_token0));
    
    let token0 = match client.0.tokens.iter()
        .find(|t| normalize_address(&t.address.to_string()) == normalized_token0) {
        Some(token) => token.clone(),
        None => {
            log("Token0 not found");
            return cx.throw_error("Token0 not found");
        }
    };
    
    log(&format!("Looking for token1: {}", token1_address));
    log(&format!("Normalized token1 address: {}", normalized_token1));
    
    let token1 = match client.0.tokens.iter()
        .find(|t| normalize_address(&t.address.to_string()) == normalized_token1) {
        Some(token) => token.clone(),
        None => {
            log("Token1 not found");
            return cx.throw_error("Token1 not found");
        }
    };

    log(&format!("Found token0: {:?}", token0));
    log(&format!("Found token1: {:?}", token1));
    
    log("Found both tokens, proceeding to get spot price");
    let channel = cx.channel();
    let (deferred, promise) = cx.promise();
    
    let client_ref = Arc::clone(&client.0);
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    std::thread::spawn(move || {
        rt.block_on(async {
            if let Some((state, _)) = client_ref.find_pool_for_pair(&token0, &token1).await {
                match state.spot_price(&token0, &token1) {
                    Ok(price) => {
                        log(&format!("Got spot price: {}", price));
                        deferred.settle_with(&channel, move |mut cx| {
                            Ok(cx.number(price))
                        });
                    }
                    Err(e) => {
                        let error_msg = format!("Failed to get spot price: {}", e);
                        log(&error_msg);
                        deferred.settle_with(&channel, move |mut cx| {
                            Ok(cx.error(error_msg)?)
                        });
                    }
                }
            } else {
                log("No pool found for token pair");
                deferred.settle_with(&channel, move |mut cx| {
                    Ok(cx.error("No pool found for token pair")?)
                });
            }
        });
    });

    Ok(promise)
}

fn get_amount_out(mut cx: FunctionContext) -> JsResult<JsPromise> {
    log("get_amount_out called");
    let client = cx.argument::<JsBox<JsSimulationClient>>(0)?;
    let token0_address = cx.argument::<JsString>(1)?.value(&mut cx);
    let token1_address = cx.argument::<JsString>(2)?.value(&mut cx);
    let amounts_in = cx.argument::<JsArray>(3)?;
    
    // Convert JS arrays to Rust vectors
    let amounts: Vec<f64> = (0..amounts_in.len(&mut cx))
        .map(|i| amounts_in.get::<JsNumber, _, u32>(&mut cx, i).unwrap().value(&mut cx))
        .collect();

    log(&format!("Processing {} amounts", amounts.len()));

    // Find tokens
    let normalized_token0 = normalize_address(&token0_address);
    let normalized_token1 = normalize_address(&token1_address);
    
    let token0 = match client.0.tokens.iter()
        .find(|t| normalize_address(&t.address.to_string()) == normalized_token0) {
        Some(token) => token.clone(),
        None => return cx.throw_error("Token0 not found"),
    };
    
    let token1 = match client.0.tokens.iter()
        .find(|t| normalize_address(&t.address.to_string()) == normalized_token1) {
        Some(token) => token.clone(),
        None => return cx.throw_error("Token1 not found"),
    };

    let channel = cx.channel();
    let (deferred, promise) = cx.promise();
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let client_ref = Arc::clone(&client.0);

    std::thread::spawn(move || {
        rt.block_on(async {
            // Get all the data we need in one go
            let states = client_ref.states.lock().await;
            let mut results = Vec::new();
            
            // Process pools and collect results
            for (pool_id, (state, comp)) in states.iter() {
                // Log pool token count
                let pool_tokens = &comp.tokens;
                log(&format!("\nChecking pool {} with {} tokens:", pool_id, pool_tokens.len()));
                
                // Check if both tokens exist in the pool in any position
                let is_matching_pair = pool_tokens.iter().any(|t| t.address == token0.address) && 
                                      pool_tokens.iter().any(|t| t.address == token1.address);

                if is_matching_pair {
                    log(&format!("\nMatching pool details:"));
                    log(&format!("Pool ID: {}", pool_id));
                    log(&format!("Pool address: {}", comp.address));
                    log(&format!("Number of tokens: {}", pool_tokens.len()));
                    
                    // Log token details
                    log("Tokens in pool:");
                    for token in pool_tokens {
                        log(&format!("  - Symbol: {}", token.symbol));
                        log(&format!("    Address: {}", token.address));
                        log(&format!("    Decimals: {}", token.decimals));
                        if token.address == token0.address || token.address == token1.address {
                            log(&format!("    ** Matched token for swap **"));
                        }
                    }

                    log(&format!("\nProcessing amounts for pool {}:", pool_id));
                    let mut amounts_out = Vec::new();
                    let mut gas_estimates = Vec::new();  // Add vector for gas values
                    let mut has_error = false;

                    // Calculate amounts out for each amount in
                    for &amount_in in &amounts {
                        let amount_in_scaled = (amount_in * 10f64.powi(token0.decimals as i32)) as u128;
                        match state.get_amount_out(amount_in_scaled.into(), &token0, &token1) {
                            Ok(result) => {
                                let amount_out = result.amount.to_f64().unwrap() / 10f64.powi(token1.decimals as i32);
                                amounts_out.push(amount_out);
                                let gas = result.gas.clone();  // Clone the gas value
                                gas_estimates.push(gas.clone());  // Store gas estimate
                            }
                            Err(e) => {
                                log(&format!("Error calculating amount out for pool {}: {}", pool_id, e));
                                has_error = true;
                                break;
                            }
                        }
                    }

                    if !has_error {
                        // Include protocol name in results
                        let protocol = get_protocol_name(state);
                        results.push((pool_id.clone(), amounts_out, gas_estimates, protocol));
                    }
                }
            }
            drop(states); // Release the lock before settling

            deferred.settle_with(&channel, move |mut cx| {
                let js_array = JsArray::new(&mut cx, results.len());
                
                for (i, (pool_id, amounts_out, gas_estimates, protocol)) in results.into_iter().enumerate() {
                    let obj = cx.empty_object();
                    
                    // Create all JS values first
                    let js_pool = cx.string(pool_id);
                    let js_protocol = cx.string(protocol);
                    let js_amounts = JsArray::new(&mut cx, amounts_out.len());
                    let js_gas = JsArray::new(&mut cx, gas_estimates.len());
                    
                    // Fill arrays
                    for (j, amount) in amounts_out.into_iter().enumerate() {
                        let js_amount = cx.number(amount);
                        js_amounts.set(&mut cx, j as u32, js_amount)?;
                    }

                    for (j, gas) in gas_estimates.into_iter().enumerate() {
                        let js_gas_value = cx.number(gas.to_f64().unwrap_or(0.0));
                        js_gas.set(&mut cx, j as u32, js_gas_value)?;
                    }

                    // Set all object properties
                    obj.set(&mut cx, "poolAddress", js_pool)?;
                    obj.set(&mut cx, "protocol", js_protocol)?;
                    obj.set(&mut cx, "amountsOut", js_amounts)?;
                    obj.set(&mut cx, "gasEstimates", js_gas)?;
                    
                    js_array.set(&mut cx, i as u32, obj)?;
                }

                Ok(js_array)
            });
        });
    });

    Ok(promise)
}

fn log(msg: &str) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    if let Some(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("tycho_node.log")
        .ok() {
        writeln!(file, "[{}] {}", timestamp, msg).ok();
    }
    println!("{}", msg);
} 