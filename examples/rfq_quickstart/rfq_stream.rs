use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use alloy::{
    primitives::{Address, Bytes as AlloyBytes, TxKind, U256},
    providers::Provider,
    rpc::types::{TransactionInput, TransactionRequest},
    sol_types::SolValue,
};
use dialoguer::{theme::ColorfulTheme, Select};
use miette::{IntoDiagnostic, Result};
use num_bigint::BigUint;
use tokio::{sync::mpsc, time::timeout};
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};
use tycho_execution::encoding::{
    evm::encoder_builders::TychoRouterEncoderBuilder, models::UserTransferType,
};
use tycho_simulation::{
    protocol::models::{ProtocolComponent, Update},
    rfq::{
        protocols::bebop::{client_builder::BebopClientBuilder, state::BebopState},
        stream::RFQStreamBuilder,
    },
};

use crate::{
    encode_input,
    env::get_env,
    format_token_amount,
    swap_executor::{SignerData, SwapExecutor},
};

pub struct RFQStreamClient {
    builder: RFQStreamBuilder,
}

impl RFQStreamClient {
    pub fn new() -> Self {
        RFQStreamClient { builder: RFQStreamBuilder::new() }
    }

    pub fn add_bebop(
        mut self,
        chain: Chain,
        tvl_threshold: f64,
        sell_token: &Token,
        buy_token: &Token,
    ) -> Result<Self> {
        match (get_env("BEBOP_USER"), get_env("BEBOP_KEY")) {
            (Some(user), Some(key)) => {
                println!("Connecting to Bebop RFQ WebSocket...");
                let client = BebopClientBuilder::new(chain, user, key)
                    .tokens([sell_token.address.clone(), buy_token.address.clone()].into())
                    .tvl_threshold(tvl_threshold)
                    .build()
                    .into_diagnostic()?;
                self.builder = self
                    .builder
                    .add_client::<BebopState>("bebop", Box::new(client));
                Ok(self)
            }
            _ => Ok(self),
        }
    }

    pub async fn set_tokens(mut self, all_tokens: HashMap<Bytes, Token>) -> Result<Self> {
        self.builder = self
            .builder
            .set_tokens(all_tokens)
            .await;
        Ok(self)
    }

    pub fn finish(self) -> RFQStreamBuilder {
        self.builder
    }
}

pub struct RFQStreamProcessor;

impl RFQStreamProcessor {
    pub fn new() -> Self {
        RFQStreamProcessor
    }

    pub async fn process_rfq_stream(
        &self,
        rx: &mut mpsc::Receiver<Update>,
        sell_token: &Token,
        buy_token: &Token,
        amount_in: BigUint,
        chain: Chain,
        swapper_pk: Option<String>,
    ) -> Result<()> {
        let encoder = TychoRouterEncoderBuilder::new()
            .chain(chain)
            .user_transfer_type(UserTransferType::TransferFromPermit2)
            .build()
            .expect("Failed to build encoder");
        let swap_executor = match swapper_pk.clone() {
            Some(pk) => {
                let rpc_url = get_env("RPC_URL").expect("RPC_URL not set");
                SwapExecutor::new(chain, encoder)
                    .with_signer(pk, &rpc_url)
                    .await?
            }
            None => SwapExecutor::new(chain, encoder),
        };
        while let Some(update) = rx.recv().await {
            // Drain any additional buffered messages to get the most recent one
            //
            // ⚠️Warning: This works fine only if you assume that this message is entirely
            // representative of the current state, as done in this quickstart.
            // You should comment out this code portion if you would like to manually track removed
            // components.
            let update = Self::drain_to_latest_update(update, rx).await;
            println!(
                "Received RFQ price levels with {} new pairs for block/timestamp {}",
                &update.states.len(),
                update.block_number_or_timestamp
            );
            Self::process_update(update, sell_token, buy_token, &amount_in, &swap_executor).await?;
            println!("\nWaiting for more price levels... (Press Ctrl+C to exit)");
        }
        Ok(())
    }

    async fn drain_to_latest_update(
        mut latest_update: Update,
        rx: &mut mpsc::Receiver<Update>,
    ) -> Update {
        let mut drained_count = 0;
        while let Ok(newer_update) = timeout(Duration::from_millis(10), rx.recv()).await {
            if let Some(newer_update) = newer_update {
                latest_update = newer_update;
                drained_count += 1;
            } else {
                break;
            }
        }
        if drained_count > 0 {
            println!(
                "Fast-forwarded through {drained_count} older RFQ updates to get latest prices"
            );
        }
        latest_update
    }

    async fn process_update(
        update: Update,
        sell_token: &Token,
        buy_token: &Token,
        amount_in: &BigUint,
        swap_executor: &SwapExecutor,
    ) -> Result<()> {
        for (component, state) in Self::filter_matching_components(&update, sell_token, buy_token) {
            if let Some(amount_out) =
                Self::calculate_amount_out(state.as_ref(), amount_in, sell_token, buy_token)
            {
                println!(
                    "Best indicative price for swap {}: {} {} -> {} {}",
                    component.protocol_system,
                    format_token_amount(amount_in, sell_token),
                    sell_token.symbol,
                    format_token_amount(&amount_out, buy_token),
                    buy_token.symbol
                );

                if let Some(signer_data) = swap_executor.signer_data.as_ref() {
                    if Self::get_token_balances(
                        sell_token,
                        buy_token,
                        amount_in,
                        swap_executor,
                        signer_data,
                    )
                    .await?
                    .is_empty()
                    {
                        continue;
                    }

                    println!("Would you like to simulate or execute this swap?");
                    println!("Please be aware that the market might move while you make your decision, which might lead to a revert if you've set a min amount out or slippage.");
                    println!("Warning: slippage is set to 0.25% during execution by default.\n");

                    let user_choice = Self::handle_user_interaction();
                    match user_choice.as_str() {
                        "simulate" => {
                            swap_executor
                                .simulate_swap(
                                    &component,
                                    state.as_ref(),
                                    sell_token,
                                    buy_token,
                                    amount_in,
                                    &amount_out,
                                    signer_data,
                                )
                                .await?;
                        }
                        "execute" => {
                            swap_executor
                                .execute_swap(
                                    &component,
                                    state.as_ref(),
                                    sell_token,
                                    buy_token,
                                    amount_in,
                                    &amount_out,
                                    signer_data,
                                )
                                .await?;
                        }
                        _ => println!("Skipping this swap..."),
                    }
                } else {
                    println!("Signer private key not provided. Skipping simulation/execution.");
                }
            }
        }
        Ok(())
    }

    fn filter_matching_components<'a>(
        update: &'a Update,
        sell_token: &'a Token,
        buy_token: &'a Token,
    ) -> impl Iterator<Item = (ProtocolComponent, Box<dyn ProtocolSim>)> + 'a {
        update
            .states
            .iter()
            .filter_map(move |(comp_id, state)| {
                update
                    .new_pairs
                    .get(comp_id)
                    .and_then(|component| {
                        let tokens = &component.tokens;
                        if HashSet::from([sell_token, buy_token])
                            .is_subset(&HashSet::from_iter(tokens.iter()))
                        {
                            Some((component.clone(), state.clone_box()))
                        } else {
                            None
                        }
                    })
            })
    }

    fn calculate_amount_out(
        state: &dyn ProtocolSim,
        amount_in: &BigUint,
        sell_token: &Token,
        buy_token: &Token,
    ) -> Option<BigUint> {
        state
            .get_amount_out(amount_in.clone(), sell_token, buy_token)
            .ok()
            .map(|result| result.amount)
    }

    async fn get_token_balances(
        sell_token: &Token,
        buy_token: &Token,
        amount_in: &BigUint,
        swap_executor: &SwapExecutor,
        signer_data: &SignerData,
    ) -> Result<HashMap<Address, BigUint>> {
        let mut balances = HashMap::new();

        // Show sell token balance
        match Self::get_token_balance(
            &signer_data.provider,
            Address::from_slice(&sell_token.address),
            signer_data.signer.address(),
            Address::from_slice(
                &swap_executor
                    .chain
                    .native_token()
                    .address,
            ),
        )
        .await
        {
            Ok(balance) => {
                let formatted_balance = format_token_amount(&balance, sell_token);
                println!(
                    "\nYour balance: {formatted_balance} {sell_symbol}",
                    sell_symbol = sell_token.symbol
                );
                if &balance < amount_in {
                    let required = format_token_amount(amount_in, sell_token);
                    println!("⚠️ Warning: Insufficient balance for swap. You have {formatted_balance} {sell_symbol} but need {required} {sell_symbol}",
                             formatted_balance = formatted_balance,
                             sell_symbol = sell_token.symbol,
                    );
                    return Ok(balances);
                }
                balances.insert(Address::from_slice(&sell_token.address), balance);
            }
            Err(e) => eprintln!("Failed to get token balance: {e}"),
        }

        // Show buy token balance
        match Self::get_token_balance(
            &signer_data.provider,
            Address::from_slice(&buy_token.address),
            signer_data.signer.address(),
            Address::from_slice(
                &swap_executor
                    .chain
                    .native_token()
                    .address,
            ),
        )
        .await
        {
            Ok(balance) => {
                let formatted_balance = format_token_amount(&balance, buy_token);
                println!(
                    "Your {buy_symbol} balance: {formatted_balance} {buy_symbol}",
                    buy_symbol = buy_token.symbol
                );
                balances.insert(Address::from_slice(&buy_token.address), balance);
            }
            Err(e) => {
                eprintln!("Failed to get {buy_symbol} balance: {e}", buy_symbol = buy_token.symbol)
            }
        }
        Ok(balances)
    }

    async fn get_token_balance(
        provider: &Arc<dyn Provider>,
        token_address: Address,
        wallet_address: Address,
        native_token_address: Address,
    ) -> Result<BigUint, Box<dyn std::error::Error>> {
        let balance = if token_address == native_token_address {
            provider
                .get_balance(wallet_address)
                .await?
        } else {
            let balance_of_signature = "balanceOf(address)";
            let data = encode_input(balance_of_signature, (wallet_address,).abi_encode());

            let result = provider
                .call(TransactionRequest {
                    to: Some(TxKind::Call(token_address)),
                    input: TransactionInput { input: Some(AlloyBytes::from(data)), data: None },
                    ..Default::default()
                })
                .await?;

            U256::from_be_bytes(
                result
                    .to_vec()
                    .try_into()
                    .unwrap_or([0u8; 32]),
            )
        };
        // Convert the U256 to BigUint
        Ok(BigUint::from_bytes_be(&balance.to_be_bytes::<32>()))
    }

    fn handle_user_interaction() -> String {
        let options = vec!["Simulate the swap", "Execute the swap", "Skip this swap"];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("What would you like to do?")
            .default(0)
            .items(&options)
            .interact()
            .unwrap_or(2); // Default to skip if error

        match selection {
            0 => "simulate".to_string(),
            1 => "execute".to_string(),
            _ => "skip".to_string(),
        }
    }
}
