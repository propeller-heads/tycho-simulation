mod cli;
mod env;
mod rfq_stream;
mod swap_executor;
mod tycho_client;
mod utils;

use std::{str::FromStr, sync::Arc};

use alloy::{
    primitives::{Address, Keccak256, Signature},
    signers::{local::PrivateKeySigner, SignerSync},
    sol_types::{eip712_domain, SolStruct, SolValue},
};
use clap::Parser;
use dotenv::dotenv;
use miette::{IntoDiagnostic, Result};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use tokio::sync::mpsc;
use tracing_subscriber::EnvFilter;
use tycho_common::{models::token::Token, simulation::protocol_sim::ProtocolSim, Bytes};
use tycho_execution::encoding::{
    errors::EncodingError,
    evm::approvals::permit2::PermitSingle,
    models,
    models::{EncodedSolution, Solution, SwapBuilder, Transaction},
};
use tycho_simulation::{
    evm::protocol::u256_num::biguint_to_u256,
    protocol::models::{ProtocolComponent, Update},
};

use crate::{
    cli::RfqCommand,
    env::get_env,
    rfq_stream::{RFQStreamClient, RFQStreamProcessor},
    tycho_client::TychoClient,
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().into_diagnostic()?;
    setup_tracing();

    let cli = RfqCommand::parse().parse_args().await?;

    let tycho_client = TychoClient::from_env(cli.swap_args.chain)?;
    let all_tokens = tycho_client.load_tokens().await;
    let (sell_token, buy_token, amount_in) =
        tycho_client.get_token_info(&cli.swap_args, &all_tokens)?;

    let rfq_stream_builder = RFQStreamClient::new()
        .add_bebop(cli.swap_args.chain, cli.swap_args.tvl_threshold, &sell_token, &buy_token)?
        // .add_hashflow(&cli.swap_args)
        .set_tokens(all_tokens.clone())
        .await?
        .finish();
    let (tx, mut rx) = mpsc::channel::<Update>(100);
    println!("Connected to RFQs! Streaming live price levels...\n");
    tokio::spawn(rfq_stream_builder.build(tx));

    let swapper_pk = get_env("PRIVATE_KEY");
    let rfq_stream_processor = RFQStreamProcessor::new();
    rfq_stream_processor
        .process_rfq_stream(
            &mut rx,
            &sell_token,
            &buy_token,
            amount_in,
            cli.swap_args.chain,
            swapper_pk,
        )
        .await?;
    Ok(())
}

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .init();
}

// Format token amounts to human-readable values
fn format_token_amount(amount: &BigUint, token: &Token) -> String {
    let decimal_amount = amount.to_f64().unwrap_or(0.0) / 10f64.powi(token.decimals as i32);
    format!("{decimal_amount:.6}")
}

// Calculate price ratios in both directions
fn format_price_ratios(
    amount_in: &BigUint,
    amount_out: &BigUint,
    token_in: &Token,
    token_out: &Token,
) -> (f64, f64) {
    let decimal_in = amount_in.to_f64().unwrap_or(0.0) / 10f64.powi(token_in.decimals as i32);
    let decimal_out = amount_out.to_f64().unwrap_or(0.0) / 10f64.powi(token_out.decimals as i32);

    if decimal_in > 0.0 && decimal_out > 0.0 {
        let forward = decimal_out / decimal_in;
        let reverse = decimal_in / decimal_out;
        (forward, reverse)
    } else {
        (0.0, 0.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn create_solution(
    component: ProtocolComponent,
    state: Arc<dyn ProtocolSim>,
    sell_token: Token,
    buy_token: Token,
    sell_amount: BigUint,
    user_address: Bytes,
    expected_amount: BigUint,
) -> Solution {
    // Prepare data to encode. First we need to create a swap object
    let simple_swap =
        SwapBuilder::new(component, sell_token.address.clone(), buy_token.address.clone())
            .protocol_state(state)
            .estimated_amount_in(sell_amount.clone())
            .build();

    // Compute a minimum amount out
    //
    // # ⚠️ Important Responsibility Note
    // For maximum security, in production code, this minimum amount out should be computed
    // from a third-party source.
    let slippage = 0.0025; // 0.25% slippage
    let bps = BigUint::from(10_000u32);
    let slippage_percent = BigUint::from((slippage * 10000.0) as u32);
    let multiplier = &bps - slippage_percent;
    let min_amount_out = (expected_amount * &multiplier) / &bps;

    // Then we create a solution object with the previous swap
    Solution {
        sender: user_address.clone(),
        receiver: user_address,
        given_token: sell_token.address,
        given_amount: sell_amount,
        checked_token: buy_token.address,
        exact_out: false, // it's an exact in solution
        checked_amount: min_amount_out,
        swaps: vec![simple_swap],
        ..Default::default()
    }
}

/// Encodes a transaction for the Tycho Router using the `singleSwapPermit2` method.
///
/// # ⚠️ Important Responsibility Note
///
/// This function is intended as **an illustrative example only** and supports only the method of
/// interest of this quickstart. **Users must implement their own encoding logic** to ensure:
/// - Full control of parameters passed to the router.
/// - Proper validation and setting of critical inputs such as `minAmountOut`.
fn encode_tycho_router_call(
    chain_id: u64,
    encoded_solution: EncodedSolution,
    solution: &Solution,
    native_address: Bytes,
    signer: PrivateKeySigner,
) -> Result<Transaction, EncodingError> {
    let p = encoded_solution
        .permit
        .expect("Permit object must be set");
    let permit = PermitSingle::try_from(&p)
        .map_err(|_| EncodingError::InvalidInput("Invalid permit".to_string()))?;
    let signature = sign_permit(chain_id, &p, signer)?;
    let given_amount = biguint_to_u256(&solution.given_amount);
    let min_amount_out = biguint_to_u256(&solution.checked_amount);
    let given_token = Address::from_slice(&solution.given_token);
    let checked_token = Address::from_slice(&solution.checked_token);
    let receiver = Address::from_slice(&solution.receiver);

    let method_calldata = (
        given_amount,
        given_token,
        checked_token,
        min_amount_out,
        false,
        false,
        receiver,
        permit,
        signature.as_bytes().to_vec(),
        encoded_solution.swaps,
    )
        .abi_encode();

    let contract_interaction = encode_input(&encoded_solution.function_signature, method_calldata);
    let value = if solution.given_token == native_address {
        solution.given_amount.clone()
    } else {
        BigUint::ZERO
    };
    Ok(Transaction { to: encoded_solution.interacting_with, value, data: contract_interaction })
}

/// Signs a Permit2 `PermitSingle` struct using the EIP-712 signing scheme.
///
/// This function constructs an EIP-712 domain specific to the Permit2 contract and computes the
/// hash of the provided `PermitSingle`. It then uses the given `PrivateKeySigner` to produce
/// a cryptographic signature of the permit.
///
/// # Warning
/// This is only an **example implementation** provided for reference purposes.
/// **Do not rely on this in production.** You should implement your own version.
fn sign_permit(
    chain_id: u64,
    permit_single: &models::PermitSingle,
    signer: PrivateKeySigner,
) -> Result<Signature, EncodingError> {
    let permit2_address = Address::from_str("0x000000000022D473030F116dDEE9F6B43aC78BA3")
        .map_err(|_| EncodingError::FatalError("Permit2 address not valid".to_string()))?;
    let domain = eip712_domain! {
        name: "Permit2",
        chain_id: chain_id,
        verifying_contract: permit2_address,
    };
    let permit_single: PermitSingle = PermitSingle::try_from(permit_single)?;
    let hash = permit_single.eip712_signing_hash(&domain);
    signer
        .sign_hash_sync(&hash)
        .map_err(|e| {
            EncodingError::FatalError(format!("Failed to sign permit2 approval with error: {e}"))
        })
}

/// Encodes the input data for a function call to the given function selector.
pub fn encode_input(selector: &str, mut encoded_args: Vec<u8>) -> Vec<u8> {
    let mut hasher = Keccak256::new();
    hasher.update(selector.as_bytes());
    let selector_bytes = &hasher.finalize()[..4];
    let mut call_data = selector_bytes.to_vec();
    // Remove extra prefix if present (32 bytes for dynamic data)
    // Alloy encoding is including a prefix for dynamic data indicating the offset or length
    // but at this point we don't want that
    if encoded_args.len() > 32 &&
        encoded_args[..32] ==
            [0u8; 31]
                .into_iter()
                .chain([32].to_vec())
                .collect::<Vec<u8>>()
    {
        encoded_args = encoded_args[32..].to_vec();
    }
    call_data.extend(encoded_args);
    call_data
}
