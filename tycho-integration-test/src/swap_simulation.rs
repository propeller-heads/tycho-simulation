use std::{str::FromStr, sync::Arc};

use alloy::{
    eips::BlockNumberOrTag,
    network::Ethereum,
    providers::{Provider, RootProvider},
    rpc::types::{
        simulate::{SimBlock, SimulatePayload},
        state::AccountOverride,
        Block, TransactionRequest,
    },
    sol_types::SolValue,
};
use miette::{miette, IntoDiagnostic, WrapErr};
use num_bigint::BigUint;
use tokio_retry2::{
    strategy::{jitter, ExponentialFactorBackoff},
    Retry, RetryError,
};
use tracing::{info, warn};
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    traits::{AllowanceSlotDetector, BalanceSlotDetector},
    Bytes,
};
use tycho_ethereum::entrypoint_tracer::{
    allowance_slot_detector::{AllowanceSlotDetectorConfig, EVMAllowanceSlotDetector},
    balance_slot_detector::{BalanceSlotDetectorConfig, EVMBalanceSlotDetector},
};
use tycho_execution::encoding::{
    evm::encoder_builders::TychoRouterEncoderBuilder,
    models::{EncodedSolution, Solution, SwapBuilder, Transaction, UserTransferType},
};
use tycho_simulation::{
    evm::protocol::u256_num::biguint_to_u256,
    foundry_evm::revm::primitives::{
        alloy_primitives::Keccak256, map::AddressHashMap, Address, U256,
    },
    protocol::models::ProtocolComponent,
};

pub fn encode_swap(
    component: &ProtocolComponent,
    state: Option<Arc<dyn ProtocolSim>>,
    sell_token: &Token,
    buy_token: &Token,
    amount_in: BigUint,
    expected_amount_out: BigUint,
    chain: Chain,
) -> miette::Result<(Solution, Transaction)> {
    let solution = create_solution(
        component.clone(),
        state,
        sell_token.clone(),
        buy_token.clone(),
        amount_in.clone(),
        expected_amount_out.clone(),
    )?;
    let encoded_solution = {
        let encoder = TychoRouterEncoderBuilder::new()
            .chain(chain)
            .user_transfer_type(UserTransferType::TransferFrom)
            .build()
            .into_diagnostic()
            .wrap_err("Failed to build encoder")?;
        encoder
            .encode_solutions(vec![solution.clone()])
            .into_diagnostic()
            .wrap_err("Failed to encode router calldata")?
            .into_iter()
            .next()
            .ok_or_else(|| miette!("Missing solution"))?
    };
    let transaction =
        encoded_transaction(encoded_solution.clone(), &solution, chain.native_token().address)?;
    Ok((solution, transaction))
}

fn create_solution(
    component: ProtocolComponent,
    state: Option<Arc<dyn ProtocolSim>>,
    sell_token: Token,
    buy_token: Token,
    amount_in: BigUint,
    expected_amount_out: BigUint,
) -> miette::Result<Solution> {
    let user_address =
        Bytes::from_str("0xf847a638E44186F3287ee9F8cAF73FF4d4B80784").into_diagnostic()?;

    // Prepare data to encode. First we need to create a swap object
    let mut simple_swap =
        SwapBuilder::new(component, sell_token.address.clone(), buy_token.address.clone());
    if let Some(state) = state {
        simple_swap = simple_swap
            .protocol_state(state)
            .estimated_amount_in(amount_in.clone());
    }
    let simple_swap = simple_swap.build();

    // Compute a minimum amount out
    //
    // # ⚠️ Important Responsibility Note
    // For maximum security, in production code, this minimum amount out should be computed
    // from a third-party source.
    let slippage = 0.0025; // 0.25% slippage
    let bps = BigUint::from(10_000u32);
    let slippage_percent = BigUint::from((slippage * 10000.0) as u32);
    let multiplier = &bps - slippage_percent;
    let min_amount_out = (expected_amount_out * &multiplier) / &bps;

    // Then we create a solution object with the previous swap
    Ok(Solution {
        sender: user_address.clone(),
        receiver: user_address,
        given_token: sell_token.address,
        given_amount: amount_in,
        checked_token: buy_token.address,
        exact_out: false, // it's an exact in solution
        checked_amount: min_amount_out,
        swaps: vec![simple_swap],
        ..Default::default()
    })
}

fn encoded_transaction(
    encoded_solution: EncodedSolution,
    solution: &Solution,
    native_address: Bytes,
) -> miette::Result<Transaction> {
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
        true,
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

pub async fn simulate_swap_transaction(
    provider: RootProvider<Ethereum>,
    rpc_url: &str,
    solution: &Solution,
    transaction: &Transaction,
) -> miette::Result<BigUint> {
    let block = provider
        .get_block_by_number(BlockNumberOrTag::Latest)
        .await
        .into_diagnostic()
        .wrap_err("Failed to fetch latest block")?
        .ok_or_else(|| miette::Error::msg("No block found"))?;
    let user_address = Address::from_slice(&solution.sender[..20]);
    let request = swap_request(transaction, &block, user_address)?;
    let state_overwrites =
        setup_user_overwrites(rpc_url, solution, transaction, &block, user_address).await?;
    let payload = SimulatePayload {
        block_state_calls: vec![SimBlock {
            block_overrides: None,
            state_overrides: Some(state_overwrites),
            calls: vec![request],
        }],
        trace_transfers: true,
        validation: true,
        return_full_transactions: true,
    };
    let retry_strategy = ExponentialFactorBackoff::from_millis(1000, 2.)
        .max_delay_millis(10000)
        .map(jitter)
        .take(10);
    let output = Retry::spawn(retry_strategy, async || match provider.simulate(&payload).await {
        Ok(res) => Ok(res),
        Err(e) => Err(RetryError::transient(e)),
    })
    .await
    .into_diagnostic()
    .wrap_err(miette!("Failed to simulate transaction after retries"))?;
    let block = output
        .first()
        .ok_or_else(|| miette!("No blocks found in simulation output"))?;
    info!("Simulated block {}", block.inner.header.number);
    let transaction = block
        .calls
        .first()
        .ok_or_else(|| miette!("No transactions found in simulated block"))?;
    info!(
        "Transaction status: {status:?}, gas used: {gas_used}",
        status = transaction.status,
        gas_used = transaction.gas_used
    );
    if !transaction.status {
        warn!("Transaction: {transaction:?}");
        return Err(miette!("Transaction status is false"));
    }
    let amount_out = U256::abi_decode(&transaction.return_data)
        .map_err(|e| miette!("Failed to decode swap amount: {e:?}"))?;
    BigUint::from_str(amount_out.to_string().as_str()).into_diagnostic()
}

/// Set up all state overrides needed for simulation
async fn setup_user_overwrites(
    rpc_url: &str,
    solution: &Solution,
    transaction: &Transaction,
    block: &Block,
    user_address: Address,
) -> miette::Result<AddressHashMap<AccountOverride>> {
    let mut overwrites = AddressHashMap::default();
    // Add ETH balance override for the user to ensure they have enough gas funds
    let mut eth_balance = U256::from_str("100000000000000000000").unwrap(); // 100 ETH

    let token_address = Address::from_slice(&solution.given_token[..20]);
    // If given token is ETH, add the given amount to the balance
    if solution.given_token == Bytes::zero(20) {
        eth_balance += biguint_to_u256(&solution.given_amount);
    // if the given token is not ETH, do balance and allowance slots overwrites
    } else {
        let detector = EVMBalanceSlotDetector::new(BalanceSlotDetectorConfig {
            rpc_url: rpc_url.to_string(),
            max_retries: 10,
            ..Default::default()
        })
        .into_diagnostic()?;

        let results = detector
            .detect_balance_slots(
                std::slice::from_ref(&solution.given_token),
                (**user_address).into(),
                (*block.header.hash).into(),
            )
            .await;

        let balance_slot =
            if let Some(Ok((_storage_addr, slot))) = results.get(&solution.given_token.clone()) {
                slot
            } else {
                return Err(miette!("Couldn't find balance storage slot for token {token_address}"));
            };

        let detector = EVMAllowanceSlotDetector::new(AllowanceSlotDetectorConfig {
            rpc_url: rpc_url.to_string(),
            max_retries: 10,
            ..Default::default()
        })
        .into_diagnostic()?;

        let results = detector
            .detect_allowance_slots(
                std::slice::from_ref(&solution.given_token),
                (**user_address).into(),
                transaction.to.clone(), // tycho router
                (*block.header.hash).into(),
            )
            .await;

        let allowance_slot = if let Some(Ok((_storage_addr, slot))) =
            results.get(&solution.given_token.clone())
        {
            slot
        } else {
            return Err(miette!("Couldn't find allowance storage slot for token {token_address}"));
        };

        overwrites.insert(
            token_address,
            AccountOverride::default().with_state_diff(vec![
                (
                    alloy::primitives::B256::from_slice(allowance_slot),
                    alloy::primitives::B256::from_slice(&U256::MAX.to_be_bytes::<32>()),
                ),
                (
                    alloy::primitives::B256::from_slice(balance_slot),
                    alloy::primitives::B256::from_slice(
                        &biguint_to_u256(&solution.given_amount).to_be_bytes::<32>(),
                    ),
                ),
            ]),
        );
    }
    overwrites.insert(user_address, AccountOverride::default().with_balance(eth_balance));

    Ok(overwrites)
}

fn swap_request(
    transaction: &Transaction,
    block: &Block,
    user_address: Address,
) -> miette::Result<TransactionRequest> {
    let (max_fee_per_gas, max_priority_fee_per_gas) = calculate_gas_fees(block)?;
    Ok(TransactionRequest::default()
        .to(Address::from_slice(&transaction.to[..20]))
        .input(transaction.data.clone().into())
        .value(U256::from_str(&transaction.value.to_string()).unwrap_or_default())
        .from(user_address)
        .max_fee_per_gas(
            max_fee_per_gas
                .try_into()
                .unwrap_or(u128::MAX),
        )
        .max_priority_fee_per_gas(
            max_priority_fee_per_gas
                .try_into()
                .unwrap_or(u128::MAX),
        ))
}

/// Calculate gas fees based on block base fee
fn calculate_gas_fees(block: &Block) -> miette::Result<(U256, U256)> {
    let base_fee = block
        .header
        .base_fee_per_gas
        .ok_or_else(|| miette::miette!("Block does not have base fee (pre-EIP-1559)"))?;
    // Set max_priority_fee_per_gas to a reasonable value (2 Gwei)
    let max_priority_fee_per_gas = U256::from(2_000_000_000u64);
    // Set max_fee_per_gas to base_fee * 2 + max_priority_fee_per_gas to handle fee fluctuations
    let max_fee_per_gas = U256::from(base_fee) * U256::from(2u64) + max_priority_fee_per_gas;
    info!(
        "Gas pricing: base_fee={}, max_priority_fee_per_gas={}, max_fee_per_gas={}",
        base_fee, max_priority_fee_per_gas, max_fee_per_gas
    );
    Ok((max_fee_per_gas, max_priority_fee_per_gas))
}
