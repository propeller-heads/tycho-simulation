use std::{process::exit, str::FromStr, sync::Arc};

use alloy::{
    eips::BlockNumberOrTag,
    network::EthereumWallet,
    primitives::{Address, Bytes as AlloyBytes, TxKind, B256},
    providers::{Provider, ProviderBuilder},
    rpc::types::{
        simulate::{SimBlock, SimulatePayload},
        TransactionInput, TransactionRequest,
    },
    signers::{
        k256::ecdsa::SigningKey,
        local::{LocalSigner, PrivateKeySigner},
    },
    sol_types::SolValue,
};
use foundry_config::NamedChain;
use miette::Result;
use num_bigint::BigUint;
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};
use tycho_execution::encoding::tycho_encoder::TychoEncoder;
use tycho_simulation::{
    evm::protocol::u256_num::biguint_to_u256, protocol::models::ProtocolComponent,
};

use crate::{
    create_solution, encode_input, encode_tycho_router_call, format_price_ratios,
    format_token_amount,
};

pub struct SwapExecutor {
    pub chain: Chain,
    pub encoder: Box<dyn TychoEncoder>,
    pub signer_data: Option<SignerData>,
}

pub struct SignerData {
    pub signer: LocalSigner<SigningKey>,
    pub provider: Arc<dyn Provider>,
}

impl SwapExecutor {
    pub fn new(chain: Chain, encoder: Box<dyn TychoEncoder>) -> Self {
        Self { chain, encoder, signer_data: None }
    }

    pub async fn with_signer(mut self, pk: String, rpc_url: &str) -> Result<Self> {
        let signer =
            PrivateKeySigner::from_bytes(&B256::from_str(&pk).expect("Invalid private key"))
                .expect("Failed to create signer");
        let provider = ProviderBuilder::default()
            .with_chain(NamedChain::try_from(self.chain.id()).expect("Invalid chain"))
            .wallet(EthereumWallet::from(signer.clone()))
            .connect(rpc_url)
            .await
            .expect("Failed to connect provider");
        self.signer_data = Some(SignerData { signer, provider: Arc::new(provider) });
        Ok(self)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn simulate_swap(
        &self,
        component: &ProtocolComponent,
        state: &dyn ProtocolSim,
        sell_token: &Token,
        buy_token: &Token,
        amount_in: &BigUint,
        amount_out: &BigUint,
        signer_data: &SignerData,
    ) -> Result<()> {
        println!("\nSimulating RFQ swap...");
        println!("Step 1: Encoding the permit2 transaction...");
        let approve_function_signature = "approve(address,uint256)";
        let args = (
            Address::from_str("0x000000000022D473030F116dDEE9F6B43aC78BA3")
                .expect("Invalid address"),
            biguint_to_u256(amount_in),
        );
        let approval_data = encode_input(approve_function_signature, args.abi_encode());
        let nonce = signer_data
            .provider
            .get_transaction_count(signer_data.signer.address())
            .await
            .expect("Failed to get nonce");
        let block = signer_data
            .provider
            .get_block_by_number(BlockNumberOrTag::Latest)
            .await
            .expect("Failed to fetch latest block")
            .expect("Block not found");
        let base_fee = block
            .header
            .base_fee_per_gas
            .expect("Base fee not available");
        let max_priority_fee_per_gas = 1_000_000_000u64;
        let max_fee_per_gas = base_fee + max_priority_fee_per_gas;
        let approval_request = TransactionRequest {
            to: Some(TxKind::Call(Address::from_slice(&sell_token.address))),
            from: Some(signer_data.signer.address()),
            value: None,
            input: TransactionInput { input: Some(AlloyBytes::from(approval_data)), data: None },
            gas: Some(100_000u64),
            chain_id: Some(self.chain.id()),
            max_fee_per_gas: Some(max_fee_per_gas.into()),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas.into()),
            nonce: Some(nonce),
            ..Default::default()
        };

        println!("Step 2: Encoding the solution transaction...");

        let solution = create_solution(
            component.clone(),
            Arc::from(state.clone_box()),
            sell_token.clone(),
            buy_token.clone(),
            amount_in.clone(),
            Bytes::from(signer_data.signer.address().to_vec()),
            amount_out.clone(),
        );

        let encoded_solution = self
            .encoder
            .encode_solutions(vec![solution.clone()])
            .expect("Failed to encode router calldata")[0]
            .clone();

        let tx = encode_tycho_router_call(
            self.chain.id(),
            encoded_solution.clone(),
            &solution,
            self.chain.native_token().address,
            signer_data.signer.clone(),
        )
        .expect("Failed to encode router call");

        let swap_request = TransactionRequest {
            to: Some(TxKind::Call(Address::from_slice(&tx.to))),
            from: Some(signer_data.signer.address()),
            value: Some(biguint_to_u256(&tx.value)),
            input: TransactionInput { input: Some(AlloyBytes::from(tx.data)), data: None },
            gas: Some(800_000u64),
            chain_id: Some(self.chain.id()),
            max_fee_per_gas: Some(max_fee_per_gas.into()),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas.into()),
            nonce: Some(nonce + 1),
            ..Default::default()
        };

        println!("Step 3: Simulating approval and solution transactions together...");
        let approval_payload = SimulatePayload {
            block_state_calls: vec![SimBlock {
                block_overrides: None,
                state_overrides: None,
                calls: vec![approval_request.clone(), swap_request],
            }],
            trace_transfers: true,
            validation: true,
            return_full_transactions: true,
        };

        match signer_data
            .provider
            .simulate(&approval_payload)
            .await
        {
            Ok(output) => {
                let mut all_successful = true;
                for block in output.iter() {
                    println!(
                        "\nSimulated Block {block_num}:",
                        block_num = block.inner.header.number
                    );
                    for transaction in block.calls.iter() {
                        println!(
                            "  RFQ Swap: Status: {status:?}, Gas Used: {gas_used}",
                            status = transaction.status,
                            gas_used = transaction.gas_used
                        );
                        if !transaction.status {
                            all_successful = false;
                        }
                    }
                }

                if all_successful {
                    println!("\n✅ Simulation successful!");
                } else {
                    println!("\n❌ Simulation failed! One or more transactions reverted.");
                    println!("Consider adjusting parameters and re-simulating before execution.");
                }
            }
            Err(e) => {
                eprintln!("\n❌ Simulation failed: {e:?}");
                println!("Your RPC provider does not support transaction simulation. Consider switching RPC provider.");
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn execute_swap(
        &self,
        component: &ProtocolComponent,
        state: &dyn ProtocolSim,
        sell_token: &Token,
        buy_token: &Token,
        amount_in: &BigUint,
        amount_out: &BigUint,
        signer_data: &SignerData,
    ) -> Result<()> {
        println!("Executing RFQ swap...");

        println!("Step 1: Sending permit2 approval...");
        let approve_function_signature = "approve(address,uint256)";
        let args = (
            Address::from_str("0x000000000022D473030F116dDEE9F6B43aC78BA3")
                .expect("Invalid address"),
            biguint_to_u256(amount_in),
        );
        let approval_data = encode_input(approve_function_signature, args.abi_encode());
        let nonce = signer_data
            .provider
            .get_transaction_count(signer_data.signer.address())
            .await
            .expect("Failed to get nonce");
        let block = signer_data
            .provider
            .get_block_by_number(BlockNumberOrTag::Latest)
            .await
            .expect("Failed to fetch latest block")
            .expect("Block not found");
        let base_fee = block
            .header
            .base_fee_per_gas
            .expect("Base fee not available");
        let max_priority_fee_per_gas = 1_000_000_000u64;
        let max_fee_per_gas = base_fee + max_priority_fee_per_gas;
        let approval_request = TransactionRequest {
            to: Some(TxKind::Call(Address::from_slice(&sell_token.address))),
            from: Some(signer_data.signer.address()),
            value: None,
            input: TransactionInput { input: Some(AlloyBytes::from(approval_data)), data: None },
            gas: Some(100_000u64),
            chain_id: Some(self.chain.id()),
            max_fee_per_gas: Some(max_fee_per_gas.into()),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas.into()),
            nonce: Some(nonce),
            ..Default::default()
        };

        let approval_receipt = match signer_data
            .provider
            .send_transaction(approval_request)
            .await
        {
            Ok(receipt) => receipt,
            Err(e) => {
                eprintln!("\nFailed to send approval transaction: {e:?}\n");
                return Ok(());
            }
        };

        let approval_result = match approval_receipt.get_receipt().await {
            Ok(result) => result,
            Err(e) => {
                eprintln!("\nFailed to get approval receipt: {e:?}\n");
                return Ok(());
            }
        };

        println!(
            "Approval transaction sent with hash: {hash:?} and status: {status:?}",
            hash = approval_result.transaction_hash,
            status = approval_result.status()
        );

        if !approval_result.status() {
            eprintln!("\nApproval transaction failed! Cannot proceed with swap.\n");
            return Ok(());
        }

        println!("Step 2: Encoding solution transaction...");

        let solution = create_solution(
            component.clone(),
            Arc::from(state.clone_box()),
            sell_token.clone(),
            buy_token.clone(),
            amount_in.clone(),
            Bytes::from(signer_data.signer.address().to_vec()),
            amount_out.clone(),
        );

        let encoded_solution = self
            .encoder
            .encode_solutions(vec![solution.clone()])
            .expect("Failed to encode router calldata")[0]
            .clone();

        let tx = encode_tycho_router_call(
            self.chain.id(),
            encoded_solution.clone(),
            &solution,
            self.chain.native_token().address,
            signer_data.signer.clone(),
        )
        .expect("Failed to encode router call");

        let swap_request = TransactionRequest {
            to: Some(TxKind::Call(Address::from_slice(&tx.to))),
            from: Some(signer_data.signer.address()),
            value: Some(biguint_to_u256(&tx.value)),
            input: TransactionInput { input: Some(AlloyBytes::from(tx.data)), data: None },
            gas: Some(800_000u64),
            chain_id: Some(self.chain.id()),
            max_fee_per_gas: Some(max_fee_per_gas.into()),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas.into()),
            nonce: Some(nonce + 1),
            ..Default::default()
        };

        let swap_receipt = match signer_data
            .provider
            .send_transaction(swap_request)
            .await
        {
            Ok(receipt) => receipt,
            Err(e) => {
                eprintln!("\nFailed to send swap transaction: {e:?}\n");
                return Ok(());
            }
        };

        let swap_result = match swap_receipt.get_receipt().await {
            Ok(result) => result,
            Err(e) => {
                eprintln!("\nFailed to get swap receipt: {e:?}\n");
                return Ok(());
            }
        };

        println!(
            "Swap transaction sent with hash: {hash:?} and status: {status:?}",
            hash = swap_result.transaction_hash,
            status = swap_result.status()
        );

        if swap_result.status() {
            println!("\n✅ Swap executed successfully! Exiting the session...\n");

            // Calculate the correct price ratio
            let (forward_price, _reverse_price) =
                format_price_ratios(amount_in, amount_out, sell_token, buy_token);

            println!(
                "Summary: Swapped {formatted_in} {sell_symbol} → {formatted_out} {buy_symbol} at a price of {forward_price:.6} {buy_symbol} per {sell_symbol}",
                formatted_in = format_token_amount(amount_in, sell_token),
                sell_symbol = sell_token.symbol,
                formatted_out = format_token_amount(amount_out, buy_token),
                buy_symbol = buy_token.symbol,
            );
            exit(0); // Exit the program after successful execution
        } else {
            eprintln!("\nSwap transaction failed!\n");
            Ok(())
        }
    }
}
