use std::collections::HashMap;

use alloy::{
    primitives::{Address, Bytes as AlloyBytes, B256, U256},
    sol_types::SolCall,
};
use alloy_rpc_types_eth::state::{AccountOverride, StateOverride};
use anyhow::{bail, Context, Result};
use app_data::AppDataHash;
use ethcontract::{web3, Address as EthAddress, Bytes as ContractBytes, U256 as EthU256};
use model::{
    interaction::InteractionData,
    order::{BuyTokenDestination, OrderData, OrderKind, SellTokenSource},
    signature::{Signature, SigningScheme},
};
use num_bigint::BigUint;
use services_contracts::alloy::support::{AnyoneAuthenticator, Solver, Spardose, Trader};
use tycho_simulation::{
    protocol::models::ProtocolComponent,
    tycho_common::{models::token::Token, Bytes},
};

use crate::contracts::{BCowHelperContract, RawInteraction, RawOrder};

pub const DEFAULT_SETTLEMENT_ADDRESS: &str = "0x9008d19f58aabd9ed0d60971565aa8510560ab41";
pub const SPARDOSE: Address = Address::new([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x00,
]);
pub const DEFAULT_GAS: u64 = 12_000_000;

pub type EncodedInteraction = (Address, U256, AlloyBytes);
pub type EncodedTrade =
    (U256, U256, Address, U256, U256, u32, [u8; 32], U256, U256, U256, AlloyBytes);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EncodedSettlement {
    pub tokens: Vec<Address>,
    pub clearing_prices: Vec<U256>,
    pub trades: Vec<EncodedTrade>,
    pub interactions: [Vec<EncodedInteraction>; 3],
}

#[derive(Clone, Debug)]
pub struct Amm {
    helper: BCowHelperContract,
    address: EthAddress,
}

pub struct TemplateOrder {
    pub order: OrderData,
    pub signature: Signature,
    pub pre_interactions: Vec<InteractionData>,
    pub post_interactions: Vec<InteractionData>,
}

#[derive(Debug)]
pub struct SettleOutput {
    pub gas_used: U256,
    pub out_amount: U256,
}

impl Amm {
    pub async fn new(
        address: EthAddress,
        helper: &BCowHelperContract,
    ) -> Result<Self, ethcontract::errors::MethodError> {
        let _ = helper.tokens(address).call().await?;
        Ok(Self { helper: helper.clone(), address })
    }

    async fn template_order_from_buy_amount(
        &self,
        buy_token: EthAddress,
        buy_amount: EthU256,
    ) -> Result<TemplateOrder> {
        let (order, pre, post, signature) = self
            .helper
            .order_from_buy_amount(self.address, buy_token, buy_amount)
            .call()
            .await?;
        convert_orders_response(order, signature, pre, post)
    }

    async fn template_order_from_sell_amount(
        &self,
        sell_token: EthAddress,
        sell_amount: EthU256,
    ) -> Result<TemplateOrder> {
        let (order, pre, post, signature) = self
            .helper
            .order_from_sell_amount(self.address, sell_token, sell_amount)
            .call()
            .await?;
        convert_orders_response(order, signature, pre, post)
    }
}

impl SettleOutput {
    pub fn from_swap(output: Solver::Solver::swapReturn, kind: OrderKind) -> Result<Self> {
        let trader_balance_before = output.queriedBalances[2];
        let trader_balance_after = output.queriedBalances[3];
        let out_amount = match kind {
            OrderKind::Sell => trader_balance_after
                .checked_sub(trader_balance_before)
                .context("underflow computing sell-order out amount")?,
            OrderKind::Buy => trader_balance_before
                .checked_sub(trader_balance_after)
                .context("underflow computing buy-order out amount")?,
        };
        Ok(Self { gas_used: output.gasUsed, out_amount })
    }
}

pub async fn select_pool_order(
    amm: &Amm,
    sell_token: Address,
    buy_token: Address,
    sell_amount: U256,
    simulated_buy_amount: &BigUint,
) -> Result<TemplateOrder> {
    let direct = amm
        .template_order_from_buy_amount(to_eth_address(sell_token), to_eth_u256(sell_amount))
        .await;
    if let Ok(template) = direct {
        if order_matches_pool_side(&template.order, sell_token, buy_token) {
            return Ok(template);
        }
    }

    let fallback = amm
        .template_order_from_sell_amount(
            to_eth_address(buy_token),
            to_eth_u256(biguint_to_u256(simulated_buy_amount.clone())),
        )
        .await
        .context("helper fallback order_from_sell_amount failed")?;
    if order_matches_pool_side(&fallback.order, sell_token, buy_token) {
        return Ok(fallback);
    }

    bail!("helper returned an order with unexpected token direction")
}

#[allow(clippy::too_many_arguments)]
pub fn encode_settlement(
    sell_token: Address,
    buy_token: Address,
    sell_amount: U256,
    trader: Address,
    solver: Address,
    settlement_address: Address,
    pool_owner: Address,
    pool_template: &TemplateOrder,
) -> EncodedSettlement {
    let encoded_user = encode_fake_user_trade(sell_token, buy_token, sell_amount, trader);
    let encoded_pool = encode_trade(
        &pool_template.order,
        &pool_template.signature,
        pool_owner,
        1,
        0,
        &pool_template.order.sell_amount,
    );
    let mut settlement = EncodedSettlement {
        tokens: vec![sell_token, buy_token],
        clearing_prices: vec![pool_template.order.sell_amount, pool_template.order.buy_amount],
        trades: vec![encoded_user, encoded_pool],
        interactions: [
            pool_template
                .pre_interactions
                .iter()
                .map(interaction_data_to_encoded)
                .collect(),
            Vec::new(),
            pool_template
                .post_interactions
                .iter()
                .map(interaction_data_to_encoded)
                .collect(),
        ],
    };
    settlement.interactions[0].push(encode_trade_setup_interaction(
        sell_token,
        sell_amount,
        trader,
        settlement_address,
        solver,
    ));
    add_balance_queries(settlement, buy_token, trader, solver)
}

pub fn prepare_state_overrides(
    authenticator: Address,
    sell_token: Address,
    sell_amount: U256,
    sell_token_balance_slot: U256,
    trader_address: Address,
    solver_address: Address,
    is_solver: bool,
) -> StateOverride {
    let mut overrides = StateOverride::default();
    overrides.insert(
        sell_token,
        solidity_mapping_state_override(SPARDOSE, sell_amount, sell_token_balance_slot),
    );
    overrides.insert(
        trader_address,
        AccountOverride::default().with_code(Trader::Trader::DEPLOYED_BYTECODE.clone()),
    );
    overrides.insert(
        SPARDOSE,
        AccountOverride::default().with_code(Spardose::Spardose::DEPLOYED_BYTECODE.clone()),
    );
    overrides.insert(
        solver_address,
        AccountOverride::default()
            .with_code(Solver::Solver::DEPLOYED_BYTECODE.clone())
            .with_balance(U256::from(10u64).pow(U256::from(18u64))),
    );
    if !is_solver {
        overrides.insert(
            authenticator,
            AccountOverride::default()
                .with_code(AnyoneAuthenticator::AnyoneAuthenticator::DEPLOYED_BYTECODE.clone()),
        );
    }
    overrides
}

pub fn resolve_token(
    address: &Bytes,
    component: &ProtocolComponent,
    all_tokens: &HashMap<Bytes, Token>,
) -> Token {
    component
        .tokens
        .iter()
        .find(|token| token.address == *address)
        .cloned()
        .or_else(|| all_tokens.get(address).cloned())
        .unwrap_or_else(|| Token::new(address, "UNKNOWN", 18, 0, &[], component.chain, 0))
}

pub fn bytes_to_eth_address(bytes: &Bytes) -> Result<EthAddress> {
    if bytes.len() != 20 {
        bail!("expected 20-byte address, got {}", bytes.len());
    }
    Ok(EthAddress::from_slice(bytes.as_ref()))
}

pub fn bytes_to_address(bytes: &Bytes) -> Result<Address> {
    if bytes.len() != 20 {
        bail!("expected 20-byte address, got {}", bytes.len());
    }
    Ok(Address::from_slice(bytes.as_ref()))
}

pub fn u256_to_biguint(value: U256) -> BigUint {
    BigUint::from_bytes_be(&value.to_be_bytes::<32>())
}

pub fn biguint_to_u256(value: BigUint) -> U256 {
    U256::from_be_slice(&value.to_bytes_be())
}

pub fn format_token_amount(amount: U256, decimals: u8) -> String {
    let raw = amount.to_string();
    let decimals = decimals as usize;
    if decimals == 0 {
        return raw;
    }
    if raw.len() <= decimals {
        let fractional = format!("{raw:0>width$}", width = decimals);
        let trimmed = fractional.trim_end_matches('0');
        return if trimmed.is_empty() { "0".to_string() } else { format!("0.{trimmed}") };
    }
    let split = raw.len() - decimals;
    let integer = &raw[..split];
    let fractional = raw[split..].trim_end_matches('0');
    if fractional.is_empty() {
        integer.to_string()
    } else {
        format!("{integer}.{fractional}")
    }
}

pub fn encode_settlement_call(
    settlement: EncodedSettlement,
) -> services_contracts::alloy::GPv2Settlement::GPv2Settlement::settleCall {
    services_contracts::alloy::GPv2Settlement::GPv2Settlement::settleCall {
        tokens: settlement.tokens,
        clearingPrices: settlement.clearing_prices,
        trades: settlement
            .trades
            .into_iter()
            .map(|trade| services_contracts::alloy::GPv2Settlement::GPv2Trade::Data {
                sellTokenIndex: trade.0,
                buyTokenIndex: trade.1,
                receiver: trade.2,
                sellAmount: trade.3,
                buyAmount: trade.4,
                validTo: trade.5,
                appData: trade.6.into(),
                feeAmount: trade.7,
                flags: trade.8,
                executedAmount: trade.9,
                signature: trade.10,
            })
            .collect(),
        interactions: settlement.interactions.map(|group| {
            group
                .into_iter()
                .map(|interaction| {
                    services_contracts::alloy::GPv2Settlement::GPv2Interaction::Data {
                        target: interaction.0,
                        value: interaction.1,
                        callData: interaction.2,
                    }
                })
                .collect()
        }),
    }
}

fn order_matches_pool_side(order: &OrderData, sell_token: Address, buy_token: Address) -> bool {
    order.buy_token == sell_token && order.sell_token == buy_token
}

fn encode_fake_user_trade(
    sell_token: Address,
    buy_token: Address,
    sell_amount: U256,
    trader: Address,
) -> EncodedTrade {
    let fake_order = OrderData {
        sell_token,
        sell_amount,
        buy_token,
        buy_amount: U256::ZERO,
        receiver: Some(trader),
        valid_to: u32::MAX,
        app_data: Default::default(),
        fee_amount: U256::ZERO,
        kind: OrderKind::Sell,
        partially_fillable: false,
        sell_token_balance: SellTokenSource::Erc20,
        buy_token_balance: BuyTokenDestination::Erc20,
    };
    encode_trade(
        &fake_order,
        &Signature::default_with(SigningScheme::Eip1271),
        trader,
        0,
        1,
        &sell_amount,
    )
}

fn encode_trade_setup_interaction(
    sell_token: Address,
    sell_amount: U256,
    trader: Address,
    settlement: Address,
    solver: Address,
) -> EncodedInteraction {
    (
        solver,
        U256::ZERO,
        Solver::Solver::ensureTradePreconditionsCall {
            trader,
            settlementContract: settlement,
            sellToken: sell_token,
            sellAmount: sell_amount,
            nativeToken: Address::ZERO,
            spardose: SPARDOSE,
        }
        .abi_encode()
        .into(),
    )
}

fn add_balance_queries(
    mut settlement: EncodedSettlement,
    buy_token: Address,
    trader: Address,
    solver: Address,
) -> EncodedSettlement {
    let interaction: EncodedInteraction = (
        solver,
        U256::ZERO,
        Solver::Solver::storeBalanceCall { token: buy_token, owner: trader, countGas: true }
            .abi_encode()
            .into(),
    );
    settlement.interactions[0].push(interaction.clone());
    settlement.interactions[2].insert(0, interaction);
    settlement
}

fn solidity_mapping_state_override(holder: Address, amount: U256, slot: U256) -> AccountOverride {
    let mut buf = [0u8; 64];
    buf[12..32].copy_from_slice(holder.as_slice());
    buf[32..64].copy_from_slice(&slot.to_be_bytes::<32>());
    let key = B256::from(web3::signing::keccak256(&buf));
    let value = B256::from(amount.to_be_bytes::<32>());
    AccountOverride::default().with_state_diff([(key, value)])
}

fn interaction_data_to_encoded(interaction: &InteractionData) -> EncodedInteraction {
    (interaction.target, interaction.value, interaction.call_data.clone().into())
}

fn encode_trade(
    order: &OrderData,
    signature: &Signature,
    owner: Address,
    sell_token_index: usize,
    buy_token_index: usize,
    executed_amount: &U256,
) -> EncodedTrade {
    (
        U256::from(sell_token_index),
        U256::from(buy_token_index),
        order.receiver.unwrap_or(Address::ZERO),
        order.sell_amount,
        order.buy_amount,
        order.valid_to,
        order.app_data.0,
        order.fee_amount,
        order_flags(order, signature),
        *executed_amount,
        signature
            .encode_for_settlement(owner)
            .into(),
    )
}

fn order_flags(order: &OrderData, signature: &Signature) -> U256 {
    let mut result = 0u8;
    result |= match order.kind {
        OrderKind::Sell => 0b0,
        OrderKind::Buy => 0b1,
    };
    result |= (order.partially_fillable as u8) << 1;
    result |= match order.sell_token_balance {
        SellTokenSource::Erc20 => 0b00,
        SellTokenSource::External => 0b10,
        SellTokenSource::Internal => 0b11,
    } << 2;
    result |= match order.buy_token_balance {
        BuyTokenDestination::Erc20 => 0b0,
        BuyTokenDestination::Internal => 0b1,
    } << 4;
    result |= match signature.scheme() {
        SigningScheme::Eip712 => 0b00,
        SigningScheme::EthSign => 0b01,
        SigningScheme::Eip1271 => 0b10,
        SigningScheme::PreSign => 0b11,
    } << 5;
    U256::from(result)
}

fn convert_orders_response(
    order: RawOrder,
    signature: ContractBytes<Vec<u8>>,
    pre_interactions: Vec<RawInteraction>,
    post_interactions: Vec<RawInteraction>,
) -> Result<TemplateOrder> {
    let converted = OrderData {
        sell_token: to_address(order.0),
        buy_token: to_address(order.1),
        receiver: Some(to_address(order.2)),
        sell_amount: to_u256(order.3),
        buy_amount: to_u256(order.4),
        valid_to: order.5,
        app_data: AppDataHash(order.6 .0),
        fee_amount: to_u256(order.7),
        kind: convert_kind(&order.8 .0)?,
        partially_fillable: order.9,
        sell_token_balance: convert_sell_token_source(&order.10 .0)?,
        buy_token_balance: convert_buy_token_destination(&order.11 .0)?,
    };

    Ok(TemplateOrder {
        order: converted,
        signature: Signature::Eip1271(
            signature
                .0
                .into_iter()
                .skip(20)
                .collect(),
        ),
        pre_interactions: convert_interactions(pre_interactions),
        post_interactions: convert_interactions(post_interactions),
    })
}

fn convert_interactions(interactions: Vec<RawInteraction>) -> Vec<InteractionData> {
    interactions
        .into_iter()
        .map(|interaction| InteractionData {
            target: to_address(interaction.0),
            value: to_u256(interaction.1),
            call_data: interaction.2 .0,
        })
        .collect()
}

fn convert_kind(bytes: &[u8]) -> Result<OrderKind> {
    match hex::encode(bytes).as_str() {
        "f3b277728b3fee749481eb3e0b3b48980dbbab78658fc419025cb16eee346775" => Ok(OrderKind::Sell),
        "6ed88e868af0a1983e3886d5f3e95a2fafbd6c3450bc229e27342283dc429ccc" => Ok(OrderKind::Buy),
        other => bail!("unknown order kind: {other}"),
    }
}

const BALANCE_ERC20: &str = "5a28e9363bb942b639270062aa6bb295f434bcdfc42c97267bf003f272060dc9";
const BALANCE_INTERNAL: &str = "4ac99ace14ee0a5ef932dc609df0943ab7ac16b7583634612f8dc35a4289a6ce";
const BALANCE_EXTERNAL: &str = "abee3b73373acd583a130924aad6dc38cfdc44ba0555ba94ce2ff63980ea0632";

fn convert_sell_token_source(bytes: &[u8]) -> Result<SellTokenSource> {
    match hex::encode(bytes).as_str() {
        BALANCE_ERC20 => Ok(SellTokenSource::Erc20),
        BALANCE_INTERNAL => Ok(SellTokenSource::Internal),
        BALANCE_EXTERNAL => Ok(SellTokenSource::External),
        other => bail!("unknown sell token source: {other}"),
    }
}

fn convert_buy_token_destination(bytes: &[u8]) -> Result<BuyTokenDestination> {
    match hex::encode(bytes).as_str() {
        BALANCE_ERC20 => Ok(BuyTokenDestination::Erc20),
        BALANCE_INTERNAL => Ok(BuyTokenDestination::Internal),
        other => bail!("unknown buy token destination: {other}"),
    }
}

pub fn to_eth_address(address: Address) -> EthAddress {
    EthAddress::from_slice(address.as_slice())
}

pub fn to_address(address: EthAddress) -> Address {
    Address::from_slice(address.as_bytes())
}

pub fn to_eth_u256(value: U256) -> EthU256 {
    EthU256::from_big_endian(&value.to_be_bytes::<32>())
}

pub fn to_u256(value: EthU256) -> U256 {
    let mut bytes = [0u8; 32];
    value.to_big_endian(&mut bytes);
    U256::from_be_bytes(bytes)
}
