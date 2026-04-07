use std::{collections::HashMap, str::FromStr, time::Duration};

use alloy::{primitives::{Address, U256}, sol_types::SolCall};
use anyhow::{Context, Result, bail};
use ethcontract::web3;
use futures::StreamExt;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use tycho_simulation::{
    evm::{protocol::cowamm::state::CowAMMState, stream::ProtocolStreamBuilder},
    protocol::models::{ProtocolComponent, Update},
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_common::{
        Bytes,
        models::{Chain, ComponentId, token::Token},
        simulation::protocol_sim::ProtocolSim,
    },
    utils::{get_default_url, load_all_tokens},
};

mod support;

use support::{
    Amm, BCowHelperContract, DEFAULT_GAS, DEFAULT_SETTLEMENT_ADDRESS, GPv2AllowListAuthenticationContract,
    GPv2SettlementContract, SettleOutput, biguint_to_u256, bytes_to_address, bytes_to_eth_address,
    encode_settlement, encode_settlement_call, format_token_amount, prepare_state_overrides,
    resolve_token, select_pool_order, to_eth_address, u256_to_biguint,
};
use services_contracts::alloy::support::Solver;

const STREAM_PROTOCOL: &str = "cowamm";
const DEFAULT_HELPER_ADDRESS: &str = "0x03362f847b4fabc12e1ce98b6b59f94401e4588e";
const DEFAULT_TYCHO_URL: &str = "https://tycho-beta.propellerheads.xyz";
const DEFAULT_TARGET_POOL_ID: &str = "0x9d0e8cdf137976e03ef92ede4c30648d05e25285";
const DEFAULT_SELL_TOKEN: &str = "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0";
const DEFAULT_BUY_TOKEN: &str = "0xBAac2B4491727D78D2b78815144570b9f2Fe8899";
const DEFAULT_SELL_AMOUNT: &str = "10000000000000000";
const DEFAULT_WSTETH_BALANCE_SLOT: u64 = 0;
const SIMULATED_TRADER: &str = "0x00000000000000000000000000000000C0FfEe01";

#[derive(Clone, Debug)]
struct Config {
    chain: Chain,
    rpc_url: String,
    tycho_url: String,
    tycho_api_key: String,
    helper_address: Address,
    settlement_address: Option<Address>,
    sell_token: Address,
    buy_token: Address,
    sell_amount: U256,
    target_pool_id: String,
    solver_address: Address,
    trader_address: Address,
    sell_token_balance_slot: U256,
}

impl Config {
    fn from_env() -> Result<Self> {
        let chain = chain_from_id(parse_chain_id(&std::env::var("CHAIN_ID").unwrap_or_else(|_| "1".to_string()))?)?;
        Ok(Self {
            chain,
            rpc_url: std::env::var("RPC_URL").context("missing RPC_URL")?,
            tycho_url: std::env::var("TYCHO_URL")
                .unwrap_or_else(|_| get_default_url(&chain).unwrap_or_else(|| DEFAULT_TYCHO_URL.to_string())),
            tycho_api_key: std::env::var("TYCHO_API_KEY").context("missing TYCHO_API_KEY")?,
            helper_address: optional_address_env("HELPER_ADDRESS")?.unwrap_or(address(DEFAULT_HELPER_ADDRESS)?),
            settlement_address: optional_address_env("SETTLEMENT_ADDRESS")?,
            sell_token: optional_address_env("SELL_TOKEN")?.unwrap_or(address(DEFAULT_SELL_TOKEN)?),
            buy_token: optional_address_env("BUY_TOKEN")?.unwrap_or(address(DEFAULT_BUY_TOKEN)?),
            sell_amount: std::env::var("SELL_AMOUNT")
                .ok()
                .map(|value| U256::from_str(&value))
                .transpose()
                .context("invalid SELL_AMOUNT")?
                .unwrap_or(U256::from_str(DEFAULT_SELL_AMOUNT).expect("default sell amount")),
            target_pool_id: std::env::var("TARGET_POOL_ID").unwrap_or_else(|_| DEFAULT_TARGET_POOL_ID.to_string()),
            solver_address: address(&std::env::var("SOLVER_ADDRESS").context("missing SOLVER_ADDRESS")?)
                .context("invalid SOLVER_ADDRESS")?,
            trader_address: optional_address_env("TRADER_ADDRESS")?.unwrap_or(address(SIMULATED_TRADER)?),
            sell_token_balance_slot: std::env::var("SELL_TOKEN_BALANCE_SLOT")
                .ok()
                .map(|value| U256::from_str(&value))
                .transpose()
                .context("invalid SELL_TOKEN_BALANCE_SLOT")?
                .unwrap_or(U256::from(DEFAULT_WSTETH_BALANCE_SLOT)),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    run(Config::from_env()?).await
}

async fn run(config: Config) -> Result<()> {
    let rpc = ethrpc::Web3::new_from_url(&config.rpc_url);
    let transport = web3::transports::Http::new(&config.rpc_url).context("failed creating web3 transport")?;
    let web3 = web3::Web3::new(transport);
    let settlement = GPv2SettlementContract::at(
        &web3,
        to_eth_address(config.settlement_address.unwrap_or(address(DEFAULT_SETTLEMENT_ADDRESS)?)),
    );
    let settlement_address = support::to_address(settlement.address());
    let authenticator_address = support::to_address(
        settlement
            .authenticator()
            .call()
            .await
            .context("failed fetching authenticator address")?,
    );
    let is_solver = GPv2AllowListAuthenticationContract::at(&web3, to_eth_address(authenticator_address))
        .is_solver(to_eth_address(config.solver_address))
        .call()
        .await
        .context("failed checking solver allowlist")?;
    let all_tokens = load_all_tokens(
        &config.tycho_url,
        false,
        Some(config.tycho_api_key.as_str()),
        true,
        config.chain,
        Some(0),
        Some(365),
    )
    .await
    .context("failed loading Tycho token registry")?;
    let mut stream = build_stream(&config, &all_tokens).await?;
    let mut components = HashMap::new();

    info!(
        solver = ?config.solver_address,
        trader = ?config.trader_address,
        settlement = ?settlement_address,
        pool = %config.target_pool_id,
        authenticated_solver = is_solver,
        "starting cowamm simulation"
    );

    while let Some(message) = stream.next().await {
        let update = match message {
            Ok(update) => update,
            Err(err) => {
                warn!(error = ?err, "failed decoding stream update");
                continue;
            }
        };
        merge_components(&mut components, &update);
        let Some((pool_id, component, state)) = select_target_state(&config, &components, &update, &all_tokens)? else {
            continue;
        };
        return simulate_once(
            &config,
            &rpc,
            &web3,
            settlement_address,
            authenticator_address,
            is_solver,
            &all_tokens,
            &pool_id,
            &component,
            &state,
        )
        .await;
    }

    bail!("stream ended before producing a matching cowamm state")
}

async fn simulate_once(
    config: &Config,
    rpc: &ethrpc::Web3,
    web3: &web3::Web3<web3::transports::Http>,
    settlement_address: Address,
    authenticator_address: Address,
    is_solver: bool,
    all_tokens: &HashMap<Bytes, Token>,
    pool_id: &str,
    component: &ProtocolComponent,
    state: &CowAMMState,
) -> Result<()> {
    let sell_token = resolve_token(&Bytes::from(config.sell_token.as_slice()), component, all_tokens);
    let buy_token = resolve_token(&Bytes::from(config.buy_token.as_slice()), component, all_tokens);
    let simulated = state
        .get_amount_out(u256_to_biguint(config.sell_amount), &sell_token, &buy_token)
        .with_context(|| format!("local tycho simulation failed for pool {pool_id}"))?;

    let helper = BCowHelperContract::at(web3, to_eth_address(config.helper_address));
    let amm = Amm::new(bytes_to_eth_address(&state.address)?, &helper)
        .await
        .context("failed instantiating BCowHelper Amm")?;
    let pool_template = select_pool_order(
        &amm,
        config.sell_token,
        config.buy_token,
        config.sell_amount,
        &simulated.amount,
    )
    .await
    .context("failed building pool template order")?;

    if pool_template.order.buy_amount != config.sell_amount {
        bail!(
            "pool order buy amount {} does not match simulated sell amount {}",
            pool_template.order.buy_amount,
            config.sell_amount
        );
    }

    let settlement = encode_settlement(
        config.sell_token,
        config.buy_token,
        config.sell_amount,
        config.trader_address,
        config.solver_address,
        settlement_address,
        bytes_to_address(&state.address)?,
        &pool_template,
    );
    let overrides = prepare_state_overrides(
        authenticator_address,
        config.sell_token,
        config.sell_amount,
        config.sell_token_balance_slot,
        config.trader_address,
        config.solver_address,
        is_solver,
    );
    let output = Solver::Solver::new(config.solver_address, rpc.provider.clone())
        .swap(
            settlement_address,
            vec![config.sell_token, config.buy_token],
            config.trader_address,
            encode_settlement_call(settlement).abi_encode().into(),
        )
        .from(config.solver_address)
        .to(config.solver_address)
        .gas(DEFAULT_GAS)
        .call()
        .overrides(overrides)
        .await
        .context("eth_call simulation failed")?;

    log_result(
        config,
        &sell_token,
        &buy_token,
        &pool_template,
        simulated.amount,
        SettleOutput::from_swap(output, model::order::OrderKind::Sell)?,
        pool_id,
    );
    Ok(())
}

fn log_result(
    config: &Config,
    sell_token: &Token,
    buy_token: &Token,
    pool_template: &support::TemplateOrder,
    tycho_amount: num_bigint::BigUint,
    summary: SettleOutput,
    pool_id: &str,
) {
    let tycho_amount = biguint_to_u256(tycho_amount);
    let abs_diff = if tycho_amount >= summary.out_amount {
        tycho_amount - summary.out_amount
    } else {
        summary.out_amount - tycho_amount
    };
    let diff_bps = if tycho_amount.is_zero() { U256::ZERO } else { abs_diff * U256::from(10_000u64) / tycho_amount };
    info!(
        pool = %pool_id,
        sell_token = %sell_token.symbol,
        buy_token = %buy_token.symbol,
        sell_amount = %config.sell_amount,
        sell_amount_display = %format_token_amount(config.sell_amount, sell_token.decimals as u8),
        tycho_buy_amount = %tycho_amount,
        tycho_buy_display = %format_token_amount(tycho_amount, buy_token.decimals as u8),
        helper_pool_sell_amount = %pool_template.order.sell_amount,
        helper_pool_sell_display = %format_token_amount(pool_template.order.sell_amount, buy_token.decimals as u8),
        eth_call_buy_amount = %summary.out_amount,
        eth_call_buy_display = %format_token_amount(summary.out_amount, buy_token.decimals as u8),
        gas_used = %summary.gas_used,
        abs_diff = %abs_diff,
        diff_bps = %diff_bps,
        exact_match = abs_diff.is_zero(),
        "simulation comparison complete"
    );
}

async fn build_stream(
    config: &Config,
    all_tokens: &HashMap<Bytes, Token>,
) -> Result<impl futures::Stream<Item = Result<Update, tycho_simulation::evm::decoder::StreamDecodeError>>> {
    let filter = ComponentFilter::Ids(vec![ComponentId::from_str(&config.target_pool_id)
        .with_context(|| format!("invalid TARGET_POOL_ID: {}", config.target_pool_id))?]);
    ProtocolStreamBuilder::new(&config.tycho_url, config.chain)
        .exchange::<CowAMMState>(STREAM_PROTOCOL, filter, None)
        .auth_key(Some(config.tycho_api_key.clone()))
        .skip_state_decode_failures(true)
        .startup_timeout(Duration::from_secs(120))
        .set_tokens(all_tokens.clone())
        .await
        .build()
        .await
        .context("failed building cowamm stream")
}

fn merge_components(components: &mut HashMap<String, ProtocolComponent>, update: &Update) {
    for (id, component) in &update.new_pairs {
        components.insert(id.clone(), component.clone());
    }
    for id in update.removed_pairs.keys() {
        components.remove(id);
    }
}

fn select_target_state(
    config: &Config,
    components: &HashMap<String, ProtocolComponent>,
    update: &Update,
    all_tokens: &HashMap<Bytes, Token>,
) -> Result<Option<(String, ProtocolComponent, CowAMMState)>> {
    for (id, state) in &update.states {
        if id != &config.target_pool_id {
            continue;
        }
        let Some(component) = components.get(id) else { continue; };
        let Some(state) = state.as_any().downcast_ref::<CowAMMState>() else { continue; };
        let sell = resolve_token(&Bytes::from(config.sell_token.as_slice()), component, all_tokens);
        let buy = resolve_token(&Bytes::from(config.buy_token.as_slice()), component, all_tokens);
        if component.tokens.iter().any(|token| token.address == sell.address)
            && component.tokens.iter().any(|token| token.address == buy.address)
        {
            return Ok(Some((id.clone(), component.clone(), state.clone())));
        }
    }
    Ok(None)
}

fn parse_chain_id(value: &str) -> Result<u64> {
    match value {
        "mainnet" | "ethereum" => Ok(1),
        other => other.parse().with_context(|| format!("unsupported CHAIN_ID: {other}")),
    }
}

fn chain_from_id(chain_id: u64) -> Result<Chain> {
    match chain_id {
        1 => Ok(Chain::Ethereum),
        8453 => Ok(Chain::Base),
        other => bail!("unsupported chain id {other}"),
    }
}

fn optional_address_env(name: &str) -> Result<Option<Address>> {
    std::env::var(name).ok().map(|value| address(&value)).transpose()
}

fn address(value: &str) -> Result<Address> {
    Address::from_str(value).with_context(|| format!("invalid address: {value}"))
}
