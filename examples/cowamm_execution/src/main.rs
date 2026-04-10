use std::{collections::HashMap, env, str::FromStr, time::Duration};

use alloy::{
    primitives::{Address, U256},
    sol_types::SolCall,
};
use anyhow::{bail, Context, Result};
use clap::Parser;
use ethcontract::web3;
use futures::StreamExt;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use tycho_simulation::{
    evm::{protocol::cowamm::state::CowAMMState, stream::ProtocolStreamBuilder},
    protocol::models::{ProtocolComponent, Update},
    tycho_client::feed::component_tracker::ComponentFilter,
    tycho_common::{
        models::{token::Token, Chain, ComponentId},
        simulation::protocol_sim::ProtocolSim,
        Bytes,
    },
    utils::{get_default_url, load_all_tokens},
};

mod contracts;
mod settlement;
mod tenderly;

use contracts::{BCowHelperContract, GPv2AllowListAuthenticationContract, GPv2SettlementContract};
use services_contracts::alloy::support::Solver;
use settlement::{
    biguint_to_u256, bytes_to_address, bytes_to_eth_address, encode_settlement_call,
    format_token_amount, prepare_state_overrides, resolve_token, select_pool_order, to_eth_address,
    u256_to_biguint, Amm, SettleOutput, DEFAULT_GAS, DEFAULT_SETTLEMENT_ADDRESS,
};
use tenderly::TenderlyConfig;

const STREAM_PROTOCOL: &str = "cowamm";
const DEFAULT_HELPER_ADDRESS: &str = "0x03362f847b4fabc12e1ce98b6b59f94401e4588e";
const DEFAULT_TARGET_POOL_ID: &str = "0x9d0e8cdf137976e03ef92ede4c30648d05e25285";
const DEFAULT_SELL_TOKEN: &str = "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0";
const DEFAULT_BUY_TOKEN: &str = "0xBAac2B4491727D78D2b78815144570b9f2Fe8899";
const DEFAULT_SELL_AMOUNT: &str = "100000000000000000";
const DEFAULT_WSTETH_BALANCE_SLOT: u64 = 0;
const SIMULATED_TRADER: &str = "0x00000000000000000000000000000000C0FfEe01";

struct ExampleDefaults {
    target_pool: &'static str,
    sell_token: &'static str,
    buy_token: &'static str,
    sell_amount: &'static str,
    sell_token_balance_slot: u64,
}

const EXAMPLE_DEFAULTS: ExampleDefaults = ExampleDefaults {
    target_pool: DEFAULT_TARGET_POOL_ID,
    sell_token: DEFAULT_SELL_TOKEN,
    buy_token: DEFAULT_BUY_TOKEN,
    sell_amount: DEFAULT_SELL_AMOUNT,
    sell_token_balance_slot: DEFAULT_WSTETH_BALANCE_SLOT,
};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    sell_token: Option<String>,
    #[arg(long)]
    buy_token: Option<String>,
    #[arg(long)]
    target_pool: Option<String>,
    #[arg(long, default_value = EXAMPLE_DEFAULTS.sell_amount)]
    sell_amount: String,
    #[arg(long, default_value_t = EXAMPLE_DEFAULTS.sell_token_balance_slot)]
    sell_token_balance_slot: u64,
    #[arg(long)]
    trader_address: Option<String>,
}

impl Cli {
    fn with_defaults(mut self) -> Self {
        if self.sell_token.is_none() {
            self.sell_token = Some(EXAMPLE_DEFAULTS.sell_token.to_string());
        }
        if self.buy_token.is_none() {
            self.buy_token = Some(EXAMPLE_DEFAULTS.buy_token.to_string());
        }
        if self.target_pool.is_none() {
            self.target_pool = Some(EXAMPLE_DEFAULTS.target_pool.to_string());
        }
        self
    }
}

#[derive(Clone, Debug)]
struct Config {
    rpc_url: String,
    tycho_url: String,
    tycho_api_key: String,
    tenderly: TenderlyConfig,
    sell_token: Address,
    buy_token: Address,
    sell_amount: U256,
    target_pool: String,
    solver_address: Address,
    trader_address: Address,
    sell_token_balance_slot: U256,
}

impl Config {
    fn from_cli(cli: Cli) -> Result<Self> {
        Ok(Self {
            rpc_url: env::var("RPC_URL").context("missing RPC_URL")?,
            tycho_url: env::var("TYCHO_URL").unwrap_or_else(|_| {
                get_default_url(&Chain::Ethereum).expect("missing default Tycho URL for Ethereum")
            }),
            tycho_api_key: env::var("TYCHO_API_KEY").context("missing TYCHO_API_KEY")?,
            tenderly: TenderlyConfig::from_env(),
            sell_token: address(
                &cli.sell_token
                    .expect("sell token defaulted"),
            )?,
            buy_token: address(
                &cli.buy_token
                    .expect("buy token defaulted"),
            )?,
            sell_amount: U256::from_str(&cli.sell_amount).context("invalid --sell-amount")?,
            target_pool: cli
                .target_pool
                .expect("target pool defaulted"),
            solver_address: address(
                &env::var("SOLVER_ADDRESS").context("missing SOLVER_ADDRESS")?,
            )?,
            trader_address: cli
                .trader_address
                .as_deref()
                .map(address)
                .transpose()?
                .unwrap_or(address(SIMULATED_TRADER)?),
            sell_token_balance_slot: U256::from(cli.sell_token_balance_slot),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    run(Config::from_cli(Cli::parse().with_defaults())?).await
}

async fn run(config: Config) -> Result<()> {
    let rpc = ethrpc::Web3::new_from_url(&config.rpc_url);
    let transport =
        web3::transports::Http::new(&config.rpc_url).context("failed creating web3 transport")?;
    let web3 = web3::Web3::new(transport);
    let settlement =
        GPv2SettlementContract::at(&web3, to_eth_address(address(DEFAULT_SETTLEMENT_ADDRESS)?));
    let settlement_address = settlement::to_address(settlement.address());
    let authenticator_address = settlement::to_address(
        settlement
            .authenticator()
            .call()
            .await
            .context("failed fetching authenticator address")?,
    );
    let is_solver =
        GPv2AllowListAuthenticationContract::at(&web3, to_eth_address(authenticator_address))
            .is_solver(to_eth_address(config.solver_address))
            .call()
            .await
            .context("failed checking solver allowlist")?;

    let all_tokens = load_all_tokens(
        &config.tycho_url,
        false,
        Some(config.tycho_api_key.as_str()),
        true,
        Chain::Ethereum,
        Some(0),
        Some(365),
    )
    .await
    .context("failed loading Tycho token registry")?;
    info!(count = all_tokens.len(), "tokens loaded from Tycho");

    let mut stream = build_stream(&config, &all_tokens).await?;
    let mut components = HashMap::new();

    info!(
        solver = ?config.solver_address,
        trader = ?config.trader_address,
        settlement = ?settlement_address,
        pool = %config.target_pool,
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
        let Some((pool_id, component, state)) =
            select_target_state(&config, &components, &update, &all_tokens)?
        else {
            continue;
        };
        return simulate_once(
            &config,
            &rpc,
            &web3,
            update.block_number_or_timestamp,
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

#[allow(clippy::too_many_arguments)]
async fn simulate_once(
    config: &Config,
    rpc: &ethrpc::Web3,
    web3: &web3::Web3<web3::transports::Http>,
    block_number: u64,
    settlement_address: Address,
    authenticator_address: Address,
    is_solver: bool,
    all_tokens: &HashMap<Bytes, Token>,
    pool_id: &str,
    component: &ProtocolComponent,
    state: &CowAMMState,
) -> Result<()> {
    let sell_token =
        resolve_token(&Bytes::from(config.sell_token.as_slice()), component, all_tokens);
    let buy_token = resolve_token(&Bytes::from(config.buy_token.as_slice()), component, all_tokens);
    let simulated = state
        .get_amount_out(u256_to_biguint(config.sell_amount), &sell_token, &buy_token)
        .with_context(|| format!("local tycho simulation failed for pool {pool_id}"))?;

    let helper = BCowHelperContract::at(web3, to_eth_address(address(DEFAULT_HELPER_ADDRESS)?));
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

    let settlement = settlement::encode_settlement(
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
    let solver = Solver::Solver::new(config.solver_address, rpc.provider.clone());
    let swap_call = solver.swap(
        settlement_address,
        vec![config.sell_token, config.buy_token],
        config.trader_address,
        encode_settlement_call(settlement)
            .abi_encode()
            .into(),
    );
    let swap_call = swap_call
        .from(config.solver_address)
        .to(config.solver_address)
        .gas(DEFAULT_GAS);
    let trace_tx = swap_call
        .clone()
        .into_transaction_request();
    let output = swap_call
        .call()
        .overrides(overrides.clone())
        .await
        .context("eth_call simulation failed")?;
    config
        .tenderly
        .maybe_submit_simulation(&trace_tx, config.solver_address, block_number + 1, &overrides)
        .await?;

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
    pool_template: &settlement::TemplateOrder,
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
    let diff_bps = if tycho_amount.is_zero() {
        U256::ZERO
    } else {
        abs_diff * U256::from(10_000u64) / tycho_amount
    };
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
) -> Result<
    impl futures::Stream<Item = Result<Update, tycho_simulation::evm::decoder::StreamDecodeError>>,
> {
    let filter = ComponentFilter::Ids(vec![ComponentId::from_str(&config.target_pool)
        .with_context(|| format!("invalid --target-pool: {}", config.target_pool))?]);
    ProtocolStreamBuilder::new(&config.tycho_url, Chain::Ethereum)
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
        if id != &config.target_pool {
            continue;
        }
        let Some(component) = components.get(id) else {
            continue;
        };
        let Some(state) = state
            .as_any()
            .downcast_ref::<CowAMMState>()
        else {
            continue;
        };
        let sell = resolve_token(&Bytes::from(config.sell_token.as_slice()), component, all_tokens);
        let buy = resolve_token(&Bytes::from(config.buy_token.as_slice()), component, all_tokens);
        if component
            .tokens
            .iter()
            .any(|token| token.address == sell.address) &&
            component
                .tokens
                .iter()
                .any(|token| token.address == buy.address)
        {
            return Ok(Some((id.clone(), component.clone(), state.clone())));
        }
    }
    Ok(None)
}

fn address(value: &str) -> Result<Address> {
    Address::from_str(value).with_context(|| format!("invalid address: {value}"))
}
