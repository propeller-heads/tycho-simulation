use std::env;

use alloy::{hex, primitives::Address, rpc::types::transaction::TransactionRequest};
use alloy_rpc_types_eth::state::StateOverride;
use anyhow::{bail, Context, Result};
use serde_json::{Map as JsonMap, Value as JsonValue};
use tracing::info;

#[derive(Clone, Debug, Default)]
pub struct TenderlyConfig {
    pub account: Option<String>,
    pub project: Option<String>,
    pub access_key: Option<String>,
    pub submit_simulation: bool,
}

impl TenderlyConfig {
    pub fn from_env() -> Self {
        Self {
            account: env::var("TENDERLY_ACCOUNT").ok(),
            project: env::var("TENDERLY_PROJECT").ok(),
            access_key: env::var("TENDERLY_ACCESS_KEY").ok(),
            submit_simulation: env::var("TENDERLY_SUBMIT_SIMULATION")
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false),
        }
    }

    pub async fn maybe_submit_simulation(
        &self,
        tx: &TransactionRequest,
        caller: Address,
        block_number: u64,
        overrides: &StateOverride,
    ) -> Result<()> {
        let Some(account) = &self.account else {
            return Ok(());
        };
        let Some(project) = &self.project else {
            return Ok(());
        };
        let Some(access_key) = &self.access_key else {
            return Ok(());
        };
        if !self.submit_simulation {
            return Ok(());
        }

        let payload = build_api_payload(tx, caller, block_number, overrides);
        let api_url = build_api_url(account, project);
        let response = submit_simulation(&api_url, access_key, &payload)
            .await
            .context("tenderly simulation api request failed")?;
        if let Some(simulation_url) = build_result_url(account, project, &response) {
            info!(tenderly_simulation_url = %simulation_url, "tenderly simulation result url");
        }

        Ok(())
    }
}

fn build_api_url(account: &str, project: &str) -> String {
    format!("https://api.tenderly.co/api/v1/account/{}/project/{}/simulate", account, project)
}

fn build_result_url(account: &str, project: &str, response: &JsonValue) -> Option<String> {
    let simulation_id = response
        .get("simulation")
        .and_then(|simulation| simulation.get("id"))
        .and_then(JsonValue::as_str)
        .or_else(|| {
            response
                .get("id")
                .and_then(JsonValue::as_str)
        })?;
    Some(format!(
        "https://dashboard.tenderly.co/{}/{}/simulator/{}",
        account, project, simulation_id
    ))
}

fn build_api_payload(
    tx: &TransactionRequest,
    caller: Address,
    block_number: u64,
    overrides: &StateOverride,
) -> JsonValue {
    let contract_address = tx
        .to
        .as_ref()
        .and_then(|to| to.to().copied())
        .map(|to| format!("0x{:x}", to))
        .unwrap_or_default();
    let input = tx
        .input
        .input()
        .map(|data| format!("0x{}", hex::encode(data)))
        .unwrap_or_default();

    let mut state_objects = JsonMap::new();
    for (address, account_override) in overrides {
        let mut object = JsonMap::new();

        if let Some(balance) = account_override.balance {
            object.insert("balance".to_string(), JsonValue::String(format!("{balance:#x}")));
        }
        if let Some(nonce) = account_override.nonce {
            object.insert("nonce".to_string(), JsonValue::String(format!("0x{nonce:x}")));
        }
        if let Some(code) = &account_override.code {
            object
                .insert("code".to_string(), JsonValue::String(format!("0x{}", hex::encode(code))));
        }

        let mut storage = JsonMap::new();
        if let Some(state) = &account_override.state {
            for (slot, value) in state {
                storage.insert(format!("{slot:#x}"), JsonValue::String(format!("{value:#x}")));
            }
        }
        if let Some(state_diff) = &account_override.state_diff {
            for (slot, value) in state_diff {
                storage.insert(format!("{slot:#x}"), JsonValue::String(format!("{value:#x}")));
            }
        }
        if !storage.is_empty() {
            object.insert("storage".to_string(), JsonValue::Object(storage));
        }

        if !object.is_empty() {
            state_objects.insert(format!("0x{:x}", address), JsonValue::Object(object));
        }
    }

    let mut payload = JsonMap::new();
    payload.insert("save".to_string(), JsonValue::Bool(true));
    payload.insert("save_if_fails".to_string(), JsonValue::Bool(true));
    payload.insert("simulation_type".to_string(), JsonValue::String("full".to_string()));
    payload.insert("network_id".to_string(), JsonValue::String("1".to_string()));
    payload.insert(
        "from".to_string(),
        JsonValue::String(format!("0x{:x}", tx.from.unwrap_or(caller))),
    );
    payload.insert("to".to_string(), JsonValue::String(contract_address));
    payload.insert("input".to_string(), JsonValue::String(input));
    payload.insert(
        "value".to_string(),
        JsonValue::String(format!("{:#x}", tx.value.unwrap_or_default())),
    );
    payload.insert("block_number".to_string(), JsonValue::Number(block_number.into()));
    payload.insert("state_objects".to_string(), JsonValue::Object(state_objects));
    if let Some(gas) = tx.gas {
        payload.insert("gas".to_string(), JsonValue::Number(gas.into()));
    }
    if let Some(gas_price) = tx.gas_price {
        payload.insert("gas_price".to_string(), JsonValue::String(format!("{:#x}", gas_price)));
    }

    JsonValue::Object(payload)
}

async fn submit_simulation(
    tenderly_api_url: &str,
    tenderly_access_key: &str,
    payload: &JsonValue,
) -> Result<JsonValue> {
    let client = reqwest::Client::new();
    let response = client
        .post(tenderly_api_url)
        .header("X-Access-Key", tenderly_access_key)
        .json(payload)
        .send()
        .await
        .context("failed sending Tenderly simulation request")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed reading Tenderly simulation response body")?;
    if !status.is_success() {
        bail!("Tenderly simulation API returned {status}: {body}");
    }
    serde_json::from_str(&body).context("failed parsing Tenderly simulation response JSON")
}
