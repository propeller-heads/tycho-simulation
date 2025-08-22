use std::{collections::HashMap, str::FromStr};

use miette::{miette, IntoDiagnostic, Result, WrapErr};
use num_bigint::BigUint;
use tycho_common::{
    models::{token::Token, Chain},
    Bytes,
};
use tycho_simulation::utils::load_all_tokens;

use crate::{
    cli::SwapArgs,
    env::{get_env, get_env_with_default},
};

pub struct TychoClient {
    url: String,
    api_key: String,
    chain: Chain,
}

impl TychoClient {
    pub fn new(url: String, api_key: String, chain: Chain) -> Self {
        Self { url, api_key, chain }
    }

    pub fn from_env(chain: Chain) -> Result<Self> {
        let default_url = Self::get_default_url(&chain)
            .ok_or_else(|| miette!("No default URL for chain {chain:?}"))?;
        Ok(Self::new(
            get_env_with_default("TYCHO_URL", default_url),
            get_env("TYCHO_API_KEY").ok_or(miette!(
                "The environment variable `TYCHO_API_KEY` is not set. Please set it to your Tycho API key."
            ))?,
            chain
        ))
    }

    /// Get the default Tycho URL for the given chain.
    fn get_default_url(chain: &Chain) -> Option<String> {
        match chain {
            Chain::Ethereum => Some("tycho-beta.propellerheads.xyz".to_string()),
            Chain::Base => Some("tycho-base-beta.propellerheads.xyz".to_string()),
            Chain::Unichain => Some("tycho-unichain-beta.propellerheads.xyz".to_string()),
            _ => None,
        }
    }

    pub async fn load_tokens(&self) -> HashMap<Bytes, Token> {
        println!("Loading tokens from Tycho... {}", self.url);
        let tokens =
            load_all_tokens(&self.url, false, Some(&self.api_key), self.chain, None, None).await;
        println!("Tokens loaded: {}", tokens.len());
        tokens
    }

    pub fn get_token_info(
        &self,
        swap_args: &SwapArgs,
        all_tokens: &HashMap<Bytes, Token>,
    ) -> Result<(Token, Token, BigUint)> {
        let sell_token_address = Bytes::from_str(
            swap_args
                .sell_token
                .as_ref()
                .ok_or_else(|| miette!("Sell token not provided"))?,
        )
        .into_diagnostic()
        .wrap_err("Invalid address for sell token")?;
        let buy_token_address = Bytes::from_str(
            swap_args
                .buy_token
                .as_ref()
                .ok_or_else(|| miette!("Buy token not provided"))?,
        )
        .into_diagnostic()
        .wrap_err("Invalid address for buy token")?;
        let sell_token = all_tokens
            .get(&sell_token_address)
            .ok_or_else(|| miette!("Sell token not found"))?
            .clone();
        let buy_token = all_tokens
            .get(&buy_token_address)
            .ok_or_else(|| miette!("Buy token not found"))?
            .clone();
        let amount_in =
            BigUint::from((swap_args.sell_amount * 10f64.powi(sell_token.decimals as i32)) as u128);

        println!(
            "Looking for RFQ quotes for {amount} {sell_symbol} -> {buy_symbol} on {chain:?}",
            amount = swap_args.sell_amount,
            sell_symbol = sell_token.symbol,
            buy_symbol = buy_token.symbol,
            chain = swap_args.chain
        );
        Ok((sell_token, buy_token, amount_in))
    }
}
