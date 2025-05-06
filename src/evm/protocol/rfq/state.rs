use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
};

use alloy_primitives::Address;
use num_bigint::BigUint;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::protocol::rfq::{client::RFQClientSource, indicative_price::IndicativePrice},
    models::{Balances, GetAmountOutParams, Token},
    protocol::{
        errors::{SimulationError, TransitionError},
        models::GetAmountOutResult,
        state::ProtocolSim,
    },
};

pub struct RFQState {
    pub base_token: Token,
    pub quote_token: Token,
    pub price_data: Box<dyn IndicativePrice>,
    pub quote_provider: Box<dyn RFQClientSource>, // needed to get binding quote
}

impl RFQState {
    pub fn new(
        base_token: Token,
        quote_token: Token,
        price_data: Box<dyn IndicativePrice>,
        quote_provider: Box<dyn RFQClientSource>,
    ) -> Self {
        RFQState { base_token, quote_token, price_data, quote_provider }
    }
}

impl Debug for RFQState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl ProtocolSim for RFQState {
    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        Ok(self
            .price_data
            .get_amount_out(GetAmountOutParams {
                amount_in,
                token_in: token_in.clone(),
                token_out: token_out.clone(),
            })?)
    }

    fn fee(&self) -> f64 {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        todo!()
    }

    fn get_limits(
        &self,
        sell_token: Address,
        buy_token: Address,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        todo!()
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        tokens: &HashMap<Bytes, Token>,
        balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn ProtocolSim> {
        todo!()
    }

    fn as_any(&self) -> &dyn Any {
        todo!()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        todo!()
    }

    fn eq(&self, other: &dyn ProtocolSim) -> bool {
        todo!()
    }
}

pub struct BindingQuote {
    pub price: f64,
    pub base_amount: f64,
    pub quote_amount: f64,
    pub signature: String,
}

// I don't like this name
pub trait IndicativelyPriced: ProtocolSim {
    fn is_indicatively_priced() -> bool {
        false
    }

    async fn request_binding_quote(
        &self,
        params: GetAmountOutParams,
    ) -> Result<BindingQuote, SimulationError> {
        Err(SimulationError::NotImplemented)
    }
}

impl IndicativelyPriced for RFQState {
    fn is_indicatively_priced() -> bool {
        true
    }

    async fn request_binding_quote(
        &self,
        params: GetAmountOutParams,
    ) -> Result<BindingQuote, SimulationError> {
        let binding_quote = self
            .quote_provider
            .request_binding_quote(&params)
            .await?;
        Ok(binding_quote)
    }
}
