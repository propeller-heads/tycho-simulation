use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
};

use alloy_primitives::Address;
use num_bigint::BigUint;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::protocol::rfq::{client::RFQClient, price_estimator::PriceEstimator},
    models::{Balances, GetAmountOutParams, Token},
    protocol::{
        errors::{SimulationError, TransitionError},
        models::GetAmountOutResult,
        state::ProtocolSim,
    },
};

pub struct RFQState {
    pub price_data: Box<dyn PriceEstimator>,
    pub quote_provider: Box<dyn RFQClient>, // needed to get binding quote
}

impl RFQState {
    pub fn new(price_data: Box<dyn PriceEstimator>, quote_provider: Box<dyn RFQClient>) -> Self {
        RFQState { price_data, quote_provider }
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

pub struct SignedQuote {
    pub base_token: Address,
    pub quote_token: Address,
    pub amount_in: BigUint,
    pub amount_out: BigUint,
    // each RFQ will need different attributes
    pub quote_attributes: HashMap<String, Bytes>,
}

pub trait IndicativelyPriced: ProtocolSim {
    // this will be true when the price is only an estimation/indicative price
    fn is_indicatively_priced() -> bool {
        false
    }

    // if it is indicatively priced, then we need to request a signed quote for the final price
    async fn request_signed_quote(
        &self,
        params: GetAmountOutParams,
    ) -> Result<SignedQuote, SimulationError> {
        Err(SimulationError::NotImplemented)
    }
}

impl IndicativelyPriced for RFQState {
    fn is_indicatively_priced() -> bool {
        true
    }

    async fn request_signed_quote(
        &self,
        params: GetAmountOutParams,
    ) -> Result<SignedQuote, SimulationError> {
        let binding_quote = self
            .quote_provider
            .request_binding_quote(&params)
            .await?;
        Ok(binding_quote)
    }
}
