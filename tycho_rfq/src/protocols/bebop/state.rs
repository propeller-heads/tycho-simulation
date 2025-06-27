use std::{
    any::Any,
    collections::HashMap,
    fmt::{Debug, Formatter},
};

use num_bigint::BigUint;
use tycho_simulation::{
    models::{Balances, GetAmountOutParams, Token},
    protocol::{
        errors::{SimulationError, TransitionError},
        models::GetAmountOutResult,
        state::ProtocolSim,
    },
    tycho_common::{Bytes, dto::ProtocolStateDelta},
};

use crate::{
    client::RFQClient,
    indicatively_priced::{IndicativelyPriced, SignedQuote},
};

pub struct BebopState {
    pub quote_provider: Box<dyn RFQClient>, // needed to get binding quote
    pub maker: String,
    pub base_token: String,
    pub quote_token: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

impl Debug for BebopState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl ProtocolSim for BebopState {
    fn fee(&self) -> f64 {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        todo!()
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        todo!()
    }

    fn get_limits(
        &self,
        sell_token: Bytes,
        buy_token: Bytes,
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

impl IndicativelyPriced for BebopState {
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
