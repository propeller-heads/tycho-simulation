use async_trait::async_trait;

use crate::{
    evm::protocol::rfq::{
        client::RFQClientSource, errors::RFQError, indicative_price::IndicativePrice,
        state::BindingQuote,
    },
    models::{GetAmountOutParams, Token},
    protocol::models::GetAmountOutResult,
};

#[derive(Clone, Debug)]
pub struct BebopIndicativePrice {
    pub base_token: Token,
    pub quote_token: Token,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

#[async_trait]
impl IndicativePrice for BebopIndicativePrice {
    fn base_token(&self) -> &Token {
        &self.base_token
    }

    fn quote_token(&self) -> &Token {
        &self.quote_token
    }

    fn get_amount_out(&self, params: GetAmountOutParams) -> Result<GetAmountOutResult, RFQError> {
        // use bids and asks to calculate the amount out
        todo!()
    }

    fn spot_price(&self) -> f64 {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn IndicativePrice> {
        Box::new(self.clone())
    }
}
#[derive(Clone)]
pub struct BebopClient {
    url: String,
}

impl BebopClient {
    pub fn new() -> Self {
        let url = "wss://api.bebop.com/v1/price".to_string();
        Self { url }
    }
}

#[async_trait]
impl RFQClientSource for BebopClient {
    async fn next_price_update(&mut self) -> Result<Vec<Box<dyn IndicativePrice>>, RFQError> {
        // get price data from Bebop
        todo!()
    }

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<BindingQuote, RFQError> {
        // get binding quote from Bebop
        todo!()
    }

    fn clone_box(&self) -> Box<dyn RFQClientSource> {
        Box::new(self.clone())
    }
}
