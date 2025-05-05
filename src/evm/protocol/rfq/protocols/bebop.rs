use async_trait::async_trait;

use crate::{
    evm::protocol::rfq::{
        client::{IndicativePriceSource, RFQClientSource, WebSocketPriceStream},
        indicative_price::IndicativePrice,
        state::BindingQuote,
    },
    models::{GetAmountOutParams, Token},
    protocol::{errors::SimulationError, models::GetAmountOutResult},
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

    fn get_amount_out(
        &self,
        params: GetAmountOutParams,
    ) -> Result<GetAmountOutResult, SimulationError> {
        // use bids and asks to calculate the amount out
        todo!()
    }

    fn clone_box(&self) -> Box<dyn IndicativePrice> {
        Box::new(self.clone())
    }
}

pub struct BebopClient {
    // Fields for HTTP client, API keys, etc.
    price_source: WebSocketPriceStream<BebopIndicativePrice>,
}

impl BebopClient {
    pub fn new(price_source: WebSocketPriceStream<BebopIndicativePrice>) -> Self {
        // TODO: figure out how to make a stream from the websocket
        // let price_source = WebSocketPriceStream::new("wss://api.bebop.com/v1/price");
        Self { price_source }
    }
}

#[async_trait]
impl RFQClientSource for BebopClient {
    async fn next_price_update(
        &mut self,
    ) -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError> {
        self.price_source
            .next_price_update()
            .await
    }

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<BindingQuote, SimulationError> {
        // gte binding quote from Bebop
        todo!()
    }

    fn clone_box(&self) -> Box<dyn RFQClientSource> {
        todo!()
        //Box::new(self.clone())
        // this is problematic because the BebopClient is not cloneable at the moment
    }
}
