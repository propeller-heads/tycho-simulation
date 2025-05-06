use async_trait::async_trait;

use crate::{
    evm::protocol::rfq::{
        errors::RFQError, indicative_price::IndicativePrice, state::BindingQuote,
    },
    models::GetAmountOutParams,
};

#[async_trait]
pub trait RFQClientSource: Send + Sync {
    // This method is responsible for fetching data from the RFQ API and converting into an
    // IndicativePrice
    async fn next_price_update(&mut self) -> Result<Vec<Box<dyn IndicativePrice>>, RFQError>;

    // This method is responsible for fetching the binding quote from the RFQ API
    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<BindingQuote, RFQError>;

    fn clone_box(&self) -> Box<dyn RFQClientSource>;
}
