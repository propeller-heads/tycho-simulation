use async_trait::async_trait;

use crate::{
    evm::protocol::rfq::{indicative_price::IndicativePrice, state::BindingQuote},
    models::GetAmountOutParams,
    protocol::errors::SimulationError,
};

#[async_trait]
pub trait RFQClientSource: Send + Sync {
    async fn next_price_update(&mut self)
        -> Result<Vec<Box<dyn IndicativePrice>>, SimulationError>;

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<BindingQuote, SimulationError>;

    fn clone_box(&self) -> Box<dyn RFQClientSource>;
}
