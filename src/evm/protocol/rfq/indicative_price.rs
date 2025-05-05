use crate::{
    models::{GetAmountOutParams, Token},
    protocol::{errors::SimulationError, models::GetAmountOutResult},
};

pub trait IndicativePrice: Send + Sync {
    fn base_token(&self) -> &Token;
    fn quote_token(&self) -> &Token;
    fn get_amount_out(
        &self,
        params: GetAmountOutParams,
    ) -> Result<GetAmountOutResult, SimulationError>;

    fn clone_box(&self) -> Box<dyn IndicativePrice>;
}
