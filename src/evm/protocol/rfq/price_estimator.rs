use crate::{
    evm::protocol::rfq::errors::RFQError, models::GetAmountOutParams,
    protocol::models::GetAmountOutResult,
};

/// This trait defines the interface for a price estimator
/// It might hold price levels, order books, or any other data needed by the RFQ to compute the
/// current price.
pub trait PriceEstimator: Send + Sync {
    fn base_token(&self) -> &String;
    fn quote_token(&self) -> &String;
    fn get_amount_out(&self, params: GetAmountOutParams) -> Result<GetAmountOutResult, RFQError>;

    fn spot_price(&self) -> f64;

    fn clone_box(&self) -> Box<dyn PriceEstimator>;
}
