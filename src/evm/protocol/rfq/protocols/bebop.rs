use async_trait::async_trait;

use crate::{
    evm::protocol::rfq::{
        client::RFQClient, errors::RFQError, price_estimator::PriceEstimator, state::SignedQuote,
    },
    models::GetAmountOutParams,
    protocol::models::GetAmountOutResult,
};

#[derive(Clone, Debug)]
pub struct BebopIndicativePrice {
    pub base_token: String,
    pub quote_token: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

#[async_trait]
impl PriceEstimator for BebopIndicativePrice {
    fn base_token(&self) -> &String {
        &self.base_token
    }

    fn quote_token(&self) -> &String {
        &self.quote_token
    }

    fn get_amount_out(&self, params: GetAmountOutParams) -> Result<GetAmountOutResult, RFQError> {
        // use bids and asks to calculate the amount out
        todo!()
    }

    fn spot_price(&self) -> f64 {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn PriceEstimator> {
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
impl RFQClient for BebopClient {
    async fn next_price_update(&mut self) -> Result<Vec<Box<dyn PriceEstimator>>, RFQError> {
        // get price data from Bebop
        todo!()
    }

    async fn request_binding_quote(
        &self,
        params: &GetAmountOutParams,
    ) -> Result<SignedQuote, RFQError> {
        // get binding quote from Bebop
        // we need to set gasless=false
        // TODO: how are we going to handle approvals? https://docs.bebop.xyz/bebop/bebop-api-pmm-rfq/rfq-api-endpoints/trade/manage-approvals
        // example request:
        // curl -X 'GET' \
        // 'https://api.bebop.xyz/pmm/ethereum/v3/quote?sell_tokens=0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2&buy_tokens=0xdAC17F958D2ee523a2206206994597C13D831ec7&sell_amounts=1000000000000000&taker_address=0xabA2fC41e2dB95E77C6799D0F580034395FF2B9E&approval_type=Standard&skip_validation=true&skip_taker_checks=true&gasless=true&expiry_type=standard&fee=0&is_ui=false&gasless=false&origin_address=0x5206213Da4F6FE0E71d61cA00bB100dB2d6fe441' \
        // -H 'accept: application/json'
        todo!()
    }

    fn clone_box(&self) -> Box<dyn RFQClient> {
        Box::new(self.clone())
    }
}
