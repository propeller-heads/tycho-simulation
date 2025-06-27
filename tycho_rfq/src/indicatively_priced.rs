use std::collections::HashMap;

use alloy::primitives::Address;
use num_bigint::BigUint;
use tycho_simulation::{
    models::GetAmountOutParams,
    protocol::{errors::SimulationError, state::ProtocolSim},
    tycho_common::Bytes,
};

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
