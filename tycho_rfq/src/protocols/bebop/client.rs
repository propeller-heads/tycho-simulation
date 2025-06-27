use std::collections::HashMap;

use async_trait::async_trait;
use futures::stream::BoxStream;
use tycho_simulation::{
    models::GetAmountOutParams,
    tycho_client::feed::synchronizer::{ComponentWithState, Snapshot, StateSyncMessage},
    tycho_common::dto::{ProtocolComponent, ResponseProtocolState},
    tycho_core::dto::Chain,
};

use crate::{client::RFQClient, errors::RFQError, indicatively_priced::SignedQuote};

#[derive(Clone)]
pub struct BebopClient {
    url: String,
    // keep track of the current token pairs supported
    pairs: Vec<(String, String)>,
    chain: Chain,
}

impl BebopClient {
    pub fn new(chain: Chain) -> Self {
        let url = "wss://api.bebop.com/v1/price".to_string();
        Self { url, pairs: vec![], chain }
    }
}

#[async_trait]
impl RFQClient for BebopClient {
    fn stream(&self) -> BoxStream<'static, (String, StateSyncMessage)> {
        // connect to Bebop websocket
        // for each message received
        //   compare the token pairs that were received with the current ones.
        //   define the removed ones (if any)
        //

        // example of a protocol component for a Bebop pair
        let id = "WETH/USDC_maker1".to_string();
        let protocol_component = ProtocolComponent {
            id: id.clone(),
            protocol_system: "bebop".to_string(),
            protocol_type_name: "".to_string(), // TODO: what is this?
            chain: self.chain,
            tokens: vec![],                        // token addresses
            contract_ids: vec![],                  // empty
            static_attributes: Default::default(), // we could include the maker here
            change: Default::default(),
            creation_tx: Default::default(),
            created_at: Default::default(),
        };

        // example snapshot for Bebop
        let snapshot = Snapshot {
            states: HashMap::from([(
                id.clone(),
                ComponentWithState {
                    state: ResponseProtocolState {
                        component_id: id.clone(),
                        attributes: HashMap::new(), /* TODO: put the bids and asks here?? then we
                                                     * would have to convert them to Bytes? */
                        balances: HashMap::new(), // No balances
                    },
                    component: protocol_component,
                    component_tvl: None,
                },
            )]),
            vm_storage: HashMap::new(),
        };

        let msg = StateSyncMessage {
            header: Default::default(), /* Header is block specific rn. We need to generalise
                                         * over this to allow for a timestamp */
            snapshots: snapshot,
            deltas: None, // Deltas will be None - all the changes are absolute
            removed_components: Default::default(),
        };
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
        // in quote_attributes we need to save the whole tx and the toSign parameters
        // then in encoding we will match the sender and receiver to the taker_address and
        // maker_address.
        // in execution we will need to check if the amountIn is the same as the quoted amount:
        //   - if it's less, we need to do a partial fill https://docs.bebop.xyz/bebop/bebop-api-pmm-rfq/rfq-api-endpoints/trade/self-execute-order#partial-fills
        //   - if it's more, we can only do the quoted amount and the rest will not be used (it will
        //     either stay in the router or in the user or in the previous pool?)
    }

    fn clone_box(&self) -> Box<dyn RFQClient> {
        Box::new(self.clone())
    }
}
