use std::{collections::HashMap, sync::RwLock};

use alloy::rpc::types::Header;
use alloy_primitives::Address;
use lazy_static::lazy_static;
use tycho_common::Bytes;

use crate::{
    evm::protocol::uniswap_v4::{hook_handler::HookHandler, state::UniswapV4State},
    models::Token,
    protocol::errors::{InvalidSnapshotError, SimulationError},
};

pub struct HookCreationParams<'a> {
    block: Header,
    account_balances: &'a HashMap<Bytes, HashMap<Bytes, Bytes>>,
    all_tokens: &'a HashMap<Bytes, Token>,
    // decoded plain uniswap v4 pool
    state: UniswapV4State,
    // raw attributes from ResponseProtocolState
    pub(crate) attributes: &'a HashMap<String, Bytes>,
    // raw balances from ResponseProtocolState
    balances: &'a HashMap<Bytes, Bytes>,
}

impl HookCreationParams<'_> {
    pub fn new() -> Self {
        todo!()
    }
}

pub trait HookHandlerCreator: Send + Sync {
    fn instantiate_hook_handler(
        &self,
        params: HookCreationParams,
    ) -> Result<Box<dyn HookHandler>, InvalidSnapshotError>;
}

// Workaround for stateless decoder trait..
lazy_static! {
    static ref HANDLER_FACTORY: RwLock<HashMap<Address, Box<dyn HookHandlerCreator>>> =
        RwLock::new(HashMap::new());
}

// TODO: where are we going to call the register_hook_handler and the set_default_hook_handler?
pub fn register_hook_handler(
    hook: Address,
    handler: Box<dyn HookHandlerCreator>,
) -> Result<(), SimulationError> {
    // Add to HANDLER_FACTORY
    todo!()
}

pub fn set_default_hook_handler(
    handler: Box<dyn HookHandlerCreator>,
) -> Result<(), SimulationError> {
    // The default should be the generic VM handler.
    todo!()
}

pub fn instantiate_hook_handler(
    hook_address: &Address,
    params: HookCreationParams<'_>,
) -> Result<Box<dyn HookHandler>, InvalidSnapshotError> {
    let binding = HANDLER_FACTORY.read().unwrap();
    let creator = binding.get(hook_address).unwrap(); // be better. if it's not found, use the default handler
    creator.instantiate_hook_handler(params)
}
