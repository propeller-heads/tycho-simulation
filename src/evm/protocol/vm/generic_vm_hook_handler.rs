use std::{any::Any, collections::HashMap, fmt::Debug};

use alloy_primitives::Address;
use revm::DatabaseRef;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::{
        engine_db::{create_engine, engine_db_interface::EngineDatabaseInterface, SHARED_TYCHO_DB},
        protocol::{
            uniswap_v4::{
                hook_handler::{
                    AfterSwapParameters, AmountRanges, BeforeSwapDelta, BeforeSwapParameters,
                    HookError, HookHandler, SwapParams, WithGasEstimate,
                },
                hook_handler_creator::{HookCreationParams, HookHandlerCreator},
                state::UniswapV4State,
            },
            vm::tycho_simulation_contract::TychoSimulationContract,
        },
        simulation::SimulationEngine,
    },
    models::{Balances, Token},
    protocol::errors::{InvalidSnapshotError, SimulationError, TransitionError},
};

#[derive(Debug, Clone)]
struct GenericVMHookHandler<D: EngineDatabaseInterface + Clone + Debug>
where
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    contract: TychoSimulationContract<D>,
    address: Address,
}

impl<D: EngineDatabaseInterface + Clone + Debug> PartialEq for GenericVMHookHandler<D>
where
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    fn eq(&self, _other: &Self) -> bool {
        todo!()
    }
}

impl<D: EngineDatabaseInterface + Clone + Debug> GenericVMHookHandler<D>
where
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    pub fn new(address: Address, engine: SimulationEngine<D>) -> Result<Self, SimulationError> {
        Ok(GenericVMHookHandler {
            contract: TychoSimulationContract::new(address, engine)?,
            address,
        })
    }

    pub fn unlock_pool_manager(&self) {
        // see here how to set transient storage on the revm:
        // https://github.com/propeller-heads/tycho-v4-hooks-prototype/blob/main/revm-tstore/src/main.rs#L10
        // the slot is here https://github.com/Uniswap/v4-core/blob/main/src/libraries/Lock.sol#L8C5-L8C117
        // TODO: do we really need this?? I'm thinking no. we only need the msg.sender to be the
        // pool manager
        todo!()
    }
}

impl<D: EngineDatabaseInterface + Clone + Debug + 'static> HookHandler for GenericVMHookHandler<D>
where
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    fn address(&self) -> Address {
        self.address
    }

    fn before_swap(
        &self,
        params: BeforeSwapParameters,
    ) -> Result<WithGasEstimate<(BeforeSwapDelta, u32)>, HookError> {
        self.unlock_pool_manager();
        todo!()
    }

    fn after_swap(&self, params: AfterSwapParameters) -> Result<WithGasEstimate<i128>, HookError> {
        self.unlock_pool_manager();
        todo!()
        // self.contract.call(..)
    }

    fn fee(&self, context: &UniswapV4State, params: SwapParams) -> Result<f64, HookError> {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, HookError> {
        todo!()
    }

    fn get_amount_ranges(
        &self,
        token_in: Address,
        token_out: Address,
    ) -> Result<AmountRanges, HookError> {
        todo!()
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        tokens: &HashMap<Bytes, Token>,
        balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        todo!()
    }

    fn clone_box(&self) -> Box<dyn HookHandler> {
        Box::new((*self).clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_equal(&self, other: &dyn HookHandler) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .map_or(false, |o| self == o)
    }
}

struct GenericVMHookHandlerCreator<D>
where
    D: EngineDatabaseInterface + Clone + Debug,
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    engine: SimulationEngine<D>,
}

impl<D> HookHandlerCreator for GenericVMHookHandlerCreator<D>
where
    D: EngineDatabaseInterface + Clone + Debug,
    <D as DatabaseRef>::Error: Debug,
    <D as EngineDatabaseInterface>::Error: Debug,
{
    fn instantiate_hook_handler(
        &self,
        params: HookCreationParams,
    ) -> Result<Box<dyn HookHandler>, InvalidSnapshotError> {
        let db = SHARED_TYCHO_DB.clone();
        let hook_address = Address::from_slice(
            &*params
                .attributes
                .get("hook_address")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("hook_address".to_string()))?
                .clone(),
        );
        let engine = create_engine(db, false)?;
        Ok(Box::new(GenericVMHookHandler::new(hook_address, engine)?))
    }
}
