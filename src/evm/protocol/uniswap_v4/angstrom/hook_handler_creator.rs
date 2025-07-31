use crate::evm::protocol::uniswap_v4::hooks::hook_handler::HookHandler;
use crate::evm::protocol::uniswap_v4::hooks::hook_handler_creator::{
    HookCreationParams, HookHandlerCreator,
};
use crate::protocol::errors::InvalidSnapshotError;
use alloy::primitives::Address;

pub struct AngstromHookCreator;

impl HookHandlerCreator for AngstromHookCreator {
    fn instantiate_hook_handler(
        &self,
        params: HookCreationParams,
    ) -> Result<Box<dyn HookHandler>, InvalidSnapshotError> {
        let hook_address_bytes = params
            .attributes
            .get("hook_address")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("hook_address".to_string()))?;

        let pool_manager_address_bytes = params
            .attributes
            .get("pool_manager_address")
            .ok_or_else(|| {
                InvalidSnapshotError::MissingAttribute("pool_manager_address".to_string())
            })?;

        let hook_address = Address::from_slice(&hook_address_bytes.0);
        let pool_manager_address = Address::from_slice(&pool_manager_address_bytes.0);

        let engine = create_engine(SHARED_TYCHO_DB.clone(), true).map_err(|e| {
            InvalidSnapshotError::VMError(SimulationError::FatalError(format!(
                "Failed to create engine: {e:?}"
            )))
        })?;

        let hook_handler = GenericVMHookHandler::new(
            hook_address,
            engine,
            pool_manager_address,
            params.all_tokens.clone(),
            params.account_balances.clone(),
        )
        .map_err(InvalidSnapshotError::VMError)?;

        Ok(Box::new(hook_handler))
    }
}
