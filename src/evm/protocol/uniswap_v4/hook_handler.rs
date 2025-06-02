use std::{collections::HashMap, fmt::Debug};

use alloy::{
    primitives::{Address, I256, U256},
    sol,
};
use thiserror::Error;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::protocol::uniswap_v4::state::{UniswapV4Fees, UniswapV4State},
    models::{Balances, Token},
    protocol::errors::{SimulationError, TransitionError},
};

pub struct StateContext {
    pub currency_0: Address,
    pub currency_1: Address,
    pub fees: UniswapV4Fees,
    pub tick: i32,
}

pub struct SwapParams {
    pub zero_for_one: bool,
    pub amount_specified: I256,
    pub sqrt_price_limit: U256,
}
pub struct BeforeSwapParameters {
    pub context: StateContext,
    pub sender: Address,
    pub swap_params: SwapParams,
    pub hook_data: Bytes,
}

pub type BeforeSwapDelta = I256;

sol! {
    #[derive(Debug)]
    struct BeforeSwapReturn {
        bytes4 selector;
        int256 amountDelta;
        uint24 fee;
    }
}

pub struct AfterSwapParameters {
    context: StateContext,
    sender: Address,
    swap_params: SwapParams,
    delta: BeforeSwapDelta,
    hook_data: Bytes,
}

#[derive(Debug, Clone)]
pub struct WithGasEstimate<T> {
    pub gas_estimate: u64,
    pub result: T,
}

pub struct AmountRanges {
    amount_in_range: (U256, U256),
    amount_out_range: (U256, U256),
}

// TODO: do we really need an extra enum for hook errors?
#[derive(Error, Debug, Clone)]
pub enum HookError {
    #[error("Method {0} not provided by the hook handler")]
    MethodNotProvided(String),
    #[error("Fatal Error: {0}")]
    FatalError(String),
    // TODO: what other errors can occur?
}

impl From<HookError> for SimulationError {
    fn from(error: HookError) -> Self {
        SimulationError::FatalError(error.to_string())
    }
}

impl From<SimulationError> for HookError {
    fn from(error: SimulationError) -> Self {
        HookError::FatalError(error.to_string())
    }
}

// https://github.com/Uniswap/v4-core/blob/main/src/interfaces/IHooks.sol
pub trait HookHandler: Debug + Send + Sync + 'static {
    fn address(&self) -> Address;
    /// Simulates the beforeSwap Solidity behaviour
    fn before_swap(
        &self,
        params: BeforeSwapParameters,
        block: u64,
    ) -> Result<WithGasEstimate<BeforeSwapReturn>, HookError>;

    /// Simulates the afterSwap Solidity behaviour
    fn after_swap(
        &self,
        params: AfterSwapParameters,
    ) -> Result<WithGasEstimate<BeforeSwapDelta>, HookError>;

    // Currently fee is not accessible on v4 pools, this is for future use
    // as soon as we adapt the ProtocolSim interface
    fn fee(&self, context: &UniswapV4State, params: SwapParams) -> Result<f64, HookError>;

    /// Hooks will likely modify spot price behaviour this function
    /// allows overriding it.
    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, HookError>;

    // Advanced version also returning minimum swap amounts for future compatbility
    // with updated ProtocolSim interface
    fn get_amount_ranges(
        &self,
        token_in: Address,
        token_out: Address,
    ) -> Result<AmountRanges, HookError>;

    // Called on each state update to update the internal state of the HookHandler
    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        tokens: &HashMap<Bytes, Token>,
        balances: &Balances,
    ) -> Result<(), TransitionError<String>>;
    fn clone_box(&self) -> Box<dyn HookHandler>;

    fn as_any(&self) -> &dyn std::any::Any;

    fn is_equal(&self, other: &dyn HookHandler) -> bool;
}

impl Clone for Box<dyn HookHandler> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
