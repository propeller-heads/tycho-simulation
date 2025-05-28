use std::{collections::HashMap, fmt::Debug};

use alloy_primitives::{Address, I256, U256};
use thiserror::Error;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::protocol::uniswap_v4::state::{UniswapV4Fees, UniswapV4State},
    models::{Balances, Token},
    protocol::errors::{SimulationError, TransitionError},
};

pub type BeforeSwapDelta = I256;
pub type SwapDelta = I256;

struct StateContext {
    currency_0: Address,
    currency_1: Address,
    fees: UniswapV4Fees,
    tick: i32,
}

pub struct SwapParams {
    zero_for_one: bool,
    amount_specified: I256,
    sqrt_price_limit: U256,
}
pub struct BeforeSwapParameters {
    context: StateContext,
    sender: Address,
    swap_params: SwapParams,
    hook_data: Bytes,
}

pub struct AfterSwapParameters {
    context: StateContext,
    sender: Address,
    swap_params: SwapParams,
    delta: SwapDelta,
    hook_data: Bytes,
}

pub struct WithGasEstimate<T> {
    gas_estimate: u32,
    result: T,
}

pub struct AmountRanges {
    amount_in_range: (U256, U256),
    amount_out_range: (U256, U256),
}

#[derive(Error, Debug)]
pub enum HookError {
    #[error("Method {0} not provided by the hook handler")]
    MethodNotProvided(String),
    // TODO: what other errors can occur?
}

impl From<HookError> for SimulationError {
    fn from(error: HookError) -> Self {
        SimulationError::FatalError(error.to_string())
    }
}

// https://github.com/Uniswap/v4-core/blob/main/src/interfaces/IHooks.sol
pub trait HookHandler: Debug + Send + Sync + 'static {
    fn address(&self) -> Address;
    /// Simulates the beforeSwap Solidity behaviour
    fn before_swap(
        &self,
        params: BeforeSwapParameters,
    ) -> Result<WithGasEstimate<(BeforeSwapDelta, u32)>, HookError>;

    /// Simulates the afterSwap Solidity behaviour
    fn after_swap(&self, params: AfterSwapParameters) -> Result<WithGasEstimate<i128>, HookError>;

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
