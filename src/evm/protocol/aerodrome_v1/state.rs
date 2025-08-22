use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use alloy::primitives::U256;
use num_bigint::BigUint;
use tycho_client::feed::Header;
use tycho_client::feed::synchronizer::ComponentWithState;
use tycho_common::Bytes;
use tycho_common::dto::ProtocolStateDelta;
use crate::evm::protocol::cfmm::protocol::CFMMProtocol;
use crate::evm::protocol::cpmm::protocol::CPMMProtocol;
use crate::models::{Balances, Token};
use crate::protocol::errors::{InvalidSnapshotError, SimulationError, TransitionError};
use crate::protocol::models::{GetAmountOutResult, TryFromWithBlock};
use crate::protocol::state::ProtocolSim;

#[derive(Clone, Debug)]
pub enum AerodromeV1State {
    CPMM(AerodromeV1PoolState),
    CFMM(AerodromeV1PoolState),
}

impl AerodromeV1State {
    fn dispatch<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&dyn ProtocolSim) -> R,
    {
        match self {
            AerodromeV1State::CPMM(state) => f(state as &dyn CPMMProtocol),
            AerodromeV1State::CFMM(state) => f(state as &dyn CFMMProtocol),
        }
    }

    fn dispatch_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut dyn ProtocolSim) -> R,
    {
        match self {
            AerodromeV1State::CPMM(state) => f(state as &mut dyn CPMMProtocol),
            AerodromeV1State::CFMM(state) => f(state as &mut dyn CFMMProtocol),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AerodromeV1PoolState {
    pub reserve0: U256,
    pub reserve1: U256,
}

impl AerodromeV1PoolState {
    /// Creates a new instance of `AerodromeV1State` with the given reserves.
    ///
    /// # Arguments
    ///
    /// * `reserve0` - Reserve of token 0.
    /// * `reserve1` - Reserve of token 1.
    /// * `stable` - Whether the pool is stable or not.
    pub fn new(reserve0: U256, reserve1: U256) -> Self {
        AerodromeV1PoolState { reserve0, reserve1 }
    }
}

impl ProtocolSim for AerodromeV1State {
    fn fee(&self) -> f64 {
        self.dispatch(|ps| ps.fee())
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        self.dispatch(|ps| ps.spot_price(base, quote))
    }

    fn get_amount_out(&self, amount_in: BigUint, token_in: &Token, token_out: &Token) -> Result<GetAmountOutResult, SimulationError> {
        self.dispatch(|ps| ps.get_amount_out(amount_in, token_in, token_out))
    }

    fn get_limits(&self, sell_token: Bytes, buy_token: Bytes) -> Result<(BigUint, BigUint), SimulationError> {
        self.dispatch(|ps| ps.get_limits(sell_token, buy_token))
    }

    fn delta_transition(&mut self, delta: ProtocolStateDelta, tokens: &HashMap<Bytes, Token>, balances: &Balances) -> Result<(), TransitionError<String>> {
        self.dispatch_mut(|ps| ps.delta_transition(delta, tokens, balances))
    }

    fn clone_box(&self) -> Box<dyn ProtocolSim> {
        self.dispatch(|ps| ps.clone_box())
    }

    fn as_any(&self) -> &dyn Any {
        self.dispatch(|ps| ps.as_any())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self.dispatch_mut(|ps| ps.as_any_mut())
    }

    fn eq(&self, other: &dyn ProtocolSim) -> bool {
        self.dispatch(|ps| ps.eq(other))
    }
}

impl TryFromWithBlock<ComponentWithState> for AerodromeV1State {
    type Error = InvalidSnapshotError;

    fn try_from_with_block(snapshot: ComponentWithState, _block: Header, _account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>, all_tokens: &HashMap<Bytes, Token>) -> impl Future<Output=Result<Self, Self::Error>> + Send + Sync
    where
        Self: Sized
    {
        let reserve0 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("reserve0")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve0".to_string()))?,
        );

        let reserve1 = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("reserve1")
                .ok_or(InvalidSnapshotError::MissingAttribute("reserve1".to_string()))?,
        );

        let stable = snapshot
            .state
            .attributes
            .get("stable")
            .map(|v| v == &Bytes::from(vec![1]))
            .unwrap_or(false);

        let pool_state = AerodromeV1PoolState::new(reserve0, reserve1);

        if stable{
            Ok(AerodromeV1State::CFMM(pool_state))
        } else {
            Ok(AerodromeV1State::CPMM(pool_state))
        }
    }
}


impl CPMMProtocol for AerodromeV1PoolState {
    fn get_fee_bps(&self) -> u32 {
        todo!()
    }

    fn get_reserves(&self) -> (U256, U256) {
        (self.reserve0, self.reserve1)
    }

    fn get_reserves_mut(&mut self) -> (&mut U256, &mut U256) {
        (&mut self.reserve0, &mut self.reserve1)
    }

    fn new(reserve0: U256, reserve1: U256) -> Self {
        Self::new(reserve0, reserve1)
    }
}

impl CFMMProtocol for AerodromeV1PoolState {
    fn get_fee_bps(&self) -> u32 {
        todo!()
    }

    fn get_reserves(&self) -> (U256, U256) {
        (self.reserve0, self.reserve1)
    }

    fn get_reserves_mut(&mut self) -> (&mut U256, &mut U256) {
        (&mut self.reserve0, &mut self.reserve1)
    }

    fn new(reserve0: U256, reserve1: U256) -> Self {
        Self::new(reserve0, reserve1)
    }
}