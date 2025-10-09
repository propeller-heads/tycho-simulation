use std::{any::Any, collections::HashMap};

use alloy::primitives::U256;
use num_bigint::{BigUint, ToBigUint};
use tycho_common::{
    dto::ProtocolStateDelta,
    models::token::Token,
    simulation::{
        errors::{SimulationError, TransitionError},
        protocol_sim::{Balances, GetAmountOutResult, ProtocolSim},
    },
    Bytes,
};

use crate::evm::protocol::{
    cfmm::protocol::{cfmm_get_amount_out, cfmm_get_limits},
    cpmm::protocol::{
        cpmm_delta_transition, cpmm_fee, cpmm_get_amount_out, cpmm_get_limits, cpmm_spot_price,
    },
    safe_math::{safe_add_u256, safe_sub_u256},
    u256_num::{biguint_to_u256, u256_to_biguint},
};

const AERODROME_V1_STABLE_FEE_BPS: u32 = 5; // 0.05% fee
const AERODROME_V1_VOLATILE_FEE_BPS: u32 = 30; // 0.3% fee

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AerodromeV1PoolState {
    pub reserve0: U256,
    pub reserve1: U256,
    pub stable: bool,
    pub base_token: Token,
    pub quote_token: Token,
}

impl AerodromeV1PoolState {
    /// Creates a new instance of `AerodromeV1State` with the given reserves.
    ///
    /// # Arguments
    ///
    /// * `reserve0` - Reserve of token 0.
    /// * `reserve1` - Reserve of token 1.
    /// * `stable` - Whether the pool is stable or not.
    pub fn new(
        reserve0: U256,
        reserve1: U256,
        stable: bool,
        base_token: Token,
        quote_token: Token,
    ) -> Self {
        AerodromeV1PoolState { reserve0, reserve1, stable, base_token, quote_token }
    }
}

impl ProtocolSim for AerodromeV1PoolState {
    fn fee(&self) -> f64 {
        if self.stable {
            cpmm_fee(AERODROME_V1_STABLE_FEE_BPS)
        } else {
            cpmm_fee(AERODROME_V1_VOLATILE_FEE_BPS)
        }
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        cpmm_spot_price(base, quote, self.reserve0, self.reserve1)
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        let amount_in = biguint_to_u256(&amount_in);
        let zero2one = token_in.address < token_out.address;
        let amount_out = if self.stable {
            cfmm_get_amount_out(
                amount_in,
                zero2one,
                self.reserve0,
                self.reserve1,
                AERODROME_V1_STABLE_FEE_BPS,
                if zero2one { token_in.decimals as u8 } else { token_out.decimals as u8 },
                if zero2one { token_out.decimals as u8 } else { token_in.decimals as u8 },
            )?
        } else {
            cpmm_get_amount_out(
                amount_in,
                zero2one,
                self.reserve0,
                self.reserve1,
                AERODROME_V1_VOLATILE_FEE_BPS,
            )?
        };
        let mut new_state = self.clone();
        let (reserve0_mut, reserve1_mut) = (&mut new_state.reserve0, &mut new_state.reserve1);
        if zero2one {
            *reserve0_mut = safe_add_u256(self.reserve0, amount_in)?;
            *reserve1_mut = safe_sub_u256(self.reserve1, amount_out)?;
        } else {
            *reserve0_mut = safe_sub_u256(self.reserve0, amount_out)?;
            *reserve1_mut = safe_add_u256(self.reserve1, amount_in)?;
        };
        Ok(GetAmountOutResult::new(
            u256_to_biguint(amount_out),
            120_000
                .to_biguint()
                .expect("Expected an unsigned integer as gas value"),
            Box::new(new_state),
        ))
    }

    fn get_limits(
        &self,
        sell_token: Bytes,
        buy_token: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        if self.stable {
            cfmm_get_limits(
                sell_token,
                buy_token,
                self.reserve0,
                self.reserve1,
                self.base_token.decimals as u8,
                self.quote_token.decimals as u8,
            )
        } else {
            cpmm_get_limits(sell_token, buy_token, self.reserve0, self.reserve1)
        }
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        _tokens: &HashMap<Bytes, Token>,
        _balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        let (reserve0_mut, reserve1_mut) = (&mut self.reserve0, &mut self.reserve1);
        cpmm_delta_transition(delta, reserve0_mut, reserve1_mut)
    }

    fn clone_box(&self) -> Box<dyn ProtocolSim> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn eq(&self, other: &dyn ProtocolSim) -> bool {
        if let Some(other_state) = other.as_any().downcast_ref::<Self>() {
            let (self_reserve0, self_reserve1) = (self.reserve0, self.reserve1);
            let (other_reserve0, other_reserve1) = (other_state.reserve0, other_state.reserve1);
            self_reserve0 == other_reserve0 &&
                self_reserve1 == other_reserve1 &&
                self.fee() == other_state.fee()
        } else {
            false
        }
    }
}
