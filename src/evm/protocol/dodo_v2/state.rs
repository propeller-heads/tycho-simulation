use crate::evm::protocol::safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256};
use crate::evm::protocol::u256_num::{biguint_to_u256, u256_to_biguint};
use crate::models::Balances;
use crate::protocol::errors::{InvalidSnapshotError, SimulationError, TransitionError};
use crate::protocol::models::GetAmountOutResult;
use crate::protocol::state::ProtocolSim;
use alloy::primitives::U256;
use num_bigint::BigUint;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Sub;
use tycho_common::dto::ProtocolStateDelta;
use tycho_common::models::token::Token;
use tycho_common::Bytes;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RState {
    ONE = 0,
    AboveOne = 1,
    BelowOne = 2,
}

impl TryFrom<u8> for RState {
    type Error = InvalidSnapshotError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(RState::ONE),
            1 => Ok(RState::AboveOne),
            2 => Ok(RState::BelowOne),
            _ => Err(InvalidSnapshotError::ValueError(format!("Invalid RState value: {}", value))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DodoV2State {
    i: U256,
    k: U256,
    b: U256,
    q: U256,
    b0: U256,
    q0: U256,
    r: RState,

    base_token: Bytes,
    quote_token: Bytes,

    lp_fee_rate: U256,
    mt_fee_rate: U256,

    mt_fee_quote: U256,
    mt_fee_base: U256,
}

impl DodoV2State {
    pub fn new(
        i: U256,
        k: U256,
        b: U256,
        q: U256,
        b0: U256,
        q0: U256,
        r: u8,
        base_token: Bytes,
        quote_token: Bytes,
        lp_fee_rate: U256,
        mt_fee_rate: U256,
        mt_fee_quote: U256,
        mt_fee_base: U256,
    ) -> Self {
        let mut state = Self {
            i,
            k,
            b,
            q,
            b0,
            q0,
            r: RState::try_from(r).unwrap(),
            base_token,
            quote_token,
            lp_fee_rate,
            mt_fee_rate,
            mt_fee_quote,
            mt_fee_base,
        };
        state.adjust_target()?;
        state
    }

    fn adjust_target(&mut self) {
        match self.r {
            RState::BelowOne => {
                // BelowOne
                let delta = safe_sub_u256(self.b, self.b0)?;
                self.q0 =
                    self.solve_quadratic_function_for_target(self.q, delta, self.i, self.k)?;
            }
            RState::AboveOne => {
                // AboveOne
                let delta = safe_sub_u256(self.q, self.q0)?;
                let reciprocal_i = self.reciprocal_floor(self.i)?;
                self.b0 =
                    self.solve_quadratic_function_for_target(self.b, delta, reciprocal_i, self.k)?;
            }
            _ => {}
        }
    }

    fn reciprocal_floor(&self, a: U256) -> Result<U256, SimulationError> {
        if a.is_zero() {
            return Err(SimulationError::FatalError("Division by zero".to_string()));
        }
        safe_div_u256(U256::from(10).pow(U256::from(36)), a)
    }

    fn sell_base_token(&self, pay_base_amount: U256) -> Result<(U256, RState), SimulationError> {
        if self.r == RState::ONE {
            let receive_quote_amount = self.solve_quadratic_function_for_trade(
                self.q0,
                self.q0,
                pay_base_amount,
                self.i,
                self.k,
            )?;
            Ok((receive_quote_amount, RState::BelowOne))
        } else if self.r == RState::AboveOne {
            let back_to_one_pay_base = safe_sub_u256(self.b0, self.b)?;
            let back_to_one_receive_quote = safe_sub_u256(self.q, self.q0)?;
            if pay_base_amount < back_to_one_pay_base {
                let mut receive_quote_amount =
                    self.general_integrate(self.q0, self.q, pay_base_amount, self.i, self.k)?;
                if receive_quote_amount > back_to_one_receive_quote {
                    receive_quote_amount = back_to_one_receive_quote;
                };
                Ok((receive_quote_amount, RState::BelowOne))
            } else if pay_base_amount == back_to_one_pay_base {
                Ok((back_to_one_receive_quote, RState::ONE))
            } else {
                let receive_quote_amount = safe_add_u256(
                    back_to_one_receive_quote,
                    self.solve_quadratic_function_for_trade(
                        self.q0,
                        self.q0,
                        pay_base_amount - back_to_one_pay_base,
                        self.i,
                        self.k,
                    )?,
                )?;
                Ok((receive_quote_amount, RState::BelowOne))
            }
        } else {
            let receive_quote_amount = self.solve_quadratic_function_for_trade(
                self.q0,
                self.q,
                pay_base_amount,
                self.i,
                self.k,
            )?;
            Ok((receive_quote_amount, RState::BelowOne))
        }
    }

    fn query_sell_base_token(
        &self,
        pay_base_amount: U256,
    ) -> Result<(U256, U256, RState, U256), SimulationError> {
        let (mut receive_quote_amount, new_r_state) = self.sell_base_token(pay_base_amount)?;
        let mt_fee = self.mul_floor(receive_quote_amount, self.mt_fee_rate)?;
        receive_quote_amount -= mt_fee;
        let lp_fee = self.mul_floor(receive_quote_amount, self.lp_fee_rate)?;
        receive_quote_amount -= lp_fee;
        Ok((receive_quote_amount, mt_fee, new_r_state, self.b0))
    }

    fn query_sell_quote_token(
        &self,
        pay_quote_amount: U256,
    ) -> Result<(U256, U256, RState, U256), SimulationError> {
        let (mut receive_base_amount, new_r_state) = self.sell_quote_token(pay_quote_amount)?;
        let mt_fee = self.mul_floor(receive_base_amount, self.mt_fee_rate)?;
        receive_base_amount -= mt_fee;
        let lp_fee = self.mul_floor(receive_base_amount, self.lp_fee_rate)?;
        receive_base_amount -= lp_fee;
        Ok((receive_base_amount, mt_fee, new_r_state, self.q0))
    }

    fn sell_quote_token(&self, pay_quote_amount: U256) -> Result<(U256, RState), SimulationError> {
        if self.r == RState::ONE {
            let receive_base_amount = self.solve_quadratic_function_for_trade(
                self.b0,
                self.b0,
                pay_quote_amount,
                self.reciprocal_floor(self.i)?,
                self.k,
            )?;
            Ok((receive_base_amount, RState::AboveOne))
        } else if self.r == RState::AboveOne {
            let receive_quote_amount = self.solve_quadratic_function_for_trade(
                self.b0,
                self.b,
                pay_quote_amount,
                self.reciprocal_floor(self.i)?,
                self.k,
            )?;
            Ok((receive_quote_amount, RState::AboveOne))
        } else {
            let back_to_one_pay_quote = safe_sub_u256(self.q0, self.q)?;
            let back_to_one_receive_base = safe_sub_u256(self.b, self.b0)?;
            if pay_quote_amount < back_to_one_pay_quote {
                let mut receive_base_amount = self.general_integrate(
                    self.q0,
                    safe_add_u256(self.q, pay_quote_amount)?,
                    self.q,
                    self.reciprocal_floor(self.i)?,
                    self.k,
                )?;
                if receive_base_amount > back_to_one_receive_base {
                    receive_base_amount = back_to_one_receive_base;
                };
                Ok((receive_base_amount, RState::BelowOne))
            } else if pay_quote_amount == back_to_one_pay_quote {
                Ok((back_to_one_receive_base, RState::ONE))
            } else {
                let receive_quote_amount = safe_add_u256(
                    back_to_one_receive_base,
                    self.solve_quadratic_function_for_trade(
                        self.b0,
                        self.b0,
                        safe_sub_u256(pay_quote_amount, back_to_one_pay_quote)?,
                        self.reciprocal_floor(self.i)?,
                        self.k,
                    )?,
                )?;
                Ok((receive_quote_amount, RState::AboveOne))
            }
        }
    }

    fn mul_floor(&self, a: U256, b: U256) -> Result<U256, SimulationError> {
        safe_div_u256(safe_mul_u256(a, b)?, U256::from(10).pow(U256::from(18)))
    }

    fn div_floor(&self, a: U256, b: U256) -> Result<U256, SimulationError> {
        if b.is_zero() {
            return Err(SimulationError::FatalError("Division by zero".to_string()));
        }
        safe_div_u256(safe_mul_u256(a, U256::from(10).pow(U256::from(18)))?, b)
    }

    fn solve_quadratic_function_for_target(
        &self,
        v1: U256,
        delta: U256,
        i: U256,
        k: U256,
    ) -> Result<U256, SimulationError> {
        if k.is_zero() {
            return safe_add_u256(v1, self.mul_floor(i, delta)?);
        }
        if v1.is_zero() {
            return Ok(U256::ZERO);
        }

        let four = U256::from(4);
        let two = U256::from(2);
        let one = U256::from(10).pow(U256::from(18));
        let one2 = U256::from(10).pow(U256::from(36));

        let ki = safe_mul_u256(four, safe_mul_u256(k, i)?)?;

        let sqrt = if ki.is_zero() {
            one2
        } else {
            let ki_mul_delta = safe_mul_u256(ki, delta)?;
            let ratio = if safe_div_u256(ki_mul_delta, ki)? == delta {
                safe_add_u256(safe_div_u256(ki_mul_delta, v1)?, one2)?
            } else {
                let ki_div_v1 = safe_div_u256(ki, v1)?;
                safe_add_u256(safe_mul_u256(ki_div_v1, delta)?, one2)?
            };
            biguint_to_u256(u256_to_biguint(ratio).sqrt()?)
        };

        let two_k = safe_mul_u256(two, k)?;
        let premium = safe_add_u256(safe_div_u256(safe_sub_u256(sqrt, one2)?, two_k)?, one)?;

        self.mul_floor(v1, premium)
    }

    pub fn solve_quadratic_function_for_trade(
        &self,
        v0: U256,
        v1: U256,
        delta: U256,
        i: U256,
        k: U256,
    ) -> Result<U256, SimulationError> {
        let one = U256::from(10).pow(U256::from(18));
        if v0.is_zero() {
            return Err(SimulationError::FatalError("TARGET_IS_ZERO".to_string()));
        }
        if delta.is_zero() {
            return Ok(U256::ZERO);
        }
        if k.is_zero() {
            let tmp = safe_mul_u256(i, delta)?;
            return if tmp > v1 { Ok(v1) } else { Ok(tmp) };
        }
        if k == one {
            let v0_sq = safe_mul_u256(v0, v0)?;
            let i_delta = safe_mul_u256(i, delta)?;
            let temp = if i_delta.is_zero() {
                U256::ZERO
            } else {
                let tmp1 = safe_mul_u256(i_delta, v1)?;
                let tmp2 = safe_div_u256(tmp1, i_delta)?;
                if tmp2 == v1 {
                    safe_div_u256(tmp1, v0_sq)?
                } else {
                    safe_div_u256(safe_mul_u256(delta, v1)?, v0)?
                        .checked_mul(i)
                        .ok_or_else(|| SimulationError::FatalError("Overflow in temp".to_string()))?
                        .checked_div(v0)
                        .ok_or_else(|| {
                            SimulationError::FatalError("Overflow in temp".to_string())
                        })?
                }
            };
            let numerator = safe_mul_u256(v1, temp)?;
            let denominator = safe_add_u256(temp, one)?;
            safe_div_u256(numerator, denominator)
        } else {
            let k_v0_sq = safe_mul_u256(k, safe_mul_u256(v0, v0)?)?;
            let part2 = safe_add_u256(safe_div_u256(k_v0_sq, v1)?, safe_mul_u256(i, delta)?)?;
            let one_minus_k = safe_sub_u256(one, k)?;
            let mut b_abs = safe_mul_u256(one_minus_k, v1)?;

            let b_sig: bool;
            if b_abs >= part2 {
                b_abs = safe_sub_u256(b_abs, part2)?;
                b_sig = false;
            } else {
                b_abs = safe_sub_u256(part2, b_abs)?;
                b_sig = true;
            }

            let four = U256::from(4);
            let four_one_minus_k = safe_mul_u256(four, one_minus_k)?;
            let four_one_minus_k_k = safe_mul_u256(four_one_minus_k, k)?;
            let four_one_minus_k_k_v0_sq =
                safe_mul_u256(four_one_minus_k_k, safe_mul_u256(v0, v0)?)?;
            let b_abs_sq = safe_mul_u256(b_abs, b_abs)?;
            let rhs = safe_add_u256(b_abs_sq, four_one_minus_k_k_v0_sq)?;

            let square_root = biguint_to_u256(u256_to_biguint(rhs).sqrt()?);
            let denominator = safe_mul_u256(one_minus_k, U256::from(2))?;
            let numerator = if b_sig {
                let diff = safe_sub_u256(square_root, b_abs)?;
                if diff.is_zero() {
                    return Err(SimulationError::FatalError(
                        "DODOMath: should not be zero".to_string(),
                    ));
                }
                diff
            } else {
                safe_add_u256(b_abs, square_root)?
            };

            let v2 = safe_div_u256(numerator, denominator)?;
            if v2 > v1 {
                Ok(U256::ZERO)
            } else {
                safe_sub_u256(v1, v2)
            }
        }
    }

    pub fn general_integrate(
        &self,
        v0: U256,
        v1: U256,
        v2: U256,
        i: U256,
        k: U256,
    ) -> Result<U256, SimulationError> {
        if v0.is_zero() {
            return Err(SimulationError::FatalError("TARGET_IS_ZERO".to_string()));
        };
        let fair_amount = safe_mul_u256(i, safe_sub_u256(v1, v2)?)?;
        let one = U256::from(10).pow(U256::from(18));
        let one2 = U256::from(10).pow(U256::from(36));
        if k.is_zero() {
            return safe_div_u256(fair_amount, one);
        };
        let v0_v1_v2 = self.div_floor(v0 * v0 / v1, v2)?;
        let penalty = self.mul_floor(k, v0_v1_v2)?;
        safe_div_u256(
            safe_mul_u256(safe_add_u256(safe_sub_u256(one, k)?, penalty)?, fair_amount)?,
            one2,
        )
    }
}

impl ProtocolSim for DodoV2State {
    fn fee(&self) -> f64 {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        todo!()
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        _token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        if token_in.address == self.base_token {
            let base_input = amount_in.sub(self.mt_fee_base);
            let (receive_quote_amount, mt_fee, new_r_state, new_base_target) =
                self.query_sell_base_token(biguint_to_u256(&base_input))?;

            let mut new_state = self.clone();

            new_state.mt_fee_quote = safe_add_u256(new_state.mt_fee_quote, mt_fee)?;

            if new_state.r != new_r_state {
                new_state.b0 = new_base_target;
                new_state.r = new_r_state;
            }

            new_state.b = new_state.b + base_input;

            // todo update reserve value

            Ok(GetAmountOutResult::new(
                u256_to_biguint(receive_quote_amount),
                u256_to_biguint(U256::from(0)), // todo gas
                Box::new(new_state),
            ))
        } else if token_in.address == self.quote_token {
            let quote_input = amount_in.sub(self.mt_fee_quote);
            let (receive_base_amount, mt_fee, new_r_state, new_quote_target) =
                self.query_sell_quote_token(biguint_to_u256(&quote_input))?;

            let mut new_state = self.clone();

            new_state.mt_fee_base += mt_fee;

            if new_state.r != new_r_state {
                new_state.q0 = new_quote_target;
                new_state.r = new_r_state;
            }

            new_state.q = new_state.q + quote_input;
            // todo update reserve value

            // new_state.r = new_r_state;
            Ok(GetAmountOutResult::new(
                u256_to_biguint(receive_base_amount),
                u256_to_biguint(U256::from(0)), // todo gas
                Box::new(new_state),
            ))
        } else {
            Err(SimulationError::InvalidInput("Invalid token input".to_string(), None))
        }
    }

    fn get_limits(
        &self,
        sell_token: Bytes,
        buy_token: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
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
        if let Some(other_state) = other
            .as_any()
            .downcast_ref::<DodoV2State>()
        {
            self.i == other_state.i
                && self.k == other_state.k
                && self.b == other_state.b
                && self.q == other_state.q
                && self.b0 == other_state.b0
                && self.q0 == other_state.q0
                && self.r == other_state.r
        } else {
            false
        }
    }
}
