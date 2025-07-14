use std::any::Any;
use std::collections::HashMap;
use alloy::primitives::U256;
use num_bigint::BigUint;
use tycho_common::Bytes;
use tycho_common::dto::ProtocolStateDelta;
use tycho_common::models::token::Token;
use crate::evm::protocol::safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256};
use crate::evm::protocol::u256_num::{biguint_to_u256, u256_to_biguint};
use crate::models::Balances;
use crate::protocol::errors::{SimulationError, TransitionError};
use crate::protocol::models::GetAmountOutResult;
use crate::protocol::state::ProtocolSim;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DodoV2State {
    i: U256,
    k: U256,
    b: U256,
    q: U256,
    b0: U256,
    q0: U256,
    r: u8, // 0: ONE, 1: ABOVE_ONE, 2: BELOW_ONE
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
    ) -> Self {
        let mut state = Self { i, k, b, q, b0, q0, r };
        state.adjust_target()?;
        state
    }

    fn adjust_target(&mut self){
        match self.r {
            2 => {
                // BELOW_ONE
                let delta = safe_sub_u256(self.b, self.b0)?;
                self.q0 = solve_quadratic_function_for_target(self.q, delta, self.i, self.k);
            }
            1 => {
                // ABOVE_ONE
                let delta = safe_sub_u256(self.q, self.q0)?;
                let reciprocal_i = safe_div_u256(U256::from(10).pow(U256::from(36)), self.i)?;
                self.b0 = solve_quadratic_function_for_target(self.b, delta, reciprocal_i, self.k);
            }
            _ => {}
        }
    }
}

fn mul_floor(a: U256, b: U256) -> Result<U256, SimulationError> {
    safe_div_u256(safe_mul_u256(a, b)?, U256::from(10).pow(U256::from(18)))
}

fn solve_quadratic_function_for_target(
    v1: U256,
    delta: U256,
    i: U256,
    k: U256,
) -> U256 {
    if k.is_zero() {
        // V0 = V1 + i * delta / 1e18
        return safe_add_u256(
            v1,
            mul_floor(i,delta)?
        )?;
    }
    if v1.is_zero() {
        return U256::ZERO;
    }

    let four = U256::from(4);
    let two = U256::from(2);
    let one = U256::from(10).pow(U256::from(18));
    let one2 = U256::from(10).pow(U256::from(36));

    let ki = safe_mul_u256(four, safe_mul_u256(k,i)?)?;

    let sqrt = if ki.is_zero() {
        one2
    } else {
        let ki_mul_delta = safe_mul_u256(ki, delta)?;
        let ratio = if safe_div_u256(ki_mul_delta, ki)? == delta {
            // (ki * delta) / V1
            safe_add_u256(safe_div_u256(ki_mul_delta, v1)?, one2)?
        } else {
            // (ki / V1) * delta
            let ki_div_v1 = safe_div_u256(ki, v1)?;
            safe_add_u256(safe_mul_u256(ki_div_v1, delta)?, one2)?
        };
        biguint_to_u256(u256_to_biguint(ratio).sqrt()?)
    };

    // premium = 1 + (sqrt - ONE2) / (2k)
    let two_k = safe_mul_u256(two, k)?;
    let premium = safe_add_u256(safe_div_u256(safe_sub_u256(sqrt, one2)?, two_k)?, one)?;

    // result = V1 * premium / 1e18
    mul_floor(v1, premium)?
}

pub fn solve_quadratic_function_for_trade(
    v0: U256,
    v1: U256,
    delta: U256,
    i: U256,
    k: U256,
) -> Result<U256, SimulationError> {
    // 常量：1e18
    let one = U256::from(10).pow(U256::from(18));

    // require(V0 > 0, "TARGET_IS_ZERO");
    if v0.is_zero() {
        return Err(SimulationError::FatalError("TARGET_IS_ZERO".to_string()));
    }

    // 如果 delta==0，直接返回 0
    if delta.is_zero() {
        return Ok(U256::ZERO);
    }

    // k == 0
    if k.is_zero() {
        let tmp = safe_mul_u256(i, delta)?; // i * delta
        if tmp > v1 {
            return Ok(v1);
        } else {
            return Ok(tmp);
        }
    }

    // k == 1
    if k == one {
        // 计算 temp = i * delta * V1 / V0^2
        let v0_sq = safe_mul_u256(v0, v0)?;               // V0 * V0
        let i_delta = safe_mul_u256(i, delta)?;           // i * delta
        let temp = if i_delta.is_zero() {
            U256::ZERO
        } else {
            // 检查 (i_delta * v1) / i_delta == v1 是否成立
            let tmp1 = safe_mul_u256(i_delta, v1)?;
            let tmp2 = safe_div_u256(tmp1, i_delta)?;
            if tmp2 == v1 {
                safe_div_u256(tmp1, v0_sq)?
            } else {
                safe_div_u256(
                    safe_mul_u256(delta, v1)?,
                    v0,
                )?
                    .checked_mul(i)
                    .ok_or_else(|| SimulationError::FatalError("Overflow in temp".to_string()))?
                    .checked_div(v0)
                    .ok_or_else(|| SimulationError::FatalError("Overflow in temp".to_string()))?
            }
        };

        // return V1 * temp / (temp + ONE)
        let numerator = safe_mul_u256(v1, temp)?;
        let denominator = safe_add_u256(temp, one)?;
        safe_div_u256(numerator, denominator)
    } else {
        // 通用二次方程分支
        // part2 = k * V0^2 / V1 + i * delta
        let k_v0_sq = safe_mul_u256(k, safe_mul_u256(v0, v0)?)?;
        let part2 = safe_add_u256(
            safe_div_u256(k_v0_sq, v1)?,
            safe_mul_u256(i, delta)?,
        )?;

        // b_abs = (1 - k) * V1
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

        // 计算 sqrt
        // squareRoot = sqrt(b_abs^2 + 4*(1-k)*k*V0^2)
        let four = U256::from(4);
        let four_one_minus_k = safe_mul_u256(four, one_minus_k)?;
        let four_one_minus_k_k = safe_mul_u256(four_one_minus_k, k)?;
        let four_one_minus_k_k_v0_sq = safe_mul_u256(four_one_minus_k_k, safe_mul_u256(v0, v0)?)?;
        let b_abs_sq = safe_mul_u256(b_abs, b_abs)?;
        let rhs = safe_add_u256(b_abs_sq, four_one_minus_k_k_v0_sq)?;

        // 使用 ruint::algorithms::sqrt 计算 sqrt(rhs)
        // sqrt 返回 (root, remainder)
        let square_root = biguint_to_u256(u256_to_biguint(rhs).sqrt()?);
        
        let denominator = safe_mul_u256(one_minus_k, U256::from(2))?;
        let numerator = if b_sig {
            let diff = safe_sub_u256(square_root, b_abs)?;
            if diff.is_zero() {
                return Err(SimulationError::FatalError("DODOMath: should not be zero".to_string()));
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

impl ProtocolSim for DodoV2State {
    fn fee(&self) -> f64 {
        todo!()
    }

    fn spot_price(&self, base: &Token, quote: &Token) -> Result<f64, SimulationError> {
        todo!()
    }

    fn get_amount_out(&self, amount_in: BigUint, token_in: &Token, token_out: &Token) -> Result<GetAmountOutResult, SimulationError> {
        todo!()
    }

    fn get_limits(&self, sell_token: Bytes, buy_token: Bytes) -> Result<(BigUint, BigUint), SimulationError> {
        todo!()
    }

    fn delta_transition(&mut self, delta: ProtocolStateDelta, tokens: &HashMap<Bytes, Token>, balances: &Balances) -> Result<(), TransitionError<String>> {
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
        if let Some(other_state) = other.as_any().downcast_ref::<DodoV2State>() {
            self.i == other_state.i &&
            self.k == other_state.k &&
            self.b == other_state.b &&
            self.q == other_state.q &&
            self.b0 == other_state.b0 &&
            self.q0 == other_state.q0 &&
            self.r == other_state.r
        } else {
            false
        }
    }
}