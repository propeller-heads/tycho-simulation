use std::{any::Any, collections::HashMap};

use alloy::primitives::{Address, U256};
use num_bigint::BigUint;
use tycho_common::{dto::ProtocolStateDelta, models::token::Token, Bytes};

use crate::{
    evm::protocol::{
        safe_math::{safe_add_u256, safe_sub_u256},
        u256_num::{biguint_to_u256, u256_to_biguint, u256_to_f64},
        utils::dodo::{
            decimal_math::{mul_floor, reciprocal_floor, ONE},
            dodo_math::{
                general_integrate, solve_quadratic_function_for_target,
                solve_quadratic_function_for_trade,
            },
        },
    },
    models::Balances,
    protocol::{
        errors::{InvalidSnapshotError, SimulationError, TransitionError},
        models::GetAmountOutResult,
        state::ProtocolSim,
    },
};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RState {
    ONE = 0,
    AboveOne = 1,
    BelowOne = 2,
}

impl TryFrom<U256> for RState {
    type Error = InvalidSnapshotError;

    fn try_from(value: U256) -> Result<Self, Self::Error> {
        let v: u8 = value.try_into().map_err(|_| {
            InvalidSnapshotError::ValueError(format!("U256 too large for RState: {value}"))
        })?;

        match v {
            0 => Ok(RState::ONE),
            1 => Ok(RState::AboveOne),
            2 => Ok(RState::BelowOne),
            _ => Err(InvalidSnapshotError::ValueError(format!("Invalid RState value: {v}"))),
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
    balances: HashMap<Address, U256>,
    component_id: String,
}

impl DodoV2State {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        i: U256,
        k: U256,
        b: U256,
        q: U256,
        b0: U256,
        q0: U256,
        r: RState,
        lp_fee_rate: U256,
        mt_fee_rate: U256,
        mt_fee_quote: U256,
        mt_fee_base: U256,
        base_token: Bytes,
        quote_token: Bytes,
        balances: HashMap<Address, U256>,
        component_id: String,
    ) -> Result<Self, SimulationError> {
        let mut state = Self {
            i,
            k,
            b,
            q,
            b0,
            q0,
            r,
            base_token,
            quote_token,
            lp_fee_rate,
            mt_fee_rate,
            mt_fee_quote,
            mt_fee_base,
            balances,
            component_id,
        };
        let (maybe_b0, maybe_q0) = state.adjust_target()?;
        if let Some(new_b0) = maybe_b0 {
            state.b0 = new_b0;
        }
        if let Some(new_q0) = maybe_q0 {
            state.q0 = new_q0;
        }

        Ok(state)
    }

    fn adjust_target(&mut self) -> Result<(Option<U256>, Option<U256>), SimulationError> {
        match self.r {
            RState::BelowOne => {
                let q0 = solve_quadratic_function_for_target(
                    self.q,
                    safe_sub_u256(self.b, self.b0)?,
                    self.i,
                    self.k,
                )?;
                Ok((None, Some(q0)))
            }
            RState::AboveOne => {
                let b0 = solve_quadratic_function_for_target(
                    self.b,
                    safe_sub_u256(self.q, self.q0)?,
                    reciprocal_floor(self.i)?,
                    self.k,
                )?;
                Ok((Some(b0), None))
            }
            _ => Ok((None, None)),
        }
    }

    /// ============ R = 1 cases ============
    fn r_one_sell_base_token(&self, pay_base_amount: U256) -> Result<U256, SimulationError> {
        solve_quadratic_function_for_trade(self.q0, self.q0, pay_base_amount, self.i, self.k)
    }

    fn r_one_sell_quote_token(&self, pay_quote_amount: U256) -> Result<U256, SimulationError> {
        solve_quadratic_function_for_trade(
            self.b0,
            self.b0,
            pay_quote_amount,
            reciprocal_floor(self.i)?,
            self.k,
        )
    }

    /// ============ R > 1 cases ============
    fn r_above_sell_base_token(&self, pay_base_amount: U256) -> Result<U256, SimulationError> {
        general_integrate(self.b0, safe_add_u256(self.b, pay_base_amount)?, self.b, self.i, self.k)
    }

    fn r_above_sell_quote_token(&self, pay_quote_amount: U256) -> Result<U256, SimulationError> {
        solve_quadratic_function_for_trade(
            self.b0,
            self.b,
            pay_quote_amount,
            reciprocal_floor(self.i)?,
            self.k,
        )
    }

    /// ============ R < 1 cases ============
    fn r_below_sell_base_token(&self, pay_base_amount: U256) -> Result<U256, SimulationError> {
        solve_quadratic_function_for_trade(self.q0, self.q, pay_base_amount, self.i, self.k)
    }

    fn r_below_sell_quote_token(&self, pay_quote_amount: U256) -> Result<U256, SimulationError> {
        general_integrate(
            self.q0,
            safe_add_u256(self.q, pay_quote_amount)?,
            self.q,
            reciprocal_floor(self.i)?,
            self.k,
        )
    }

    /// Sells base token and returns the amount of quote token received and the new RState.
    /// # Arguments
    /// * `pay_base_amount`: The amount of base token to sell.
    /// # Returns
    /// * `Result<(U256, RState), SimulationError>`: A tuple containing the amount of quote token
    ///   received and the new RState. Returns an error if the operation fails.
    fn sell_base_token(&self, pay_base_amount: U256) -> Result<(U256, RState), SimulationError> {
        if self.r == RState::ONE {
            Ok((self.r_one_sell_base_token(pay_base_amount)?, RState::BelowOne))
        } else if self.r == RState::AboveOne {
            let back_to_one_pay_base = safe_sub_u256(self.b0, self.b)?;
            let back_to_one_receive_quote = safe_sub_u256(self.q, self.q0)?;
            if pay_base_amount < back_to_one_pay_base {
                let mut receive_quote_amount = self.r_above_sell_base_token(pay_base_amount)?;
                if receive_quote_amount > back_to_one_receive_quote {
                    receive_quote_amount = back_to_one_receive_quote;
                };
                Ok((receive_quote_amount, RState::AboveOne))
            } else if pay_base_amount == back_to_one_pay_base {
                Ok((back_to_one_receive_quote, RState::ONE))
            } else {
                let receive_quote_amount = safe_add_u256(
                    back_to_one_receive_quote,
                    self.r_one_sell_base_token(safe_sub_u256(
                        pay_base_amount,
                        back_to_one_pay_base,
                    )?)?,
                )?;
                Ok((receive_quote_amount, RState::BelowOne))
            }
        } else {
            Ok((self.r_below_sell_base_token(pay_base_amount)?, RState::BelowOne))
        }
    }

    /// Sells quote token and returns the amount of base token received and the new RState.
    /// # Arguments
    /// * `pay_quote_amount`: The amount of quote token to sell.
    /// # Returns
    /// * `Result<(U256, RState), SimulationError>`: A tuple containing the amount of base token
    ///   received and the new RState. Returns an error if the operation fails.
    fn sell_quote_token(&self, pay_quote_amount: U256) -> Result<(U256, RState), SimulationError> {
        if self.r == RState::ONE {
            Ok((self.r_one_sell_quote_token(pay_quote_amount)?, RState::AboveOne))
        } else if self.r == RState::AboveOne {
            Ok((self.r_above_sell_quote_token(pay_quote_amount)?, RState::AboveOne))
        } else {
            let back_to_one_pay_quote = safe_sub_u256(self.q0, self.q)?;
            let back_to_one_receive_base = safe_sub_u256(self.b, self.b0)?;
            if pay_quote_amount < back_to_one_pay_quote {
                let mut receive_base_amount = self.r_below_sell_quote_token(pay_quote_amount)?;
                if receive_base_amount > back_to_one_receive_base {
                    receive_base_amount = back_to_one_receive_base;
                };
                Ok((receive_base_amount, RState::BelowOne))
            } else if pay_quote_amount == back_to_one_pay_quote {
                Ok((back_to_one_receive_base, RState::ONE))
            } else {
                let receive_quote_amount = safe_add_u256(
                    back_to_one_receive_base,
                    self.r_one_sell_quote_token(safe_sub_u256(
                        pay_quote_amount,
                        back_to_one_pay_quote,
                    )?)?,
                )?;
                Ok((receive_quote_amount, RState::AboveOne))
            }
        }
    }

    /// Queries the amount of quote token received when selling a base token.
    /// Returns the amount of quote token received, the market maker fee, the new RState, and the
    /// base target.
    ///
    /// # Arguments
    /// * `pay_base_amount`: The amount of base token to sell.
    /// # Returns
    /// * `Result<(U256, U256, RState, U256), SimulationError>`: A tuple containing the amount of
    ///   quote token received, the market maker fee, the new RState, and the base target. Returns
    ///   an error if the operation fails.
    /// # Errors
    /// * `SimulationError`: If the operation fails due to an error in the simulation, such as an
    ///   invalid input or a mathematical error.
    fn query_sell_base_token(
        &self,
        pay_base_amount: U256,
    ) -> Result<(U256, U256, RState, U256), SimulationError> {
        let (mut receive_quote_amount, new_r_state) = self.sell_base_token(pay_base_amount)?;
        let mt_fee = mul_floor(receive_quote_amount, self.mt_fee_rate)?;
        let lp_fee = mul_floor(receive_quote_amount, self.lp_fee_rate)?;
        receive_quote_amount = receive_quote_amount - lp_fee - mt_fee;
        Ok((receive_quote_amount, mt_fee, new_r_state, self.b0))
    }

    /// Queries the amount of base token received when selling a quote token.
    /// Returns the amount of base token received, the market maker fee, the new RState, and the
    /// quote target.
    ///
    /// # Arguments
    /// * `pay_quote_amount`: The amount of quote token to sell.
    /// # Returns
    /// * `Result<(U256, U256, RState, U256), SimulationError>`: A tuple containing the amount of
    ///   base token received, the market maker fee, the new RState, and the quote target. Returns
    ///   an error if the operation fails.
    /// # Errors
    /// * `SimulationError`: If the operation fails due to an error in the simulation, such as an
    ///   invalid input or a mathematical error.
    fn query_sell_quote_token(
        &self,
        pay_quote_amount: U256,
    ) -> Result<(U256, U256, RState, U256), SimulationError> {
        let (mut receive_base_amount, new_r_state) = self.sell_quote_token(pay_quote_amount)?;
        let mt_fee = mul_floor(receive_base_amount, self.mt_fee_rate)?;
        receive_base_amount -= mt_fee;
        let lp_fee = mul_floor(receive_base_amount, self.lp_fee_rate)?;
        receive_base_amount -= lp_fee;
        Ok((receive_base_amount, mt_fee, new_r_state, self.q0))
    }
}

impl ProtocolSim for DodoV2State {
    fn fee(&self) -> f64 {
        let sum = self.mt_fee_rate + self.lp_fee_rate;
        let fee_times_100 = sum * U256::from(100u64);
        u256_to_f64(fee_times_100) / u256_to_f64(ONE)
    }

    fn spot_price(&self, a: &Token, b: &Token) -> Result<f64, SimulationError> {
        if a < b {
            let token_correction = 10f64.powi(a.decimals as i32 - b.decimals as i32);
            Ok((u256_to_f64(self.q) / u256_to_f64(self.b)) * token_correction)
        } else {
            let token_correction = 10f64.powi(b.decimals as i32 - a.decimals as i32);
            Ok((u256_to_f64(self.b) / u256_to_f64(self.q)) * token_correction)
        }
    }

    fn get_amount_out(
        &self,
        amount_in: BigUint,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountOutResult, SimulationError> {
        let token_in_address = Address::from_slice(&token_in.address);
        let token_out_address = Address::from_slice(&token_out.address);

        let mut new_state = self.clone();
        let amount_in_u256 = biguint_to_u256(&amount_in);

        let get_balance = |addr: &Address| {
            self.balances
                .get(addr)
                .copied()
                .ok_or_else(|| {
                    SimulationError::InvalidInput(
                        format!("Token address {addr} not found in balances"),
                        None,
                    )
                })
        };

        if token_in.address == self.base_token {
            let base_balance = safe_sub_u256(
                safe_add_u256(get_balance(&token_in_address)?, amount_in_u256)?,
                self.mt_fee_base,
            )?;
            let mut quote_balance = get_balance(&token_out_address)?;

            let base_input = safe_sub_u256(base_balance, self.b)?;
            let (receive_quote_amount, mt_fee, new_r_state, new_base_target) =
                self.query_sell_base_token(base_input)?;

            quote_balance = safe_sub_u256(quote_balance, receive_quote_amount)?;
            let quote_reserve = safe_sub_u256(quote_balance, self.mt_fee_quote)?;

            new_state.mt_fee_quote = safe_add_u256(new_state.mt_fee_quote, mt_fee)?;

            if new_state.r != new_r_state {
                new_state.b0 = new_base_target;
                new_state.r = new_r_state;
            }

            new_state.b = base_balance;
            new_state.q = quote_reserve;

            new_state
                .balances
                .insert(token_in_address, base_balance);
            new_state
                .balances
                .insert(token_out_address, quote_balance);

            Ok(GetAmountOutResult::new(
                u256_to_biguint(receive_quote_amount),
                u256_to_biguint(U256::from(128000)),
                Box::new(new_state),
            ))
        } else if token_in.address == self.quote_token {
            let quote_balance = safe_sub_u256(
                safe_add_u256(get_balance(&token_in_address)?, amount_in_u256)?,
                self.mt_fee_quote,
            )?;
            let mut base_balance = get_balance(&token_out_address)?;

            let quote_input = safe_sub_u256(quote_balance, self.q)?;
            let (receive_base_amount, mt_fee, new_r_state, new_quote_target) =
                self.query_sell_quote_token(quote_input)?;

            base_balance = safe_sub_u256(base_balance, receive_base_amount)?;
            let base_reserve = safe_sub_u256(base_balance, self.mt_fee_base)?;

            new_state.mt_fee_base = safe_add_u256(new_state.mt_fee_base, mt_fee)?;

            if new_state.r != new_r_state {
                new_state.q0 = new_quote_target;
                new_state.r = new_r_state;
            }

            new_state.b = base_reserve;
            new_state.q = quote_balance;

            new_state
                .balances
                .insert(token_in_address, quote_balance);
            new_state
                .balances
                .insert(token_out_address, base_balance);

            Ok(GetAmountOutResult::new(
                u256_to_biguint(receive_base_amount),
                u256_to_biguint(U256::from(116000)),
                Box::new(new_state),
            ))
        } else {
            Err(SimulationError::InvalidInput("Invalid token input".to_string(), None))
        }
    }

    fn get_limits(
        &self,
        _sell_token: Bytes,
        _buy_token: Bytes,
    ) -> Result<(BigUint, BigUint), SimulationError> {
        todo!()
    }

    fn delta_transition(
        &mut self,
        delta: ProtocolStateDelta,
        _tokens: &HashMap<Bytes, Token>,
        balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        if let Some(base_reserve) = delta.updated_attributes.get("B") {
            self.b = U256::from_be_slice(base_reserve);
        };
        if let Some(quote_reserve) = delta.updated_attributes.get("Q") {
            self.q = U256::from_be_slice(quote_reserve);
        };
        if let Some(base_target) = delta.updated_attributes.get("B0") {
            self.b0 = U256::from_be_slice(base_target);
        };
        if let Some(quote_target) = delta.updated_attributes.get("Q0") {
            self.q0 = U256::from_be_slice(quote_target);
        };
        if let Some(r) = delta.updated_attributes.get("R") {
            let r_uint = U256::from_be_slice(r);
            self.r = RState::try_from(r_uint)
                .map_err(|_| TransitionError::DecodeError("Unsupported R value".to_string()))?;
        };
        if let Some(i) = delta.updated_attributes.get("I") {
            self.i = U256::from_be_slice(i);
        };
        if let Some(k) = delta.updated_attributes.get("K") {
            self.k = U256::from_be_slice(k);
        };
        if let Some(lp_fee_rate) = delta
            .updated_attributes
            .get("LP_FEE_RATE")
        {
            self.lp_fee_rate = U256::from_be_slice(lp_fee_rate);
        };
        if let Some(mt_fee_rate) = delta
            .updated_attributes
            .get("MT_FEE_RATE")
        {
            self.mt_fee_rate = U256::from_be_slice(mt_fee_rate);
        };
        if let Some(mt_fee_quote) = delta
            .updated_attributes
            .get("MT_FEE_QUOTE")
        {
            self.mt_fee_quote = U256::from_be_slice(mt_fee_quote);
        };
        if let Some(mt_fee_base) = delta
            .updated_attributes
            .get("MT_FEE_BASE")
        {
            self.mt_fee_base = U256::from_be_slice(mt_fee_base);
        };

        let component_balances = balances
            .component_balances
            .get(&self.component_id)
            .ok_or_else(|| {
                TransitionError::DecodeError(format!(
                    "Component ID {} not found",
                    self.component_id
                ))
            })?;
        if let Some(balance) = component_balances.get(&self.base_token) {
            let balance_u256 = U256::from_be_slice(balance);
            let address = Address::from_slice(&self.base_token);
            self.balances
                .insert(address, balance_u256);
        }

        if let Some(balance) = component_balances.get(&self.quote_token) {
            let balance_u256 = U256::from_be_slice(balance);
            let address = Address::from_slice(&self.quote_token);
            self.balances
                .insert(address, balance_u256);
        }

        Ok(())
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
            self.i == other_state.i &&
                self.k == other_state.k &&
                self.b == other_state.b &&
                self.q == other_state.q &&
                self.b0 == other_state.b0 &&
                self.q0 == other_state.q0 &&
                self.r == other_state.r &&
                self.lp_fee_rate == other_state.lp_fee_rate &&
                self.mt_fee_rate == other_state.mt_fee_rate &&
                self.mt_fee_quote == other_state.mt_fee_quote &&
                self.mt_fee_base == other_state.mt_fee_base &&
                self.base_token == other_state.base_token &&
                self.quote_token == other_state.quote_token &&
                self.balances == other_state.balances
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, str::FromStr};

    use alloy::primitives::Address;
    use num_bigint::BigUint;
    use revm::primitives::U256;
    use tycho_common::{
        dto::ProtocolStateDelta,
        models::{token::Token, Chain},
        Bytes,
    };

    use crate::{
        evm::protocol::dodo_v2::state::{DodoV2State, RState},
        models::Balances,
        protocol::state::ProtocolSim,
    };

    struct TestEnv {
        state: DodoV2State,
        base_token: Token,
        quote_token: Token,
    }

    #[allow(clippy::too_many_arguments)]
    fn build_test_state(
        i: &str,
        k: &str,
        b: &str,
        q: &str,
        b0: &str,
        q0: &str,
        lp_fee_rate: &str,
        mt_fee_rate: &str,
        mt_fee_quote: &str,
        mt_fee_base: &str,
        base_balance: &str,
        quote_balance: &str,
        r: RState,
    ) -> TestEnv {
        let base_token = Token::new(
            &Bytes::from_str("0xdAC17F958D2ee523a2206206994597C13D831ec7").unwrap(),
            "USDT",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let quote_token = Token::new(
            &Bytes::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap(),
            "USDC",
            6,
            0,
            &[Some(10_000)],
            Chain::Ethereum,
            100,
        );
        let component_balance = HashMap::from_iter([
            (Address::from_slice(&base_token.address), U256::from_str(base_balance).unwrap()),
            (Address::from_slice(&quote_token.address), U256::from_str(quote_balance).unwrap()),
        ]);

        let state = DodoV2State::new(
            U256::from_str(i).unwrap(),
            U256::from_str(k).unwrap(),
            U256::from_str(b).unwrap(),
            U256::from_str(q).unwrap(),
            U256::from_str(b0).unwrap(),
            U256::from_str(q0).unwrap(),
            r,
            U256::from_str(lp_fee_rate).unwrap(),
            U256::from_str(mt_fee_rate).unwrap(),
            U256::from_str(mt_fee_quote).unwrap(),
            U256::from_str(mt_fee_base).unwrap(),
            base_token.address.clone(),
            quote_token.address.clone(),
            component_balance,
            "dodo_v2_test_component".to_string(),
        )
        .unwrap();

        TestEnv { state, base_token, quote_token }
    }

    #[test]
    fn test_fee() {
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "673062485699",
            "750288026085",
            "711890324071",
            "711460008520",
            "6400000000000",
            "1600000000000",
            "95863902",
            "95361908",
            "673157847607",
            "750383889987",
            RState::AboveOne,
        );

        let fee = env.state.fee();
        assert_eq!(fee, 0.0008);
    }

    #[test]
    fn test_sell_quote_token_only() {
        // reference tx trace: 0x8ab71d02d0d29958a619757ee64225b19d9f34c6043a680efd42def4e0c57076
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "673062485699",
            "750288026085",
            "711890324071",
            "711460008520",
            "6400000000000",
            "1600000000000",
            "95863902",
            "95361908",
            "673157847607",
            "750383889987",
            RState::AboveOne,
        );

        let pay_quote_amount = U256::from_str("1475279565").unwrap();
        let (receive_base_amount, new_r_state) = env
            .state
            .sell_quote_token(pay_quote_amount)
            .unwrap();
        assert_eq!(receive_base_amount, U256::from_str("1475265266").unwrap());
        assert_eq!(new_r_state, RState::AboveOne);
    }

    #[test]
    fn test_query_sell_quote_token_only() {
        // reference tx trace: 0x8ab71d02d0d29958a619757ee64225b19d9f34c6043a680efd42def4e0c57076
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "673062485699",
            "750288026085",
            "711890324071",
            "711460008520",
            "6400000000000",
            "1600000000000",
            "95863902",
            "95361908",
            "673157847607",
            "750383889987",
            RState::AboveOne,
        );

        let pay_quote_amount = U256::from_str("1475279565").unwrap();
        let (amount, mt_fee, r_state, q0) = env
            .state
            .query_sell_quote_token(pay_quote_amount)
            .unwrap();
        assert_eq!(amount, U256::from_str("1475253465").unwrap());
        assert_eq!(mt_fee, U256::from_str("2360").unwrap());
        assert_eq!(r_state, RState::AboveOne);
        assert_eq!(q0, U256::from_str("711460008520").unwrap());
    }

    #[test]
    fn test_get_amount_out_quote_to_base() {
        // reference tx trace: 0x8ab71d02d0d29958a619757ee64225b19d9f34c6043a680efd42def4e0c57076
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "673062485699",
            "750288026085",
            "711890324071",
            "711460008520",
            "6400000000000",
            "1600000000000",
            "95863902",
            "95361908",
            "673157847607",
            "750383889987",
            RState::AboveOne,
        );

        let pay_quote_amount = BigUint::from_str("1475279565").unwrap();
        let res = env
            .state
            .get_amount_out(pay_quote_amount, &env.quote_token, &env.base_token)
            .unwrap();
        assert_eq!(res.amount, BigUint::from_str("1475253465").unwrap());
    }

    #[test]
    fn test_sell_base_token_only() {
        // reference tx trace: 0x2d83a1740f5a2eef4f3a423f1553619f0c2c64b52c8fc9be29ad974597884112
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "670745193540",
            "752725766257",
            "711943314208",
            "711527443153",
            "6400000000000",
            "1600000000000",
            "110733284",
            "110230253",
            "670855423793",
            "752836499541",
            RState::AboveOne,
        );

        let pay_base_amount = U256::from_str("5415998652").unwrap();
        let (receive_quote_amount, new_r_state) = env
            .state
            .sell_base_token(pay_base_amount)
            .unwrap();
        assert_eq!(receive_quote_amount, U256::from_str("5416049601").unwrap());
        assert_eq!(new_r_state, RState::AboveOne);
    }

    #[test]
    fn test_query_sell_base_token_only() {
        // reference tx trace: 0x2d83a1740f5a2eef4f3a423f1553619f0c2c64b52c8fc9be29ad974597884112
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "670745193540",
            "752725766257",
            "711943314208",
            "711527443153",
            "6400000000000",
            "1600000000000",
            "110733284",
            "110230253",
            "670855423793",
            "752836499541",
            RState::AboveOne,
        );

        let pay_base_amount = U256::from_str("5415998652").unwrap();
        let (amount, mt_fee, r_state, q0) = env
            .state
            .query_sell_base_token(pay_base_amount)
            .unwrap();
        assert_eq!(amount, U256::from_str("5416006274").unwrap());
        assert_eq!(mt_fee, U256::from_str("8665").unwrap());
        assert_eq!(r_state, RState::AboveOne);
        assert_eq!(q0, U256::from_str("711943314208").unwrap());
    }

    #[test]
    fn test_get_amount_out_base_to_quote() {
        // reference tx trace: 0x2d83a1740f5a2eef4f3a423f1553619f0c2c64b52c8fc9be29ad974597884112
        let env = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "670745193540",
            "752725766257",
            "711943314208",
            "711527443153",
            "6400000000000",
            "1600000000000",
            "110733284",
            "110230253",
            "670855423793",
            "752836499541",
            RState::AboveOne,
        );

        let pay_base_amount = BigUint::from_str("5415998652").unwrap();
        let res = env
            .state
            .get_amount_out(pay_base_amount, &env.base_token, &env.quote_token)
            .unwrap();
        assert_eq!(res.amount, BigUint::from_str("5416006274").unwrap());
    }

    #[test]
    fn test_delta_transition() {
        let before_transition_base_balance = 670855423793u64;
        let before_transition_quote_balance = 752836499541u64;

        let mut state = build_test_state(
            "1000000000000000000",
            "80000000000000",
            "670745193540",
            "752725766257",
            "711943314208",
            "711527443153",
            "6400000000000",
            "1600000000000",
            "110733284",
            "110230253",
            &before_transition_base_balance.to_string(),
            &before_transition_quote_balance.to_string(),
            RState::AboveOne,
        )
        .state;

        let attributes: HashMap<String, Bytes> = [
            ("B".to_string(), Bytes::from(1000u64.to_be_bytes().to_vec())),
            ("Q".to_string(), Bytes::from(2000u64.to_be_bytes().to_vec())),
            (
                "B0".to_string(),
                Bytes::from(
                    711_943_314_208u64
                        .to_be_bytes()
                        .to_vec(),
                ),
            ),
            (
                "Q0".to_string(),
                Bytes::from(
                    711_527_443_153u64
                        .to_be_bytes()
                        .to_vec(),
                ),
            ),
            (
                "R".to_string(),
                Bytes::from(
                    (RState::BelowOne as u8)
                        .to_be_bytes()
                        .to_vec(),
                ),
            ),
            (
                "I".to_string(),
                Bytes::from(
                    1_000_000_000_000u64
                        .to_be_bytes()
                        .to_vec(),
                ),
            ),
            ("K".to_string(), Bytes::from(80_000_000_000u64.to_be_bytes().to_vec())),
            ("LP_FEE_RATE".to_string(), Bytes::from(6_400_000_000u64.to_be_bytes().to_vec())),
            ("MT_FEE_RATE".to_string(), Bytes::from(1_600_000_000u64.to_be_bytes().to_vec())),
            ("MT_FEE_QUOTE".to_string(), Bytes::from(110_733_284u64.to_be_bytes().to_vec())),
            ("MT_FEE_BASE".to_string(), Bytes::from(110_230_253u64.to_be_bytes().to_vec())),
        ]
        .into_iter()
        .collect();

        let delta = ProtocolStateDelta {
            component_id: "dodo_v2_test_component".to_string(),
            updated_attributes: attributes,
            deleted_attributes: Default::default(),
        };

        let updated_base_balance = 670855423803u64;
        let updated_quote_balance = 752836499531u64;
        let balances = Balances {
            component_balances: HashMap::from_iter([(
                "dodo_v2_test_component".to_string(),
                HashMap::from_iter([
                    (
                        state.base_token.clone(),
                        Bytes::from(
                            updated_base_balance
                                .to_be_bytes()
                                .to_vec(),
                        ),
                    ),
                    (
                        state.quote_token.clone(),
                        Bytes::from(
                            updated_quote_balance
                                .to_be_bytes()
                                .to_vec(),
                        ),
                    ),
                ]),
            )]),
            account_balances: Default::default(),
        };

        state
            .delta_transition(delta, &HashMap::new(), &balances)
            .expect("Failed to transition state");

        let after_transition_base_balance = state
            .balances
            .get(&Address::from_slice(&state.base_token))
            .expect("Base token balance not found");

        let after_transition_quote_balance = state
            .balances
            .get(&Address::from_slice(&state.quote_token))
            .expect("Quote token balance not found");

        assert_eq!(*after_transition_base_balance, U256::from(updated_base_balance));
        assert_eq!(*after_transition_quote_balance, U256::from(updated_quote_balance));
        assert_eq!(state.b, U256::from(1000u64));
        assert_eq!(state.q, U256::from(2000u64));
        assert_eq!(state.r, RState::BelowOne);
    }
}
