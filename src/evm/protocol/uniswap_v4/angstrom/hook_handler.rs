use std::{any::Any, collections::HashMap, fmt::Debug};

use alloy::primitives::{aliases::U24, keccak256, Address, I128, I256, U256};
use revm::primitives::FixedBytes;
use tycho_common::{dto::ProtocolStateDelta, Bytes};

use crate::{
    evm::{
        engine_db::simulation_db::BlockHeader,
        protocol::uniswap_v4::hooks::{
            hook_handler::HookHandler,
            models::{
                AfterSwapDelta, AfterSwapParameters, AmountRanges, BeforeSwapOutput,
                BeforeSwapParameters, BeforeSwapSolOutput, SwapParams, WithGasEstimate,
            },
        },
    },
    models::{Balances, Token},
    protocol::errors::{SimulationError, TransitionError},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AngstromHookHandler {
    address: Address,
    pool_manager: Address,
    angstrom_address: Address,
    current_trading_pairs: HashMap<(Address, Address), AngstromFees>,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct AngstromFees {
    unlock: U24,
    protocol_unlock: U24,
}
impl HookHandler for AngstromHookHandler {
    fn address(&self) -> Address {
        self.angstrom_address
    }
    fn fee(
        &self,
        _: &crate::evm::protocol::uniswap_v4::state::UniswapV4State,
        _: SwapParams,
    ) -> Result<f64, SimulationError> {
        todo!("Currently tycho grabs the fee from before_swap")
    }

    fn before_swap(
        &self,
        params: BeforeSwapParameters,
        _: BlockHeader,
        _: Option<HashMap<Address, HashMap<U256, U256>>>,
        _: Option<HashMap<Address, HashMap<U256, U256>>>,
    ) -> Result<WithGasEstimate<BeforeSwapOutput>, SimulationError> {
        // start 400 for call
        let mut gas_estimate = 400;
        if !params.hook_data.is_empty() {
            // Another call + sstore + ec-recover
            gas_estimate += 10_000;
        }

        let Some(fee) = self
            .current_trading_pairs
            .get(&(params.context.currency_0, params.context.currency_1))
            .copied()
        else {
            return Err(SimulationError::FatalError(
                "pair is not enabled for trading on angstrom".to_string(),
            ));
        };

        // amount of gas to do a asset lookup;
        gas_estimate += 1000;

        let selector = &keccak256("beforeSwap(address,(address,address,uint24,int24,address),(bool,int256,uint160),bytes)")[..4];

        Ok(WithGasEstimate {
            gas_estimate,
            result: BeforeSwapOutput::new(
                BeforeSwapSolOutput {
                    selector: FixedBytes::from_slice(selector),
                    amountDelta: I256::ZERO,
                    fee: fee.unlock,
                },
                HashMap::default(),
                HashMap::default(),
            ),
        })
    }

    fn after_swap(
        &self,
        params: AfterSwapParameters,
        _: BlockHeader,
        _: Option<HashMap<Address, HashMap<U256, U256>>>,
        _: Option<HashMap<Address, HashMap<U256, U256>>>,
    ) -> Result<WithGasEstimate<AfterSwapDelta>, SimulationError> {
        let Some(fee) = self
            .current_trading_pairs
            .get(&(params.context.currency_0, params.context.currency_1))
            .copied()
        else {
            return Err(SimulationError::FatalError(
                "pair is not enabled for trading on angstrom".to_string(),
            ));
        };

        let protocol_fee = fee.protocol_unlock;

        let exact_input = params.swap_params.amount_specified < I256::ZERO;
        let target_amount = if exact_input != params.swap_params.zero_for_one {
            params.delta.amount0()
        } else {
            params.delta.amount1()
        };

        // Take absolute value of target amount
        let p_target_amount =
            if target_amount < I128::ZERO { target_amount.wrapping_neg() } else { target_amount };

        let fee_rate_e6 = I128::unchecked_from(protocol_fee.to::<u32>());
        let one_e6 = I128::unchecked_from(1_000_000);

        let protocol_fee_amount = if exact_input {
            p_target_amount * fee_rate_e6 / one_e6
        } else {
            p_target_amount * one_e6 / (one_e6 - fee_rate_e6) - p_target_amount
        };

        Ok(WithGasEstimate {
            // lookup + mint + tick updates post swap.
            gas_estimate: 15_000,
            result: protocol_fee_amount,
        })
    }

    fn spot_price(&self, _base: &Token, _quote: &Token) -> Result<f64, SimulationError> {
        Err(SimulationError::RecoverableError(
            "spot_price is not implemented for AngstromHook".to_string(),
        ))
    }

    fn get_amount_ranges(
        &self,
        _token_in: Address,
        _token_out: Address,
    ) -> Result<AmountRanges, SimulationError> {
        Err(SimulationError::RecoverableError(
            "get_amount_ranges is not implemented for AngstromHook".to_string(),
        ))
    }

    fn delta_transition(
        &mut self,
        _delta: ProtocolStateDelta,
        _tokens: &HashMap<Bytes, Token>,
        _balances: &Balances,
    ) -> Result<(), TransitionError<String>> {
        Err(TransitionError::SimulationError(SimulationError::RecoverableError(
            "delta_transition is not implemented for AngstromHook".to_string(),
        )))
    }

    fn clone_box(&self) -> Box<dyn HookHandler> {
        Box::new((*self).clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_equal(&self, other: &dyn HookHandler) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
}
