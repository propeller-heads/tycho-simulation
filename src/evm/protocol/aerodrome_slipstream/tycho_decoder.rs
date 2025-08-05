use std::collections::HashMap;
use std::future::Future;
use alloy::primitives::U256;
use tycho_client::feed::Header;
use tycho_client::feed::synchronizer::ComponentWithState;
use tycho_common::Bytes;
use crate::evm::protocol::aerodrome_slipstream::state::AerodromeSlipstreamState;
use crate::evm::protocol::utils::uniswap::i24_be_bytes_to_i32;
use crate::evm::protocol::utils::uniswap::tick_list::TickInfo;
use crate::models::Token;
use crate::protocol::errors::InvalidSnapshotError;
use crate::protocol::models::TryFromWithBlock;

impl TryFromWithBlock<ComponentWithState> for AerodromeSlipstreamState {
    type Error = InvalidSnapshotError;

    /// Decodes a `ComponentWithState` into an `AerodromeSlipstreamState`. Errors with a `InvalidSnapshotError`
    /// if the snapshot is missing any required attributes or if the fee amount is not supported.
    fn try_from_with_block(snapshot: ComponentWithState, block: Header, account_balances: &HashMap<Bytes, HashMap<Bytes, Bytes>>, all_tokens: &HashMap<Bytes, Token>) -> impl Future<Output=Result<Self, Self::Error>> + Send + Sync
    where
        Self: Sized
    {
        let liq = snapshot
            .state
            .attributes
            .get("liquidity")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("liquidity".to_string()))?
            .clone();

        // This is a hotfix because if the liquidity has never been updated after creation, it's
        // currently encoded as H256::zero(), therefore, we can't decode this as u128.
        // We can remove this once it has been fixed on the tycho side.
        let liq_16_bytes = if liq.len() == 32 {
            // Make sure it only happens for 0 values, otherwise error.
            if liq == Bytes::zero(32) {
                Bytes::from([0; 16])
            } else {
                return Err(InvalidSnapshotError::ValueError(format!(
                    "Liquidity bytes too long for {liq}, expected 16"
                )));
            }
        } else {
            liq
        };

        let liquidity = u128::from(liq_16_bytes);

        let sqrt_price = U256::from_be_slice(
            snapshot
                .state
                .attributes
                .get("sqrt_price_x96")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("sqrt_price".to_string()))?,
        );

        let fee = 0; // todo require DCI

        let tick_spacing = u16::from(
            snapshot
                .component
                .static_attributes
                .get("tick_spacing")
                .ok_or_else(|| InvalidSnapshotError::MissingAttribute("tick_spacing".to_string()))?
                .clone(),
        );

        let tick = snapshot
            .state
            .attributes
            .get("tick")
            .ok_or_else(|| InvalidSnapshotError::MissingAttribute("tick".to_string()))?
            .clone();

        // This is a hotfix because if the tick has never been updated after creation, it's
        // currently encoded as H256::zero(), therefore, we can't decode this as i32. We can
        // remove this this will be fixed on the tycho side.
        let ticks_4_bytes = if tick.len() == 32 {
            // Make sure it only happens for 0 values, otherwise error.
            if tick == Bytes::zero(32) {
                Bytes::from([0; 4])
            } else {
                return Err(InvalidSnapshotError::ValueError(format!(
                    "Tick bytes too long for {tick}, expected 4"
                )));
            }
        } else {
            tick
        };
        let tick = i24_be_bytes_to_i32(&ticks_4_bytes);

        let ticks: Result<Vec<_>, _> = snapshot
            .state
            .attributes
            .iter()
            .filter_map(|(key, value)| {
                if key.starts_with("ticks/") {
                    Some(
                        key.split('/')
                            .nth(1)?
                            .parse::<i32>()
                            .map(|tick_index| TickInfo::new(tick_index, i128::from(value.clone())))
                            .map_err(|err| InvalidSnapshotError::ValueError(err.to_string())),
                    )
                } else {
                    None
                }
            })
            .collect();

        let mut ticks = match ticks {
            Ok(ticks) if !ticks.is_empty() => ticks
                .into_iter()
                .filter(|t| t.net_liquidity != 0)
                .collect::<Vec<_>>(),
            _ => return Err(InvalidSnapshotError::MissingAttribute("tick_liquidities".to_string())),
        };

        ticks.sort_by_key(|tick| tick.index);

        Ok(AerodromeSlipstreamState::new(liquidity, sqrt_price, tick, tick_spacing, fee, ticks))
    }
}