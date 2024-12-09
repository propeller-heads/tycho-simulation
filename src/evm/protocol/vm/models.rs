use alloy_primitives::U256;
use strum_macros::Display;

use crate::protocol::errors::SimulationError;

/// Represents a distinct functionality or feature that an `EVMPoolState` can support.
///
/// Each `Capability` variant corresponds to a specific functionality that influences how the
/// simulation behaves or interacts with the `EVMPoolState`.
///
/// # Variants
///
/// TODO double-check these. Very GPT-generated!!!!!!!!!!!!
///
/// - `SellSide`: Indicates the entity supports sell-side operations.
/// - `BuySide`: Indicates the entity supports buy-side operations.
/// - `PriceFunction`: Indicates the use of dynamic pricing based on a function.
/// - `FeeOnTransfer`: Indicates fees are applied during token transfers.
/// - `ConstantPrice`: Indicates the pricing mechanism is constant.
/// - `TokenBalanceIndependent`: Indicates the entity's price determination is independent of token
///   balances.
/// - `ScaledPrice`: Indicates the use of scaled pricing mechanisms.
/// - `HardLimits`: Indicates hard operational limits are in place.
/// - `MarginalPrice`: Indicates the use of marginal pricing strategies.
///
/// # Usage
///
/// Capabilities can be used to query or enforce feature sets on simulation entities.
/// For example, they can determine which operations are valid for a given pool state.
///
/// ```
/// 
/// use crate::evm::protocol::vm::models::Capability;
/// use crate::protocol::errors::SimulationError;
///
/// let capability = Capability::SellSide;
///
/// match capability {
///     Ok(Capability::SellSide) => println!("Supports sell-side operations."),
///     Ok(cap) => println!("Other capability: {:?}", cap),
///     Err(err) => println!("Error: {}", err),
/// }
/// ```
#[derive(Eq, PartialEq, Hash, Debug, Display, Clone)]
pub enum Capability {
    SellSide = 1,
    BuySide = 2,
    PriceFunction = 3,
    FeeOnTransfer = 4,
    ConstantPrice = 5,
    TokenBalanceIndependent = 6,
    ScaledPrice = 7,
    HardLimits = 8,
    MarginalPrice = 9,
}

impl Capability {
    pub fn from_u256(value: U256) -> Result<Self, SimulationError> {
        let value_as_u8 = value.to_le_bytes::<32>()[0];
        match value_as_u8 {
            1 => Ok(Capability::SellSide),
            2 => Ok(Capability::BuySide),
            3 => Ok(Capability::PriceFunction),
            4 => Ok(Capability::FeeOnTransfer),
            5 => Ok(Capability::ConstantPrice),
            6 => Ok(Capability::TokenBalanceIndependent),
            7 => Ok(Capability::ScaledPrice),
            8 => Ok(Capability::HardLimits),
            9 => Ok(Capability::MarginalPrice),
            _ => {
                Err(SimulationError::FatalError(format!("Unexpected Capability value: {}", value)))
            }
        }
    }
}
