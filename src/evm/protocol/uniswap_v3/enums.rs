use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeeAmount {
    Lowest = 100,  // 0.01%
    Lowest2 = 200, // 0.02%
    Lowest3 = 300, // 0.03%
    Lowest4 = 400, // 0.04%
    Low = 500,     // 0.05%
    /// 0.075% fee tier with tick_spacing=15, used by AlienBase V3 on Base.
    MediumLow2 = 750,
    MediumLow = 2500,  // 0.25% [Pancakeswap V3]
    Medium = 3000,     // 0.3%
    MediumHigh = 5000, // 0.5% [Pancakeswap V3]
    High = 10_000,     // 1%
}

impl std::convert::TryFrom<i32> for FeeAmount {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            100 => Ok(FeeAmount::Lowest),
            200 => Ok(FeeAmount::Lowest2),
            300 => Ok(FeeAmount::Lowest3),
            400 => Ok(FeeAmount::Lowest4),
            500 => Ok(FeeAmount::Low),
            750 => Ok(FeeAmount::MediumLow2),
            2500 => Ok(FeeAmount::MediumLow),
            3000 => Ok(FeeAmount::Medium),
            5000 => Ok(FeeAmount::MediumHigh),
            10_000 => Ok(FeeAmount::High),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_amount_750_round_trip() {
        let fee = FeeAmount::try_from(750).expect("750 should be a valid fee tier");
        assert_eq!(fee, FeeAmount::MediumLow2);
        assert_eq!(fee as i32, 750);
    }

    #[test]
    fn test_fee_amount_unsupported_returns_err() {
        assert!(FeeAmount::try_from(999).is_err());
    }
}
