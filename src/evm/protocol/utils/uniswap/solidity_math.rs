use alloy_primitives::{U256, U512};

use crate::{
    evm::protocol::safe_math::{div_mod_u512, safe_div_u512, safe_mul_u512},
    protocol::errors::SimulationError,
};

pub(super) fn mul_div_rounding_up(a: U256, b: U256, denom: U256) -> Result<U256, SimulationError> {
    let a_big = U512::from(a);
    let b_big = U512::from(b);
    let product = safe_mul_u512(a_big, b_big)?;
    let (mut result, rest) = div_mod_u512(product, U512::from(denom))?;
    if rest > U512::from(0u64) {
        result += U512::from(1u64);
    }
    truncate_to_u256(result)
}

pub(super) fn mul_div(a: U256, b: U256, denom: U256) -> Result<U256, SimulationError> {
    let a_big = U512::from(a);
    let b_big = U512::from(b);
    let product = safe_mul_u512(a_big, b_big)?;
    let result = safe_div_u512(product, U512::from(denom))?;
    truncate_to_u256(result)
}

fn truncate_to_u256(value: U512) -> Result<U256, SimulationError> {
    // Access the limbs of the U512 value
    let limbs = value.as_limbs();

    // Check if the upper 256 bits are non-zero
    if limbs[4] != 0 || limbs[5] != 0 || limbs[6] != 0 || limbs[7] != 0 {
        return Err(SimulationError::FatalError("Overflow: Value exceeds 256 bits".to_string()));
    }

    // Extract the lower 256 bits
    Ok(U256::from_limbs([limbs[0], limbs[1], limbs[2], limbs[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_div_rounding_up() {
        let a = U256::from(23);
        let b = U256::from(10);
        let denom = U256::from(50);
        let res = mul_div_rounding_up(a, b, denom).unwrap();

        assert_eq!(res, U256::from(5));
    }

    #[test]
    fn test_mul_div_rounding_up_exact() {
        let a = U256::from(20);
        let b = U256::from(10);
        let denom = U256::from(40);
        let res = mul_div_rounding_up(a, b, denom).unwrap();

        assert_eq!(res, U256::from(5));
    }

    #[test]
    fn test_mul_div_rounding_up_with_remainder() {
        let a = U256::from(7);
        let b = U256::from(7);
        let denom = U256::from(10);
        let res = mul_div_rounding_up(a, b, denom).unwrap();

        assert_eq!(res, U256::from(5)); // 49/10 = 4.9, rounds up to 5
    }

    #[test]
    fn test_mul_div_rounding_up_large_numbers() {
        let a = U256::MAX;
        let b = U256::from(2);
        let denom = U256::MAX;
        let res = mul_div_rounding_up(a, b, denom).unwrap();

        assert_eq!(res, U256::from(2));
    }

    #[test]
    fn test_mul_div_rounding_up_overflow_u256() {
        let (a, b) = (U256::MAX, U256::MAX);
        let denom = U256::from(1);

        let result = mul_div_rounding_up(a, b, denom);

        assert!(matches!(result, Err(SimulationError::FatalError(_))));
    }

    #[test]
    fn test_mul_div() {
        let a = U256::from(23);
        let b = U256::from(10);
        let denom = U256::from(50);
        let res = mul_div(a, b, denom).unwrap();

        assert_eq!(res, U256::from(4));
    }

    #[test]
    fn test_mul_div_exact() {
        let a = U256::from(20);
        let b = U256::from(10);
        let denom = U256::from(40);
        let res = mul_div(a, b, denom).unwrap();

        assert_eq!(res, U256::from(5));
    }

    #[test]
    fn test_mul_div_with_remainder() {
        let a = U256::from(7);
        let b = U256::from(7);
        let denom = U256::from(10);
        let res = mul_div(a, b, denom).unwrap();

        assert_eq!(res, U256::from(4)); // 49/10 = 4.9, truncates to 4
    }

    #[test]
    fn test_mul_div_large_numbers() {
        let a = U256::MAX;
        let b = U256::from(2);
        let denom = U256::MAX;
        let res = mul_div(a, b, denom).unwrap();

        assert_eq!(res, U256::from(2));
    }

    #[test]
    fn test_mul_div_overflow_u256() {
        let (a, b) = (U256::MAX, U256::MAX);
        let denom = U256::from(1);

        let result = mul_div(a, b, denom);

        assert!(matches!(result, Err(SimulationError::FatalError(_))));
    }
}
