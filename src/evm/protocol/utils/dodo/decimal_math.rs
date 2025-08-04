use alloy::primitives::U256;

use crate::{
    evm::protocol::safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256},
    protocol::errors::SimulationError,
};

// ONE = 10^18 = 1,000,000,000,000,000,000
pub(crate) const ONE: U256 = U256::from_limbs([
    0x0DE0_B6B3_A764_0000u64, // low 64 bits
    0,
    0,
    0,
]);

// ONE2 = 10^36 = 1,000,000,000,000,000,000,000,000,000,000,000,000
pub(crate) const ONE2: U256 = U256::from_limbs([
    0xb34b_9f10_0000_0000u64,
    0x00c0_97ce_7bc9_0715u64,
    0x0000_0000_0000_0000u64,
    0x0000_0000_0000_0000u64,
]);

pub(crate) fn reciprocal_floor(target: U256) -> Result<U256, SimulationError> {
    // uint256(10 ** 36) / target;
    safe_div_u256(ONE2, target)
}

pub(crate) fn mul_floor(target: U256, d: U256) -> Result<U256, SimulationError> {
    // target * d / (10 ** 18);
    safe_div_u256(safe_mul_u256(target, d)?, ONE)
}

pub(crate) fn div_floor(target: U256, d: U256) -> Result<U256, SimulationError> {
    // target * (10 ** 18) / d;
    safe_div_u256(safe_mul_u256(target, ONE)?, d)
}

pub(crate) fn div_ceil(target: U256, d: U256) -> Result<U256, SimulationError> {
    let a = safe_mul_u256(target, ONE)?;
    _div_ceil(a, d)
}

fn _div_ceil(a: U256, b: U256) -> Result<U256, SimulationError> {
    let quotient = safe_div_u256(a, b)?;
    let remainder = safe_sub_u256(a, safe_mul_u256(quotient, b)?)?;

    if remainder > U256::ZERO {
        safe_add_u256(quotient, U256::ONE)
    } else {
        Ok(quotient)
    }
}

pub fn log2(value: U256) -> u32 {
    if value.is_zero() {
        return 0;
    }
    (255 - value.leading_zeros()) as u32
}

pub(crate) fn sqrt(a: U256) -> Result<U256, SimulationError> {
    if a.is_zero() {
        return Ok(U256::ZERO);
    }

    let log2_a = log2(a);
    let shift = log2_a / 2;
    let mut result = U256::ONE << shift;

    for _ in 0..7 {
        let div = safe_div_u256(a, result)?;
        let sum = safe_add_u256(result, div)?;
        result = sum >> 1;
    }

    let alt = safe_div_u256(a, result)?;
    Ok(std::cmp::min(result, alt))
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::evm::protocol::safe_math::safe_sub_u256;

    #[test]
    fn test_one_constant() {
        let expected = U256::from(10).pow(U256::from(18));
        assert_eq!(ONE, expected, "ONE (10^18) does not match expected value");
    }

    #[test]
    fn test_one2_constant() {
        let expected = U256::from(10).pow(U256::from(36));
        assert_eq!(ONE2, expected, "ONE (10^36) does not match expected value");
    }

    #[test]
    fn test_log2() {
        // https://github.com/OpenZeppelin/openzeppelin-contracts/blob/3790c59623e99cb0272ddf84e6a17a5979d06b35/test/utils/math/Math.t.sol#L148
        let mut rng = rand::rng();

        for _ in 0..1000 {
            let rand_val: u128 = rng.random();
            let input = U256::from(rand_val);
            let result = log2(input);
            if input.is_zero() {
                assert_eq!(result, 0);
            } else if power_of_2_bigger(result, input) {
                assert!(power_of_2_smaller(result - 1, input));
            } else if power_of_2_smaller(result, input) {
                assert!(power_of_2_bigger(result + 1, input));
            } else {
                assert!(is_exact_power_of_two(result, input));
            }
        }
    }

    /// Returns true if 2^value > ref, or value >= 256 (since 2^256 overflows uint256 in Solidity).
    fn power_of_2_bigger(value: u32, reference: U256) -> bool {
        value >= 256 || (U256::from(1u8) << value) > reference
    }

    /// Returns true if 2^value < ref.
    fn power_of_2_smaller(value: u32, reference: U256) -> bool {
        (U256::from(1u8) << value) < reference
    }

    /// Returns true if 2^result == input.
    fn is_exact_power_of_two(result: u32, input: U256) -> bool {
        (U256::from(1) << result) == input
    }

    #[test]
    fn test_sqrt() -> Result<(), SimulationError> {
        let test_cases = [
            U256::ZERO,
            U256::ONE,
            U256::from(4),
            U256::from(10),
            U256::from(15),
            U256::from(16),
            U256::from(100),
            U256::from(1_000_000),
            U256::from(u64::MAX),
            U256::MAX,
        ];

        for &input in &test_cases {
            let result = sqrt(input)?;

            if square_bigger(result, input) {
                assert!(square_smaller(safe_sub_u256(result, U256::ONE)?, input));
            } else if square_smaller(result, input) {
                assert!(square_bigger(safe_add_u256(result, U256::ONE)?, input));
            } else {
                // perfect square
                let square = result
                    .checked_mul(result)
                    .expect("no overflow");
                assert_eq!(square, input);
            }
        }
        Ok(())
    }

    /// Returns true if `value * value` would overflow or be greater than `reference`
    fn square_bigger(value: U256, reference: U256) -> bool {
        match value.checked_mul(value) {
            Some(square) => square > reference,
            None => true, // overflow
        }
    }

    /// Returns true if `value * value < reference`
    fn square_smaller(value: U256, reference: U256) -> bool {
        match value.checked_mul(value) {
            Some(square) => square < reference,
            None => false, // overflow implies not smaller
        }
    }
}
