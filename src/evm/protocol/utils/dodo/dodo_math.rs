use alloy::primitives::U256;

use crate::{
    evm::protocol::{
        safe_math::{safe_add_u256, safe_div_u256, safe_mul_u256, safe_sub_u256},
        utils::dodo::decimal_math::{div_ceil, div_floor, mul_floor, sqrt, ONE, ONE2},
    },
    protocol::errors::SimulationError,
};

pub(crate) fn solve_quadratic_function_for_target(
    v1: U256,
    delta: U256,
    i: U256,
    k: U256,
) -> Result<U256, SimulationError> {
    if k.is_zero() {
        return safe_add_u256(v1, mul_floor(i, delta)?)
    };
    // V0 = V1*(1+(sqrt-1)/2k)
    // sqrt = √(1+4kidelta/V1)
    // premium = 1+(sqrt-1)/2k
    // uint256 sqrt = (4 * k).mul(i).mul(delta).div(V1).add(DecimalMath.ONE2).sqrt();
    if v1.is_zero() {
        return Ok(U256::ZERO)
    };

    let sqrt_result;
    let ki = safe_mul_u256(U256::from(4), safe_mul_u256(k, i)?)?;

    if ki.is_zero() {
        sqrt_result = ONE;
    } else if safe_div_u256(safe_mul_u256(ki, delta)?, ki)? == delta {
        let ratio = safe_add_u256(safe_div_u256(safe_mul_u256(ki, delta)?, v1)?, ONE2)?;
        sqrt_result = sqrt(ratio)?;
    } else {
        let ratio = safe_add_u256(safe_mul_u256(safe_div_u256(ki, v1)?, delta)?, ONE2)?;
        sqrt_result = sqrt(ratio)?;
    }
    let premium = safe_add_u256(
        div_floor(safe_sub_u256(sqrt_result, ONE)?, safe_mul_u256(U256::from(2), k)?)?,
        ONE,
    )?;
    // V0 is greater than or equal to V1 according to the solution
    mul_floor(v1, premium)
}

pub(crate) fn solve_quadratic_function_for_trade(
    v0: U256,
    v1: U256,
    delta: U256,
    i: U256,
    k: U256,
) -> Result<U256, SimulationError> {
    if v0.is_zero() {
        return Err(SimulationError::FatalError("TARGET_IS_ZERO".to_string()));
    }
    if delta.is_zero() {
        return Ok(U256::ZERO);
    }
    if k.is_zero() {
        let m = mul_floor(i, delta)?;
        return if m > v1 { Ok(v1) } else { Ok(m) };
    }
    if k == ONE {
        // if k==1
        // Q2=Q1/(1+ideltaBQ1/Q0/Q0)
        // temp = ideltaBQ1/Q0/Q0
        // Q2 = Q1/(1+temp)
        // Q1-Q2 = Q1*(1-1/(1+temp)) = Q1*(temp/(1+temp))
        // uint256 temp = i.mul(delta).mul(V1).div(V0.mul(V0));
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
                safe_div_u256(safe_mul_u256(safe_div_u256(safe_mul_u256(delta, v1)?, v0)?, i)?, v0)?
            }
        };
        let numerator = safe_mul_u256(v1, temp)?;
        let denominator = safe_add_u256(temp, ONE)?;
        safe_div_u256(numerator, denominator)
    } else {
        // calculate -b value and sig
        // b = kQ0^2/Q1-i*deltaB-(1-k)Q1
        // part1 = (1-k)Q1 >=0
        // part2 = kQ0^2/Q1-i*deltaB >=0
        // bAbs = abs(part1-part2)
        // if part1>part2 => b is negative => bSig is false
        // if part2>part1 => b is positive => bSig is true

        // k_v0_sq = k * Q0^2
        let k_v0_sq = safe_mul_u256(k, safe_mul_u256(v0, v0)?)?;
        // part2 = kQ0^2/Q1-i*deltaB
        let part2 = safe_add_u256(safe_div_u256(k_v0_sq, v1)?, safe_mul_u256(i, delta)?)?;

        let one_minus_k = safe_sub_u256(ONE, k)?;
        let mut b_abs = safe_mul_u256(one_minus_k, v1)?;

        let b_sig: bool;
        if b_abs >= part2 {
            b_abs = safe_sub_u256(b_abs, part2)?;
            b_sig = false;
        } else {
            b_abs = safe_sub_u256(part2, b_abs)?;
            b_sig = true;
        }
        b_abs = safe_div_u256(b_abs, ONE)?;

        let square_root_result = mul_floor(
            safe_mul_u256(safe_sub_u256(ONE, k)?, U256::from(4))?,
            safe_mul_u256(mul_floor(k, v0)?, v0)?,
        )?;

        // sqrt(b*b+4(1-k)kQ0*Q0)
        let square_root = sqrt(safe_add_u256(safe_mul_u256(b_abs, b_abs)?, square_root_result)?)?;

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

        let v2 = div_ceil(numerator, denominator)?;
        if v2 > v1 {
            Ok(U256::ZERO)
        } else {
            safe_sub_u256(v1, v2)
        }
    }
}

pub(crate) fn general_integrate(
    v0: U256,
    v1: U256,
    v2: U256,
    i: U256,
    k: U256,
) -> Result<U256, SimulationError> {
    /*
        Integrate dodo curve from V1 to V2
        require V0>=V1>=V2>0
        res = (1-k)i(V1-V2)+ikV0*V0(1/V2-1/V1)
        let V1-V2=delta
        res = i*delta*(1-k+k(V0^2/V1/V2))

        i is the price of V-res trading pair

        support k=1 & k=0 case

        [round down]
    */
    if v0.is_zero() {
        return Err(SimulationError::FatalError("TARGET_IS_ZERO".to_string()));
    };
    let fair_amount = safe_mul_u256(i, safe_sub_u256(v1, v2)?)?;
    if k.is_zero() {
        return safe_div_u256(fair_amount, ONE);
    };
    let v0_v1_v2 = div_floor(safe_div_u256(safe_mul_u256(v0, v0)?, v1)?, v2)?;
    let penalty = mul_floor(k, v0_v1_v2)?;
    safe_div_u256(
        safe_mul_u256(safe_add_u256(safe_sub_u256(ONE, k)?, penalty)?, fair_amount)?,
        ONE2,
    )
}

#[cfg(test)]
mod tests {
    use std::{ops::Div, str::FromStr};

    use super::*;

    fn eth(n: u64) -> U256 {
        U256::from(n) * U256::pow(U256::from(10), U256::from(18))
    }

    #[test]
    fn test_solve_quadratic_function_for_target_solidity() {
        // k = 0
        assert_eq!(
            solve_quadratic_function_for_target(eth(100), eth(40), eth(1), U256::ZERO).unwrap(),
            eth(140)
        );

        // Q1 = 0, return 0
        assert_eq!(
            solve_quadratic_function_for_target(U256::ZERO, eth(40), eth(1), eth(1)).unwrap(),
            U256::ZERO
        );

        // i = 0, return Q1
        assert_eq!(
            solve_quadratic_function_for_target(eth(100), eth(40), U256::ZERO, eth(1)).unwrap(),
            eth(100)
        );

        // deltaB = 0, return Q1
        assert_eq!(
            solve_quadratic_function_for_target(eth(100), U256::ZERO, eth(1), eth(1)).unwrap(),
            eth(100)
        );

        // normal case
        let res = solve_quadratic_function_for_target(
            eth(100),
            eth(40),
            eth(1),
            eth(1).div(U256::from(10)),
        )
        .unwrap();
        assert_eq!(res, U256::from_str("138516480713450403000").unwrap());
    }

    #[test]
    fn test_solve_quadratic_function_for_trade_solidity() {
        // if V0 == 0 -> should revert or return error
        let res =
            solve_quadratic_function_for_trade(U256::ZERO, eth(100), eth(40), eth(1), U256::ZERO);
        assert!(res.is_err());

        // if delta == 0 -> always return 0
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), U256::ZERO, eth(1), U256::ZERO)
                .unwrap(),
            U256::ZERO
        );
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(1), U256::ZERO, eth(1))
                .unwrap(),
            U256::ZERO
        );

        // k = 0, constant price
        // Q0 = 120 ether, Q1 = 100 ether, deltaB = 40 ether, i = 1 ether;
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(40), eth(1), U256::ZERO)
                .unwrap(),
            eth(40)
        );
        // Q0 = 120 ether, Q1 = 100 ether, deltaB = 40 ether, i = 2 ether;
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(40), eth(2), U256::ZERO)
                .unwrap(),
            eth(80)
        );
        // Q0 = 120 ether, Q1 = 100 ether, deltaB = 140 ether, i = 1 ether; maximum return is Q1
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(140), eth(1), U256::ZERO)
                .unwrap(),
            eth(100)
        );
        // Q0 = 120 ether, Q1 = 100 ether, deltaB = 60 ether, i = 2 ether; maximum return is Q1
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(60), eth(2), U256::ZERO)
                .unwrap(),
            eth(100)
        );

        // k = 1
        // Q0 = 120 ether, Q1 = 100 ether, deltaB = 40 ether, i = 0;
        assert_eq!(
            solve_quadratic_function_for_trade(eth(120), eth(100), eth(40), U256::ZERO, U256::ZERO)
                .unwrap(),
            U256::ZERO
        );
        // Q0 = 100 ether, Q1 = 100 ether, deltaB = 40 ether, i = 1;
        assert_eq!(
            solve_quadratic_function_for_trade(eth(100), eth(100), eth(40), U256::ONE, eth(1))
                .unwrap(),
            U256::ZERO
        );

        // normal case, b is negative
        let res = solve_quadratic_function_for_trade(
            eth(120),
            eth(100),
            eth(40),
            U256::ONE,
            eth(1).div(U256::from(10)),
        )
        .unwrap();
        assert_eq!(res, U256::from(38u64));

        // normal case, b is positive
        let res = solve_quadratic_function_for_trade(
            eth(1),
            eth(2),
            eth(1),
            eth(1),
            eth(5).div(U256::from(10)),
        )
        .unwrap();
        assert_eq!(res, U256::from_str("1219223593595584863").unwrap());
    }

    // reference tx: 0x8ab71d02d0d29958a619757ee64225b19d9f34c6043a680efd42def4e0c57076
    #[test]
    fn test_solve_quadratic_function_for_target() {
        let v1 = U256::from_str("673062485699").unwrap();
        let delta = U256::from_str("38828017565").unwrap();
        let i = U256::from_str("1000000000000000000").unwrap();
        let k = U256::from_str("80000000000000").unwrap();

        let result = solve_quadratic_function_for_target(v1, delta, i, k);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), U256::from(711890324071u64));
    }

    #[test]
    fn test_solve_quadratic_function_for_trade() {
        let v0 = U256::from_str("711890324071").unwrap();
        let v1 = U256::from_str("673062485699").unwrap();
        let delta = U256::from_str("1475279565").unwrap();
        let i = U256::from_str("1000000000000000000").unwrap();
        let k = U256::from_str("80000000000000").unwrap();

        let result = solve_quadratic_function_for_trade(v0, v1, delta, i, k);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), U256::from(1475265266));
    }

    // reference tx: 0xba4f820f677b3df7c3d29964b6a971891b4b835b01c7c9d25d679b7c930e4594
    #[test]
    fn test_general_integrate() {
        let v0 = U256::from_str("711487341741").unwrap();
        let v1 = U256::from_str("677767654931").unwrap();
        let v2 = U256::from_str("671776002323").unwrap();
        let i = U256::from_str("1000000000000000000").unwrap();
        let k = U256::from_str("80000000000000").unwrap();

        let result = general_integrate(v0, v1, v2, i, k);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), U256::from_str("5991706200").unwrap());
    }
}
