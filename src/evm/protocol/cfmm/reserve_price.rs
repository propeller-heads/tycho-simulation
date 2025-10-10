use alloy::primitives::U256;
use crate::evm::protocol::u256_num::u256_to_f64;

fn one_e18() -> U256 {
    U256::from(10u64).pow(18.into())
}

/// Helper: a * b / 1e18
fn mul_div_1e18(a: U256, b: U256) -> U256 {
    a.checked_mul(b).unwrap() / one_e18()
}

/// Helper: a^2 / 1e18
fn square_div_1e18(a: U256) -> U256 {
    a.checked_mul(a).unwrap() / one_e18()
}

fn compute_k(x: U256, y: U256) -> U256 {
    let x = mul_div_1e18(x, U256::from(1)); // scaled
    let y = mul_div_1e18(y, U256::from(1)); // scaled

    let a = mul_div_1e18(x, y);
    let b = square_div_1e18(x) + square_div_1e18(y);
    mul_div_1e18(a, b)
}

fn df_dy(x: U256, y: U256) -> U256 {
    // f(x, y) = x^3y + y^3x => ∂f/∂y = x^3 + 3xy^2
    let x3 = x.checked_mul(x).unwrap().checked_mul(x).unwrap();
    let y2 = y.checked_mul(y).unwrap();
    let three_xy2 = U256::from(3) * x.checked_mul(y2).unwrap();
    x3 + three_xy2
}

fn compute_f(x: U256, y: U256) -> U256 {
    // x^3*y + y^3*x
    let x3 = x.checked_mul(x).unwrap().checked_mul(x).unwrap();
    let y3 = y.checked_mul(y).unwrap().checked_mul(y).unwrap();
    x3.checked_mul(y).unwrap() + y3.checked_mul(x).unwrap()
}

fn get_y(x0: U256, xy: U256, mut y: U256) -> U256 {
    for _ in 0..255 {
        let k = compute_f(x0, y);

        if k < xy {
            let dy = ((xy - k) * one_e18()) / df_dy(x0, y);
            if dy.is_zero() {
                if k == xy {
                    return y;
                }
                let k_plus1 = compute_f(x0, y + U256::from(1));
                if k_plus1 > xy {
                    return y + U256::from(1);
                }
                y += U256::from(1);
            } else {
                y += dy;
            }
        } else {
            let dy = ((k - xy) * one_e18()) / df_dy(x0, y);
            if dy.is_zero() {
                if k == xy || compute_f(x0, y - U256::from(1)) < xy {
                    return y;
                }
                y -= U256::from(1);
            } else {
                y -= dy;
            }
        }
    }
    panic!("!y: iteration failed")
}

fn get_amount_out(
    amount_in: U256,
    token_in_is_token0: bool,
    reserve0: U256,
    reserve1: U256,
    decimals0: U256,
    decimals1: U256,
) -> U256 {

        let xy = compute_k(reserve0, reserve1);

        let r0 = reserve0 * one_e18() / decimals0;
        let r1 = reserve1 * one_e18() / decimals1;

        let (reserve_a, reserve_b, decimals_in, decimals_out) = if token_in_is_token0 {
            (r0, r1, decimals0, decimals1)
        } else {
            (r1, r0, decimals1, decimals0)
        };

        let adjusted_amount_in = amount_in * one_e18() / decimals_in;
        let y = reserve_b - get_y(adjusted_amount_in + reserve_a, xy, reserve_b);
        y * decimals_out / one_e18()
}

pub(super) fn spot_price_from_reserves(
    r0: U256,
    r1: U256,
    token_0_decimals: u32,
    token_1_decimals: u32,
) -> f64 {
    let token_correction = 10f64.powi(token_0_decimals as i32 - token_1_decimals as i32);
    (u256_to_f64(r1) / u256_to_f64(r0)) * token_correction
}
