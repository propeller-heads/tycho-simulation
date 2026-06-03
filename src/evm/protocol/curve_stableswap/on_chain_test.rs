//! On-chain verification tests for Curve StableSwap.
//!
//! These tests call a real Curve pool on mainnet and compare the output
//! with our native implementation. Run with:
//!
//!   RPC_URL=https://eth-mainnet.alchemyapi.io/v2/YOUR_KEY cargo test -p tycho-simulation -- curve_stableswap::on_chain_test --ignored
//!
//! Requires the `evm` feature (enabled by default).
use std::str::FromStr;

use alloy::{
    primitives::{Address, U256},
    providers::ProviderBuilder,
    sol,
};
use num_bigint::BigUint;
use tycho_common::{
    models::{token::Token, Chain},
    simulation::protocol_sim::ProtocolSim,
    Bytes,
};

use super::{math, state::CurveStableSwapState};

sol! {
    #[sol(rpc)]
    interface ICurvePool {
        function get_dy(int128 i, int128 j, uint256 dx) external view returns (uint256);
        function balances(uint256 i) external view returns (uint256);
        function A() external view returns (uint256);
        function fee() external view returns (uint256);
    }
}

/// FRAX/USDC plain 2-token StableSwap pool on Ethereum mainnet.
/// FRAX = token0 (18 decimals), USDC = token1 (6 decimals).
const FRAX_USDC_POOL: &str = "0xDcEF968d416a41Cdac0ED8702fAC8128A64241A2";

fn frax_token() -> Token {
    Token::new(
        &Bytes::from_str("0x853d955aCEf822Db058eb8505911ED77F175b99e").unwrap(),
        "FRAX",
        18,
        0,
        &[Some(10_000)],
        Chain::Ethereum,
        100,
    )
}

fn usdc_token() -> Token {
    Token::new(
        &Bytes::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap(),
        "USDC",
        6,
        0,
        &[Some(10_000)],
        Chain::Ethereum,
        100,
    )
}

#[tokio::test]
#[ignore = "requires RPC_URL env var pointing to Ethereum mainnet"]
async fn verify_against_on_chain_frax_usdc() {
    let rpc_url = std::env::var("RPC_URL").expect("RPC_URL must be set");
    let provider = ProviderBuilder::new().connect_http(
        rpc_url
            .parse()
            .expect("invalid RPC_URL"),
    );

    let pool_address = Address::from_str(FRAX_USDC_POOL).unwrap();
    let pool = ICurvePool::new(pool_address, &provider);

    // Read on-chain state
    let reserve0 = pool
        .balances(U256::from(0))
        .call()
        .await
        .expect("balances(0)");
    let reserve1 = pool
        .balances(U256::from(1))
        .call()
        .await
        .expect("balances(1)");
    let amp = pool.A().call().await.expect("A()");
    let fee = pool.fee().call().await.expect("fee()");

    println!("On-chain state:");
    println!("  reserve0 (FRAX): {reserve0}");
    println!("  reserve1 (USDC): {reserve1}");
    println!("  A: {amp}");
    println!("  fee: {fee}");

    // A() returns raw A. Internal math requires A * A_PRECISION.
    let amp_precise = amp * math::A_PRECISION;

    // Build our native state
    let state = CurveStableSwapState::new(
        reserve0,
        reserve1,
        amp_precise,
        fee,
        math::rate_from_decimals(18).unwrap(),
        math::rate_from_decimals(6).unwrap(),
    );

    // Test multiple swap amounts
    let test_amounts_frax: Vec<U256> = vec![
        U256::from(1_000_000_000_000_000_000u128),       // 1 FRAX
        U256::from(1_000_000_000_000_000_000_000u128),   // 1,000 FRAX
        U256::from(100_000_000_000_000_000_000_000u128), // 100,000 FRAX
    ];

    println!("\nFRAX -> USDC swaps:");
    for amount_in in &test_amounts_frax {
        // On-chain result
        let on_chain = pool
            .get_dy(0.into(), 1.into(), *amount_in)
            .call()
            .await
            .expect("get_dy");

        // Our result
        let our_result = state
            .get_amount_out(
                BigUint::from_bytes_be(&amount_in.to_be_bytes::<32>()),
                &frax_token(),
                &usdc_token(),
            )
            .expect("get_amount_out");

        let our_amount = U256::from_be_slice(&our_result.amount.to_bytes_be());
        println!("  amount_in={amount_in}, on_chain={on_chain}, ours={our_amount}");
        assert_eq!(our_amount, on_chain, "Output mismatch for amount_in={amount_in}");
    }

    // Test reverse direction: USDC -> FRAX
    let test_amounts_usdc: Vec<U256> = vec![
        U256::from(1_000_000u128),       // 1 USDC
        U256::from(1_000_000_000u128),   // 1,000 USDC
        U256::from(100_000_000_000u128), // 100,000 USDC
    ];

    println!("\nUSDC -> FRAX swaps:");
    for amount_in in &test_amounts_usdc {
        let on_chain = pool
            .get_dy(1.into(), 0.into(), *amount_in)
            .call()
            .await
            .expect("get_dy");

        let our_result = state
            .get_amount_out(
                BigUint::from_bytes_be(&amount_in.to_be_bytes::<32>()),
                &usdc_token(),
                &frax_token(),
            )
            .expect("get_amount_out");

        let our_amount = U256::from_be_slice(&our_result.amount.to_bytes_be());
        println!("  amount_in={amount_in}, on_chain={on_chain}, ours={our_amount}");
        assert_eq!(our_amount, on_chain, "Output mismatch for amount_in={amount_in}");
    }

    // --- Verify spot_price() ---
    // For a stablecoin pool, spot price should be in a reasonable range (0.5 - 2.0)
    let price_frax_in_usdc = state
        .spot_price(&frax_token(), &usdc_token())
        .unwrap();
    let price_usdc_in_frax = state
        .spot_price(&usdc_token(), &frax_token())
        .unwrap();
    println!("\nspot_price:");
    println!("  FRAX priced in USDC: {price_frax_in_usdc}");
    println!("  USDC priced in FRAX: {price_usdc_in_frax}");
    // Prices must be positive and finite
    assert!(price_frax_in_usdc > 0.0 && price_frax_in_usdc.is_finite());
    assert!(price_usdc_in_frax > 0.0 && price_usdc_in_frax.is_finite());
    // Prices should be approximate inverses (within 1% for analytical formula)
    let product = price_frax_in_usdc * price_usdc_in_frax;
    println!("  product (should be ~1.0): {product}");
    assert!(
        product > 0.99 && product < 1.01,
        "spot prices should be approximate inverses, product={product}"
    );

    // --- Verify get_limits() ---
    let frax_addr = Bytes::from_str("0x853d955aCEf822Db058eb8505911ED77F175b99e").unwrap();
    let usdc_addr = Bytes::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48").unwrap();

    let (max_in_frax, max_out_usdc) = state
        .get_limits(frax_addr.clone(), usdc_addr.clone())
        .unwrap();
    let (max_in_usdc, max_out_frax) = state
        .get_limits(usdc_addr, frax_addr)
        .unwrap();
    println!("\nget_limits:");
    println!("  FRAX->USDC: max_in={max_in_frax}, max_out={max_out_usdc}");
    println!("  USDC->FRAX: max_in={max_in_usdc}, max_out={max_out_frax}");

    // max_out must not exceed the output reserve
    let reserve1_biguint = BigUint::from_bytes_be(&reserve1.to_be_bytes::<32>());
    let reserve0_biguint = BigUint::from_bytes_be(&reserve0.to_be_bytes::<32>());
    assert!(
        max_out_usdc <= reserve1_biguint,
        "max_out USDC ({max_out_usdc}) exceeds reserve1 ({reserve1_biguint})"
    );
    assert!(
        max_out_frax <= reserve0_biguint,
        "max_out FRAX ({max_out_frax}) exceeds reserve0 ({reserve0_biguint})"
    );

    // get_amount_out at max_in should succeed (not exceed pool capacity)
    let result_at_limit = state.get_amount_out(max_in_frax, &frax_token(), &usdc_token());
    assert!(result_at_limit.is_ok(), "get_amount_out should succeed at max_in");

    println!("\nAll on-chain verifications passed!");
}
