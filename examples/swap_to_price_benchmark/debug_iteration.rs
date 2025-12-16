//! Debug script to compare swap_to_price vs query_supply iteration behavior
//!
//! Run with: cargo run --release --example swap_to_price_benchmark -- debug

use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::path::Path;
use tycho_common::simulation::protocol_sim::ProtocolSim;

use crate::snapshot;

/// Detailed iteration data for analysis
#[derive(Debug, Clone)]
pub struct IterationData {
    pub iteration: usize,
    pub amount_in: f64,
    pub amount_out: f64,
    pub price: f64,        // The metric being optimized (spot or trade)
    pub error_bps: f64,    // Distance from target in bps
    pub bracket_low: f64,  // Lower bound of amount bracket
    pub bracket_high: f64, // Upper bound of amount bracket
    pub t_param: f64,      // Chandrupatla t parameter
}

/// Run detailed comparison between SpotPrice and TradePrice metrics
pub async fn run_debug_comparison(snapshot_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading snapshot...");
    let loaded = snapshot::load_and_process_snapshot(snapshot_path).await?;

    // Target pool: WETH/WBTC Balancer V2
    let pool_id = "0xa6f548df93de924d73be7d25dc02554c6bd66db500020000000000000000000e";

    let state = loaded.states.get(pool_id)
        .ok_or_else(|| format!("Pool {} not found", pool_id))?;

    let component = loaded.components.get(pool_id)
        .ok_or_else(|| format!("Component {} not found", pool_id))?;

    // Find WETH and WBTC tokens
    let tokens: Vec<_> = component.tokens.iter().collect();
    println!("\nPool tokens:");
    for t in &tokens {
        println!("  {} ({}): {} decimals", t.symbol, t.address, t.decimals);
    }

    let weth = tokens.iter().find(|t| t.symbol == "WETH")
        .ok_or("WETH not found")?;
    let wbtc = tokens.iter().find(|t| t.symbol == "WBTC")
        .ok_or("WBTC not found")?;

    // Get spot price
    let spot_price = state.spot_price(wbtc, weth)?;
    println!("\nSpot price (WBTC/WETH): {:.8}", spot_price);

    // Target: +50bps
    let target_price = spot_price * 1.0050;
    println!("Target price (+50bps): {:.8}", target_price);

    // Get limits - use 95% to avoid edge cases where get_limits overestimates
    let (max_amount_in, _) = state.get_limits(weth.address.clone(), wbtc.address.clone())?;
    let max_amount_f64 = max_amount_in.to_f64().unwrap_or(f64::MAX) * 0.95;
    println!("Max amount in (95%): {:.4} WETH", max_amount_f64 / 1e18);

    println!("\n{}", "=".repeat(80));
    println!("COMPARING SPOT PRICE vs TRADE PRICE METRICS");
    println!("{}", "=".repeat(80));

    // Run with SpotPrice metric
    println!("\n>>> SPOT PRICE METRIC (swap_to_price):");
    let spot_iterations = run_chandrupatla_with_logging(
        state.as_ref(),
        weth,
        wbtc,
        target_price,
        max_amount_f64,
        false, // use_trade_price = false
    )?;

    // Run with TradePrice metric
    println!("\n>>> TRADE PRICE METRIC (query_supply):");
    let trade_iterations = run_chandrupatla_with_logging(
        state.as_ref(),
        weth,
        wbtc,
        target_price,
        max_amount_f64,
        true, // use_trade_price = true
    )?;

    // Print comparison table
    println!("\n{}", "=".repeat(80));
    println!("ITERATION COMPARISON");
    println!("{}", "=".repeat(80));

    println!("\n{:<5} {:>15} {:>15} {:>12} {:>12} {:>12} {:>12}",
             "Iter", "Amount (WETH)", "SpotPrice", "Spot Err", "TradePrice", "Trade Err", "Diff");
    println!("{}", "-".repeat(95));

    let max_iters = spot_iterations.len().max(trade_iterations.len());
    for i in 0..max_iters {
        let spot = spot_iterations.get(i);
        let trade = trade_iterations.get(i);

        let amount_str = spot.or(trade)
            .map(|d| format!("{:.6}", d.amount_in / 1e18))
            .unwrap_or_default();

        let spot_price_str = spot.map(|d| format!("{:.6}", d.price)).unwrap_or_default();
        let spot_err_str = spot.map(|d| format!("{:.4}bps", d.error_bps)).unwrap_or_default();

        let trade_price_str = trade.map(|d| format!("{:.6}", d.price)).unwrap_or_default();
        let trade_err_str = trade.map(|d| format!("{:.4}bps", d.error_bps)).unwrap_or_default();

        let diff_str = if let (Some(s), Some(t)) = (spot, trade) {
            format!("{:.4}", (s.price - t.price).abs())
        } else {
            String::new()
        };

        println!("{:<5} {:>15} {:>15} {:>12} {:>12} {:>12} {:>12}",
                 i + 1, amount_str, spot_price_str, spot_err_str, trade_price_str, trade_err_str, diff_str);
    }

    // Generate CSV for plotting
    println!("\n{}", "=".repeat(80));
    println!("CSV DATA FOR PLOTTING");
    println!("{}", "=".repeat(80));
    println!("\nSpot Price iterations:");
    println!("iteration,amount_weth,price,error_bps,bracket_low,bracket_high");
    for d in &spot_iterations {
        println!("{},{:.8},{:.8},{:.6},{:.8},{:.8}",
                 d.iteration, d.amount_in/1e18, d.price, d.error_bps, d.bracket_low/1e18, d.bracket_high/1e18);
    }

    println!("\nTrade Price iterations:");
    println!("iteration,amount_weth,price,error_bps,bracket_low,bracket_high");
    for d in &trade_iterations {
        println!("{},{:.8},{:.8},{:.6},{:.8},{:.8}",
                 d.iteration, d.amount_in/1e18, d.price, d.error_bps, d.bracket_low/1e18, d.bracket_high/1e18);
    }

    // Analysis
    println!("\n{}", "=".repeat(80));
    println!("ANALYSIS");
    println!("{}", "=".repeat(80));

    if let Some(last_spot) = spot_iterations.last() {
        println!("\nSpot Price metric:");
        println!("  Final amount: {:.6} WETH", last_spot.amount_in / 1e18);
        println!("  Final price: {:.8}", last_spot.price);
        println!("  Final error: {:.6} bps", last_spot.error_bps);
        println!("  Converged: {}", last_spot.error_bps.abs() < 0.001);
    }

    if let Some(last_trade) = trade_iterations.last() {
        println!("\nTrade Price metric:");
        println!("  Final amount: {:.6} WETH", last_trade.amount_in / 1e18);
        println!("  Final price: {:.8}", last_trade.price);
        println!("  Final error: {:.6} bps", last_trade.error_bps);
        println!("  Converged: {}", last_trade.error_bps.abs() < 0.001);
    }

    // Key insight
    println!("\n>>> KEY INSIGHT:");
    println!("The issue is that SpotPrice and TradePrice are DIFFERENT functions of amount_in.");
    println!("- SpotPrice: marginal rate at state AFTER swapping amount_in");
    println!("- TradePrice: average rate = amount_out / amount_in");
    println!("\nFor the SAME target price, they require DIFFERENT amounts to achieve it!");

    Ok(())
}

/// Run Chandrupatla algorithm with detailed logging
fn run_chandrupatla_with_logging(
    state: &dyn ProtocolSim,
    token_in: &tycho_common::models::token::Token,
    token_out: &tycho_common::models::token::Token,
    target_price: f64,
    max_amount: f64,
    use_trade_price: bool,
) -> Result<Vec<IterationData>, Box<dyn std::error::Error>> {
    let mut iterations = Vec::new();

    let decimal_adjustment = 10_f64.powi(token_in.decimals as i32 - token_out.decimals as i32);

    // Helper to calculate price at a given amount
    let calc_price = |amount: f64| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        if amount <= 0.0 {
            let spot = state.spot_price(token_out, token_in)?;
            return Ok((spot, spot)); // At 0, spot and trade are same (theoretical)
        }

        let amount_in = BigUint::from(amount as u128);
        let result = state.get_amount_out(amount_in, token_in, token_out)?;

        let amount_out_f64 = result.amount.to_f64().unwrap_or(0.0);
        let spot_price = result.new_state.spot_price(token_out, token_in)?;
        let trade_price = (amount_out_f64 / amount) * decimal_adjustment;

        Ok((spot_price, trade_price))
    };

    // Initialize bracket [0, max_amount]
    let mut a = 0.0;
    let mut b = max_amount;

    // Get function values at bracket endpoints
    let (spot_a, trade_a) = calc_price(a)?;
    let (spot_b, trade_b) = calc_price(b)?;

    let f_a = if use_trade_price { trade_a - target_price } else { spot_a - target_price };
    let f_b = if use_trade_price { trade_b - target_price } else { spot_b - target_price };

    println!("  Initial bracket: [{:.6}, {:.6}] WETH", a/1e18, b/1e18);
    println!("  f(a) = {:.8}, f(b) = {:.8}", f_a, f_b);
    println!("  At a=0: spot={:.8}, trade={:.8}", spot_a, trade_a);
    println!("  At b=max: spot={:.8}, trade={:.8}", spot_b, trade_b);

    // Check if root exists in bracket
    if f_a * f_b > 0.0 {
        println!("  WARNING: No sign change in bracket! Cannot converge.");
        println!("  Target {:.8} is outside achievable range [{:.8}, {:.8}]",
                 target_price,
                 if use_trade_price { trade_a.min(trade_b) } else { spot_a.min(spot_b) },
                 if use_trade_price { trade_a.max(trade_b) } else { spot_a.max(spot_b) });
        return Ok(iterations);
    }

    // Ensure a has the negative value
    let (mut a, mut b, mut f_a, mut f_b) = if f_a > 0.0 {
        (b, a, f_b, f_a)
    } else {
        (a, b, f_a, f_b)
    };

    let mut c = a;
    let mut f_c = f_a;
    let mut t = 0.5;

    let tolerance = 0.001 / 100.0; // 0.001%
    let max_iterations = 30;

    for iter in 1..=max_iterations {
        // Calculate new point
        let x = a + t * (b - a);
        let (spot_x, trade_x) = calc_price(x)?;
        let price_x = if use_trade_price { trade_x } else { spot_x };
        let f_x = price_x - target_price;
        let error_bps = (f_x / target_price).abs() * 10000.0;

        iterations.push(IterationData {
            iteration: iter,
            amount_in: x,
            amount_out: 0.0, // Could calculate if needed
            price: price_x,
            error_bps,
            bracket_low: a.min(b),
            bracket_high: a.max(b),
            t_param: t,
        });

        println!("  Iter {}: amount={:.6} WETH, price={:.8}, error={:.4}bps, t={:.4}",
                 iter, x/1e18, price_x, error_bps, t);

        // Check convergence
        if error_bps < tolerance * 10000.0 {
            println!("  CONVERGED at iteration {}", iter);
            break;
        }

        // Update bracket (Chandrupatla logic)
        if f_x.signum() == f_a.signum() {
            // x replaces a
            c = a;
            f_c = f_a;
            a = x;
            f_a = f_x;
        } else {
            // x replaces b, and old b becomes c
            c = b;
            f_c = f_b;
            b = a;
            f_b = f_a;
            a = x;
            f_a = f_x;
        }

        // Calculate t for next iteration (simplified Chandrupatla)
        let phi = (f_a - f_b) / (f_c - f_b);
        let xi = (a - b) / (c - b);

        // IQI criterion
        let use_iqi = phi * phi < xi && (1.0 - phi) * (1.0 - phi) < 1.0 - xi;

        if use_iqi {
            // Inverse quadratic interpolation
            let r = f_a / f_c;
            let s = f_a / f_b;
            let p = s * (r * (r - 1.0) * (c - a) + (1.0 - r) * (b - a));
            let q = (r - 1.0) * (s - 1.0) * (1.0 - r);
            t = if q.abs() > 1e-10 { p / q } else { 0.5 };
            t = t.clamp(0.0, 1.0);
        } else {
            // Bisection
            t = 0.5;
        }
    }

    Ok(iterations)
}
