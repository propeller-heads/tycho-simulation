use std::{env, io::{self, Write}};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;
use tycho_simulation::{
    utils::load_all_tokens,
};

use super::{get_canonical_address, get_initial_state, SwapSimulation};

fn prompt(text: &str) -> io::Result<String> {
    print!("{}", text);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

pub async fn run_interactive() -> anyhow::Result<()> {
    let tycho_url = env::var("TYCHO_URL").unwrap_or_else(|_| "tycho-beta.propellerheads.xyz".to_string());
    let tycho_api_key = env::var("TYCHO_API_KEY").unwrap_or_else(|_| "sampletoken".to_string());
    env::var("RPC_URL").expect("RPC_URL env variable should be set");

    // Load tokens once at startup
    println!("Loading tokens...");
    let all_tokens = load_all_tokens(tycho_url.as_str(), false, Some(tycho_api_key.as_str())).await;
    
    loop {
        println!("\nSimple Solver Interactive Mode");
        println!("---------------------------");
        
        let token_in = prompt("Enter input token (e.g. WETH): ")?;
        let token_out = prompt("Enter output token (e.g. USDC): ")?;
        let amount = prompt("Enter amount to swap: ")?;
        let amount: f64 = amount.parse().unwrap_or(0.0);
        
        let tvl_threshold = prompt("Enter minimum TVL threshold [1000]: ")?;
        let tvl_threshold: f64 = tvl_threshold.parse().unwrap_or(1000.0);

        // Find tokens using the same logic as main.rs
        let token_in_addr = get_canonical_address(&token_in);
        let token_out_addr = get_canonical_address(&token_out);

        let (_, token_in) = if let Some(addr) = token_in_addr {
            all_tokens
                .iter()
                .find(|(_, t)| t.address.to_string().eq_ignore_ascii_case(addr))
                .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_in))?
        } else {
            all_tokens
                .iter()
                .find(|(_, t)| t.symbol.eq_ignore_ascii_case(&token_in))
                .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_in))?
        };

        let (_, token_out) = if let Some(addr) = token_out_addr {
            all_tokens
                .iter()
                .find(|(_, t)| t.address.to_string().eq_ignore_ascii_case(addr))
                .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_out))?
        } else {
            all_tokens
                .iter()
                .find(|(_, t)| t.symbol.eq_ignore_ascii_case(&token_out))
                .ok_or_else(|| anyhow::anyhow!("Token {} not found", token_out))?
        };

        println!("\nLooking for tokens:");
        println!("  {} ({})", token_in.symbol, token_in.address);
        println!("  {} ({})", token_out.symbol, token_out.address);

        let amount_in = BigUint::from((amount * 10f64.powi(token_in.decimals as i32)) as u64);
        let initial_state = get_initial_state(
            &tycho_url,
            &tycho_api_key,
            tvl_threshold,
            all_tokens.clone(),
        ).await?;

        let mut simulations = Vec::new();

        // Find pools that have both tokens and simulate swaps
        for (id, comp) in initial_state.new_pairs.iter() {
            // Check both token orderings
            let has_tokens = (comp.tokens.iter().any(|t| t.address == token_in.address) 
                             && comp.tokens.iter().any(|t| t.address == token_out.address))
                            || (comp.tokens.iter().any(|t| t.address == token_out.address) 
                                && comp.tokens.iter().any(|t| t.address == token_in.address));

            if has_tokens {
                if let Some(state) = initial_state.states.get(id) {
                    // Try both directions
                    let result = state.get_amount_out(amount_in.clone(), token_in, token_out)
                        .or_else(|_| state.get_amount_out(amount_in.clone(), token_out, token_in));
                    
                    if let Ok(result) = result {
                        let effective_price = result.amount.to_f64().unwrap() 
                            / (amount_in.to_f64().unwrap() * 10f64.powi(token_out.decimals as i32 - token_in.decimals as i32));
                        
                        simulations.push(SwapSimulation {
                            pool_name: format!("{:?}", state),
                            pool_address: format!("{:#x}", comp.address),
                            amount_out: result.amount,
                            gas: result.gas.to_u64().unwrap(),
                            effective_price,
                        });
                    }
                }
            }
        }

        // Sort by amount out (best to worst)
        simulations.sort_by(|a, b| b.amount_out.cmp(&a.amount_out));

        println!("\nFound {} pools with this pair. Results sorted by best price:", simulations.len());
        println!("{:<40} {:<15} {:<10} {:<10}", "Pool", "Amount Out", "Price", "Gas");
        println!("{}", "-".repeat(75));

        for sim in simulations {
            // Truncate the pool name to make it more readable
            let pool_name = if sim.pool_name.len() > 40 {
                format!("{}...", &sim.pool_name[..37])
            } else {
                sim.pool_name
            };

            println!(
                "{:<40} {:<15.6} {:<10.6} {:<10}",
                format!("{} ({})", pool_name, sim.pool_address),
                sim.amount_out.to_f64().unwrap() / 10f64.powi(token_out.decimals as i32),
                sim.effective_price,
                sim.gas
            );
        }

        let continue_prompt = prompt("\nTry another swap? (y/n): ")?;
        if continue_prompt.to_lowercase() != "y" {
            break;
        }
    }

    Ok(())
} 