//! Curve StableSwap 2-token pool (native implementation)
//!
//! Native Rust implementation of the Curve StableSwap invariant for 2-token plain pools
//! (e.g., DAI/USDC, USDS/USDC). ~100x faster than the VM/revm fallback.
//!
//! Reference: <https://github.com/curvefi/curve-contract/blob/master/contracts/pool-templates/base/SwapTemplateBase.vy>
mod decoder;
mod math;
#[cfg(test)]
mod on_chain_test;
pub mod state;
