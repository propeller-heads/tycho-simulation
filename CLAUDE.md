# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tycho Simulation is a Rust library for off-chain simulation of decentralized exchange protocols. It enables fast simulation of token swaps, arbitrage detection, and price discovery without blockchain interaction. The library supports major DEX protocols (Uniswap V2/V3/V4, PancakeSwap, Balancer, Curve, etc.) and includes Python bindings.

## Development Commands

### Building and Testing
```bash
# Format code (requires nightly toolchain)
cargo +nightly fmt

# Lint code
cargo +nightly clippy --workspace --lib --all-targets --all-features -- -D warnings

# Run tests
cargo nextest run --workspace --lib --all-targets --all-features

# Full CI check (runs all of the above)
./check.sh

# Build with specific features
cargo build --features "evm rfq"
cargo build --no-default-features --features "evm"
```

### Python Bindings
```bash
# Build Python bindings
cd tycho_simulation_py
maturin develop

# Run Python tests
pytest python/test/
```

### Examples
```bash
# Quickstart example (requires RPC_URL environment variable)
export RPC_URL=<your-rpc-url>
cargo run --release --example quickstart

# Price printer TUI
cargo run --release --example price_printer

# RFQ quickstart
cargo run --release --example rfq_quickstart
```

### Benchmarks
```bash
# Run all protocol swap benchmarks
cargo bench

# Run benchmarks with custom configuration
BENCH_N_SWAPS=1000 BENCH_TVL_THRESHOLD=5000.0 cargo bench

# Run benchmarks and generate HTML report
cargo bench -- --output-format html

# Run specific protocol benchmarks
cargo bench uniswap_v3_swaps
```

## Architecture

### Core Structure
- `src/evm/` - EVM protocol simulations using Foundry/Revm
- `src/rfq/` - Request-for-Quote protocol implementations
- `src/protocol/` - Common protocol models and errors
- `tycho_simulation_py/` - Python bindings via PyO3

### Protocol Support
**EVM Protocols**: Uniswap V2/V3/V4, PancakeSwap V2, Balancer V2/V3, Curve, Maverick V2, Ekubo (Starknet)
**RFQ Protocols**: Bebop

### Feature Flags
- `evm` (default) - EVM protocol simulations
- `rfq` (default) - Request-for-Quote protocols  
- `network_tests` - Network-dependent integration tests

### Key Dependencies
- **EVM**: foundry-evm, revm, alloy for blockchain simulation
- **Async**: tokio for async runtime
- **Python**: PyO3 for Python bindings
- **Testing**: rstest for parameterized tests, mockall for mocking

## Testing

Tests are organized by protocol with JSON snapshots for various states. Use `cargo nextest run` for faster parallel test execution. Network tests require the `network_tests` feature flag.

## Python Integration

The Python bindings expose core simulation functionality. Build with `maturin develop` and use `pytest` for testing. Python package depends on `tycho-indexer-client` for data retrieval.