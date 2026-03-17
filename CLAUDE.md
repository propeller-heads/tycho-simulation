# CLAUDE.md

## Commands

```bash
# Run all checks (format + clippy + tests)
./check.sh

# Tests
cargo nextest run --workspace --lib --all-targets --all-features
cargo nextest run --workspace --all-targets --all-features -E 'test(my_test_name)'

# Format (requires nightly)
cargo +nightly fmt --all

# Lint
cargo clippy --workspace --lib --all-targets --all-features -- -D warnings
```

## Architecture

Off-chain DeFi protocol simulation library. Computes swap outputs, spot prices, and price impact for solvers.

### Workspace
- **tycho-simulation** (root): Main library
- **tycho-integration-test**: E2E tests against live Tycho
- **tycho-test**: Shared test utilities
- **tycho_simulation_py**: Python bindings (PyO3)

### Key Modules (`src/`)
- **`protocol/`**: Core traits and models (`ProtocolComponent`, `Update`)
- **`evm/simulation.rs`**: `SimulationEngine` — runs EVM transactions via `revm`
- **`evm/engine_db/`**: Database backends (in-memory, RPC-backed)
- **`evm/stream.rs`**: Tycho feed integration — decodes live state into protocol instances
- **`evm/protocol/`**: Protocol implementations
  - **Native**: Pure Rust math (`uniswap_v2`, `uniswap_v3`, `uniswap_v4`, `ekubo`, `cowamm`, `fluid`, etc.)
  - **VM** (`vm/`): Generic Solidity adapter executed in `revm` for protocols without native impls
  - Shared: `cpmm.rs`, `clmm.rs`, `safe_math.rs`, `u256_num.rs`
- **`rfq/`**: RFQ client for off-chain market makers (Bebop, Hashflow)

### Simulation Approaches (order of preference)
1. **Native**: Pure Rust protocol math (fastest)
2. **VM**: Solidity adapter in `revm` (any EVM protocol, slower)
3. **RFQ**: Off-chain quotes via API (non-simulatable protocols)

## Conventions
- Nightly required for `cargo fmt`, stable for everything else
- `evm` and `rfq` features enabled by default
- Conventional Commits (`release.config.js`)
- `rstest`: name each case with `#[case::descriptive_name(...)]`
- `network_tests` feature gates tests requiring network access
