# Swap-to-Price Algorithm Benchmark

Benchmark for evaluating algorithms that find the swap amount needed to move a pool's spot price to a target value.

## Overview

This benchmark tests **multiple interpolation search strategies** across multiple protocols and price movements to measure:
- **Iterations**: Number of search steps needed
- **Gas**: Estimated gas cost
- **Convergence**: Accuracy of reaching target price
- **Performance**: Execution time

### Strategies Tested

- **binary**: Classic binary search (always picks midpoint)
- **linear**: Linear interpolation using price information
- **bounded_linear**: Linear interpolation clamped to 50% deviation from midpoint

## Quick Start

```bash
# 1. Set environment variables
export TYCHO_AUTH_KEY="your_api_key_here"

# 2. Create snapshot (captures live Tycho stream data)
cargo run --example swap_to_price_benchmark_benchmark --features swap_to_price,evm --release -- snapshot

# 3. Run benchmark (auto-finds latest snapshot)
cargo run --example swap_to_price_benchmark_benchmark --features swap_to_price,evm --release -- run
```

## Prerequisites

### Environment Variables

```bash
# Required: Tycho API credentials
export TYCHO_AUTH_KEY="your_api_key_here"
export TYCHO_URL="tycho-beta.propellerheads.xyz"

# Optional: For VM protocols (future use)
export RPC_URL="https://eth-mainnet.g.alchemy.com/v2/your_key"
```

### Build

```bash
cargo build --example swap_to_price_benchmark --features swap_to_price,evm --release
```

## Commands

### `snapshot` - Create Pool State Snapshot

Captures a self-contained snapshot from live Tycho stream for reproducible benchmarking.

```bash
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  snapshot [OPTIONS]
```

**Options:**

| Flag | Env Variable | Default | Description |
|------|--------------|---------|-------------|
| `--output-folder <PATH>` | - | `examples/swap_to_price_benchmark/data` | Output directory (creates snapshot_<block>.json) |
| `--tycho-url <URL>` | `TYCHO_URL` | `tycho-beta.propellerheads.xyz` | Tycho API URL |
| `--api-key <KEY>` | `TYCHO_AUTH_KEY` | - | Tycho authentication key |
| `--min-tvl <ETH>` | - | `500` | Minimum pool TVL in ETH |

**What it does:**
1. Connects to Tycho websocket stream
2. Subscribes to UniswapV2 and UniswapV3 protocols
3. Receives first FeedMessage with pool states
4. Extracts tokens from component data
5. Saves self-contained snapshot to JSON with:
   - Raw FeedMessage data
   - All tokens referenced in pools
   - Protocol list
   - Metadata (block, chain, TVL filter, timestamp)

**Example output:**
```
Creating snapshot...
INFO  Connecting to Tycho at tycho-beta.propellerheads.xyz...
INFO  Adding protocol: uniswap_v2
INFO  Adding protocol: uniswap_v3
INFO  Building stream...
INFO  Waiting for first message...
INFO  Received FeedMessage for block 23868409
INFO    Protocols: uniswap_v2, uniswap_v3
INFO    Total components: 686
INFO  Extracting tokens from feed...
INFO  Found 586 unique token addresses in components
INFO  Extracted 586 tokens total
INFO  Saving snapshot to snapshot.json...
INFO  Snapshot saved successfully!

✅ Snapshot saved successfully!
   File: examples/swap_to_price_benchmark/data/snapshot_23868409.json
   Block: 23868409
   Chain: Ethereum
   Protocols: uniswap_v2, uniswap_v3
   Components: 686
   Tokens: 586
```

**Examples:**

```bash
# Use default settings (100 ETH min TVL)
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- snapshot

# Higher TVL threshold
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  snapshot --min-tvl 500

# Custom output folder
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  snapshot --output-folder my_snapshots/
```

**Snapshot Format:**

The snapshot is a self-contained JSON file with everything needed to restore states:

```json
{
  "feed_message": { /* Raw FeedMessage from Tycho */ },
  "protocols": ["uniswap_v2", "uniswap_v3"],
  "tokens": { /* All tokens referenced in pools */ },
  "metadata": {
    "captured_at": "2025-11-24T11:25:37Z",
    "block_number": 23868409,
    "chain": "Ethereum",
    "min_tvl": 100.0,
    "total_components": 686
  }
}
```

---

### `run` - Load and Test Snapshot

Loads a snapshot, decodes all states, and displays pool information. Can also run benchmarks if swap_to_price feature is enabled.

```bash
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  run --snapshot <PATH>
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--snapshot <PATH>` | *(auto-finds latest)* | Path to snapshot file (optional) |

**What it tests:**

For each pool and token pair direction:
- **8 price movements**: +0.1, +0.5, +1, +5, +10, +50, +100, +500 bps
- **3 strategies**: binary, linear, bounded_linear
  - Tolerance: 0.001% (0.1 basis point)
  - Max iterations: 96

**Example output (without benchmark):**
```
Loading snapshot...
   File: snapshot.json
INFO  Loading snapshot from snapshot.json...
INFO  Loaded snapshot:
INFO    Block: 23868409
INFO    Protocols: uniswap_v2, uniswap_v3
INFO    Tokens: 586
INFO    Components: 686
INFO  Registering decoder for uniswap_v2
INFO  Registering decoder for uniswap_v3
INFO  Decoding snapshot...
INFO  Snapshot decoded successfully!
INFO    States: 686
INFO    New pairs: 686
INFO    Removed pairs: 0

✅ Snapshot loaded and decoded successfully!
   Block: 23868409
   Chain: Ethereum
   Captured: 2025-11-24 11:25:37 UTC
   States: 686
   Components: 686

📊 Sample pools:
   uniswap_v2 - WETH/USDC (0x0d4a11d5...)
   uniswap_v3 - WETH/USDT (0x4e68cdb1...)
   uniswap_v2 - WETH/DAI (0xa478c2c8...)
   ... and 683 more
```

**Example output (with swap_to_price benchmark):**
```
Loaded snapshot with 80 pools
Block: 23869809
Chain: Ethereum
Testing 3 strategies: binary, linear, bounded_linear

Benchmarking pool 1/80: 0xabc... (uniswap_v2)
  USDC -> WETH (spot: 0.000308, limit: 0.000350, testing 8 movements)
    +  1bps [binary        ]:   5 iters
    +  1bps [linear        ]:   3 iters
    +  1bps [bounded_linear]:   4 iters
    +  5bps [binary        ]:   6 iters
    +  5bps [linear        ]:   4 iters
    +  5bps [bounded_linear]:   5 iters
    ...

================================================================================
BENCHMARK SUMMARY
================================================================================

Configuration:
  Tolerance: 0.00%
  Max iterations: 96

Overall:
  Total scenarios: 1620
  Converged: 1600
  Not converged: 15
  Failed (errors): 5
  Convergence rate: 98.8%

By Strategy:
  Strategy           Total  Converged  Not Conv     Min    Mean  Median     P95     P99
  -------------------------------------------------------------------------------------
  binary               540        535         5      10    45.2      44      62      68
  bounded_linear       540        533         7       6    25.3      24      38      45
  linear               540        532         8       5    22.1      20      35      42

By Protocol:
  Protocol              Total  Converged  Not Conv     Min    Mean  Median     P95     P99
  -------------------------------------------------------------------------------------
  uniswap_v2              720        715         5       4     6.2       6       8       9
  uniswap_v3              900        885        15       5     8.5       8      12      15

By Price Movement:
       Bps    Total  Converged  Not Conv     Min    Mean  Median     P95     P99
  -------------------------------------------------------------------------------------
         0      270        268         2       4     5.1       5       7       8
         1      270        270         0       4     5.5       5       7       8
         5      270        270         0       5     6.2       6       8       9
        10      270        268         2       6     7.8       7      10      12
        50      270        267         3       7     8.9       8      12      14
       100      270        265         5       9    11.2      11      15      18

Worst 10 Scenarios by Iteration Count:
   Iters Strategy       Protocol         In       Out    Price(bps)
  --------------------------------------------------------------------------------
      68 binary         uniswap_v3       WBTC     DAI          500
      45 bounded_linear uniswap_v3       WETH     USDT         500
      42 linear         uniswap_v3       DAI      WBTC         500

Results saved to: examples/swap_to_price_benchmark/runs/run_20250124_143052.json
```

**Examples:**

```bash
# Load latest snapshot (auto-finds in examples/swap_to_price_benchmark/data/)
cargo run --example swap_to_price_benchmark --release -- run

# Run full benchmark on latest snapshot (requires swap_to_price feature)
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- run

# Use specific snapshot
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- run \
  --snapshot examples/swap_to_price_benchmark/data/snapshot_23868409.json
```

**Key Benefits:**
- ✅ **No reconstruction needed** - States are fully decoded `Box<dyn ProtocolSim>` objects
- ✅ **Self-contained** - Includes all tokens, no external dependencies
- ✅ **Fast loading** - ~1-2 seconds to deserialize and decode
- ✅ **Production-tested** - Uses same FeedMessage format as Tycho stream

---

## Understanding Results

### Metrics

- **Iterations**: Number of binary search steps needed
  - Lower is better
  - Typical range: 4-15 for most scenarios

- **Convergence error**: Distance from target price
  - Should be within tolerance (0.01% = 1 bps)

- **Success rate**: % of scenarios that converged
  - Should be >95%
  - Failures usually due to insufficient liquidity

- **P95/P99**: 95th and 99th percentile iterations
  - Important for worst-case performance

### JSON Output

Full results saved to `examples/swap_to_price_benchmark/runs/run_<timestamp>.json`:

```json
{
  "timestamp": "2025-01-24T14:30:52Z",
  "results": [
    {
      "strategy": "linear",
      "pool_id": "0xabc...",
      "protocol": "uniswap_v2",
      "token_in": "USDC",
      "token_out": "WETH",
      "spot_price": 0.000308,
      "target_price": 0.000309,
      "target_movement_bps": 1,
      "actual_price": 0.000309,
      "amount_in": "1000000",
      "iterations": 5,
      "gas": "120000",
      "elapsed_micros": 234,
      "convergence_error_bps": 0.5,
      "status": "success"
    }
  ],
  "summary": {
    "total_scenarios": 1620,
    "converged": 1600,
    "not_converged": 15,
    "failed": 5,
    "by_strategy": {...},
    "by_protocol": {...},
    "by_movement": {...}
  }
}
```

**Analyze with jq:**

```bash
# View summary
cat examples/swap_to_price_benchmark/runs/run_*.json | jq '.summary'

# Filter successful results
cat examples/swap_to_price_benchmark/runs/run_*.json | jq '.results[] | select(.status == "success")'

# Get average iterations by protocol
cat examples/swap_to_price_benchmark/runs/run_*.json | jq '.summary.by_protocol'
```

---

## Troubleshooting

### Issue: "No data received from protocol stream"
**Solution:**
- Verify `TYCHO_AUTH_KEY` is set and valid
- Check network connectivity to Tycho

### Issue: "Target price is above limit price"
**Expected behavior** - Pool doesn't have enough liquidity for that price movement. These scenarios are marked as failed but don't indicate algorithm issues.

### Issue: Few pools captured
**Solutions:**
- Lower `--min-tvl` threshold (try `--min-tvl 10`)
- Check that you're on the correct chain
- Verify target tokens are correct for your use case

### Issue: High iteration counts (>15)
**This may be normal for:**
- Large price movements (>100 bps)
- UniswapV3 pools with sparse tick distribution
- Low liquidity pools

If consistently high, consider:
- Adjusting tolerance in `src/swap_to_price/mod.rs`
- Testing different algorithms

---

## Configuration

### Algorithm Parameters

Edit `src/swap_to_price/mod.rs`:

```rust
pub const SWAP_TO_PRICE_TOLERANCE: f64 = 0.0001;  // 0.01%
pub const SWAP_TO_PRICE_MAX_ITERATIONS: u32 = 20;
```

### Supported Protocols

Currently hardcoded in `examples/swap_to_price/snapshot.rs`:

```rust
let protocols = vec!["uniswap_v2", "uniswap_v3"];
```

To add more protocols, update this list and ensure the decoder is registered in `create_decoder_for_protocols()`.

### Price Movements

Edit `examples/swap_to_price/benchmark.rs`:

```rust
const PRICE_MOVEMENTS: &[f64] = &[
    1.0001,  // +0.01% = +1 bps
    1.0005,  // +0.05% = +5 bps
    1.001,   // +0.1% = +10 bps
    1.005,   // +0.5% = +50 bps
    1.01,    // +1% = +100 bps
    1.05,    // +5% = +500 bps
];
```

---

## Development

### Project Structure

```
examples/swap_to_price/
   README.md           # This file
   main.rs             # CLI entry point
   snapshot.rs         # Pool state capture and persistence
   benchmark.rs        # Algorithm execution and metrics
   reporting.rs        # Results formatting and output

src/swap_to_price/
   mod.rs              # Core types, traits, and constants
   strategies/
       binary_search.rs   # BinarySearchStrategy implementation
```

### Adding New Algorithms

1. Implement `SwapToPriceStrategy` trait in `src/swap_to_price/strategies/`
2. Add to benchmark.rs for testing
3. Compare results against BinarySearchStrategy baseline

Example:

```rust
pub struct NewtonRaphsonStrategy;

impl SwapToPriceStrategy for NewtonRaphsonStrategy {
    fn get_amount_in(
        &self,
        state: &dyn ProtocolSim,
        target_price: f64,
        token_in: &Token,
        token_out: &Token,
    ) -> Result<GetAmountInResult, SwapToPriceError> {
        // Your implementation here
    }
}
```

### Running Tests

```bash
# Run all tests
cargo test --features swap_to_price,evm

# Run integration test (creates live snapshot and loads it)
cargo test --example swap_to_price_benchmark test_create_and_load_snapshot -- --ignored --nocapture
```

**Integration Test:**

The `test_create_and_load_snapshot` test:
1. Connects to live Tycho stream
2. Creates a snapshot with 600+ pools
3. Saves to temp file
4. Loads it back and decodes all states
5. Verifies states work by calling `fee()` and `spot_price()`

This test ensures the entire snapshot workflow is functional end-to-end.

---

## Use Cases

### 1. Algorithm Comparison

Create one snapshot, then test multiple algorithms:

```bash
# Create snapshot once
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- snapshot

# Run with different algorithms (after implementing them)
cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  run --snapshot snapshot_X.json --algorithm binary_search

cargo run --example swap_to_price_benchmark --features swap_to_price,evm --release -- \
  run --snapshot snapshot_X.json --algorithm newton_raphson
```

### 2. Parameter Tuning

Test different tolerance values:

1. Modify `SWAP_TO_PRICE_TOLERANCE` in `src/swap_to_price/mod.rs`
2. Run benchmark
3. Compare iteration counts and convergence errors

### 3. Protocol Analysis

Focus on specific protocols by adjusting TVL threshold or modifying `snapshot.rs` to filter specific protocols.

---

## Performance Tips

### Faster Snapshot Creation

- Use higher `--min-tvl` to reduce pool count
- Focus on specific block instead of `latest`

### Faster Benchmark Execution

- Use `--release` flag (10x faster)
- Reduce number of price movements in `PRICE_MOVEMENTS`
- Test subset of pools by creating smaller snapshot

---

## FAQ

**Q: Why are some scenarios marked as "unreachable"?**
A: The pool doesn't have enough liquidity to reach that price. This is expected behavior, especially for large price movements on low-liquidity pools.

**Q: Can I test on different chains?**
A: Yes, modify `chain` in `snapshot.rs` and update `TYCHO_URL` to the appropriate Tycho instance.

**Q: How do I compare algorithm performance?**
A: Run the same snapshot through different algorithms and compare the JSON outputs, focusing on iteration counts and success rates.

**Q: What's the difference between spot_price and target_price?**
A: `spot_price` is the current pool price. `target_price` is what we want to achieve. The algorithm finds how much to trade to reach `target_price`.

---

## Support

For issues or questions:
1. Check this README
2. Review source code documentation
3. Open an issue on GitHub
