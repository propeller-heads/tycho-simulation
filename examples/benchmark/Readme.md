# Benchmark

This example benchmarks swap simulation times for given protocols

## How to run

```bash
export RPC_URL=<your-eth-rpc-url>
cargo run --release --example benchmark -- --exchange uniswap_v2 --exchange uniswap_v3
```

### To see all config options:
```bash
cargo run --release --example benchmark -- help
```

### To print out individual swap logs:
```bash
RUST_LOG=info cargo run --release --example benchmark -- --exchange uniswap_v2
```