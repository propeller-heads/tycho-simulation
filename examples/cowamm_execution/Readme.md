# CowAMM Execution

This example shows how to:

1. Subscribe to a CowAMM pool through the Tycho stream.
2. Simulate a swap locally with Tycho.
3. Run the CoW settlement path through `eth_call` with state overrides.
4. Compare the `eth_call` result against the Tycho quote.

## Run

```bash
export RPC_URL=<your-rpc-url>
export TYCHO_API_KEY=<your-tycho-api-key>
export SOLVER_ADDRESS=<a-solver-address>
cargo run --release
```

Logging defaults to `info`. Set `RUST_LOG` to override it, for example:

```bash
RUST_LOG=debug cargo run --release
```

By default, the example:

- subscribes to the target CowAMM pool through Tycho
- runs a local Tycho quote
- simulates the settlement path through `eth_call` with state overrides
- logs the Tycho quote, helper-derived amount, `eth_call` output, gas used, and diff

By default, the example simulates the default CowAMM pool and trade configured in `main.rs`.
If you want a different pool or trade, you can do:

```bash
export RPC_URL=<your-rpc-url>
export TYCHO_API_KEY=<your-tycho-api-key>
export SOLVER_ADDRESS=<a-solver-address>
cargo run --release -- \
  --target-pool 0x9d0e8cdf137976e03ef92ede4c30648d05e25285 \
  --sell-token 0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0 \
  --buy-token 0xBAac2B4491727D78D2b78815144570b9f2Fe8899 \
  --sell-amount 100000000000000000 \
  --sell-token-balance-slot 0
```

## Save To Tenderly

If you want the example to submit the same simulated call to Tenderly and print the saved simulation URL, set:

```bash
export TENDERLY_ACCOUNT=<your-tenderly-account>
export TENDERLY_PROJECT=<your-tenderly-project>
export TENDERLY_ACCESS_KEY=<your-tenderly-access-key>
export TENDERLY_SUBMIT_SIMULATION=true
```

When those variables are present, the example submits the Simulation API request and logs a single `tenderly simulation result url`.
