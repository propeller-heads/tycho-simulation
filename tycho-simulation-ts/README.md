# Tycho Simulation TypeScript

TypeScript bindings for the Tycho simulation engine, providing DeFi protocol simulation capabilities.

## Features

- Spot price calculations for various DeFi protocols
- Support for Uniswap V2, V3, Curve, and Balancer pools
- Gas estimation for swaps
- TVL-based pool filtering

## Installation

1. Prerequisites:
   - Node.js 16.x or higher
   - Rust toolchain (latest stable)
   - Cargo package manager
   - Platform: macOS, Linux, or Windows
   - Architecture: x64 or arm64

2. Install dependencies:
```bash
npm install
```

1. Build the project:
```bash
npm run build-all
```

## Usage

1. Create a `.env` file in the project root with your Tycho API key:
```env
TYCHO_API_KEY=your_api_key_here
```

2. Run the benchmark example:
```bash
npm run example
```

### Running the Spot Price Example

The spot price example demonstrates how to fetch prices between USDC and WETH:

```typescript
// examples/spot_price.ts
import { createClient } from '../';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function main() {
  try {
    // USDC and WETH addresses on Ethereum mainnet
    const USDC = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48';
    const WETH = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2';

    // Create client with your API key
    const client = await createClient(
      'https://api.tycho.xyz',
      process.env.TYCHO_API_KEY
    );

    // Get spot price
    const price = await client.getSpotPrice(USDC, WETH);
    console.log(`Current USDC/WETH price: ${price}`);

    // Get amounts out for different USDC amounts
    const amounts = [1000000, 10000000, 100000000]; // 1, 10, 100 USDC
    const result = await client.getAmountOut(USDC, WETH, amounts);
    
    console.log('\nSwap simulations:');
    result.forEach((pool, i) => {
      console.log(`\nPool ${i + 1} (${pool.poolAddress}) [${pool.protocol}]:`);
      pool.amountsOut.forEach((amount, j) => {
        console.log(`${amounts[j] / 1e6} USDC -> ${amount / 1e18} WETH (Gas: ${pool.gasEstimates[j]})`);
      });
    });
  } catch (error) {
    console.error('Error:', error);
    console.error('Node version:', process.version);
    console.error('Platform:', process.platform);
    console.error('Architecture:', process.arch);
    process.exit(1);
  }
}

main().catch(console.error);
```