import dotenv from 'dotenv';
dotenv.config();

interface SimulationClient {
    // Opaque client type
}

interface SwapResult {
    poolAddress: string;
    amountsOut: number[];
    gasEstimates: number[];
    protocol: string;
}

interface TychoSimulation {
    createClient: (
        url: string,
        apiKey: string,
        tvlThreshold: number
    ) => Promise<SimulationClient>;
    
    getAmountOut: (
        client: SimulationClient,
        tokenIn: string,
        tokenOut: string,
        amountsIn: number[]
    ) => Promise<SwapResult[]>;
}

const tychoSim = require('../index.node') as TychoSimulation;

console.log("Loaded tycho simulation module:", Object.keys(tychoSim));

function generateRandomAmounts(count: number, min: number, max: number): number[] {
    return Array.from({ length: count }, () => 
        min + Math.random() * (max - min)
    );
}

async function runBenchmark(): Promise<void> {
    try {
        console.log("Creating simulation client...");
        const tvlThreshold = 1000;
        
        const apiKey = process.env.TYCHO_API_KEY;
        if (!apiKey) {
            throw new Error("TYCHO_API_KEY environment variable is not set");
        }

        const client = await tychoSim.createClient(
            "tycho-beta.propellerheads.xyz",
            apiKey,
            tvlThreshold
        );
        console.log("Client created successfully");

        const WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2";  // Ethereum Mainnet WETH
        const USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7";  // Ethereum Mainnet USDT

        // Generate 10000 random amounts between 0.1 and 100 WETH
        const amountsIn = generateRandomAmounts(1000, 1, 10);
        console.log(`Generated ${amountsIn.length} random amounts`);
        console.log("Sample amounts:", amountsIn.slice(0, 5));

        console.log("\nStarting benchmark...");
        const startTime = process.hrtime.bigint();

        const results = await tychoSim.getAmountOut(client, WETH, USDT, amountsIn);
        
        const endTime = process.hrtime.bigint();
        const duration = Number(endTime - startTime) / 1e6; // Convert to milliseconds

        console.log("\nBenchmark Results:");
        console.log(`Total time: ${duration.toFixed(2)}ms`);
        console.log(`Average time per amount: ${(duration / amountsIn.length).toFixed(2)}ms`);
        console.log(`Amounts processed per second: ${(amountsIn.length / (duration / 1000)).toFixed(2)}`);
        
        if (results && results.length > 0) {
            console.log(`\nFound ${results.length} pools`);
            console.log("Sample results from first pool:");
            const firstPool = results[0];
            console.log(`Pool: ${firstPool.poolAddress}`);
            console.log(`Protocol: ${firstPool.protocol}`);
            console.log("First 5 amounts:");
            for (let i = 0; i < 5; i++) {
                console.log(`  ${amountsIn[i]} WETH -> ${firstPool.amountsOut[i]} USDT (gas: ${firstPool.gasEstimates[i]})`);
            }
        }

    } catch (error) {
        console.error("Benchmark error:", error);
    }
}

// Call the benchmark
async function main(): Promise<void> {
    console.log("Starting benchmark test...");
    const startTime = process.hrtime.bigint();
    
    await runBenchmark();
    
    const endTime = process.hrtime.bigint();
    const totalDuration = Number(endTime - startTime) / 1e9; // Convert to seconds
    console.log(`\nTotal execution time: ${totalDuration.toFixed(2)} seconds`);
    
    process.exit(0);
}

process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

main().catch(error => {
    console.error("Unhandled error:", error);
    process.exit(1);
}); 