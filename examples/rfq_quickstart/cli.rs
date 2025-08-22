use clap::{Args, Parser};
use miette::{miette, Result};
use tycho_common::models::Chain;

#[derive(Clone, Debug, Parser)]
pub struct RfqCommand {
    #[command(flatten)]
    pub swap_args: SwapArgs,
}

#[derive(Clone, Debug, Args, Default)]
pub struct SwapArgs {
    #[arg(long)]
    pub sell_token: Option<String>,
    #[arg(long)]
    pub buy_token: Option<String>,
    #[arg(long, default_value_t = 10.0)]
    pub sell_amount: f64,
    /// The minimum TVL threshold for RFQ quotes in USD
    #[arg(long, default_value_t = 1000.0)]
    pub tvl_threshold: f64,
    #[arg(long, default_value = "ethereum")]
    pub chain: Chain,
}

impl SwapArgs {
    fn parse_args(mut self) -> Result<Self> {
        // By default, we request quotes for USDC to WETH on whatever chain we choose
        if self.buy_token.is_none() {
            self.buy_token = Some(match self.chain {
                Chain::Ethereum => "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".to_string(),
                Chain::Base => "0x4200000000000000000000000000000000000006".to_string(),
                _ => {
                    return Err(miette!(
                        "RFQ quickstart does not yet support chain {chain}",
                        chain = self.chain
                    ))
                }
            });
        }
        if self.sell_token.is_none() {
            self.sell_token = Some(match self.chain {
                Chain::Ethereum => "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48".to_string(),
                Chain::Base => "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913".to_string(),
                _ => {
                    return Err(miette!(
                        "RFQ quickstart does not yet support chain {chain}",
                        chain = self.chain
                    ))
                }
            });
        }
        Ok(self)
    }
}

impl RfqCommand {
    pub async fn parse_args(mut self) -> Result<Self> {
        self.swap_args = self.swap_args.parse_args()?;
        Ok(self)
    }
}
