use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowPriceLevelsResponse {
    pub status: String, // "success" or "fail"
    #[serde(rename = "baseChain")]
    pub base_chain: Option<HashflowChain>,
    pub levels: Option<HashMap<String, Vec<HashflowMarketMakerLevel>>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowChain {
    #[serde(rename = "chainType")]
    pub chain_type: String, // "evm" or "solana"
    #[serde(rename = "chainId")]
    pub chain_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowMarketMakerLevel {
    pub pair: HashflowPair,
    pub levels: Vec<HashflowPriceLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowPair {
    #[serde(rename = "baseToken")]
    pub base_token: String,
    #[serde(rename = "quoteToken")]
    pub quote_token: String,
    #[serde(rename = "baseTokenName")]
    pub base_token_name: String,
    #[serde(rename = "quoteTokenName")]
    pub quote_token_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowPriceLevel {
    pub q: String, // quantity level
    pub p: String, // price per unit at that level
}

impl HashflowPriceLevel {
    pub fn quantity(&self) -> Result<f64, std::num::ParseFloatError> {
        self.q.parse()
    }
    
    pub fn price(&self) -> Result<f64, std::num::ParseFloatError> {
        self.p.parse()
    }
}

impl HashflowMarketMakerLevel {
    /// Calculate Total Value Locked (TVL) for this market maker level
    pub fn calculate_tvl(&self) -> f64 {
        self.levels.iter()
            .filter_map(|level| {
                let quantity = level.quantity().ok()?;
                let price = level.price().ok()?;
                Some(quantity * price)
            })
            .sum()
    }
    
    /// Get the pair identifier in format "baseToken/quoteToken"
    pub fn pair_key(&self) -> String {
        format!("{}/{}", self.pair.base_token, self.pair.quote_token)
    }
    
    /// Get estimated price for a given quantity (simple linear interpolation)
    pub fn get_estimated_price(&self, target_quantity: f64) -> Option<f64> {
        if self.levels.is_empty() || target_quantity <= 0.0 {
            return None;
        }
        
        let mut remaining_quantity = target_quantity;
        let mut total_value = 0.0;
        let mut filled_quantity = 0.0;
        
        for level in &self.levels {
            if remaining_quantity <= 0.0 {
                break;
            }
            
            let level_quantity = level.quantity().ok()?;
            let level_price = level.price().ok()?;
            
            let quantity_to_fill = remaining_quantity.min(level_quantity);
            total_value += quantity_to_fill * level_price;
            filled_quantity += quantity_to_fill;
            remaining_quantity -= quantity_to_fill;
        }
        
        if filled_quantity > 0.0 {
            Some(total_value / filled_quantity)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowMarketMakersResponse {
    #[serde(rename = "marketMakers")]
    pub market_makers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HashflowQuoteResponse {
    Success(Box<HashflowQuotePartial>),
    Error(HashflowApiError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowQuotePartial {
    // Add quote response fields when needed for RFQ functionality
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowApiError {
    pub error: HashflowErrorDetail,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashflowErrorDetail {
    #[serde(rename = "errorCode")]
    pub error_code: u32,
    pub message: String,
    #[serde(rename = "requestId")]
    pub request_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_level_parsing() {
        let level = HashflowPriceLevel {
            q: "1.5".to_string(),
            p: "3000.0".to_string(),
        };
        
        assert_eq!(level.quantity().unwrap(), 1.5);
        assert_eq!(level.price().unwrap(), 3000.0);
    }
    
    #[test]
    fn test_market_maker_level_tvl() {
        let mm_level = HashflowMarketMakerLevel {
            pair: HashflowPair {
                base_token: "0xETH".to_string(),
                quote_token: "0xUSDC".to_string(),
                base_token_name: "ETH".to_string(),
                quote_token_name: "USDC".to_string(),
            },
            levels: vec![
                HashflowPriceLevel { q: "1.0".to_string(), p: "3000.0".to_string() },
                HashflowPriceLevel { q: "2.0".to_string(), p: "2999.0".to_string() },
            ],
        };
        
        let tvl = mm_level.calculate_tvl();
        // 1.0 * 3000.0 + 2.0 * 2999.0 = 3000.0 + 5998.0 = 8998.0
        assert_eq!(tvl, 8998.0);
    }
    
    #[test]
    fn test_pair_key() {
        let mm_level = HashflowMarketMakerLevel {
            pair: HashflowPair {
                base_token: "0xETH".to_string(),
                quote_token: "0xUSDC".to_string(),
                base_token_name: "ETH".to_string(),
                quote_token_name: "USDC".to_string(),
            },
            levels: vec![],
        };
        
        assert_eq!(mm_level.pair_key(), "0xETH/0xUSDC");
    }
    
    #[test]
    fn test_estimated_price() {
        let mm_level = HashflowMarketMakerLevel {
            pair: HashflowPair {
                base_token: "0xETH".to_string(),
                quote_token: "0xUSDC".to_string(),
                base_token_name: "ETH".to_string(),
                quote_token_name: "USDC".to_string(),
            },
            levels: vec![
                HashflowPriceLevel { q: "1.0".to_string(), p: "3000.0".to_string() },
                HashflowPriceLevel { q: "2.0".to_string(), p: "2999.0".to_string() },
            ],
        };
        
        // Test exact quantity match
        assert_eq!(mm_level.get_estimated_price(1.0).unwrap(), 3000.0);
        
        // Test quantity spanning multiple levels
        let estimated = mm_level.get_estimated_price(2.0).unwrap();
        // 1.0 * 3000.0 + 1.0 * 2999.0 = 5999.0 / 2.0 = 2999.5
        assert_eq!(estimated, 2999.5);
    }
}
