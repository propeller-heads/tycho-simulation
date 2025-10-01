use alloy::{
    hex,
    primitives::{keccak256, Bytes, map::AddressHashMap},
    providers::{ProviderBuilder, Provider},
    rpc::types::{
        TransactionRequest,
        BlockId,
        state::AccountOverride,
    },
    sol_types::SolValue,
    transports::http::reqwest,
};
use alloy_rpc_types_trace::geth::{
    GethDebugTracingCallOptions, GethDebugTracingOptions,
    GethDebugTracerType, GethDebugBuiltInTracerType,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{error, info};

pub type BlockTag = alloy::eips::BlockNumberOrTag;

/// Result of a simulation with trace
#[derive(Debug)]
pub enum SimulationResult {
    Success {
        return_data: Vec<u8>,
        gas_used: u64,
    },
    Revert {
        reason: String,
    },
}

/// Raw JSON-RPC call for lower-level access
#[allow(dead_code)]
pub async fn raw_rpc_call(
    rpc_url: &str,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    });

    let response = client.post(rpc_url).json(&payload).send().await?;
    let result: serde_json::Value = response.json().await?;
    Ok(result)
}

/// Client for fetching 4byte error signatures.
/// 4byte.directory is a service that maps 4-byte function signatures to human-readable function names.
/// See https://www.4byte.directory/ for more.
pub struct FourBytesClient {
    url: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct FourByteResponse {
    results: Vec<SignatureResult>,
}

#[derive(Debug, Deserialize)]
struct SignatureResult {
    text_signature: String,
}

impl FourBytesClient {
    pub fn new() -> Self {
        Self {
            url: "https://www.4byte.directory/api/v1/signatures/".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_signature(&self, hex_sig: &str) -> Result<String, Box<dyn std::error::Error>> {
        let response = self
            .client
            .get(&self.url)
            .query(&[("hex_signature", hex_sig)])
            .send()
            .await?;

        let result: FourByteResponse = response.json().await?;

        if result.results.is_empty() {
            return Err("No signatures found!".into());
        }

        Ok(result.results[0].text_signature.clone())
    }
}

impl Default for FourBytesClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a Solidity error signature
#[derive(Debug, Clone)]
struct SolidityError {
    name: String,
    bytes_sig: [u8; 4],
    types: Vec<String>,
}

impl SolidityError {
    fn new(name: String, bytes_sig: [u8; 4], types: Vec<String>) -> Self {
        Self {
            name,
            bytes_sig,
            types,
        }
    }

    fn decode(&self, data: &Bytes) -> String {
        if data.len() <= 4 {
            return format!("{}()", self.name);
        }

        // Try to decode the arguments
        match self.decode_args(&data[4..]) {
            Ok(args_str) => format!("{}({})", self.name, args_str),
            Err(_) => format!("{}(<decode error>)", self.name),
        }
    }

    fn decode_args(&self, data: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
        // This is a simplified version. For full ABI decoding, you'd need alloy-sol-types
        let mut args = Vec::new();
        let mut offset = 0;

        for typ in &self.types {
            match typ.as_str() {
                "string" => {
                    // Decode ABI-encoded string: offset (32 bytes) + length (32 bytes) + data
                    if data.len() >= 64 {
                        // Skip the offset (first 32 bytes) and read length
                        let length = alloy::primitives::U256::from_be_slice(&data[32..64]).to::<usize>();
                        if data.len() >= 64 + length {
                            let string_data = &data[64..64 + length];
                            if let Ok(s) = std::str::from_utf8(string_data) {
                                args.push(format!("\"{}\"", s));
                            } else {
                                // The string data is not UTF-8, it might be nested error bytes
                                // Recursively decode until we get the actual string
                                if let Ok(nested) = Self::decode_nested_error_recursive(string_data) {
                                    args.push(format!("\"{}\"", nested));
                                } else {
                                    args.push(format!("0x{}", hex::encode(string_data)));
                                }
                            }
                        }
                    }
                }
                "uint256" | "uint" => {
                    if data.len() >= offset + 32 {
                        let val = alloy::primitives::U256::from_be_slice(&data[offset..offset + 32]);
                        args.push(val.to_string());
                        offset += 32;
                    }
                }
                "address" => {
                    if data.len() >= offset + 32 {
                        let addr = alloy::primitives::Address::from_slice(&data[offset + 12..offset + 32]);
                        args.push(format!("{:?}", addr));
                        offset += 32;
                    }
                }
                _ => {
                    args.push(format!("0x{}", hex::encode(data)));
                }
            }
        }

        Ok(args.join(", "))
    }

    fn decode_nested_error_recursive(data: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
        // Check if it starts with Error(string) signature (0x08c379a0)
        if data.len() >= 4 {
            let mut sig = [0u8; 4];
            sig.copy_from_slice(&data[..4]);

            if sig == [0x08, 0xc3, 0x79, 0xa0] {
                // Decode nested Error(string): selector (4 bytes) + offset (32 bytes) + length (32 bytes) + string data
                if data.len() >= 68 {
                    let length = alloy::primitives::U256::from_be_slice(&data[36..68]).to::<usize>();
                    if data.len() >= 68 + length {
                        let string_data = &data[68..68 + length];

                        // Try to decode as UTF-8 string
                        if let Ok(s) = std::str::from_utf8(string_data) {
                            return Ok(s.to_string());
                        }

                        // If not UTF-8, it might be another nested error - recurse
                        if let Ok(nested) = Self::decode_nested_error_recursive(string_data) {
                            return Ok(nested);
                        }
                    }
                }
            }
        }

        Err("Failed to decode nested error".into())
    }
}

/// Fetches revert reasons for transactions.
/// It runs an eth_call on the transaction and tries to decode the revert reason.
/// If the revert reason is not known and found in error_signatures, it tries to fetch the signature
/// from 4byte.directory. If it still fails, it returns "EmptyMessage".
pub struct RevertReasonFetcher {
    rpc_url: String,
    error_signatures: HashMap<[u8; 4], SolidityError>,
}

impl RevertReasonFetcher {
    pub fn new(rpc_url: String, error_signatures: Vec<String>) -> Self {
        let parsed_signatures = Self::parse_signatures(&error_signatures);
        Self {
            rpc_url,
            error_signatures: parsed_signatures,
        }
    }

    fn parse_signatures(error_signatures: &[String]) -> HashMap<[u8; 4], SolidityError> {
        let mut result = HashMap::new();
        for signature in error_signatures {
            if let Ok(error) = Self::parse_signature_from_str(signature) {
                result.insert(error.bytes_sig, error);
            }
        }
        result
    }

    fn parse_signature_from_str(signature: &str) -> Result<SolidityError, Box<dyn std::error::Error>> {
        if signature.contains(' ') {
            return Err(format!("Found whitespace in signature: {}", signature).into());
        }

        // Parse signature like "Error(string)" or "Panic(uint256)"
        let re = regex::Regex::new(r"(.*?)\((.*?)\)").unwrap();
        let captures = re
            .captures(signature)
            .ok_or("Failed to parse signature")?;

        let name = captures.get(1).unwrap().as_str().to_string();
        let types_str = captures.get(2).unwrap().as_str();

        let types: Vec<String> = if types_str.is_empty() {
            Vec::new()
        } else {
            types_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        };

        let byte_sig = keccak256(signature.as_bytes());
        let mut sig_array = [0u8; 4];
        sig_array.copy_from_slice(&byte_sig[..4]);

        Ok(SolidityError::new(name, sig_array, types))
    }

    #[allow(dead_code)]
    pub async fn fetch(
        &mut self,
        tx: TransactionRequest,
        block: BlockTag,
        state_overrides: Option<AddressHashMap<AccountOverride>>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let reason = self.fetch_revert_reason(tx, block, state_overrides).await?;
        Ok(reason)
    }

    /// Decode a Panic(uint256) error code to a human-readable message
    fn decode_panic_code(code: u64) -> String {
        match code {
            0x01 => "assertion failed".to_string(),
            0x11 => "arithmetic underflow or overflow".to_string(),
            0x12 => "division or modulo by zero".to_string(),
            0x21 => "invalid enum value".to_string(),
            0x22 => "storage byte array incorrectly encoded".to_string(),
            0x31 => "pop on empty array".to_string(),
            0x32 => "array index out of bounds".to_string(),
            0x41 => "out of memory".to_string(),
            0x51 => "invalid internal function".to_string(),
            _ => format!("panic code 0x{:x}", code),
        }
    }

    /// Recursively find the deepest revert reason in a trace
    fn find_deepest_revert_reason(trace: &Value) -> Option<String> {
        // Check for output field that might contain revert data
        if let Some(output) = trace.get("output").and_then(|o| o.as_str()) {
            if !output.is_empty() && output != "0x" {
                // Try to decode using the recursive decoder for nested errors
                if let Ok(data) = hex::decode(output.trim_start_matches("0x")) {
                    if data.len() >= 4 {
                        // Check if it's Error(string) selector (0x08c379a0)
                        if data[0..4] == [0x08, 0xc3, 0x79, 0xa0] {
                            // Use the recursive decoder to handle nested Error(string) wrapping
                            if let Ok(reason) = SolidityError::decode_nested_error_recursive(&data) {
                                return Some(reason);
                            }
                        }
                        // Check if it's Panic(uint256) selector (0x4e487b71)
                        else if data[0..4] == [0x4e, 0x48, 0x7b, 0x71] && data.len() >= 36 {
                            // Decode the panic code (uint256)
                            let panic_code = alloy::primitives::U256::from_be_slice(&data[4..36]).to::<u64>();
                            return Some(Self::decode_panic_code(panic_code));
                        }
                    }
                }
                // Return raw output if we can't decode it
                return Some(output.to_string());
            }
        }

        // Check for revertReason field
        if let Some(revert_reason) = trace.get("revertReason").and_then(|r| r.as_str()) {
            if !revert_reason.is_empty() {
                // The revertReason might contain raw bytes with error encoding
                let bytes = revert_reason.as_bytes();
                if bytes.len() >= 4 {
                    // Check if it's Error(string) selector (0x08c379a0)
                    if bytes[0..4] == [0x08, 0xc3, 0x79, 0xa0] {
                        if let Ok(decoded) = SolidityError::decode_nested_error_recursive(bytes) {
                            return Some(decoded);
                        }
                    }
                    // Check if it's Panic(uint256) selector (0x4e487b71)
                    else if bytes[0..4] == [0x4e, 0x48, 0x7b, 0x71] && bytes.len() >= 36 {
                        let panic_code = alloy::primitives::U256::from_be_slice(&bytes[4..36]).to::<u64>();
                        return Some(Self::decode_panic_code(panic_code));
                    }
                }
                return Some(revert_reason.to_string());
            }
        }

        // Recursively check nested calls
        if let Some(calls) = trace.get("calls").and_then(|c| c.as_array()) {
            for call in calls {
                if let Some(reason) = Self::find_deepest_revert_reason(call) {
                    return Some(reason);
                }
            }
        }

        None
    }

    /// Simulate a transaction with trace and return either success or revert reason
    pub async fn simulate_with_trace(
        &mut self,
        tx: TransactionRequest,
        block: BlockTag,
        state_overrides: Option<AddressHashMap<AccountOverride>>,
    ) -> Result<SimulationResult, Box<dyn std::error::Error>> {
        let trace_result = self.debug_trace_call_internal(tx, block, state_overrides, false).await?;

        // Check for error in response
        if let Some(error) = trace_result.get("error") {
            // Print the full trace only on failure
            error!("=== Transaction Trace (FAILURE) ===");
            crate::traces::print_call_trace(&trace_result, 0).await;
            error!("=== End Trace ===");

            // Try to find the deepest revert reason in the trace
            let reason = if let Some(deepest_reason) = Self::find_deepest_revert_reason(&trace_result) {
                // Try to decode it if it looks like hex data
                if deepest_reason.starts_with("0x") {
                    self.match_error(&deepest_reason).await?
                } else {
                    deepest_reason
                }
            } else if let Some(data) = error.get("data").and_then(|d| d.as_str()) {
                self.match_error(data).await?
            } else if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                message.to_string()
            } else {
                "EmptyMessage".to_string()
            };

            return Ok(SimulationResult::Revert { reason });
        }

        // Extract return data and gas used from successful trace
        let return_data = if let Some(output) = trace_result.get("output").and_then(|o| o.as_str()) {
            hex::decode(output.trim_start_matches("0x"))?
        } else {
            Vec::new()
        };

        let gas_used = trace_result
            .get("gasUsed")
            .and_then(|g| g.as_str())
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);

        Ok(SimulationResult::Success { return_data, gas_used })
    }

    #[allow(dead_code)]
    async fn fetch_revert_reason(
        &mut self,
        tx: TransactionRequest,
        block: BlockTag,
        state_overrides: Option<AddressHashMap<AccountOverride>>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let response = self.debug_trace_call_internal(tx, block, state_overrides, true).await?;

        // Check for error in response
        if let Some(error) = response.get("error") {
            if let Some(data) = error.get("data").and_then(|d| d.as_str()) {
                return self.match_error(data).await;
            } else if let Some(message) = error.get("message").and_then(|m| m.as_str()) {
                return Ok(message.to_string());
            }
            return Ok("EmptyMessage".to_string());
        }
        Ok(String::new())
    }

    async fn debug_trace_call_internal(
        &self,
        tx: TransactionRequest,
        block: BlockTag,
        state_overrides: Option<AddressHashMap<AccountOverride>>,
        print_trace: bool,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let provider = ProviderBuilder::new().connect_http(self.rpc_url.parse()?);

        // Configure tracing options - use callTracer for better formatted results
        let tracing_options = GethDebugTracingOptions {
            tracer: Some(GethDebugTracerType::BuiltInTracer(
                GethDebugBuiltInTracerType::CallTracer,
            )),
            config: Default::default(),
            tracer_config: Default::default(),
            timeout: None,
        };

        let trace_options = GethDebugTracingCallOptions {
            tracing_options,
            state_overrides,
            block_overrides: None,
        };

        let block_id = match block {
            BlockTag::Latest => BlockId::latest(),
            BlockTag::Number(n) => BlockId::from(n),
            _ => BlockId::latest(),
        };

        let result: Value = provider
            .client()
            .request("debug_traceCall", (tx, block_id, trace_options))
            .await?;

        // Print the full trace only if requested (for legacy fetch_revert_reason path)
        if print_trace {
            error!("=== Transaction Trace ===");
            crate::traces::print_call_trace(&result, 0).await;
            error!("=== End Trace ===");
        }

        Ok(result)
    }

    async fn match_error(&mut self, data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let data = hex::decode(data.trim_start_matches("0x"))?;
        let data_bytes = Bytes::from(data.clone());

        let result = format!("Revert: 0x{}", hex::encode(&data));

        // Try to match known error signatures
        if data.len() >= 4 {
            let mut sig = [0u8; 4];
            sig.copy_from_slice(&data[..4]);

            if let Some(error) = self.error_signatures.get(&sig) {
                return Ok(error.decode(&data_bytes));
            }
        }

        // Try to decode as string (standard revert reason)
        if data.len() > 4 {
            if let Ok(decoded) = String::abi_decode(&data[4..]) {
                return Ok(decoded);
            }
        }

        // Try fetching from 4bytes
        if data.len() >= 4 {
            info!("Fetching signature from 4bytes...");
            let hex_sig = format!("0x{}", hex::encode(&data[..4]));

            match FourBytesClient::new().get_signature(&hex_sig).await {
                Ok(fetched_sig) => {
                    if let Ok(error) = Self::parse_signature_from_str(&fetched_sig) {
                        let decoded = error.decode(&data_bytes);
                        self.error_signatures.insert(error.bytes_sig, error);
                        return Ok(decoded);
                    }
                }
                Err(e) => {
                    error!("Failed to fetch from 4bytes: {}", e);
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_signature() {
        let sig = "Error(string)";
        let error = RevertReasonFetcher::parse_signature_from_str(sig).unwrap();
        assert_eq!(error.name, "Error");
        assert_eq!(error.types, vec!["string"]);
    }

    #[test]
    fn test_parse_signature_no_args() {
        let sig = "EmptyError()";
        let error = RevertReasonFetcher::parse_signature_from_str(sig).unwrap();
        assert_eq!(error.name, "EmptyError");
        assert!(error.types.is_empty());
    }

    #[test]
    fn test_parse_signature_multiple_args() {
        let sig = "ComplexError(uint256,address,string)";
        let error = RevertReasonFetcher::parse_signature_from_str(sig).unwrap();
        assert_eq!(error.name, "ComplexError");
        assert_eq!(error.types, vec!["uint256", "address", "string"]);
    }

    #[tokio::test]
    async fn test_fourbyte_client() {
        let client = FourBytesClient::new();
        let signature = client.get_signature("0x90bfb865").await;

        assert!(signature.is_ok());
        let sig_text = signature.unwrap();
        assert_eq!(sig_text, "WrappedError(address,bytes4,bytes,bytes)");
    }

    #[tokio::test]
    async fn test_match_error_with_fourbyte_lookup() {
        // Create a fetcher with no pre-loaded signatures
        let mut fetcher = RevertReasonFetcher::new(
            "http://dummy".to_string(),
            vec![]
        );

        // Test data with signature 0x90bfb865 (WrappedError)
        let error_data = "0x90bfb865";

        let result = fetcher.match_error(error_data).await;
        assert!(result.is_ok());

        let decoded = result.unwrap();
        // Should contain WrappedError after 4byte lookup
        assert!(decoded.contains("WrappedError"));
    }
}