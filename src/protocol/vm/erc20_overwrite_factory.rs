// TODO: remove skip for clippy dead_code check
#![allow(dead_code)]
use crate::protocol::vm::utils::{get_contract_bytecode, get_storage_slot_index_at_key, SlotHash};
use ethers::{addressbook::Address, prelude::U256};
use std::{collections::HashMap, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FileError {
    /// Occurs when the ABI file cannot be read
    #[error("Malformed ABI error: {0}")]
    MalformedABI(String),
    /// Occurs when the parent directory of the current file cannot be retrieved
    #[error("Structure error {0}")]
    Structure(String),
    /// Occurs when a bad file path was given, which cannot be converted to string.
    #[error("File path conversion error {0}")]
    FilePath(String),
}

pub struct GethOverwrite {
    /// the formatted overwrites
    pub state_diff: HashMap<String, String>,
    /// the bytecode as a string
    pub code: String,
}

pub type Overwrites = HashMap<SlotHash, U256>;

pub struct ERC20OverwriteFactory {
    token_address: Address,
    overwrites: Overwrites,
    balance_slot: SlotHash,
    allowance_slot: SlotHash,
    total_supply_slot: SlotHash,
}

impl ERC20OverwriteFactory {
    pub fn new(token_address: Address, token_slots: (SlotHash, SlotHash)) -> Self {
        ERC20OverwriteFactory {
            token_address,
            overwrites: HashMap::new(),
            balance_slot: token_slots.0,
            allowance_slot: token_slots.1,
            total_supply_slot: SlotHash::from_low_u64_be(2),
        }
    }

    pub fn set_balance(&mut self, balance: U256, owner: Address) {
        let storage_index = get_storage_slot_index_at_key(owner, self.balance_slot);
        self.overwrites
            .insert(storage_index, balance);
    }

    pub fn set_allowance(&mut self, allowance: U256, spender: Address, owner: Address) {
        let owner_slot = get_storage_slot_index_at_key(owner, self.allowance_slot);
        let storage_index = get_storage_slot_index_at_key(spender, owner_slot);
        self.overwrites
            .insert(storage_index, allowance);
    }

    pub fn set_total_supply(&mut self, supply: U256) {
        self.overwrites
            .insert(self.total_supply_slot, supply);
    }

    pub fn get_protosim_overwrites(&self) -> HashMap<Address, Overwrites> {
        let mut result = HashMap::new();
        result.insert(self.token_address, self.overwrites.clone());
        result
    }

    pub fn get_geth_overwrites(&self) -> Result<HashMap<Address, GethOverwrite>, FileError> {
        let mut formatted_overwrites = HashMap::new();

        for (key, val) in &self.overwrites {
            let hex_key = hex::encode(key.as_bytes());

            let mut bytes = [0u8; 32];
            val.to_big_endian(&mut bytes);
            let hex_val = format!("0x{:0>64}", hex::encode(bytes));

            formatted_overwrites.insert(hex_key, hex_val);
        }

        let erc20_abi_path = Path::new(file!())
            .parent()
            .ok_or_else(|| {
                FileError::Structure(
                    "Failed to obtain parent directory of current file.".to_string(),
                )
            })?
            .join("assets")
            .join("ERC20.abi");

        let code = format!(
            "0x{}",
            hex::encode(
                get_contract_bytecode(erc20_abi_path.to_str().ok_or_else(|| {
                    FileError::FilePath("Failed to convert file path to string.".to_string())
                })?)
                .map_err(|_err| FileError::MalformedABI(
                    "Failed to read contract bytecode.".to_string()
                ))?
            )
        );

        let mut result = HashMap::new();
        result.insert(self.token_address, GethOverwrite { state_diff: formatted_overwrites, code });

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::vm::utils::SlotHash;

    fn setup_factory() -> ERC20OverwriteFactory {
        let token_address = Address::random();
        let balance_slot = SlotHash::random();
        let allowance_slot = SlotHash::random();
        ERC20OverwriteFactory::new(token_address, (balance_slot, allowance_slot))
    }

    #[test]
    fn test_set_balance() {
        let mut factory = setup_factory();
        let owner = Address::random();
        let balance = U256::from(1000);

        factory.set_balance(balance, owner);

        assert_eq!(factory.overwrites.len(), 1);
        assert!(factory
            .overwrites
            .values()
            .any(|&v| v == balance));
    }

    #[test]
    fn test_set_allowance() {
        let mut factory = setup_factory();
        let owner = Address::random();
        let spender = Address::random();
        let allowance = U256::from(500);

        factory.set_allowance(allowance, spender, owner);

        assert_eq!(factory.overwrites.len(), 1);
        assert!(factory
            .overwrites
            .values()
            .any(|&v| v == allowance));
    }

    #[test]
    fn test_set_total_supply() {
        let mut factory = setup_factory();
        let supply = U256::from(1_000_000);

        factory.set_total_supply(supply);

        assert_eq!(factory.overwrites.len(), 1);
        assert_eq!(factory.overwrites[&factory.total_supply_slot], supply);
    }

    #[test]
    fn test_get_protosim_overwrites() {
        let mut factory = setup_factory();
        let supply = U256::from(1_000_000);
        factory.set_total_supply(supply);

        let overwrites = factory.get_protosim_overwrites();

        assert_eq!(overwrites.len(), 1);
        assert!(overwrites.contains_key(&factory.token_address));
        assert_eq!(overwrites[&factory.token_address].len(), 1);
        assert_eq!(overwrites[&factory.token_address][&factory.total_supply_slot], supply);
    }

    #[test]
    fn test_get_geth_overwrites() {
        let mut factory = setup_factory();

        let storage_slot = SlotHash::from_low_u64_be(1);
        let val = U256::from(123456);
        factory
            .overwrites
            .insert(storage_slot, val);

        let result = factory
            .get_geth_overwrites()
            .expect("Failed to get geth overwrites");

        assert_eq!(result.len(), 1);

        let geth_overwrite = result
            .get(&factory.token_address)
            .expect("Missing token address");
        assert_eq!(geth_overwrite.state_diff.len(), 1);

        let expected_key =
            String::from("0000000000000000000000000000000000000000000000000000000000000001");
        let expected_val =
            String::from("0x000000000000000000000000000000000000000000000000000000000001e240");
        assert_eq!(
            geth_overwrite
                .state_diff
                .get(&expected_key),
            Some(&expected_val)
        );
        assert_eq!(geth_overwrite.code.len(), 8752);
    }
}