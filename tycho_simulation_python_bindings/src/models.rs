use std::collections::HashMap;

use pyo3::prelude::*;
use tycho_simulation::{
    models::Token,
    protocol::models::{BlockUpdate, ProtocolComponent},
    tycho_common::models::Chain,
};

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyToken {
    #[pyo3(get)]
    pub address: String,
    #[pyo3(get)]
    pub decimals: usize,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub gas: String, // TODO: this should be a biguint why can't I?
}

impl From<Token> for PyToken {
    fn from(token: Token) -> Self {
        Self {
            address: format!("0x{}", hex::encode(token.address)),
            decimals: token.decimals,
            symbol: token.symbol,
            gas: token.gas.to_string(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyProtocolComponent {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub tokens: Vec<PyToken>,
    #[pyo3(get)]
    pub protocol_system: String,
    #[pyo3(get)]
    pub protocol_type_name: String,
    #[pyo3(get)]
    pub chain: PyChain,
    #[pyo3(get)]
    pub contract_ids: Vec<String>,
    #[pyo3(get)]
    pub static_attributes: HashMap<String, String>,
    #[pyo3(get)]
    pub creation_tx: String,
    #[pyo3(get)]
    pub created_at: String,
}

impl From<ProtocolComponent> for PyProtocolComponent {
    fn from(component: ProtocolComponent) -> Self {
        Self {
            id: format!("0x{}", hex::encode(component.id)),
            tokens: component
                .tokens
                .into_iter()
                .map(PyToken::from)
                .collect(),
            protocol_system: component.protocol_system,
            protocol_type_name: component.protocol_type_name,
            chain: (&component.chain).into(),
            contract_ids: component
                .contract_ids
                .iter()
                .map(|id| format!("0x{}", hex::encode(id)))
                .collect(),
            static_attributes: component
                .static_attributes
                .into_iter()
                .map(|(k, v)| (k, format!("0x{}", hex::encode(v))))
                .collect(),
            creation_tx: format!("0x{}", hex::encode(component.creation_tx)),
            created_at: component.created_at.to_string(),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum PyChain {
    Ethereum,
    Starknet,
    ZkSync,
    Arbitrum,
    Base,
    Unichain,
}

impl From<&Chain> for PyChain {
    fn from(chain: &Chain) -> Self {
        match chain {
            Chain::Ethereum => PyChain::Ethereum,
            Chain::Base => PyChain::Base,
            Chain::Unichain => PyChain::Unichain,
            Chain::Starknet => PyChain::Starknet,
            Chain::ZkSync => PyChain::ZkSync,
            Chain::Arbitrum => PyChain::Arbitrum,
        }
    }
}

#[pyclass]
pub struct PyBlockUpdate {
    pub block_number: u64,
    #[pyo3(get)]
    pub states: HashMap<String, PyObject>, // TODO: map ProtocolSim
    #[pyo3(get)]
    pub new_pairs: HashMap<String, PyProtocolComponent>,
    #[pyo3(get)]
    pub removed_pairs: HashMap<String, PyProtocolComponent>,
}

impl From<BlockUpdate> for PyBlockUpdate {
    fn from(update: BlockUpdate) -> Self {
        Self {
            block_number: update.block_number,
            states: update
                .states
                .into_iter()
                .map(|(id, state)| {})
                .collect(),
            new_pairs: update
                .new_pairs
                .into_iter()
                .map(|(k, component)| (k, component.into()))
                .collect(),
            removed_pairs: update
                .removed_pairs
                .into_iter()
                .map(|(k, component)| (k, component.into()))
                .collect(),
        }
    }
}
