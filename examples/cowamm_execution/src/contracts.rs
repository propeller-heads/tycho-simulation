#![allow(unused_imports)]

use ethcontract::{Address as EthAddress, Bytes as ContractBytes, U256 as EthU256};

include!(concat!(env!("OUT_DIR"), "/b_cow_helper.rs"));
include!(concat!(env!("OUT_DIR"), "/g_pv_2_allow_list_authentication.rs"));
include!(concat!(env!("OUT_DIR"), "/g_pv_2_settlement.rs"));

pub use self::{
    b_cow_helper::Contract as BCowHelperContract,
    g_pv_2_allow_list_authentication::Contract as GPv2AllowListAuthenticationContract,
    g_pv_2_settlement::Contract as GPv2SettlementContract,
};

pub type RawOrder = (
    EthAddress,
    EthAddress,
    EthAddress,
    EthU256,
    EthU256,
    u32,
    ContractBytes<[u8; 32]>,
    EthU256,
    ContractBytes<[u8; 32]>,
    bool,
    ContractBytes<[u8; 32]>,
    ContractBytes<[u8; 32]>,
);

pub type RawInteraction = (EthAddress, EthU256, ContractBytes<Vec<u8>>);
