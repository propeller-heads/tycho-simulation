use std::{env, path::PathBuf};

use ethcontract_generate::{loaders::TruffleLoader, ContractBuilder};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("missing OUT_DIR"));

    generate_contract(
        "BCowHelper",
        manifest_dir.join("artifacts/BCowHelper.json"),
        out_dir.join("b_cow_helper.rs"),
    );
    generate_contract(
        "GPv2AllowListAuthentication",
        manifest_dir.join("artifacts/GPv2AllowListAuthentication.json"),
        out_dir.join("g_pv_2_allow_list_authentication.rs"),
    );
    generate_contract(
        "GPv2Settlement",
        manifest_dir.join("artifacts/GPv2Settlement.json"),
        out_dir.join("g_pv_2_settlement.rs"),
    );
}

fn generate_contract(contract_name: &str, source: PathBuf, destination: PathBuf) {
    println!("cargo:rerun-if-changed={}", source.display());

    let contract = TruffleLoader::new()
        .name(contract_name)
        .load_contract_from_file(&source)
        .unwrap_or_else(|err| {
            panic!("failed loading contract artifact {}: {err}", source.display())
        });

    ContractBuilder::new()
        .generate(&contract)
        .unwrap_or_else(|err| panic!("failed generating bindings for {}: {err}", source.display()))
        .write_to_file(&destination)
        .unwrap_or_else(|err| panic!("failed writing bindings {}: {err}", destination.display()));
}
