use std::{env, env::VarError};

use miette::{miette, Result};

fn _get_env(var_name: &str) -> Result<Option<String>> {
    match env::var(var_name) {
        Ok(val) => {
            Ok(Some(val))
        },
        Err(e) => match e {
            VarError::NotPresent => Ok(None),
            VarError::NotUnicode(_) => Err(miette!("The environment variable `{var_name}` cannot be decoded because it is not some valid Unicode")),
        },
    }
}

pub fn get_env(var_name: &str) -> Option<String> {
    _get_env(var_name).ok().flatten()
}

pub fn get_env_with_default(var_name: &str, default_value: String) -> String {
    _get_env(var_name)
        .ok()
        .flatten()
        .unwrap_or(default_value)
}
