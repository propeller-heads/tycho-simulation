use alloy_primitives::Address;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HookOptions {
    AfterRemoveLiquidityReturnsDelta = 0,
    AfterAddLiquidityReturnsDelta = 1,
    AfterSwapReturnsDelta = 2,
    BeforeSwapReturnsDelta = 3,
    AfterDonate = 4,
    BeforeDonate = 5,
    AfterSwap = 6,
    BeforeSwap = 7,
    AfterRemoveLiquidity = 8,
    BeforeRemoveLiquidity = 9,
    AfterAddLiquidity = 10,
    BeforeAddLiquidity = 11,
    AfterInitialize = 12,
    BeforeInitialize = 13,
}

// from https://github.com/shuhuiluo/uniswap-v4-sdk-rs/blob/main/src/utils/hook.rs#L69
pub const fn has_permission(address: Address, hook_option: HookOptions) -> bool {
    let mask = ((address.0 .0[18] as u64) << 8) | (address.0 .0[19] as u64);
    let hook_flag_index = hook_option as u64;
    mask & (1 << hook_flag_index) != 0
}
