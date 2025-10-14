use num_bigint::BigUint;
use opentelemetry::{global, KeyValue};
use tycho_client::feed::SynchronizerState;
use tycho_common::Bytes;

/// Record the duration between block timestamp and component receipt
pub fn record_block_processing_duration(duration_seconds: f64) {
    let meter = global::meter("tycho-integration-test");
    let histogram = meter
        .f64_histogram("tycho_integration_block_processing_duration_seconds")
        .with_description("Time between block timestamp and when protocol components are received")
        .init();
    histogram.record(duration_seconds, &[]);
}

/// Record a failed get_limits operation
pub fn record_get_limits_failure(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    block_number: u64,
    token_in: &Bytes,
    token_out: &Bytes,
    error_message: String,
) {
    let meter = global::meter("tycho-integration-test");
    let counter = meter
        .u64_counter("tycho_integration_simulation_get_limits_failures_total")
        .with_description("Total number of failed get_limits operations")
        .init();

    counter.add(
        1,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
            KeyValue::new("block", block_number.to_string()),
            KeyValue::new("token_in", token_in.to_string()),
            KeyValue::new("token_out", token_out.to_string()),
            KeyValue::new("error_message", error_message),
        ],
    );
}

/// Record a failed get_amount_out operation
#[allow(clippy::too_many_arguments)]
pub fn record_get_amount_out_failure(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    block_number: u64,
    token_in: &Bytes,
    token_out: &Bytes,
    amount_in: &BigUint,
    error_message: String,
) {
    let meter = global::meter("tycho-integration-test");
    let counter = meter
        .u64_counter("tycho_integration_simulation_get_amount_out_failures_total")
        .with_description("Total number of failed get_amount_out operations")
        .init();

    counter.add(
        1,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
            KeyValue::new("block", block_number.to_string()),
            KeyValue::new("token_in", token_in.to_string()),
            KeyValue::new("token_out", token_out.to_string()),
            KeyValue::new("amount_in", amount_in.to_string()),
            KeyValue::new("error_message", error_message),
        ],
    );
}

/// Record the duration of a get_amount_out operation
pub fn record_get_amount_out_duration(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    duration_seconds: f64,
) {
    let meter = global::meter("tycho-integration-test");
    let histogram = meter
        .f64_histogram("tycho_integration_simulation_get_amount_out_duration_seconds")
        .with_description("Time taken to execute get_amount_out operations")
        .init();

    histogram.record(
        duration_seconds,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
        ],
    );
}

/// Record a successful execution simulation
pub fn record_simulation_execution_success(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    block_number: u64,
) {
    let meter = global::meter("tycho-integration-test");
    let counter = meter
        .u64_counter("tycho_integration_simulation_execution_success_total")
        .with_description("Total number of successful execution simulations")
        .init();

    counter.add(
        1,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
            KeyValue::new("block", block_number.to_string()),
        ],
    );
}

/// Record a failed execution simulation
#[allow(clippy::too_many_arguments)]
pub fn record_simulation_execution_failure(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    block_number: u64,
    error_message: &str,
    error_name: &str,
    tenderly_url: &str,
    overwrites: &str,
) {
    let meter = global::meter("tycho-integration-test");
    let counter = meter
        .u64_counter("tycho_integration_simulation_execution_failures_total")
        .with_description("Total number of failed execution simulations")
        .init();

    counter.add(
        1,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
            KeyValue::new("block", block_number.to_string()),
            KeyValue::new("error_message", error_message.to_string()),
            KeyValue::new("error_name", error_name.to_string()),
            KeyValue::new("tenderly_url", tenderly_url.to_string()),
            KeyValue::new("overwrites", overwrites.to_string()),
        ],
    );
}

/// Record slippage between simulation and execution
pub fn record_execution_slippage(
    simulation_id: &str,
    protocol: &str,
    component_id: &str,
    block_number: u64,
    slippage_ratio: f64,
) {
    let meter = global::meter("tycho-integration-test");
    let histogram = meter
        .f64_histogram("tycho_integration_simulation_execution_slippage_ratio")
        .with_description("Slippage ratio between simulated and actual execution amounts")
        .init();

    histogram.record(
        slippage_ratio,
        &[
            KeyValue::new("simulation_id", simulation_id.to_string()),
            KeyValue::new("protocol", protocol.to_string()),
            KeyValue::new("component_id", component_id.to_string()),
            KeyValue::new("block", block_number.to_string()),
        ],
    );
}

/// Record the current synchronization state of a protocol
pub fn record_protocol_sync_state(protocol: &str, sync_state: &SynchronizerState) {
    let state_value = match sync_state {
        SynchronizerState::Started => 1.0,
        SynchronizerState::Ready(_) => 2.0,
        SynchronizerState::Delayed(_) => 3.0,
        SynchronizerState::Stale(_) => 4.0,
        SynchronizerState::Advanced(_) => 5.0,
        SynchronizerState::Ended(_) => 6.0,
    };

    let meter = global::meter("tycho-integration-test");
    let gauge = meter
        .f64_up_down_counter("tycho_integration_protocol_sync_state")
        .with_description("Current synchronization state per protocol (1=Started, 2=Ready, 3=Delayed, 4=Stale, 5=Advanced, 6=Ended)")
        .init();

    gauge.add(state_value, &[KeyValue::new("protocol", protocol.to_string())]);
}

/// Record when a protocol update is skipped because it's behind the current block
pub fn record_protocol_update_skipped() {
    let meter = global::meter("tycho-integration-test");
    let counter = meter
        .u64_counter("tycho_integration_protocol_updates_skipped_total")
        .with_description(
            "Total number of protocol updates skipped due to being behind current block",
        )
        .init();

    counter.add(1, &[]);
}

/// Record the block delay of protocol updates
pub fn record_protocol_update_block_delay(block_delay: u64) {
    let meter = global::meter("tycho-integration-test");
    let histogram = meter
        .f64_histogram("tycho_integration_protocol_update_block_delay_blocks")
        .with_description("Number of blocks behind current that protocol updates are received")
        .init();

    histogram.record(block_delay as f64, &[]);
}
