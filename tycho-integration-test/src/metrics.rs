use actix_web::{rt::System, web, App, HttpResponse, HttpServer, Responder};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use miette::{Context, IntoDiagnostic, Result};
use tracing::info;
use tycho_client::feed::SynchronizerState;

/// Initialize the metrics registry and describe all metrics
pub fn initialize_metrics() {
    describe_histogram!(
        "tycho_integration_block_processing_duration_seconds",
        "Time between block timestamp and when protocol components are received"
    );
    describe_counter!(
        "tycho_integration_simulation_get_limits_failures_total",
        "Total number of failed get_limits operations"
    );
    describe_counter!(
        "tycho_integration_simulation_get_limits_success_total",
        "Total number of successful get_limits operations"
    );
    describe_counter!(
        "tycho_integration_simulation_get_amount_out_failures_total",
        "Total number of failed get_amount_out operations"
    );
    describe_counter!(
        "tycho_integration_simulation_get_amount_out_success_total",
        "Total number of successful get_amount_out operations"
    );
    describe_histogram!(
        "tycho_integration_simulation_get_amount_out_duration_seconds",
        "Time taken to execute get_amount_out operations"
    );
    describe_counter!(
        "tycho_integration_simulation_execution_success_total",
        "Total number of successful execution simulations"
    );
    describe_counter!(
        "tycho_integration_simulation_execution_failures_total",
        "Total number of failed execution simulations"
    );
    describe_histogram!(
        "tycho_integration_simulation_execution_slippage_ratio",
        "Slippage ratio between simulated and actual execution amounts"
    );
    describe_gauge!(
        "tycho_integration_protocol_sync_state",
        "Current synchronization state per protocol (1=Started, 2=Ready, 3=Delayed, 4=Stale, 5=Advanced, 6=Ended)"
    );
    describe_counter!(
        "tycho_integration_protocol_updates_skipped_total",
        "Total number of protocol updates skipped due to being behind current block"
    );
    describe_histogram!(
        "tycho_integration_protocol_update_block_delay_blocks",
        "Number of blocks behind current that protocol updates are received"
    );
}

/// Record the duration between block timestamp and component receipt
pub fn record_block_processing_duration(duration_seconds: f64) {
    histogram!("tycho_integration_block_processing_duration_seconds").record(duration_seconds);
}

/// Record a failed get_limits operation
pub fn record_get_limits_failure(protocol: &str) {
    counter!(
        "tycho_integration_simulation_get_limits_failures_total",
        "protocol" => protocol.to_string(),
    )
    .increment(1);
}

/// Record a successful get_limits operation
pub fn record_get_limits_success(protocol: &str) {
    counter!(
        "tycho_integration_simulation_get_limits_success_total",
        "protocol" => protocol.to_string(),
    )
    .increment(1);
}

/// Record a failed get_amount_out operation
pub fn record_get_amount_out_failure(protocol: &str) {
    counter!(
        "tycho_integration_simulation_get_amount_out_failures_total",
        "protocol" => protocol.to_string(),
    )
    .increment(1);
}

/// Record a successful get_amount_out operation
pub fn record_get_amount_out_success(protocol: &str) {
    counter!(
        "tycho_integration_simulation_get_amount_out_success_total",
        "protocol" => protocol.to_string(),
    )
    .increment(1);
}

/// Record the duration of a get_amount_out operation
pub fn record_get_amount_out_duration(protocol: &str, duration_seconds: f64) {
    histogram!(
        "tycho_integration_simulation_get_amount_out_duration_seconds",
        "protocol" => protocol.to_string(),
    )
    .record(duration_seconds);
}

/// Record a successful execution simulation
pub fn record_simulation_execution_success(protocol: &str) {
    counter!(
        "tycho_integration_simulation_execution_success_total",
        "protocol" => protocol.to_string(),
    )
    .increment(1);
}

/// Record a failed execution simulation
pub fn record_simulation_execution_failure(protocol: &str, error_name: &str) {
    // We can add more categories here when we find new meaningful ones
    let error_category = match error_name {
        e if e.contains("Couldn't find balance storage slot") => "Storage slot not found",
        e if e.contains("TychoRouter__NegativeSlippage") => "TychoRouter__NegativeSlippage",
        _ => "other",
    };

    counter!(
        "tycho_integration_simulation_execution_failures_total",
        "protocol" => protocol.to_string(),
        "error_category" => error_category.to_string(),
    )
    .increment(1);
}

/// Record slippage between simulation and execution
pub fn record_execution_slippage(protocol: &str, slippage_ratio: f64) {
    histogram!(
        "tycho_integration_simulation_execution_slippage_ratio",
        "protocol" => protocol.to_string(),
    )
    .record(slippage_ratio);
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
    gauge!(
        "tycho_integration_protocol_sync_state",
        "protocol" => protocol.to_string()
    )
    .set(state_value);
}

/// Record when a protocol update is skipped because it's behind the current block
pub fn record_protocol_update_skipped() {
    counter!("tycho_integration_protocol_updates_skipped_total").increment(1);
}

/// Record the block delay of protocol updates
pub fn record_protocol_update_block_delay(block_delay: u64) {
    histogram!("tycho_integration_protocol_update_block_delay_blocks").record(block_delay as f64);
}

/// Creates and runs the Prometheus metrics exporter using Actix Web.
/// Returns a JoinHandle that should be awaited to detect server failures.
pub async fn create_metrics_exporter(port: u16) -> Result<tokio::task::JoinHandle<Result<()>>> {
    let exporter_builder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("tycho_integration_block_processing_duration_seconds".to_string()),
            &[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0],
        )
        .map_err(|e| miette::miette!("Failed to set buckets: {}", e))?
        .set_buckets_for_metric(
            Matcher::Full(
                "tycho_integration_simulation_get_amount_out_duration_seconds".to_string(),
            ),
            &[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        )
        .map_err(|e| miette::miette!("Failed to set buckets: {}", e))?
        .set_buckets_for_metric(
            Matcher::Full("tycho_integration_simulation_execution_slippage_ratio".to_string()),
            &[
                -0.0025, -0.002, -0.0015, -0.001, -0.0005, -0.0001, 0.0, 0.0001, 0.0005, 0.001,
                0.0015, 0.002, 0.0025,
            ],
        )
        .map_err(|e| miette::miette!("Failed to set buckets: {}", e))?
        .set_buckets_for_metric(
            Matcher::Full("tycho_integration_protocol_update_block_delay_blocks".to_string()),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0],
        )
        .map_err(|e| miette::miette!("Failed to set buckets: {}", e))?;
    let handle = exporter_builder
        .install_recorder()
        .into_diagnostic()
        .wrap_err("Failed to install Prometheus recorder")?;

    // Verify port is available by attempting to bind
    let test_bind = std::net::TcpListener::bind(("0.0.0.0", port))
        .into_diagnostic()
        .wrap_err(format!("Failed to bind metrics server to port {} - port may be in use", port))?;
    drop(test_bind);

    info!("Starting Prometheus metrics server on 0.0.0.0:{}", port);

    // Spawn in a separate thread with its own Actix runtime
    let task = std::thread::spawn(move || {
        System::new().block_on(async move {
            HttpServer::new(move || {
                App::new().route(
                    "/metrics",
                    web::get().to({
                        let handle = handle.clone();
                        move || metrics_handler(handle.clone())
                    }),
                )
            })
            .bind(("0.0.0.0", port))
            .into_diagnostic()
            .wrap_err(format!("Failed to bind metrics server to port {}", port))?
            .run()
            .await
            .into_diagnostic()
            .wrap_err("Metrics server failed")
        })
    });

    // Wrap the thread handle in a tokio task
    let join_handle = tokio::spawn(async move {
        task.join()
            .map_err(|_| miette::miette!("Metrics server thread panicked"))?
    });

    Ok(join_handle)
}

/// Handles requests to the /metrics endpoint, rendering Prometheus metrics.
async fn metrics_handler(handle: PrometheusHandle) -> impl Responder {
    let metrics = handle.render();
    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4; charset=utf-8")
        .body(metrics)
}
