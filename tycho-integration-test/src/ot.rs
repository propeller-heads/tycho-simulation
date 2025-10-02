use miette::{miette, IntoDiagnostic, Result};
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{propagation::TraceContextPropagator, runtime, trace, Resource};
use serde::Deserialize;
use tracing::{error, Subscriber};
use tracing_subscriber::{
    layer::SubscriberExt, registry::LookupSpan, util::SubscriberInitExt, EnvFilter, Layer,
};

#[derive(Debug, Clone, Deserialize)]
pub struct TracingConfig {
    pub otlp_exporter_endpoint: String,
}

/// Initialize tracing: apply an `EnvFilter` using the `RUST_LOG` environment variable to define the
/// log levels, add a formatter layer logging trace events as JSON and on OpenTelemetry layer
/// exporting trace data.
pub fn init_tracing(config: TracingConfig) -> Result<()> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    global::set_error_handler(|error| error!(error = format!("{error:#}"), "otel error"))
        .map_err(|e| miette!("Failed to set error handler: {e}"))?;

    let format = tracing_subscriber::fmt::format()
        .with_level(true)
        .with_target(false)
        .compact();

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(otlp_layer(config)?)
        .with(tracing_subscriber::fmt::layer().event_format(format))
        .try_init()
        .map_err(|e| miette!("Failed to initialize tracing subscriber: {e}"))
}

/// Create an OTLP layer exporting tracing data.
fn otlp_layer<S>(config: TracingConfig) -> Result<impl Layer<S>>
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(config.otlp_exporter_endpoint);

    let trace_config = trace::config().with_resource(Resource::default());

    let batch_config = trace::BatchConfig::default()
        .with_max_queue_size(20_480)
        .with_max_export_batch_size(2_560);

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(trace_config)
        .with_batch_config(batch_config)
        .install_batch(runtime::Tokio)
        .into_diagnostic()?;

    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}
