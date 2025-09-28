use tracing_subscriber::EnvFilter;

/// Initialize tracing subscribers using `RUST_LOG` when provided.
pub fn init() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("microdrop=info"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}
