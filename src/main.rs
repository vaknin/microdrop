use clap::Parser;
use tracing::error;

use microdrop::cli::Cli;
use microdrop::telemetry;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    telemetry::init();

    let cli = Cli::parse();

    if let Err(err) = cli.run().await {
        error!(error = %err, "microdrop command failed");
        std::process::exit(1);
    }
}
