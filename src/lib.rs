pub mod audio;
pub mod cli;
pub mod config;
pub mod model;
pub mod notify;
pub mod output;
pub mod telemetry;
pub mod transcribe;
pub mod workflow;

mod error;

pub use error::{MicrodropError, Result};
