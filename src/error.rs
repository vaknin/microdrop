use thiserror::Error;

#[derive(Debug, Error)]
pub enum MicrodropError {
    #[error("{feature} is not implemented yet")]
    Unimplemented { feature: &'static str },
    #[error("Audio error: {0}")]
    Audio(String),
    #[error("Transcription error: {0}")]
    Transcription(String),
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    #[error("Model download error: {0}")]
    ModelDownload(String),
    #[error("Model cache error: {0}")]
    ModelCache(String),
    #[error("Model registry error: {0}")]
    ModelRegistry(String),
    #[error("Configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, MicrodropError>;

impl MicrodropError {
    pub fn unimplemented(feature: &'static str) -> Self {
        MicrodropError::Unimplemented { feature }
    }
}
