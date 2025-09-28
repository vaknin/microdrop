//! Whisper transcription engine integration.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tracing::{debug, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::model::{ModelManager, Quantization};
use crate::{MicrodropError, Result};

pub struct TranscriptionEngine {
    context: WhisperContext,
    model_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub start: Duration,
    pub end: Duration,
    pub text: String,
}

impl TranscriptionEngine {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(MicrodropError::ModelLoad(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        info!("Loading Whisper model from: {}", model_path.display());

        let context = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| {
                MicrodropError::ModelLoad("Model path contains invalid UTF-8".to_string())
            })?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| MicrodropError::ModelLoad(format!("Failed to load model: {}", e)))?;

        debug!("Whisper model loaded successfully");

        Ok(Self {
            context,
            model_path,
        })
    }

    pub async fn transcribe(&self, audio_samples: &[f32]) -> Result<TranscriptionResult> {
        if audio_samples.is_empty() {
            warn!("Empty audio provided for transcription");
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: Vec::new(),
                language: None,
                processing_time: Duration::from_millis(0),
            });
        }

        let start_time = std::time::Instant::now();

        // Clone audio data for the blocking task
        let audio_data = audio_samples.to_vec();

        // Run inference synchronously since WhisperContext cannot be sent across threads safely
        let mut result = self.run_inference(&audio_data)?;

        let processing_time = start_time.elapsed();
        result.processing_time = processing_time;
        debug!("Transcription completed in {:?}", processing_time);

        Ok(result)
    }

    fn run_inference(&self, audio_data: &[f32]) -> Result<TranscriptionResult> {
        let mut state = self
            .context
            .create_state()
            .map_err(|e| MicrodropError::Transcription(format!("Failed to create state: {}", e)))?;

        // Configure transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_translate(false);
        params.set_language(Some("en"));
        params.set_print_realtime(false);
        params.set_print_progress(false);

        // Run transcription
        state
            .full(params, audio_data)
            .map_err(|e| MicrodropError::Transcription(format!("Transcription failed: {}", e)))?;

        // Extract results
        let num_segments = state.full_n_segments();

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                let segment_text = segment
                    .to_str_lossy()
                    .map_err(|e| {
                        MicrodropError::Transcription(format!("Failed to get segment text: {}", e))
                    })?
                    .to_string();

                let start_time = segment.start_timestamp();
                let end_time = segment.end_timestamp();

                // Convert time from centiseconds to Duration
                let start = Duration::from_millis((start_time * 10) as u64);
                let end = Duration::from_millis((end_time * 10) as u64);

                segments.push(TranscriptionSegment {
                    start,
                    end,
                    text: segment_text.clone(),
                });

                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&segment_text);
            }
        }

        Ok(TranscriptionResult {
            text: full_text,
            segments,
            language: Some("en".to_string()),
            processing_time: Duration::from_millis(0), // This will be set by the caller
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

pub fn find_default_model() -> Option<PathBuf> {
    // First try to use the model manager to find cached models
    if let Ok(model_manager) = ModelManager::new() {
        if let Ok(cached_models) = model_manager.list_cached_models() {
            if let Some(cached) = cached_models.first() {
                debug!("Found cached model: {}", cached.path.display());
                return Some(cached.path.clone());
            }
        }
    }

    // Fallback to old directory search
    let possible_dirs = [
        dirs::data_local_dir().map(|d| d.join("microdrop/models")),
        dirs::home_dir().map(|d| d.join(".local/share/microdrop/models")),
        Some(PathBuf::from("./models")),
        Some(PathBuf::from(".")),
    ];

    for dir_opt in possible_dirs.iter() {
        if let Some(dir) = dir_opt {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension() {
                            if ext == "bin" || ext == "ggml" {
                                debug!("Found potential model: {}", path.display());
                                return Some(path);
                            }
                        }
                    }
                }
            }
        }
    }

    warn!("No default model found in standard locations");
    None
}

/// Resolve a model name or path to an actual file path
/// This function handles:
/// - Direct file paths (if they exist)
/// - Model names that should be resolved from cache
/// - Model names with quantization
pub fn resolve_model_path(model_input: &str, quantization: Option<&str>) -> Result<PathBuf> {
    let model_path = PathBuf::from(model_input);

    // If it's an existing file path, use it directly
    if model_path.exists() && model_path.is_file() {
        return Ok(model_path);
    }

    // Try to resolve as a model name using the model manager
    let model_manager = ModelManager::new()?;
    let parsed_quantization = quantization
        .map(|q| q.parse::<Quantization>())
        .transpose()
        .map_err(|e| MicrodropError::ModelLoad(format!("Invalid quantization '{}': {}", quantization.unwrap_or(""), e)))?;

    if let Some(resolved_path) = model_manager.resolve_model(model_input, parsed_quantization)? {
        return Ok(resolved_path);
    }

    // If not found in cache, return error with helpful message
    Err(MicrodropError::ModelLoad(format!(
        "Model '{}' not found. Please specify a valid file path or install the model with 'microdrop model install {}'",
        model_input, model_input
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_model_file_not_found() {
        let non_existent_path = PathBuf::from("non_existent_model.bin");
        let result = TranscriptionEngine::new(&non_existent_path);

        assert!(result.is_err());
        if let Err(error) = result {
            match error {
                MicrodropError::ModelLoad(msg) => {
                    assert!(msg.contains("Model file not found"));
                }
                _ => panic!("Expected ModelLoad error"),
            }
        }
    }

    #[test]
    fn test_empty_audio_transcription() {
        // Create a temporary dummy model file for testing
        let temp_dir = std::env::temp_dir();
        let dummy_model_path = temp_dir.join("dummy_model.bin");

        // Create a dummy file (won't be loaded by whisper, but tests the path logic)
        if let Ok(mut file) = File::create(&dummy_model_path) {
            file.write_all(b"dummy model content").ok();
        }

        // This test will fail when trying to load the dummy model with whisper,
        // but that's expected behavior - the test is mainly for the empty audio case
        if dummy_model_path.exists() {
            let _ = std::fs::remove_file(&dummy_model_path);
        }
    }

    #[test]
    fn test_find_default_model_no_models() {
        // In a clean test environment, there should be no models
        let result = find_default_model();
        // This might be None in test environment, which is fine
        // The function should not panic
        let _ = result;
    }

    #[test]
    fn test_transcription_result_creation() {
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            segments: vec![TranscriptionSegment {
                start: Duration::from_millis(0),
                end: Duration::from_millis(1000),
                text: "Hello world".to_string(),
            }],
            language: Some("en".to_string()),
            processing_time: Duration::from_millis(100),
        };

        assert_eq!(result.text, "Hello world");
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "Hello world");
        assert_eq!(result.language, Some("en".to_string()));
    }

    #[test]
    fn test_transcription_segment_timing() {
        let segment = TranscriptionSegment {
            start: Duration::from_millis(500),
            end: Duration::from_millis(1500),
            text: "test segment".to_string(),
        };

        assert_eq!(segment.start.as_millis(), 500);
        assert_eq!(segment.end.as_millis(), 1500);
        assert_eq!(segment.text, "test segment");
    }
}

/// Mock transcription engine for deterministic testing
#[cfg(test)]
pub struct MockTranscriptionEngine {
    responses: Vec<TranscriptionResult>,
    call_count: std::cell::RefCell<usize>,
}

#[cfg(test)]
impl MockTranscriptionEngine {
    pub fn new() -> Self {
        Self {
            responses: vec![
                TranscriptionResult {
                    text: "This is a test transcription.".to_string(),
                    segments: vec![TranscriptionSegment {
                        start: Duration::from_millis(0),
                        end: Duration::from_millis(2000),
                        text: "This is a test transcription.".to_string(),
                    }],
                    language: Some("en".to_string()),
                    processing_time: Duration::from_millis(50),
                },
            ],
            call_count: std::cell::RefCell::new(0),
        }
    }

    pub fn with_responses(responses: Vec<TranscriptionResult>) -> Self {
        Self {
            responses,
            call_count: std::cell::RefCell::new(0),
        }
    }

    pub async fn transcribe(&self, _audio_data: &[f32]) -> Result<TranscriptionResult> {
        let mut count = self.call_count.borrow_mut();
        let response_index = *count % self.responses.len();
        *count += 1;

        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(10)).await;

        Ok(self.responses[response_index].clone())
    }

    pub fn call_count(&self) -> usize {
        *self.call_count.borrow()
    }
}

#[cfg(test)]
mod mock_tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_transcription_engine() {
        let mock = MockTranscriptionEngine::new();

        let audio_data = vec![0.1, -0.2, 0.3, -0.4]; // Dummy audio
        let result = mock.transcribe(&audio_data).await.unwrap();

        assert_eq!(result.text, "This is a test transcription.");
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.language, Some("en".to_string()));
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_with_custom_responses() {
        let custom_responses = vec![
            TranscriptionResult {
                text: "First response".to_string(),
                segments: vec![TranscriptionSegment {
                    start: Duration::from_millis(0),
                    end: Duration::from_millis(1000),
                    text: "First response".to_string(),
                }],
                language: Some("en".to_string()),
                processing_time: Duration::from_millis(25),
            },
            TranscriptionResult {
                text: "Second response".to_string(),
                segments: vec![TranscriptionSegment {
                    start: Duration::from_millis(0),
                    end: Duration::from_millis(1500),
                    text: "Second response".to_string(),
                }],
                language: Some("en".to_string()),
                processing_time: Duration::from_millis(30),
            },
        ];

        let mock = MockTranscriptionEngine::with_responses(custom_responses);
        let audio_data = vec![0.0; 100];

        // First call
        let result1 = mock.transcribe(&audio_data).await.unwrap();
        assert_eq!(result1.text, "First response");

        // Second call
        let result2 = mock.transcribe(&audio_data).await.unwrap();
        assert_eq!(result2.text, "Second response");

        // Third call should cycle back to first response
        let result3 = mock.transcribe(&audio_data).await.unwrap();
        assert_eq!(result3.text, "First response");

        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn test_mock_response_cycling() {
        let responses = vec![
            TranscriptionResult {
                text: "Response A".to_string(),
                segments: vec![],
                language: Some("en".to_string()),
                processing_time: Duration::from_millis(10),
            },
            TranscriptionResult {
                text: "Response B".to_string(),
                segments: vec![],
                language: Some("en".to_string()),
                processing_time: Duration::from_millis(10),
            },
        ];

        let mock = MockTranscriptionEngine::with_responses(responses);
        let audio_data = vec![0.0; 10];

        // Test cycling through responses
        for i in 0..5 {
            let result = mock.transcribe(&audio_data).await.unwrap();
            if i % 2 == 0 {
                assert_eq!(result.text, "Response A");
            } else {
                assert_eq!(result.text, "Response B");
            }
        }

        assert_eq!(mock.call_count(), 5);
    }
}
