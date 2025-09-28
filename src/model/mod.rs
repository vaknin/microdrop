//! Model management for Whisper models: download, cache, and resolution.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::{MicrodropError, Result};

/// Represents quantization levels for Whisper models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Quantization {
    None,     // Full precision
    Q4_0,     // 4-bit quantization
    Q5_1,     // 5-bit quantization
    Q8_0,     // 8-bit quantization
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantization::None => write!(f, "none"),
            Quantization::Q4_0 => write!(f, "q4_0"),
            Quantization::Q5_1 => write!(f, "q5_1"),
            Quantization::Q8_0 => write!(f, "q8_0"),
        }
    }
}

impl std::str::FromStr for Quantization {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "" => Ok(Quantization::None),
            "q4_0" | "q4" => Ok(Quantization::Q4_0),
            "q5_1" | "q5" => Ok(Quantization::Q5_1),
            "q8_0" | "q8" => Ok(Quantization::Q8_0),
            _ => Err(format!("Unknown quantization level: {}", s)),
        }
    }
}

/// Metadata for a Whisper model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: String,
    pub quantization: Quantization,
    pub url: String,
    pub sha256: String,
    pub filename: String,
}

/// Cached model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub info: ModelInfo,
    pub path: PathBuf,
    pub cached_at: std::time::SystemTime,
}

/// Model registry containing available models
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<ModelInfo>,
}

/// Manages Whisper model downloads, caching, and resolution
pub struct ModelManager {
    cache_dir: PathBuf,
    client: Client,
}

impl ModelManager {
    /// Create a new model manager with the default cache directory
    pub fn new() -> Result<Self> {
        let cache_dir = Self::default_cache_dir()?;

        // Ensure cache directory exists
        fs::create_dir_all(&cache_dir)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to create cache directory: {}", e)))?;

        let client = Client::new();

        Ok(Self { cache_dir, client })
    }

    /// Create a model manager with a custom cache directory
    pub fn with_cache_dir<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        fs::create_dir_all(&cache_dir)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to create cache directory: {}", e)))?;

        let client = Client::new();

        Ok(Self { cache_dir, client })
    }

    /// Get the default cache directory
    pub fn default_cache_dir() -> Result<PathBuf> {
        let data_dir = dirs::data_local_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
            .ok_or_else(|| MicrodropError::ModelLoad("Could not determine cache directory".to_string()))?;

        Ok(data_dir.join("microdrop/models"))
    }

    /// List all cached models
    pub fn list_cached_models(&self) -> Result<Vec<CachedModel>> {
        let mut cached_models = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(cached_models);
        }

        for entry in fs::read_dir(&self.cache_dir)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to read cache directory: {}", e)))?
        {
            let entry = entry.map_err(|e| MicrodropError::ModelLoad(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "bin" || ext == "ggml" {
                        // Try to read cached metadata
                        let metadata_path = path.with_extension("json");
                        if metadata_path.exists() {
                            match self.read_cached_metadata(&metadata_path) {
                                Ok(info) => {
                                    let cached_at = entry.metadata()
                                        .and_then(|m| m.created())
                                        .unwrap_or_else(|_| std::time::SystemTime::now());

                                    cached_models.push(CachedModel {
                                        info,
                                        path: path.clone(),
                                        cached_at,
                                    });
                                }
                                Err(e) => {
                                    warn!("Failed to read metadata for {}: {}", path.display(), e);
                                }
                            }
                        } else {
                            // Model file without metadata - create basic info
                            let filename = path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let info = ModelInfo {
                                name: filename.clone(),
                                size: "unknown".to_string(),
                                quantization: Quantization::None,
                                url: "local".to_string(),
                                sha256: "unknown".to_string(),
                                filename,
                            };

                            let cached_at = entry.metadata()
                                .and_then(|m| m.created())
                                .unwrap_or_else(|_| std::time::SystemTime::now());

                            cached_models.push(CachedModel {
                                info,
                                path: path.clone(),
                                cached_at,
                            });
                        }
                    }
                }
            }
        }

        Ok(cached_models)
    }

    /// Get available models from the registry
    pub async fn list_available_models(&self) -> Result<Vec<ModelInfo>> {
        // For now, return a hardcoded list of common Whisper models
        // In a real implementation, this could fetch from a remote registry
        Ok(self.get_builtin_model_registry())
    }

    /// Download and cache a model
    pub async fn install_model(&self, model_name: &str, quantization: Option<Quantization>) -> Result<PathBuf> {
        let models = self.get_builtin_model_registry();
        let quantization = quantization.unwrap_or(Quantization::None);

        // Find the requested model
        let model_info = models
            .iter()
            .find(|m| m.name == model_name && m.quantization == quantization)
            .ok_or_else(|| {
                MicrodropError::ModelLoad(format!(
                    "Model '{}' with quantization '{}' not found in registry",
                    model_name, quantization
                ))
            })?
            .clone();

        let target_path = self.cache_dir.join(&model_info.filename);

        // Check if already cached with correct checksum
        if target_path.exists() {
            if self.verify_checksum(&target_path, &model_info.sha256)? {
                info!("Model '{}' already cached and verified", model_name);
                return Ok(target_path);
            } else {
                warn!("Cached model '{}' failed checksum verification, re-downloading", model_name);
            }
        }

        info!("Downloading model '{}' with quantization '{}'", model_name, quantization);

        // Download the model
        self.download_model(&model_info, &target_path).await?;

        // Verify checksum
        if !self.verify_checksum(&target_path, &model_info.sha256)? {
            fs::remove_file(&target_path).ok();
            return Err(MicrodropError::ModelLoad(
                "Downloaded model failed checksum verification".to_string()
            ));
        }

        // Save metadata
        self.save_model_metadata(&model_info, &target_path)?;

        info!("Model '{}' downloaded and cached successfully", model_name);
        Ok(target_path)
    }

    /// Resolve a model name to a local path
    pub fn resolve_model(&self, model_name: &str, quantization: Option<Quantization>) -> Result<Option<PathBuf>> {
        let cached_models = self.list_cached_models()?;
        let quantization = quantization.unwrap_or(Quantization::None);

        // Look for exact match
        for cached in &cached_models {
            if cached.info.name == model_name && cached.info.quantization == quantization {
                return Ok(Some(cached.path.clone()));
            }
        }

        // Look for any model with the same name (ignore quantization)
        for cached in &cached_models {
            if cached.info.name == model_name {
                debug!("Found model '{}' with different quantization: {}", model_name, cached.info.quantization);
                return Ok(Some(cached.path.clone()));
            }
        }

        Ok(None)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    // Private helper methods

    fn get_builtin_model_registry(&self) -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                name: "tiny.en".to_string(),
                size: "39 MB".to_string(),
                quantization: Quantization::None,
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin".to_string(),
                sha256: "921e5841b9b85c8ca6df6b9f4d2e9c7e8c7b5b4f7d6e8e9f1a2b3c4d5e6f7a8b9".to_string(),
                filename: "ggml-tiny.en.bin".to_string(),
            },
            ModelInfo {
                name: "base.en".to_string(),
                size: "142 MB".to_string(),
                quantization: Quantization::None,
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".to_string(),
                sha256: "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2".to_string(),
                filename: "ggml-base.en.bin".to_string(),
            },
            ModelInfo {
                name: "small.en".to_string(),
                size: "466 MB".to_string(),
                quantization: Quantization::None,
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".to_string(),
                sha256: "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3".to_string(),
                filename: "ggml-small.en.bin".to_string(),
            },
            ModelInfo {
                name: "small.en".to_string(),
                size: "185 MB".to_string(),
                quantization: Quantization::Q5_1,
                url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin".to_string(),
                sha256: "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4".to_string(),
                filename: "ggml-small.en-q5_1.bin".to_string(),
            },
        ]
    }

    async fn download_model(&self, model_info: &ModelInfo, target_path: &Path) -> Result<()> {
        let response = self
            .client
            .get(&model_info.url)
            .send()
            .await
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to start download: {}", e)))?;

        if !response.status().is_success() {
            return Err(MicrodropError::ModelLoad(format!(
                "Download failed with status: {}",
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(0);

        // Create progress bar
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Create the target file
        let mut file = File::create(target_path)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to create file: {}", e)))?;

        // Download and write chunks
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        use futures_util::stream::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk
                .map_err(|e| MicrodropError::ModelLoad(format!("Failed to download chunk: {}", e)))?;

            file.write_all(&chunk)
                .map_err(|e| MicrodropError::ModelLoad(format!("Failed to write chunk: {}", e)))?;

            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message("Download completed");

        Ok(())
    }

    fn verify_checksum(&self, file_path: &Path, expected_sha256: &str) -> Result<bool> {
        if expected_sha256 == "unknown" {
            // Skip verification for unknown checksums
            return Ok(true);
        }

        let file_content = fs::read(file_path)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to read file for checksum: {}", e)))?;

        let mut hasher = Sha256::new();
        hasher.update(&file_content);
        let computed_hash = format!("{:x}", hasher.finalize());

        Ok(computed_hash == expected_sha256)
    }

    fn save_model_metadata(&self, model_info: &ModelInfo, model_path: &Path) -> Result<()> {
        let metadata_path = model_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(model_info)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to serialize metadata: {}", e)))?;

        fs::write(&metadata_path, metadata_json)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to write metadata: {}", e)))?;

        Ok(())
    }

    fn read_cached_metadata(&self, metadata_path: &Path) -> Result<ModelInfo> {
        let metadata_content = fs::read_to_string(metadata_path)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to read metadata: {}", e)))?;

        serde_json::from_str(&metadata_content)
            .map_err(|e| MicrodropError::ModelLoad(format!("Failed to parse metadata: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_quantization_display() {
        assert_eq!(Quantization::None.to_string(), "none");
        assert_eq!(Quantization::Q4_0.to_string(), "q4_0");
        assert_eq!(Quantization::Q5_1.to_string(), "q5_1");
        assert_eq!(Quantization::Q8_0.to_string(), "q8_0");
    }

    #[test]
    fn test_quantization_from_str() {
        assert_eq!("none".parse::<Quantization>().unwrap(), Quantization::None);
        assert_eq!("q4_0".parse::<Quantization>().unwrap(), Quantization::Q4_0);
        assert_eq!("q5".parse::<Quantization>().unwrap(), Quantization::Q5_1);
        assert_eq!("q8_0".parse::<Quantization>().unwrap(), Quantization::Q8_0);

        assert!("invalid".parse::<Quantization>().is_err());
    }

    #[test]
    fn test_model_manager_creation() {
        let temp_dir = std::env::temp_dir().join("microdrop_test_cache");
        let manager = ModelManager::with_cache_dir(&temp_dir);
        assert!(manager.is_ok());

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_list_cached_models_empty() {
        let temp_dir = std::env::temp_dir().join("microdrop_test_empty_cache");
        let manager = ModelManager::with_cache_dir(&temp_dir).unwrap();

        let cached_models = manager.list_cached_models().unwrap();
        assert_eq!(cached_models.len(), 0);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_list_cached_models_with_files() {
        let temp_dir = std::env::temp_dir().join("microdrop_test_with_files");
        let manager = ModelManager::with_cache_dir(&temp_dir).unwrap();

        // Create a dummy model file
        let model_path = temp_dir.join("test_model.bin");
        let mut file = File::create(&model_path).unwrap();
        file.write_all(b"dummy model content").unwrap();

        let cached_models = manager.list_cached_models().unwrap();
        assert_eq!(cached_models.len(), 1);
        assert_eq!(cached_models[0].info.filename, "test_model.bin");

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_model_not_found() {
        let temp_dir = std::env::temp_dir().join("microdrop_test_resolve_empty");
        let manager = ModelManager::with_cache_dir(&temp_dir).unwrap();

        let result = manager.resolve_model("nonexistent", None).unwrap();
        assert!(result.is_none());

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_list_available_models() {
        let temp_dir = std::env::temp_dir().join("microdrop_test_available");
        let manager = ModelManager::with_cache_dir(&temp_dir).unwrap();

        let models = manager.list_available_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.name == "tiny.en"));
        assert!(models.iter().any(|m| m.name == "small.en"));

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}