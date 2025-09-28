//! Configuration loading and merging primitives.

use std::path::{Path, PathBuf};
use std::fs;

use dirs;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::{MicrodropError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub output: OutputConfig,
    #[serde(default)]
    pub behavior: BehaviorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Preferred audio input device name (None = system default)
    pub device: Option<String>,
    /// Maximum recording duration in seconds (None = unlimited)
    pub max_duration: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model name or path
    pub default_model: Option<String>,
    /// Default quantization level
    pub default_quantization: Option<String>,
    /// Directory for cached models (None = default ~/.local/share/microdrop/models)
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Enable clipboard by default
    pub enable_clipboard: bool,
    /// Enable paste by default
    pub enable_paste: bool,
    /// Default timestamp format
    pub timestamp_format: String,
    /// Default file to append transcripts to
    pub append_file: Option<PathBuf>,
    /// Command to run for notifications
    pub notify_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    /// Enable audio feedback cues
    pub audio_cues: bool,
    /// Minimum silence duration before stopping auto-record (seconds)
    pub silence_threshold: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            model: ModelConfig::default(),
            output: OutputConfig::default(),
            behavior: BehaviorConfig::default(),
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: None,
            max_duration: None,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            default_quantization: None,
            cache_dir: None,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            enable_clipboard: true,
            enable_paste: false,
            timestamp_format: "none".to_string(),
            append_file: None,
            notify_command: None,
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            audio_cues: false,
            silence_threshold: None,
        }
    }
}

impl Config {
    /// Load configuration from the default location
    pub fn load() -> Result<Self> {
        let config_path = Self::default_config_path()?;
        Self::load_from_path(&config_path)
    }

    /// Load configuration from a specific file path
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            debug!("Config file not found at {}, using defaults", path.display());
            return Ok(Self::default());
        }

        let content = fs::read_to_string(path)
            .map_err(|e| MicrodropError::Config(format!("Failed to read config file: {}", e)))?;

        let config: Config = toml::from_str(&content)
            .map_err(|e| MicrodropError::Config(format!("Failed to parse config file: {}", e)))?;

        debug!("Loaded config from {}", path.display());
        Ok(config)
    }

    /// Write default configuration to the default location
    pub fn write_default(force: bool) -> Result<PathBuf> {
        let config_path = Self::default_config_path()?;
        Self::write_default_to_path(&config_path, force)
    }

    /// Write default configuration to a specific file path
    pub fn write_default_to_path<P: AsRef<Path>>(path: P, force: bool) -> Result<PathBuf> {
        let path = path.as_ref();

        if path.exists() && !force {
            return Err(MicrodropError::Config(format!(
                "Config file already exists at {}. Use --force to overwrite.",
                path.display()
            )));
        }

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| MicrodropError::Config(format!("Failed to create config directory: {}", e)))?;
        }

        let default_config = Config::default();
        let content = toml::to_string_pretty(&default_config)
            .map_err(|e| MicrodropError::Config(format!("Failed to serialize default config: {}", e)))?;

        fs::write(path, content)
            .map_err(|e| MicrodropError::Config(format!("Failed to write config file: {}", e)))?;

        debug!("Wrote default config to {}", path.display());
        Ok(path.to_path_buf())
    }

    /// Get the default configuration file path
    pub fn default_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| MicrodropError::Config("Unable to determine config directory".to_string()))?;

        Ok(config_dir.join("microdrop").join("config.toml"))
    }

    /// Merge CLI arguments into this configuration
    pub fn merge_cli_args(&mut self,
        device: Option<String>,
        duration: Option<u64>,
        model: Option<String>,
        quantized: Option<String>,
        paste: bool,
        no_clipboard: bool,
        timestamps: Option<String>,
        append: Option<PathBuf>,
        notify: Option<String>,
    ) {
        // Audio settings
        if device.is_some() {
            self.audio.device = device;
        }
        if duration.is_some() {
            self.audio.max_duration = duration;
        }

        // Model settings
        if model.is_some() {
            self.model.default_model = model;
        }
        if quantized.is_some() {
            self.model.default_quantization = quantized;
        }

        // Output settings - CLI args override config
        if paste {
            self.output.enable_paste = true;
        }
        if no_clipboard {
            self.output.enable_clipboard = false;
        }
        if let Some(ts) = timestamps {
            self.output.timestamp_format = ts;
        }
        if append.is_some() {
            self.output.append_file = append;
        }
        if notify.is_some() {
            self.output.notify_command = notify;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.output.enable_clipboard);
        assert!(!config.output.enable_paste);
        assert_eq!(config.output.timestamp_format, "none");
        assert!(config.audio.device.is_none());
        assert!(config.model.default_model.is_none());
    }

    #[test]
    fn test_load_nonexistent_config() {
        let result = Config::load_from_path("/nonexistent/path/config.toml");
        assert!(result.is_ok());
        let config = result.unwrap();
        // Should return default config
        assert!(config.output.enable_clipboard);
    }

    #[test]
    fn test_load_valid_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, r#"
[audio]
device = "test-device"
max_duration = 300

[model]
default_model = "small.en"
default_quantization = "q5_1"

[output]
enable_clipboard = false
enable_paste = true
timestamp_format = "simple"

[behavior]
audio_cues = true
silence_threshold = 2.0
"#).unwrap();

        let config = Config::load_from_path(temp_file.path()).unwrap();
        assert_eq!(config.audio.device, Some("test-device".to_string()));
        assert_eq!(config.audio.max_duration, Some(300));
        assert_eq!(config.model.default_model, Some("small.en".to_string()));
        assert_eq!(config.model.default_quantization, Some("q5_1".to_string()));
        assert!(!config.output.enable_clipboard);
        assert!(config.output.enable_paste);
        assert_eq!(config.output.timestamp_format, "simple");
        assert!(config.behavior.audio_cues);
        assert_eq!(config.behavior.silence_threshold, Some(2.0));
    }

    #[test]
    fn test_load_invalid_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "invalid toml content [").unwrap();

        let result = Config::load_from_path(temp_file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to parse config file"));
    }

    #[test]
    fn test_merge_cli_args() {
        let mut config = Config::default();

        config.merge_cli_args(
            Some("custom-device".to_string()),
            Some(120),
            Some("base.en".to_string()),
            Some("q8_0".to_string()),
            true,  // paste
            true,  // no_clipboard
            Some("detailed".to_string()),
            Some("/tmp/output.txt".into()),
            Some("notify-send".to_string()),
        );

        assert_eq!(config.audio.device, Some("custom-device".to_string()));
        assert_eq!(config.audio.max_duration, Some(120));
        assert_eq!(config.model.default_model, Some("base.en".to_string()));
        assert_eq!(config.model.default_quantization, Some("q8_0".to_string()));
        assert!(config.output.enable_paste);
        assert!(!config.output.enable_clipboard);
        assert_eq!(config.output.timestamp_format, "detailed");
        assert_eq!(config.output.append_file, Some("/tmp/output.txt".into()));
        assert_eq!(config.output.notify_command, Some("notify-send".to_string()));
    }

    #[test]
    fn test_write_and_read_default_config() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write default config
        let written_path = Config::write_default_to_path(path, true).unwrap();
        assert_eq!(written_path, path);

        // Read it back
        let config = Config::load_from_path(path).unwrap();

        // Should match defaults
        let default_config = Config::default();
        assert_eq!(config.output.enable_clipboard, default_config.output.enable_clipboard);
        assert_eq!(config.output.enable_paste, default_config.output.enable_paste);
        assert_eq!(config.output.timestamp_format, default_config.output.timestamp_format);
    }

    #[test]
    fn test_write_default_without_force_fails_if_exists() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write some content to the file to make it exist
        std::fs::write(path, "existing content").unwrap();

        // Write without force should fail
        let result = Config::write_default_to_path(path, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }
}
