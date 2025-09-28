use std::io;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use tracing::{debug, info};

use crate::audio::{AudioEngine, AudioProcessor};
use crate::model::{ModelManager, Quantization};
use crate::output::{OutputManager, TimestampFormat};
use crate::transcribe::{find_default_model, TranscriptionEngine};
use crate::{MicrodropError, Result};

#[derive(Debug, Clone, ValueEnum)]
pub enum TimestampFormatArg {
    None,
    Simple,
    Detailed,
}

impl From<TimestampFormatArg> for TimestampFormat {
    fn from(arg: TimestampFormatArg) -> Self {
        match arg {
            TimestampFormatArg::None => TimestampFormat::None,
            TimestampFormatArg::Simple => TimestampFormat::Simple,
            TimestampFormatArg::Detailed => TimestampFormat::Detailed,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "microdrop",
    version,
    about = "On-demand speech-to-text transcription"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Toggle(ToggleCommand),
    Model(ModelCommand),
    Config(ConfigCommand),
}

#[derive(Debug, Args)]
pub struct ToggleCommand {
    #[arg(long)]
    pub device: Option<String>,
    #[arg(long)]
    pub duration: Option<u64>,
    #[arg(long)]
    pub paste: bool,
    #[arg(long)]
    pub append: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub quantized: Option<String>,
    #[arg(long)]
    pub notify: Option<String>,
    #[arg(long)]
    pub no_clipboard: bool,
    #[arg(long, value_enum)]
    pub timestamps: Option<TimestampFormatArg>,
}

#[derive(Debug, Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub command: ModelSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelSubcommand {
    List,
    Install(ModelInstallCommand),
}

#[derive(Debug, Args)]
pub struct ModelInstallCommand {
    pub model: String,
    #[arg(long)]
    pub quantized: Option<String>,
}

#[derive(Debug, Args)]
pub struct ConfigCommand {
    #[command(subcommand)]
    pub command: ConfigSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum ConfigSubcommand {
    WriteDefault {
        #[arg(long)]
        force: bool,
    },
}

impl Cli {
    pub async fn run(&self) -> Result<()> {
        match &self.command {
            Commands::Toggle(command) => {
                info!(?command, "toggle command invoked");
                command.run().await
            }
            Commands::Model(command) => command.run().await,
            Commands::Config(command) => command.run().await,
        }
    }
}

impl ModelCommand {
    async fn run(&self) -> Result<()> {
        match &self.command {
            ModelSubcommand::List => {
                info!("model list command invoked");
                let model_manager = ModelManager::new()?;

                // List cached models
                let cached_models = model_manager.list_cached_models()?;

                if cached_models.is_empty() {
                    println!("No cached models found.");
                    println!("Use 'microdrop model install <model>' to download models.");
                } else {
                    println!("Cached models:");
                    for cached in &cached_models {
                        println!("  {} ({})", cached.info.name, cached.info.quantization);
                        println!("    Path: {}", cached.path.display());
                        println!("    Size: {}", cached.info.size);
                        println!();
                    }
                }

                // List available models
                println!("Available models for download:");
                let available_models = model_manager.list_available_models().await?;
                for model in &available_models {
                    println!("  {} ({}) - {}", model.name, model.quantization, model.size);
                }

                Ok(())
            }
            ModelSubcommand::Install(command) => {
                info!(?command, "model install command invoked");

                let model_manager = ModelManager::new()?;

                // Parse quantization if provided
                let quantization = if let Some(ref q) = command.quantized {
                    Some(q.parse::<Quantization>().map_err(|e| {
                        MicrodropError::ModelLoad(format!("Invalid quantization '{}': {}", q, e))
                    })?)
                } else {
                    None
                };

                // Install the model
                let model_path = model_manager.install_model(&command.model, quantization).await?;

                println!("Model '{}' installed successfully!", command.model);
                println!("Path: {}", model_path.display());

                Ok(())
            }
        }
    }
}

impl ConfigCommand {
    async fn run(&self) -> Result<()> {
        match &self.command {
            ConfigSubcommand::WriteDefault { force } => {
                info!(force = *force, "config write-default command invoked");
                let config_path = crate::config::Config::write_default(*force)?;
                println!("Default configuration written to: {}", config_path.display());
                Ok(())
            }
        }
    }
}

impl ToggleCommand {
    async fn run(&self) -> Result<()> {
        info!("Starting audio capture session");

        // Initialize audio engine
        let mut audio_engine = AudioEngine::new();

        // Select audio device
        audio_engine.select_device(self.device.as_deref())?;

        // Configure the stream
        audio_engine.configure_stream()?;

        // Start capture
        audio_engine.start_capture()?;

        // Wait for user input to stop (simple implementation for MVP)
        println!("Audio capture started. Press Enter to stop...");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| MicrodropError::Audio(format!("Failed to read input: {}", e)))?;

        // Stop capture and get samples
        let raw_samples = audio_engine.stop_capture()?;

        if raw_samples.is_empty() {
            println!("No audio captured");
            return Ok(());
        }

        // Get basic stats before processing
        let raw_stats = audio_engine.get_stats(&raw_samples);

        // Process audio (downmix to mono, resample to 16kHz)
        let mut processor = AudioProcessor::new(raw_stats.sample_rate, raw_stats.channels)?;
        let processed_samples = processor.process(&raw_samples)?;

        if processed_samples.is_empty() {
            println!("No processed audio available for transcription");
            return Ok(());
        }

        // Initialize transcription engine
        let model_path = if let Some(ref model) = self.model {
            // User specified a model path or name
            crate::transcribe::resolve_model_path(model, self.quantized.as_deref())?
        } else {
            // Try to find a default model
            find_default_model().ok_or_else(|| {
                MicrodropError::ModelLoad(
                    "No model specified and no default model found. \
                     Please specify a model with --model <path> or install a model with 'microdrop model install <model>'"
                        .to_string(),
                )
            })?
        };

        info!("Loading transcription model: {}", model_path.display());
        let transcription_engine = TranscriptionEngine::new(&model_path)?;

        // Run transcription
        info!("Running transcription...");
        let result = transcription_engine.transcribe(&processed_samples).await?;

        // Initialize output manager
        let mut output_manager = OutputManager::new()?;

        // Determine output settings
        let enable_clipboard = !self.no_clipboard;
        let enable_paste = self.paste;
        let timestamp_format = self
            .timestamps
            .as_ref()
            .map(|t| t.clone().into())
            .unwrap_or(TimestampFormat::None);

        // Output transcript using the output manager
        output_manager.output_transcript(
            &result,
            enable_clipboard,
            enable_paste,
            self.append.as_deref(),
            timestamp_format,
        )?;

        // Debug information goes to stderr
        debug!(
            "Transcription completed: {} segments, {:.2}s processing time",
            result.segments.len(),
            result.processing_time.as_secs_f64()
        );

        debug!("Toggle command completed successfully");
        Ok(())
    }
}
