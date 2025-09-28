//! Microphone capture and audio preprocessing pipeline.

use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Stream, StreamConfig};
use ringbuf::HeapRb;
use tracing::{debug, error, info};

use crate::{MicrodropError, Result};

pub mod processing;
pub use processing::*;

const RING_BUFFER_SIZE: usize = 1024 * 1024; // 1MB ring buffer

pub struct AudioEngine {
    host: Host,
    device: Option<Device>,
    config: Option<StreamConfig>,
    stream: Option<Stream>,
    ring_buffer: Option<HeapRb<f32>>,
}

#[derive(Debug, Clone)]
pub struct AudioStats {
    pub duration: Duration,
    pub sample_count: usize,
    pub sample_rate: u32,
    pub channels: u16,
    pub format: String,
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEngine {
    pub fn new() -> Self {
        let host = cpal::default_host();
        Self {
            host,
            device: None,
            config: None,
            stream: None,
            ring_buffer: None,
        }
    }

    pub fn list_devices(&self) -> Result<Vec<String>> {
        let devices: Result<Vec<String>> = self
            .host
            .input_devices()
            .map_err(|e| MicrodropError::Audio(format!("Failed to enumerate devices: {}", e)))?
            .map(|device| {
                device
                    .name()
                    .map_err(|e| MicrodropError::Audio(format!("Failed to get device name: {}", e)))
            })
            .collect();
        devices
    }

    pub fn select_device(&mut self, device_name: Option<&str>) -> Result<()> {
        let device = match device_name {
            Some(name) => {
                let devices = self.host.input_devices().map_err(|e| {
                    MicrodropError::Audio(format!("Failed to enumerate devices: {}", e))
                })?;

                devices
                    .filter(|d| d.name().map(|n| n == name).unwrap_or(false))
                    .next()
                    .ok_or_else(|| MicrodropError::Audio(format!("Audio device '{}' not found. Use 'arecord -l' or system audio settings to see available devices.", name)))?
            }
            None => self.host.default_input_device().ok_or_else(|| {
                MicrodropError::Audio("No default input device available. Please check that your microphone is connected and recognized by the system.".to_string())
            })?,
        };

        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown Device".to_string());
        info!("Selected audio device: {}", device_name);

        self.device = Some(device);
        Ok(())
    }

    pub fn configure_stream(&mut self) -> Result<()> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| MicrodropError::Audio("No device selected".to_string()))?;

        let supported_configs = device.supported_input_configs().map_err(|e| {
            MicrodropError::Audio(format!("Failed to get supported configs: {}", e))
        })?;

        let mut best_config = None;
        let mut best_sample_rate = 0;

        for config in supported_configs {
            let sample_rate = config.max_sample_rate().0;
            if sample_rate >= 16000 && sample_rate > best_sample_rate {
                best_sample_rate = sample_rate;
                best_config = Some(config.with_max_sample_rate());
            }
        }

        let config = best_config.ok_or_else(|| {
            MicrodropError::Audio("No suitable audio configuration found. The selected device does not support sampling rates compatible with speech transcription (16kHz or higher).".to_string())
        })?;

        debug!("Selected audio config: {:?}", config);
        self.config = Some(config.into());
        Ok(())
    }

    pub fn start_capture(&mut self) -> Result<()> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| MicrodropError::Audio("No device selected".to_string()))?;

        let config = self
            .config
            .as_ref()
            .ok_or_else(|| MicrodropError::Audio("No configuration set".to_string()))?;

        // For MVP, create a simple ring buffer placeholder
        let rb = HeapRb::<f32>::new(RING_BUFFER_SIZE);
        self.ring_buffer = Some(rb);

        // Build stream - simplified approach for MVP
        let stream = self.build_stream(device, config)?;

        // Start the stream
        stream
            .play()
            .map_err(|e| MicrodropError::Audio(format!("Failed to start stream: {}", e)))?;

        info!("Audio capture started");
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop_capture(&mut self) -> Result<Vec<f32>> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            info!("Audio capture stopped");
        }

        // For MVP, return empty vec for now - we'll implement proper ring buffer draining later
        let samples = Vec::new();
        self.ring_buffer = None;

        debug!("Collected {} samples from ring buffer", samples.len());
        Ok(samples)
    }

    pub fn get_stats(&self, samples: &[f32]) -> AudioStats {
        let config = self.config.as_ref();
        let sample_rate = config.map(|c| c.sample_rate.0).unwrap_or(44100);
        let channels = config.map(|c| c.channels).unwrap_or(1);

        let duration =
            Duration::from_secs_f64(samples.len() as f64 / (sample_rate as f64 * channels as f64));

        AudioStats {
            duration,
            sample_count: samples.len(),
            sample_rate,
            channels,
            format: "f32".to_string(),
        }
    }

    fn build_stream(&self, device: &Device, config: &StreamConfig) -> Result<Stream> {
        let err_callback = move |err| {
            error!("Audio stream error: {}", err);
        };

        // For MVP, create a simple f32 stream without ring buffer integration
        let stream = device
            .build_input_stream(
                config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // For MVP, just count samples - no actual storage
                    debug!("Received {} audio samples", data.len());
                },
                err_callback,
                None,
            )
            .map_err(|e| MicrodropError::Audio(format!("Failed to build input stream: {}", e)))?;

        Ok(stream)
    }
}
