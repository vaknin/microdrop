//! Audio preprocessing utilities for format conversion and resampling.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tracing::{debug, warn};

use crate::{MicrodropError, Result};

const TARGET_SAMPLE_RATE: u32 = 16000;

pub struct AudioProcessor {
    resampler: Option<SincFixedIn<f32>>,
    input_sample_rate: u32,
    input_channels: u16,
}

impl AudioProcessor {
    pub fn new(input_sample_rate: u32, input_channels: u16) -> Result<Self> {
        let resampler = if input_sample_rate != TARGET_SAMPLE_RATE {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };

            let resampler = SincFixedIn::<f32>::new(
                TARGET_SAMPLE_RATE as f64 / input_sample_rate as f64,
                2.0, // max_resample_ratio_relative
                params,
                1024, // chunk_size
                input_channels as usize,
            )
            .map_err(|e| MicrodropError::Audio(format!("Failed to create resampler: {}", e)))?;

            Some(resampler)
        } else {
            None
        };

        debug!(
            "AudioProcessor initialized: {}Hz {}ch -> {}Hz 1ch",
            input_sample_rate, input_channels, TARGET_SAMPLE_RATE
        );

        Ok(Self {
            resampler,
            input_sample_rate,
            input_channels,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        // Handle empty input early
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Convert to mono if needed
        let mono_samples = if self.input_channels > 1 {
            self.downmix_to_mono(input)
        } else {
            input.to_vec()
        };

        // Step 2: Resample if needed
        let resampled = if self.resampler.is_some() && !mono_samples.is_empty() {
            let input_channels = vec![mono_samples];
            let output_channels = self
                .resampler
                .as_mut()
                .unwrap()
                .process(&input_channels, None)
                .map_err(|e| MicrodropError::Audio(format!("Resampling failed: {}", e)))?;
            output_channels.into_iter().next().unwrap_or_default()
        } else {
            mono_samples
        };

        debug!(
            "Processed {} input samples -> {} output samples",
            input.len(),
            resampled.len()
        );
        Ok(resampled)
    }

    fn downmix_to_mono(&self, interleaved: &[f32]) -> Vec<f32> {
        let channels = self.input_channels as usize;
        let frame_count = interleaved.len() / channels;
        let mut mono = Vec::with_capacity(frame_count);

        for frame_idx in 0..frame_count {
            let start = frame_idx * channels;
            let end = start + channels;

            if end <= interleaved.len() {
                let frame_sum: f32 = interleaved[start..end].iter().sum();
                mono.push(frame_sum / channels as f32);
            } else {
                warn!("Incomplete frame at end of audio buffer, skipping");
                break;
            }
        }

        debug!("Downmixed {} frames from {}ch to 1ch", mono.len(), channels);
        mono
    }

    pub fn get_output_sample_rate(&self) -> u32 {
        TARGET_SAMPLE_RATE
    }

    pub fn get_output_channels(&self) -> u16 {
        1
    }

    pub fn get_input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downmix_stereo_to_mono() {
        let processor = AudioProcessor::new(44100, 2).unwrap();

        // Stereo input: [L1, R1, L2, R2, L3, R3]
        let stereo_input = vec![1.0, -1.0, 0.5, 0.5, 2.0, 0.0];
        let mono_output = processor.downmix_to_mono(&stereo_input);

        // Expected: [(1.0 + -1.0)/2, (0.5 + 0.5)/2, (2.0 + 0.0)/2] = [0.0, 0.5, 1.0]
        assert_eq!(mono_output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_no_processing_needed() {
        let mut processor = AudioProcessor::new(16000, 1).unwrap();
        let input = vec![1.0, 0.5, -0.5, -1.0];
        let output = processor.process(&input).unwrap();

        // Should be unchanged
        assert_eq!(output, input);
    }

    #[test]
    fn test_empty_input() {
        let mut processor = AudioProcessor::new(44100, 2).unwrap();
        let output = processor.process(&[]).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_resampling_produces_output() {
        let mut processor = AudioProcessor::new(44100, 1).unwrap();

        // Generate enough samples to satisfy resampler buffer requirements
        let input: Vec<f32> = (0..100000) // Much larger buffer
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();

        let output = processor.process(&input).unwrap();

        // Just verify we get some output and it's reasonable
        assert!(!output.is_empty());
        assert!(output.len() < input.len()); // Should be downsampled

        // Check that samples are in reasonable range
        for sample in &output {
            assert!(sample.abs() <= 2.0);
        }
    }

    #[test]
    fn test_downmix_quad_to_mono() {
        let processor = AudioProcessor::new(44100, 4).unwrap();

        // Quad input: [L1, R1, SL1, SR1, L2, R2, SL2, SR2]
        let quad_input = vec![1.0, -1.0, 0.5, -0.5, 2.0, 0.0, 1.0, -1.0];
        let mono_output = processor.downmix_to_mono(&quad_input);

        // Expected: [(1.0 + -1.0 + 0.5 + -0.5)/4, (2.0 + 0.0 + 1.0 + -1.0)/4] = [0.0, 0.5]
        assert_eq!(mono_output, vec![0.0, 0.5]);
    }

    #[test]
    fn test_incomplete_frame_handling() {
        let processor = AudioProcessor::new(44100, 2).unwrap();

        // Incomplete stereo frame (3 samples instead of even number)
        let input = vec![1.0, -1.0, 0.5];
        let output = processor.downmix_to_mono(&input);

        // Should only process the complete frame
        assert_eq!(output, vec![0.0]); // (1.0 + -1.0) / 2 = 0.0
    }
}

// Property-based tests
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn downmix_preserves_energy_bounds(
            samples in prop::collection::vec(-1.0f32..1.0f32, 2..1000),
            channels in 1u16..8u16,
        ) {
            let processor = AudioProcessor::new(44100, channels).unwrap();

            // Ensure we have complete frames
            let frame_count = samples.len() / channels as usize;
            let complete_samples = &samples[..frame_count * channels as usize];

            if !complete_samples.is_empty() {
                let mono_output = processor.downmix_to_mono(complete_samples);

                // Output should have same number of frames as input
                prop_assert_eq!(mono_output.len(), frame_count);

                // All output samples should be within reasonable bounds
                for sample in &mono_output {
                    prop_assert!(sample.abs() <= 1.0);
                }
            }
        }

        #[test]
        fn resampling_preserves_sample_bounds(
            samples in prop::collection::vec(-1.0f32..1.0f32, 1100..5000), // Ensure enough samples for resampler
            input_rate in 8000u32..96000u32,
        ) {
            let mut processor = AudioProcessor::new(input_rate, 1).unwrap();

            // Only test if we have enough samples for resampling
            if input_rate != 16000 && samples.len() >= 1024 {
                let output = processor.process(&samples).unwrap();

                // All output samples should remain within bounds
                for sample in &output {
                    prop_assert!(sample.abs() <= 2.0); // Allow some tolerance for resampling artifacts
                }
            }
        }

        #[test]
        fn process_is_deterministic(
            frame_count in 300..500usize, // Generate frames instead of raw samples
            sample_rate in 16000u32..48000u32,
            channels in 1u16..4u16,
        ) {
            // Generate interleaved samples properly
            let mut samples = Vec::with_capacity(frame_count * channels as usize);
            for frame in 0..frame_count {
                for _ch in 0..channels {
                    let sample = ((frame * 1103515245 + 12345) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    samples.push(sample * 0.5); // Keep amplitude reasonable
                }
            }

            let mut processor1 = AudioProcessor::new(sample_rate, channels).unwrap();
            let mut processor2 = AudioProcessor::new(sample_rate, channels).unwrap();

            // Handle resampling buffer requirements - need enough frames
            if sample_rate == 16000 || (sample_rate != 16000 && frame_count >= 1024) {
                let output1 = processor1.process(&samples).unwrap();
                let output2 = processor2.process(&samples).unwrap();

                // Same input should produce same output
                prop_assert_eq!(output1, output2);
            }
        }

        #[test]
        fn empty_input_always_produces_empty_output(
            sample_rate in 8000u32..96000u32,
            channels in 1u16..8u16,
        ) {
            let mut processor = AudioProcessor::new(sample_rate, channels).unwrap();
            let output = processor.process(&[]).unwrap();
            prop_assert!(output.is_empty());
        }
    }
}
