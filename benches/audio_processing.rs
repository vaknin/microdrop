use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use microdrop::audio::AudioProcessor;
use std::time::Duration;

// Generate synthetic audio data for benchmarking
fn generate_sine_wave(sample_rate: u32, duration_ms: u32, frequency: f32, channels: u16) -> Vec<f32> {
    let sample_count = (sample_rate * duration_ms / 1000) as usize;
    let total_samples = sample_count * channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..sample_count {
        let t = i as f32 / sample_rate as f32;
        let amplitude = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;

        // Add sample for each channel
        for _ in 0..channels {
            samples.push(amplitude);
        }
    }

    samples
}

fn generate_noise(sample_count: usize, channels: u16) -> Vec<f32> {
    let total_samples = sample_count * channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        // Simple pseudo-random noise
        let noise = ((i.wrapping_mul(1103515245).wrapping_add(12345)) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        samples.push(noise * 0.1); // Keep amplitude low
    }

    samples
}

fn bench_downmix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("downmix");

    // Test different channel configurations
    let configs = vec![
        (2, "stereo"),
        (4, "quad"),
        (6, "5.1"),
        (8, "7.1"),
    ];

    for (channels, name) in configs {
        let processor = AudioProcessor::new(44100, channels).unwrap();

        // Test different durations
        for duration_ms in [100, 500, 1000, 5000] {
            let samples = generate_sine_wave(44100, duration_ms, 440.0, channels);

            group.throughput(Throughput::Elements(samples.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}ms", name, duration_ms), samples.len()),
                &samples,
                |b, samples| {
                    b.iter(|| {
                        black_box(processor.downmix_to_mono(black_box(samples)));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_resampling_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("resampling");

    // Test different sample rate conversions
    let sample_rates = vec![
        (8000, "8kHz"),
        (22050, "22kHz"),
        (44100, "44kHz"),
        (48000, "48kHz"),
        (96000, "96kHz"),
    ];

    for (sample_rate, name) in sample_rates {
        if sample_rate == 16000 {
            continue; // Skip target rate (no resampling needed)
        }

        let mut processor = AudioProcessor::new(sample_rate, 1).unwrap();

        // Test different durations
        for duration_ms in [100, 500, 1000] {
            let samples = generate_sine_wave(sample_rate, duration_ms, 440.0, 1);

            group.throughput(Throughput::Elements(samples.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}ms", name, duration_ms), samples.len()),
                &samples,
                |b, samples| {
                    b.iter(|| {
                        let _ = black_box(processor.process(black_box(samples)));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_full_audio_processing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.measurement_time(Duration::from_secs(10));

    // Test realistic scenarios
    let scenarios = vec![
        (44100, 2, 1000, "cd_quality_stereo_1s"),
        (48000, 2, 5000, "studio_quality_stereo_5s"),
        (22050, 1, 10000, "voice_quality_mono_10s"),
        (96000, 4, 500, "hires_quad_0.5s"),
    ];

    for (sample_rate, channels, duration_ms, name) in scenarios {
        let mut processor = AudioProcessor::new(sample_rate, channels).unwrap();

        // Generate more realistic audio with multiple frequencies
        let samples = generate_mixed_content(sample_rate, duration_ms, channels);

        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(name, samples.len()),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let _ = black_box(processor.process(black_box(samples)));
                });
            },
        );
    }

    group.finish();
}

fn generate_mixed_content(sample_rate: u32, duration_ms: u32, channels: u16) -> Vec<f32> {
    let sample_count = (sample_rate * duration_ms / 1000) as usize;
    let total_samples = sample_count * channels as usize;
    let mut samples = Vec::with_capacity(total_samples);

    // Mix multiple frequencies to simulate real speech/audio
    let frequencies = [200.0, 440.0, 800.0, 1200.0];
    let amplitudes = [0.3, 0.4, 0.2, 0.1];

    for i in 0..sample_count {
        let t = i as f32 / sample_rate as f32;

        // Mix multiple sine waves
        let mut mixed_sample = 0.0;
        for (freq, amp) in frequencies.iter().zip(amplitudes.iter()) {
            mixed_sample += (2.0 * std::f32::consts::PI * freq * t).sin() * amp;
        }

        // Add some noise
        let noise = ((i.wrapping_mul(1103515245).wrapping_add(12345)) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        mixed_sample += noise * 0.05;

        // Add sample for each channel with slight variations
        for ch in 0..channels {
            let channel_variation = if ch == 0 { 1.0 } else { 0.8 + (ch as f32 * 0.1) };
            samples.push(mixed_sample * channel_variation);
        }
    }

    samples
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    // Test different buffer sizes to understand allocation patterns
    let buffer_sizes = vec![1024, 4096, 16384, 65536];

    for buffer_size in buffer_sizes {
        let mut processor = AudioProcessor::new(44100, 2).unwrap();
        let samples = generate_noise(buffer_size, 2);

        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("process_buffer", buffer_size),
            &samples,
            |b, samples| {
                b.iter(|| {
                    let _ = black_box(processor.process(black_box(samples)));
                });
            },
        );
    }

    group.finish();
}

fn bench_extreme_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    // Test edge cases that might occur in real usage

    // Very short buffers
    let mut processor_short = AudioProcessor::new(44100, 2).unwrap();
    let short_samples = generate_sine_wave(44100, 1, 440.0, 2); // 1ms
    group.bench_function("very_short_buffer", |b| {
        b.iter(|| {
            let _ = black_box(processor_short.process(black_box(&short_samples)));
        });
    });

    // Empty buffer
    let mut processor_empty = AudioProcessor::new(44100, 2).unwrap();
    let empty_samples: Vec<f32> = vec![];
    group.bench_function("empty_buffer", |b| {
        b.iter(|| {
            let _ = black_box(processor_empty.process(black_box(&empty_samples)));
        });
    });

    // Large buffer
    let mut processor_large = AudioProcessor::new(44100, 2).unwrap();
    let large_samples = generate_sine_wave(44100, 30000, 440.0, 2); // 30 seconds
    group.bench_function("large_buffer_30s", |b| {
        b.iter(|| {
            let _ = black_box(processor_large.process(black_box(&large_samples)));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_downmix_operations,
    bench_resampling_operations,
    bench_full_audio_processing_pipeline,
    bench_memory_allocation_patterns,
    bench_extreme_edge_cases
);
criterion_main!(benches);