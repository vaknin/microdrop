# Microdrop Design

## Vision
Deliver a single Rust binary that performs high-quality speech-to-text transcription on-demand. The binary captures microphone audio, streams it to an embedded Whisper transcription engine, and publishes the transcript to stdout by default. Optional flags let users push the transcript to the clipboard, simulate `Ctrl+Shift+V` paste into the focused window. Users can bind the binary to any external hotkey system (i3, KDE, AutoHotkey, etc.) to automate start/stop without any resident daemon.

## Key Workflows
- `microdrop toggle` — start capturing microphone audio, stop with the same command (second invocation or signal), transcribe, print to stdout.
- `microdrop toggle --paste` — capture, transcribe, and then emit `Ctrl+Shift+V` plus copy the transcript into the clipboard so graphical applications receive the text immediately.
- `microdrop model install small.en --quantized q5_1` — download and prepare Whisper models (optional but recommended for smoother UX).

## Functional Requirements
- Capture microphone audio with low latency and without temporary files.
- Convert raw audio frames into the mono 16 kHz PCM format Whisper expects.
- Run Whisper inference locally, returning the best-effort transcript with timestamps.
- Emit transcript to stdout; optional integrations include clipboard updates and simulated `Ctrl+Shift+V` paste.
- Allow users to configure model path, default output behaviour, and notes file via CLI arguments or a config file at `~/.config/microdrop/config.toml`.
- Guarantee robust error reporting (exit codes, stderr messaging) while keeping stdout clean for transcript piping.

## Non-Goals
- No long-running daemon or background service.
- No built-in global hotkey manager; external tooling is expected to launch/terminate the binary.
- No cloud transcription, streaming to external APIs, or GUI.

## Quality Targets
- Maintain latency under ~2s for small/medium models on modern CPUs.
- Avoid audible glitches or missed frames during capture (buffer backpressure handling).
- Prevent data loss on abrupt termination; ensure audio buffers flush deterministically before transcription starts.
- Deliver reproducible builds with `cargo check`, `cargo fmt`, `cargo clippy`, and full test coverage on CI.

## Architecture Overview
### Crate Layout
- `microdrop` (binary crate)
  - `main.rs`: CLI entry via `clap` 4.x, command dispatch.
  - `config/`: load/merge settings from CLI, env, and TOML.
  - `audio/`: microphone session management (`cpal`), buffering, resampling.
  - `transcribe/`: Whisper engine wrapper (`whisper-rs` bindings), quantization support, inference pipeline.
  - `output/`: stdout printer, clipboard writer (`arboard`), and synthetic keypress (`enigo`).
  - `workflow/`: state machine orchestrating record→stop→transcribe flow with cancellation and error propagation.
  - `notify/`: optional audible or desktop notifications (future enhancement).

### Audio Capture
- Use `cpal` for cross-platform audio input. Configure stream for nearest supported format ≥16 kHz, 16-bit or 32-bit.
- In capture callback, push frames into a lock-free ring buffer (`ringbuf` crate) owned by the controller.
- Upon stop signal, drain the buffer into a contiguous `Vec<f32>`, downmix to mono if needed, and resample to 16 kHz with `rubato` or `speexdsp` (depending on latency vs. CPU trade-offs).
- This approach skips any temporary WAV files, minimizing disk I/O and latency while avoiding sample precision loss.

### Transcription Engine
- Leverage `whisper-rs`, the actively maintained Rust binding to `whisper.cpp`.
- Load models in GGML format; support both full-precision (`.bin`) and quantized files (`ggml-small.en-q5_1.bin`, etc.).
- Provide a thin async-friendly wrapper that receives prepared PCM buffers and returns structured transcripts (text + timestamps + confidence metrics).
- Implement a `TranscriptionError` enum via `thiserror` to distinguish model load, inference, and audio pre-processing faults.

### Output Targets
- `stdout`: default sink, streaming or single blob depending on inference mode.
- Clipboard: use `arboard` for cross-platform clipboard access; fall back gracefully if clipboard operations fail.
- Simulated paste: use `enigo` to emulate `Ctrl+Shift+V` (Linux/X11, Windows, macOS). Offer platform overrides if needed.

### Configuration
- Default config path: `~/.config/microdrop/config.toml` with sections for audio device selection, default model, output actions, notes file, and paste behaviour.
- CLI flags override config values. Provide `microdrop config write-default` to scaffold the file.

## Concurrency & Runtime Strategy
`cpal` delivers audio via callback threads; transcription is CPU-bound; clipboard/paste operations interact with the OS. Managing these concurrently benefits from a structured async runtime. `tokio` remains the modern, high-performance choice in the Rust ecosystem, with mature ecosystem support, cooperative task scheduling, and utilities for signals and async IO. The plan:
- Run the CLI and orchestration on `tokio::main` with `current_thread` runtime (single-threaded) plus dedicated blocking tasks for Whisper inference.
- Use `tokio::sync::mpsc` for signaling start/stop events and communicating audio buffers between capture and transcription tasks.
- Wrap blocking Whisper inference in `tokio::task::spawn_blocking` to keep the runtime responsive.

## User Feedback Without a Visible Terminal
Because many users will trigger `microdrop` via desktop hotkeys without an open terminal:
- Emit concise status logs to `stderr`, which window managers can capture if desired.
- Provide optional audible cues (short WAV embedded assets played via `rodio`) to mark start/stop.
- Offer `--notify command` hook so users can integrate with `notify-send` or platform-specific notifiers; this keeps the core binary headless while allowing feedback when desired.

## Model Management & Quantization
- Ship helper commands to list, download, and cache Whisper models. Support pre-quantized GGML files (Q4_0, Q5_1, Q8_0, etc.) for lower memory usage and faster inference.
- Document trade-offs: quantized models reduce RAM/CPU requirements with minor accuracy loss, ideal for laptops.
- Allow `microdrop toggle --model small.en --quantized q5_1` to pick the cached quantized variant automatically.
- Keep model cache under `~/.local/share/microdrop/models` with checksum verification.

## CLI Surface
```
microdrop toggle [--device <name>] [--duration <seconds>] [--paste] [--model <id|path>] [--quantized <variant>] [--notify <command>] [--no-clipboard]
microdrop model list
microdrop model install <model> [--quantized <variant>]
microdrop config write-default [--force]
```
- Recording ends when stdin receives EOF (Ctrl+D), SIGINT is delivered, or a second `microdrop record --stop` invocation signals via a PID/file lock. We will implement a lockfile-based toggle so external hotkeys can bind start/stop to the same command.
- Provide helpful exit codes (0 success, 1 recoverable error, etc.).

## Milestones
1. **Scaffold**: create new `microdrop` crate, integrate `clap`, `tokio`, `thiserror`, `tracing`.
2. **Audio MVP**: capture PCM into buffer, dump stats to stdout, verify no disk I/O.
3. **Transcription**: integrate `whisper-rs`, run inference on captured buffer, print transcript.
4. **Outputs**: implement clipboard + `Ctrl+Shift+V`, configurable timestamp formatting.
5. **Model Tooling**: add model manager commands with download + checksum + quantization options.
6. **Config & UX**: load config, refine error messages, document usage.
7. **Testing & CI**: add unit tests (audio pipeline, CLI parsing), integration tests with mocked transcription, and GitHub Actions.

## Testing Strategy
- Unit-test resampling/downmixing logic with synthetic waveforms.
- Integration-test CLI commands using `assert_cmd`, stubbing Whisper with a fixture model.
- Property-test clipboard/paste fallback logic under feature flags to avoid interfering with the developer environment (use cargo features to disable for CI).
- Benchmark inference latency across quantization levels to catch regressions.
