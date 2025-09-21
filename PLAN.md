# Speech-To-Text Setup Plan for Arch Linux

## Goal
Create a global hotkey (Super+S) system that:
1. Records audio when pressed first time
2. Stops recording and transcribes when pressed again
3. Automatically inserts transcribed text at cursor position

## Technology Choice: faster-whisper
- **CPU Performance**: Up to 4x faster than openai/whisper with same accuracy
- **Accuracy**: Uses same OpenAI Whisper models
- **Memory**: Uses less RAM and supports INT8 quantization for CPU efficiency
- **Auto-download**: Models download automatically from Hugging Face on first use
- **Reference**: https://github.com/SYSTRAN/faster-whisper

## Implementation Plan

### Phase 1: Create Python Speech-to-Text Script (`speech-to-text.py`)
- Spin up a project `venv` for Python dependencies and document activation in setup notes
- Capture audio in pure Python via `sounddevice` (stream) and `soundfile` (write temp WAV) when called with "start"
- Stop capture cleanly on "stop", ensuring the temporary WAV filename is unique per run
- Transcribe with faster-whisper via its Python API (optionally `python -m faster_whisper.transcribe` for debugging)
- Keep CPU-only settings (`device="cpu"`, `compute_type="int8"`) and pin a default model name so the first download is predictable
- Output transcribed text to clipboard via xclip
- Paste text at cursor position with xdotool after a short, configurable delay
- Validate microphone/clipboard/display availability at startup and emit friendly errors

### Phase 2: Create Bash Toggle Wrapper (`stt-toggle.sh`)
- Track recording state via a lock file in `/tmp`, clearing it on normal exit and guarding against stale locks
- Call the Python script with start/stop accordingly and abort if the state file says a session is already running
- Handle hotkey press detection and state management; ensure only one instance can run at a time
- Provide visual/audio feedback for recording state (e.g., `notify-send` or a short sound)

### Phase 3: Configure i3 Hotkey
- Add Super+S hotkey binding to i3 config with an explicit command snippet
- Bind hotkey to execute the toggle script and reload i3 (`i3-msg reload`) after editing
- Configure proper key release handling for reliable text insertion and to avoid modifier-stuck issues

### Phase 4: Testing & Optimization
- Test recording quality and transcription accuracy across a few mic sources selected in pavucontrol
- Optimize model size (start with base model, upgrade to large-v3 if needed)
- Fine-tune timing delays for reliable text insertion and notification timing
- Test in various applications (terminal, text editors, browsers)

## Expected Workflow
1. Position cursor in any text input
2. Press Super+S to start recording (brief notification)
3. Speak your text
4. Press Super+S again to stop recording
5. Text automatically appears at cursor position after ~2-5 seconds

## Files to Create
- `speech-to-text.py` - Main transcription script
- `stt-toggle.sh` - Toggle wrapper script
- `~/.config/i3/config` - Add hotkey binding (modify existing)

## Dependencies (Already Installed)
- faster-whisper (Python package)
- alsa-utils or pulseaudio-alsa (audio recording)
- xdotool (text insertion)
- xclip (clipboard management)
- sounddevice (Python package)
- soundfile (Python package)
- python-virtualenv or Python 3's built-in `venv` module for isolated environment
