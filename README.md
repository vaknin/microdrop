# microdrop

🎤 Hotkey speech-to-text with auto-paste

A Python-based speech-to-text daemon that provides real-time audio transcription with automatic text insertion at the cursor position.

## Features

- **Real-time Speech Transcription**: Uses OpenAI's Whisper model via faster-whisper for high accuracy
- **Global Hotkey Support**: Designed to work with window manager hotkeys (e.g., i3wm)
- **Automatic Text Insertion**: Transcribed text is automatically copied to clipboard and pasted at cursor
- **Daemon Architecture**: Client-server model for fast response times and model caching
- **Audio Buffer Management**: Handles audio capture with overflow detection
- **Model Caching**: Intelligent model loading/unloading with configurable TTL

## Architecture

The tool consists of two main components:

1. **SpeechToText Class**: Handles audio recording, transcription, and text insertion
2. **CommandServer Class**: UNIX socket server that manages commands and maintains state

## Dependencies

### System Dependencies
- `xclip` - Clipboard management
- `xdotool` - Text insertion at cursor position
- Audio system (ALSA/PulseAudio) with input device

### Python Dependencies
- `faster-whisper` - Fast Whisper model implementation
- `numpy` - Audio data processing
- `sounddevice` - Audio capture
- `soundfile` - Audio file handling

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install faster-whisper numpy sounddevice soundfile
   ```

3. **Install system dependencies** (Arch Linux):
   ```bash
   sudo pacman -S xclip xdotool
   ```

## Usage

### Starting the Daemon
```bash
python speech-to-text.py serve
```

### Client Commands
- `start` - Begin recording audio
- `stop` - Stop recording and transcribe
- `status` - Check if currently recording
- `shutdown` - Stop the daemon
- `release_model` - Force release cached model

### Command Line Options
- `--json` - Output JSON responses
- `--no-auto-start` - Don't auto-start daemon if not running

### Examples
```bash
# Start recording
python speech-to-text.py start

# Stop recording and get transcription
python speech-to-text.py stop

# Check status
python speech-to-text.py status

# Shutdown daemon
python speech-to-text.py shutdown
```

## Configuration

### Default Settings
- **Sample Rate**: 16kHz
- **Model**: small.en (English-only, fast)
- **Socket Path**: `/tmp/stt-daemon.sock`
- **Model Cache TTL**: 600 seconds (10 minutes)
- **OpenMP Threads**: 14

### Environment Variables
- `OMP_NUM_THREADS` - Controls OpenMP threading (default: 14)
- `DISPLAY` - Required for clipboard/paste functionality

## Integration with i3wm

To set up a global hotkey in i3, add this to your i3 config:

```bash
bindsym $mod+s exec --no-startup-id "/path/to/speech-to-text.py start"
bindsym $mod+shift+s exec --no-startup-id "/path/to/speech-to-text.py stop"
```

Or create a toggle script that manages start/stop state automatically.

## Error Handling

The tool includes comprehensive error handling for:
- Missing dependencies
- Audio device unavailability
- Model loading failures
- Transcription errors
- Clipboard/paste failures
- Network connectivity issues

## Performance Notes

- Uses CPU-only inference with INT8 quantization for efficiency
- Model is cached in memory for fast subsequent transcriptions
- Automatic model release after inactivity to conserve memory
- Designed for real-time use with minimal latency

## Workflow

1. Position cursor in any text input field
2. Start recording (daemon begins audio capture)
3. Speak clearly into microphone
4. Stop recording (transcription begins automatically)
5. Transcribed text appears at cursor position

## License

This project appears to be a personal tool. No explicit license is provided.
