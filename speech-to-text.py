#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SOCKET_PATH = Path("/tmp/stt-daemon.sock")
MODEL_CACHE_TTL_SECONDS = 600
SOCKET_WAIT_SECONDS = 10
SOCKET_WAIT_INTERVAL = 0.1
CLIENT_TIMEOUT_SECONDS = 30

DEFAULT_OMP_THREADS = "14"
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_OMP_THREADS)

np: Any | None = None
sd: Any | None = None
WhisperModel: Any | None = None


def _load_server_modules() -> None:
    global np, sd, WhisperModel
    if np is not None:
        return

    try:
        import numpy as _np
        import sounddevice as _sd
        from faster_whisper import WhisperModel as _WhisperModel
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        missing = exc.name or "dependency"
        raise RuntimeError(
            f"Missing Python dependency '{missing}'. Activate the virtual environment or install requirements."
        ) from exc

    np = _np
    sd = _sd
    WhisperModel = _WhisperModel


class SpeechToText:
    """Manage audio capture and transcription with an optional model cache."""

    def __init__(self, sample_rate: int = 16000, model_name: str = "small.en") -> None:
        _load_server_modules()

        self.sample_rate = sample_rate
        self.model_name = model_name

        self.model: Optional[WhisperModel] = None
        self._model_lock = threading.RLock()
        self._model_last_used: Optional[float] = None

        self._dependencies_validated = False

        self._audio_chunks: list[np.ndarray] = []
        self._record_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._recording_lock = threading.RLock()
        self._recording_active = False
        self._record_error: Optional[Exception] = None
        self._buffer_overflow = False

    # ------------------------------------------------------------------
    # Dependency handling
    # ------------------------------------------------------------------
    def validate_dependencies(self) -> Tuple[bool, str]:
        _load_server_modules()

        if self._dependencies_validated:
            return True, "Dependencies already validated"

        missing = []
        for dep in ("xclip", "xdotool"):
            if subprocess.run(["which", dep], capture_output=True).returncode != 0:
                missing.append(dep)

        if missing:
            return False, (
                "Missing dependencies: " + ", ".join(missing) +
                ". Install with: sudo pacman -S xclip xdotool"
            )

        if not os.environ.get("DISPLAY"):
            return False, "DISPLAY environment variable is not set; clipboard/paste unavailable"

        try:
            devices = sd.query_devices()
            if not any(device.get("max_input_channels", 0) > 0 for device in devices):
                return False, "No audio input devices detected"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"Cannot query audio devices: {exc}"

        self._dependencies_validated = True
        return True, "Dependencies validated"

    # ------------------------------------------------------------------
    # Recording management
    # ------------------------------------------------------------------
    def is_recording(self) -> bool:
        with self._recording_lock:
            return self._recording_active

    def start_recording(self) -> Tuple[bool, str]:
        with self._recording_lock:
            if self._recording_active:
                return False, "Recording already active"

            ok, message = self.validate_dependencies()
            if not ok:
                return False, message

            self._audio_chunks = []
            self._record_error = None
            self._buffer_overflow = False
            self._stop_event.clear()

            thread = threading.Thread(target=self._record_audio, daemon=True)
            self._record_thread = thread
            thread.start()
            self._recording_active = True

        return True, "Recording started"

    def _record_audio(self) -> None:
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                while not self._stop_event.is_set():
                    frames, overflow = stream.read(1024)
                    if overflow:
                        self._buffer_overflow = True
                    self._audio_chunks.append(frames.copy())
        except Exception as exc:  # pragma: no cover - defensive
            self._record_error = exc
        finally:
            with self._recording_lock:
                self._recording_active = False

    def stop_recording(self) -> Tuple[bool, str, Optional[str]]:
        with self._recording_lock:
            if not self._recording_active and not self._audio_chunks:
                return False, "No recording in progress", None

            self._stop_event.set()
            thread = self._record_thread

        if thread:
            thread.join(timeout=5)

        if self._record_error:
            return False, f"Recording error: {self._record_error}", None

        if not self._audio_chunks:
            return False, "No audio captured", None

        audio = np.concatenate(self._audio_chunks, axis=0).astype(np.float32).flatten()

        ok, transcript, message = self._transcribe(audio)
        if not ok:
            return False, message, None

        self._audio_chunks = []
        self._stop_event.clear()

        overflow_note = " (audio overflow detected)" if self._buffer_overflow else ""
        return True, f"Transcription complete{overflow_note}", transcript

    # ------------------------------------------------------------------
    # Transcription and clipboard helpers
    # ------------------------------------------------------------------
    def _ensure_model(self) -> Tuple[bool, Optional[str]]:
        with self._model_lock:
            if self.model is None:
                try:
                    self.model = WhisperModel(
                        self.model_name,
                        device="cpu",
                        compute_type="int8",
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    return False, f"Failed to load model '{self.model_name}': {exc}"
            self._model_last_used = time.time()
        return True, None

    def _transcribe(self, audio: "np.ndarray") -> Tuple[bool, Optional[str], str]:
        ok, message = self._ensure_model()
        if not ok:
            return False, None, message or "Model load failed"

        try:
            segments, _info = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
            )
            pieces = []
            for segment in segments:
                pieces.append(segment.text)
            transcript = "".join(pieces).strip()
        except Exception as exc:  # pragma: no cover - defensive
            return False, None, f"Transcription failed: {exc}"

        if not transcript:
            return False, None, "No speech detected"

        try:
            self._copy_to_clipboard(transcript)
        except Exception as exc:
            return False, None, f"Clipboard error: {exc}"

        try:
            self._paste_at_cursor()
        except Exception as exc:
            return False, None, f"Paste error: {exc}"

        with self._model_lock:
            self._model_last_used = time.time()

        return True, transcript, "Transcription complete"

    @staticmethod
    def _copy_to_clipboard(text: str) -> None:
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(text)
        if proc.returncode != 0:
            raise RuntimeError(f"xclip failed with code {proc.returncode}: {stderr}")

    @staticmethod
    def _paste_at_cursor() -> None:
        # Clearing modifiers ensures Super/Alt from the hotkey doesn't interfere.
        time.sleep(0.2)
        window_id: Optional[str] = None
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                window_id = result.stdout.strip() or None
        except FileNotFoundError as exc:  # pragma: no cover - dependency already validated
            raise RuntimeError("xdotool not available") from exc

        command = ["xdotool", "key"]
        if window_id:
            command.extend(["--window", window_id])
        command.extend(["--clearmodifiers", "ctrl+shift+v"])

        if subprocess.run(command, check=False).returncode != 0:
            raise RuntimeError("xdotool could not paste using ctrl+shift+v")

    def release_model_if_idle(self, force: bool = False) -> None:
        with self._model_lock:
            if self.model is None:
                return
            if force or (
                self._model_last_used is not None
                and (time.time() - self._model_last_used) > MODEL_CACHE_TTL_SECONDS
            ):
                self.model = None
                self._model_last_used = None


class CommandServer:
    """Simple UNIX socket server that mediates transcription commands."""

    def __init__(self, stt: SpeechToText, socket_path: Path = SOCKET_PATH) -> None:
        self.stt = stt
        self.socket_path = socket_path
        self._shutdown_event = threading.Event()
        self._server_socket: Optional[socket.socket] = None
        self._reaper_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                raise SystemExit(f"Cannot remove stale socket at {self.socket_path}")

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(str(self.socket_path))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)

        self._register_signals()
        self._start_model_reaper()

        try:
            while not self._shutdown_event.is_set():
                try:
                    client, _ = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                with client:
                    payload = self._read_message(client)
                    response = self._handle_payload(payload)
                    self._send_message(client, response)
        finally:
            self._shutdown_event.set()
            if self._reaper_thread:
                self._reaper_thread.join(timeout=1)
            self.stt.release_model_if_idle(force=True)
            if self._server_socket:
                self._server_socket.close()
            if self.socket_path.exists():
                self.socket_path.unlink()

    # ------------------------------------------------------------------
    def _handle_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        command = payload.get("command")
        if not command:
            return {"ok": False, "message": "Missing command"}

        try:
            if command == "start":
                ok, message = self.stt.start_recording()
                return {"ok": ok, "message": message, "recording": self.stt.is_recording()}

            if command == "stop":
                ok, message, transcript = self.stt.stop_recording()
                response: Dict[str, Any] = {
                    "ok": ok,
                    "message": message,
                    "recording": self.stt.is_recording(),
                }
                if transcript:
                    response["transcript"] = transcript
                return response

            if command == "status":
                recording = self.stt.is_recording()
                return {
                    "ok": True,
                    "message": "Recording" if recording else "Idle",
                    "recording": recording,
                }

            if command == "shutdown":
                self._shutdown_event.set()
                return {"ok": True, "message": "Shutting down", "recording": False}

            if command == "release_model":
                self.stt.release_model_if_idle(force=True)
                return {"ok": True, "message": "Model released", "recording": self.stt.is_recording()}

            return {"ok": False, "message": f"Unknown command: {command}"}

        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "message": f"Command '{command}' failed: {exc}"}

    # ------------------------------------------------------------------
    def _read_message(self, client: socket.socket) -> Dict[str, Any]:
        buffer = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            buffer += chunk
            if b"\n" in chunk:
                break
        data = buffer.split(b"\n", 1)[0]
        if not data:
            return {}
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_message(self, client: socket.socket, payload: Dict[str, Any]) -> None:
        message = json.dumps(payload).encode("utf-8") + b"\n"
        try:
            client.sendall(message)
        except BrokenPipeError:
            # Client disconnected; ignore to keep daemon alive
            pass

    def _register_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:  # pragma: no cover - signal handling
        if self.stt.is_recording():
            ok, message, _ = self.stt.stop_recording()
            if ok:
                print("Recording stopped due to signal", file=sys.stderr)
            else:
                print(f"Failed to stop recording on signal: {message}", file=sys.stderr)
        self._shutdown_event.set()
        if self._server_socket:
            self._server_socket.close()

    def _start_model_reaper(self) -> None:
        def _reaper() -> None:
            while not self._shutdown_event.wait(timeout=60):
                self.stt.release_model_if_idle()

        self._reaper_thread = threading.Thread(target=_reaper, daemon=True)
        self._reaper_thread.start()


# ----------------------------------------------------------------------
# Client helpers
# ----------------------------------------------------------------------

def start_daemon_process() -> None:
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def wait_for_socket(timeout: float = SOCKET_WAIT_SECONDS) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if SOCKET_PATH.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect(str(SOCKET_PATH))
                    payload = json.dumps({"command": "status"}).encode("utf-8") + b"\n"
                    sock.sendall(payload)

                    buffer = b""
                    while True:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        buffer += chunk
                        if b"\n" in chunk:
                            break
                    if buffer:
                        return True
            except (
                FileNotFoundError,
                ConnectionRefusedError,
                ConnectionResetError,
                BrokenPipeError,
                json.JSONDecodeError,
            ):
                pass
        time.sleep(SOCKET_WAIT_INTERVAL)
    return False


def send_command(command: str, auto_start: bool = True) -> Dict[str, Any]:
    # Don't auto-start daemon for shutdown command - it makes no sense
    if command == "shutdown":
        auto_start = False

    attempt = 0
    while True:
        attempt += 1
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(CLIENT_TIMEOUT_SECONDS)
                sock.connect(str(SOCKET_PATH))
                payload = json.dumps({"command": command}).encode("utf-8") + b"\n"
                sock.sendall(payload)

                buffer = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    if b"\n" in chunk:
                        break
                data = buffer.split(b"\n", 1)[0]
                if not data:
                    return {"ok": False, "message": "Empty response from daemon"}
                return json.loads(data.decode("utf-8"))
        except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, json.JSONDecodeError):
            if not auto_start or attempt >= 2:
                raise
            start_daemon_process()
            if not wait_for_socket():
                raise RuntimeError("Timed out waiting for speech daemon to start")


def client_main(args: argparse.Namespace) -> int:
    try:
        response = send_command(args.command, auto_start=not args.no_auto_start)
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        if args.command == "shutdown":
            print("No daemon running", file=sys.stderr)
        else:
            print(f"Error communicating with daemon: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error communicating with daemon: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response))
    else:
        if args.command == "status":
            state = "recording" if response.get("recording") else "idle"
            print(state)
        else:
            print(response.get("message", ""))
            if args.command == "stop" and response.get("transcript"):
                print(response["transcript"])

    return 0 if response.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech-to-text daemon controller")
    parser.add_argument(
        "command",
        choices=["serve", "start", "stop", "status", "shutdown", "release_model"],
        help="Command to execute",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON responses",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not automatically start the daemon if it's not running",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        server = CommandServer(SpeechToText())
        server.run()
        return 0

    return client_main(args)


if __name__ == "__main__":
    sys.exit(main())
