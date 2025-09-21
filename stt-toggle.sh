#!/bin/bash

# Speech-to-Text Toggle Script
# Talks to the Python daemon via its CLI helper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/speech-to-text.py"
VENV_DIR="$SCRIPT_DIR/venv"
NOTIFY_ID=42420

notify_user() {
    local message="$1"
    local urgency="${2:-normal}"
    local timeout="${3:-1000}"
    local replace_id="${4:-}"

    if command -v notify-send >/dev/null 2>&1; then
        local cmd=(notify-send -t "$timeout" -u "$urgency")
        if [[ -n "$replace_id" ]]; then
            cmd+=(-r "$replace_id")
        fi
        cmd+=("Speech-to-Text" "$message")
        "${cmd[@]}"
    else
        echo "STT: $message"
    fi
}

play_sound() {
    local sound_type="$1"

    if command -v paplay >/dev/null 2>&1; then
        case "$sound_type" in
            start)
                paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null &
                ;;
            stop)
                paplay /usr/share/sounds/alsa/Front_Right.wav 2>/dev/null &
                ;;
        esac
    fi
}

ensure_environment() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        echo "Python script not found: $PYTHON_SCRIPT" >&2
        exit 1
    fi

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Virtual environment not found: $VENV_DIR" >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
}

main() {
    ensure_environment

    local status
    status=$(python "$PYTHON_SCRIPT" status 2>/dev/null)
    local status_rc=$?

    if [[ $status_rc -ne 0 ]]; then
        echo "Unable to contact speech daemon" >&2
        exit 1
    fi

    if [[ $status == "recording" ]]; then
        play_sound stop
        local output
        output=$(python "$PYTHON_SCRIPT" stop)
        local rc=$?
        local message
        message=$(echo "$output" | head -n1)
        if [[ $rc -eq 0 ]]; then
            :
        else
            echo "Stop failed: ${message:-see logs}" >&2
        fi
    else
        notify_user "Starting recording..." low 1000 "$NOTIFY_ID"
        play_sound start
        local output
        output=$(python "$PYTHON_SCRIPT" start)
        local rc=$?
        local message
        message=$(echo "$output" | head -n1)
        if [[ $rc -eq 0 ]]; then
            :
        else
            notify_user "Start failed: ${message:-see logs}" critical 5000 "$NOTIFY_ID"
        fi
    fi
}

main "$@"
