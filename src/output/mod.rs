//! Output handling for transcripts: stdout, clipboard, paste simulation, and file append.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use tracing::{debug, info, warn};

use crate::transcribe::TranscriptionResult;
use crate::{MicrodropError, Result};

#[derive(Debug, Clone)]
pub enum TimestampFormat {
    None,
    Simple,
    Detailed,
}

pub struct OutputManager {
    clipboard: Option<Clipboard>,
    enigo: Option<Enigo>,
}

impl OutputManager {
    pub fn new() -> Result<Self> {
        let clipboard = match Clipboard::new() {
            Ok(clipboard) => {
                debug!("Clipboard initialized successfully");
                Some(clipboard)
            }
            Err(e) => {
                warn!("Failed to initialize clipboard: {}", e);
                None
            }
        };

        let enigo = match Enigo::new(&Settings::default()) {
            Ok(enigo) => {
                debug!("Input simulation initialized successfully");
                Some(enigo)
            }
            Err(e) => {
                warn!("Failed to initialize input simulation: {}", e);
                None
            }
        };

        Ok(Self { clipboard, enigo })
    }

    pub fn output_transcript(
        &mut self,
        result: &TranscriptionResult,
        enable_clipboard: bool,
        enable_paste: bool,
        append_file: Option<&Path>,
        timestamp_format: TimestampFormat,
    ) -> Result<()> {
        let formatted_text = self.format_transcript(result, &timestamp_format);

        // Always output to stdout (clean for piping)
        println!("{}", result.text);

        // Copy to clipboard if enabled and available
        if enable_clipboard {
            if let Err(e) = self.copy_to_clipboard(&formatted_text) {
                warn!("Failed to copy to clipboard: {}", e);
            }
        }

        // Simulate paste if enabled and available
        if enable_paste {
            if let Err(e) = self.simulate_paste(&formatted_text) {
                warn!("Failed to simulate paste: {}", e);
            }
        }

        // Append to file if specified
        if let Some(path) = append_file {
            if let Err(e) = self.append_to_file(&formatted_text, path) {
                warn!("Failed to append to file {}: {}", path.display(), e);
            }
        }

        Ok(())
    }

    fn format_transcript(&self, result: &TranscriptionResult, format: &TimestampFormat) -> String {
        match format {
            TimestampFormat::None => result.text.clone(),
            TimestampFormat::Simple => {
                if result.segments.is_empty() {
                    result.text.clone()
                } else {
                    let mut formatted = String::new();
                    for segment in &result.segments {
                        formatted.push_str(&format!(
                            "[{:.1}s] {}\n",
                            segment.start.as_secs_f64(),
                            segment.text
                        ));
                    }
                    formatted.trim_end().to_string()
                }
            }
            TimestampFormat::Detailed => {
                if result.segments.is_empty() {
                    result.text.clone()
                } else {
                    let mut formatted = String::new();
                    for segment in &result.segments {
                        formatted.push_str(&format!(
                            "[{:.1}s - {:.1}s] {}\n",
                            segment.start.as_secs_f64(),
                            segment.end.as_secs_f64(),
                            segment.text
                        ));
                    }
                    formatted.trim_end().to_string()
                }
            }
        }
    }

    fn copy_to_clipboard(&mut self, text: &str) -> Result<()> {
        match &mut self.clipboard {
            Some(clipboard) => {
                clipboard
                    .set_text(text)
                    .map_err(|e| MicrodropError::Audio(format!("Clipboard error: {}", e)))?;
                info!("Text copied to clipboard");
                Ok(())
            }
            None => Err(MicrodropError::Audio("Clipboard not available".to_string())),
        }
    }

    fn simulate_paste(&mut self, text: &str) -> Result<()> {
        match &mut self.clipboard {
            Some(clipboard) => {
                // First copy to clipboard
                clipboard
                    .set_text(text)
                    .map_err(|e| MicrodropError::Audio(format!("Clipboard error: {}", e)))?;

                // Then simulate Ctrl+Shift+V
                match &mut self.enigo {
                    Some(enigo) => {
                        // Small delay to ensure clipboard is ready
                        std::thread::sleep(Duration::from_millis(50));

                        // Simulate Ctrl+Shift+V using the new enigo API
                        enigo.key(Key::Control, Direction::Press).map_err(|e| {
                            MicrodropError::Audio(format!("Key press failed: {}", e))
                        })?;
                        enigo.key(Key::Shift, Direction::Press).map_err(|e| {
                            MicrodropError::Audio(format!("Key press failed: {}", e))
                        })?;
                        enigo
                            .key(Key::Unicode('v'), Direction::Click)
                            .map_err(|e| {
                                MicrodropError::Audio(format!("Key press failed: {}", e))
                            })?;
                        enigo.key(Key::Shift, Direction::Release).map_err(|e| {
                            MicrodropError::Audio(format!("Key press failed: {}", e))
                        })?;
                        enigo.key(Key::Control, Direction::Release).map_err(|e| {
                            MicrodropError::Audio(format!("Key press failed: {}", e))
                        })?;

                        info!("Simulated Ctrl+Shift+V paste");
                        Ok(())
                    }
                    None => Err(MicrodropError::Audio(
                        "Input simulation not available on this platform. Paste functionality requires X11 on Linux, or running on Windows/macOS.".to_string(),
                    )),
                }
            }
            None => Err(MicrodropError::Audio(
                "Clipboard not available for paste simulation. Please ensure your system supports clipboard operations.".to_string(),
            )),
        }
    }

    fn append_to_file(&self, text: &str, path: &Path) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| MicrodropError::Audio(format!("Failed to open file: {}", e)))?;

        writeln!(file, "{}", text)
            .map_err(|e| MicrodropError::Audio(format!("Failed to write to file: {}", e)))?;

        info!("Text appended to file: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::TranscriptionSegment;
    use std::time::Duration;

    fn create_test_result() -> TranscriptionResult {
        TranscriptionResult {
            text: "Hello world".to_string(),
            segments: vec![
                TranscriptionSegment {
                    start: Duration::from_millis(0),
                    end: Duration::from_millis(1000),
                    text: "Hello".to_string(),
                },
                TranscriptionSegment {
                    start: Duration::from_millis(1000),
                    end: Duration::from_millis(2000),
                    text: "world".to_string(),
                },
            ],
            language: Some("en".to_string()),
            processing_time: Duration::from_millis(100),
        }
    }

    #[test]
    fn test_format_transcript_none() {
        let manager = OutputManager::new().unwrap();
        let result = create_test_result();
        let formatted = manager.format_transcript(&result, &TimestampFormat::None);
        assert_eq!(formatted, "Hello world");
    }

    #[test]
    fn test_format_transcript_simple() {
        let manager = OutputManager::new().unwrap();
        let result = create_test_result();
        let formatted = manager.format_transcript(&result, &TimestampFormat::Simple);
        assert_eq!(formatted, "[0.0s] Hello\n[1.0s] world");
    }

    #[test]
    fn test_format_transcript_detailed() {
        let manager = OutputManager::new().unwrap();
        let result = create_test_result();
        let formatted = manager.format_transcript(&result, &TimestampFormat::Detailed);
        assert_eq!(formatted, "[0.0s - 1.0s] Hello\n[1.0s - 2.0s] world");
    }

    #[test]
    fn test_format_empty_segments() {
        let manager = OutputManager::new().unwrap();
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            segments: vec![],
            language: Some("en".to_string()),
            processing_time: Duration::from_millis(100),
        };

        let formatted_simple = manager.format_transcript(&result, &TimestampFormat::Simple);
        let formatted_detailed = manager.format_transcript(&result, &TimestampFormat::Detailed);

        assert_eq!(formatted_simple, "Hello world");
        assert_eq!(formatted_detailed, "Hello world");
    }

    #[test]
    fn test_append_to_file() {
        let manager = OutputManager::new().unwrap();
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("microdrop_test_append.txt");

        // Clean up any existing file
        let _ = std::fs::remove_file(&temp_file);

        // Test appending
        manager.append_to_file("First line", &temp_file).unwrap();
        manager.append_to_file("Second line", &temp_file).unwrap();

        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert_eq!(content, "First line\nSecond line\n");

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }
}
