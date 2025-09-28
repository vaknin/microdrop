use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_help_command() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("On-demand speech-to-text transcription"));
}

#[test]
fn test_version_command() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.arg("--version");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("microdrop"));
}

#[test]
fn test_model_list_command() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.args(["model", "list"]);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Available models for download:"));
}

#[test]
fn test_config_write_default_command() {
    let temp_dir = TempDir::new().unwrap();

    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.args(["config", "write-default", "--force"]);
    cmd.env("HOME", temp_dir.path()); // Override home directory for test
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Default configuration written to:"));
}

#[test]
fn test_config_write_default_without_force_fails_when_exists() {
    let temp_dir = TempDir::new().unwrap();
    let config_dir = temp_dir.path().join(".config").join("microdrop");
    fs::create_dir_all(&config_dir).unwrap();
    let config_path = config_dir.join("config.toml");
    fs::write(&config_path, "# existing config").unwrap();

    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.args(["config", "write-default"]);
    cmd.env("HOME", temp_dir.path());
    cmd.assert()
        .failure()
        .stdout(predicate::str::contains("already exists"));
}

#[test]
fn test_toggle_command_basic_functionality() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.args(["toggle"]);
    cmd.write_stdin(""); // Simulate immediate enter to stop capture
    cmd.assert()
        .success() // This should succeed and capture/stop immediately
        .stdout(predicate::str::contains("Audio capture started"));
}

#[test]
fn test_invalid_subcommand() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.arg("invalid_command");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand"));
}

#[test]
fn test_model_install_invalid_model() {
    let mut cmd = Command::cargo_bin("microdrop").unwrap();
    cmd.args(["model", "install", "nonexistent-model"]);
    cmd.assert()
        .failure()
        .stdout(predicate::str::contains("Model loading error"));
}