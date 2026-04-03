//! Corrupt / hostile CLI inputs: garbage files, truncation, path traversal, bad JSON input.

mod common;

use common::minimal_relu_container;
use std::fs;
use std::process::Command;

fn coelanox_bin() -> &'static str {
    option_env!("CARGO_BIN_EXE_coelanox")
        .expect("run with `cargo test -p coelanox-cli` so CARGO_BIN_EXE_coelanox is set")
}

#[test]
fn validate_rejects_empty_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("empty.cnox");
    fs::write(&p, []).expect("write");
    let st = Command::new(coelanox_bin())
        .args(["validate", "--file"])
        .arg(&p)
        .status()
        .expect("spawn");
    assert!(!st.success(), "empty file should not validate");
}

#[test]
fn validate_rejects_random_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("noise.cnox");
    let noise: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
    fs::write(&p, &noise).expect("write");
    let st = Command::new(coelanox_bin())
        .args(["validate", "--file"])
        .arg(&p)
        .status()
        .expect("spawn");
    assert!(!st.success(), "random bytes should not validate");
}

#[test]
fn validate_rejects_truncated_real_container() {
    let (cnox, _) = minimal_relu_container().expect("container");
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("trunc.cnox");
    let bytes = fs::read(cnox.path()).expect("read good cnox");
    assert!(bytes.len() > 32);
    fs::write(&p, &bytes[..32]).expect("write truncated");
    let st = Command::new(coelanox_bin())
        .args(["validate", "--file"])
        .arg(&p)
        .status()
        .expect("spawn");
    assert!(!st.success(), "truncated container should not validate");
}

#[test]
fn run_rejects_path_traversal_dotdot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (cnox, _) = minimal_relu_container().expect("container");
    let target = dir.path().join("model.cnox");
    fs::copy(cnox.path(), &target).expect("copy");
    let sub = dir.path().join("sub");
    fs::create_dir_all(&sub).expect("mkdir");
    let out = sub.join("out.json");
    // String contains `..` — runtime rejects regardless of filesystem resolution.
    let st = Command::new(coelanox_bin())
        .current_dir(&sub)
        .args([
            "run",
            "--file",
            "../model.cnox",
            "--output",
            out.to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
        ])
        .status()
        .expect("spawn");
    assert!(
        !st.success(),
        "literal .. in --file should be rejected (exit {st:?})"
    );
}

#[test]
fn run_rejects_malformed_input_json() {
    let (cnox, _) = minimal_relu_container().expect("container");
    let dir = tempfile::tempdir().expect("tempdir");
    let bad_json = dir.path().join("bad_in.json");
    fs::write(&bad_json, b"{not json").expect("write");
    let out = dir.path().join("out.json");
    let st = Command::new(coelanox_bin())
        .args([
            "run",
            "--file",
            cnox.path().to_str().unwrap(),
            "--input",
            bad_json.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
        ])
        .status()
        .expect("spawn");
    assert!(!st.success(), "malformed input JSON should fail");
}
