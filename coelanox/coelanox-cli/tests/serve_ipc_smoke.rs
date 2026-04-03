//! Regression: `coelanox serve` IPC v1 (CREQ/CROk/CNXT) and per-request audit flush.

mod common;

use common::{f32_payload, minimal_relu_container, read_ipc_ok_only, wait_serve_ready};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn serve_ipc_creq_then_cnxt_returns_crok() {
    let bin = option_env!("CARGO_BIN_EXE_coelanox")
        .expect("run with `cargo test -p coelanox-cli` so CARGO_BIN_EXE_coelanox is set");

    let (cnox, input_elems) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = Command::new(bin)
        .args([
            "serve",
            "-f",
            &path,
            "-b",
            "scalar",
            "--no-verify",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn coelanox serve");

    let stderr = child.stderr.take().expect("stderr");
    wait_serve_ready(stderr, Duration::from_secs(10));

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    let input: Vec<f32> = (0..input_elems).map(|i| i as f32 + 1.0).collect();
    let payload = f32_payload(&input);
    stdin.write_all(b"CREQ").expect("CREQ");
    stdin
        .write_all(&(payload.len() as u32).to_le_bytes())
        .expect("len");
    stdin.write_all(&payload).expect("payload");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    let out = read_ipc_ok_only(&mut stdout).expect("IPC response");
    assert_eq!(out.len(), input_elems);

    let status = child.wait().expect("wait");
    assert!(status.success(), "serve exit {:?}", status.code());
}

#[test]
fn serve_ipc_audit_flush_writes_file_before_shutdown() {
    let bin = option_env!("CARGO_BIN_EXE_coelanox")
        .expect("run with `cargo test -p coelanox-cli` so CARGO_BIN_EXE_coelanox is set");

    let dir = tempdir().expect("tempdir");
    let audit_path = dir.path().join("serve_audit.json");

    let (cnox, input_elems) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let audit_arg = format!("--audit-output={}", audit_path.display());
    let mut child = Command::new(bin)
        .args([
            "serve",
            "-f",
            &path,
            "-b",
            "scalar",
            "--no-verify",
            "--audit",
            &audit_arg,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn coelanox serve");

    let stderr = child.stderr.take().expect("stderr");
    wait_serve_ready(stderr, Duration::from_secs(10));

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    let input: Vec<f32> = (0..input_elems).map(|i| i as f32 + 1.0).collect();
    let payload = f32_payload(&input);
    stdin.write_all(b"CREQ").expect("CREQ");
    stdin
        .write_all(&(payload.len() as u32).to_le_bytes())
        .expect("len");
    stdin.write_all(&payload).expect("payload");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    let out = read_ipc_ok_only(&mut stdout).expect("IPC response");
    assert_eq!(out.len(), input_elems);

    let status = child.wait().expect("wait");
    assert!(status.success(), "serve exit {:?}", status.code());

    // Per-request flush plus shutdown flush should leave non-trivial audit on disk.
    let meta = std::fs::metadata(&audit_path).expect("audit file");
    assert!(
        meta.len() > 2,
        "audit JSON should be non-trivial after inference"
    );
}
