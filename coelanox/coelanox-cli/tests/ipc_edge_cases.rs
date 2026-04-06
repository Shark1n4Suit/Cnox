//! IPC protocol edges: bad framing, wrong shapes, sequential requests, stdin EOF.
//! Aligns with `coelanox-ipc` error codes (400 bad magic / framing, 422 shape, 500 exec).

mod common;

use common::{
    f32_payload, minimal_relu_container, read_ipc_frame, read_ipc_ok_only, wait_serve_ready,
    IpcFrame,
};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

fn spawn_serve(path: &str) -> std::process::Child {
    let bin = env!("CARGO_BIN_EXE_coelanox");
    Command::new(bin)
        .args(["serve", "-f", path, "-b", "scalar", "--no-verify"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn coelanox serve")
}

#[test]
fn serve_ipc_unknown_magic_returns_cre_400() {
    let (cnox, _) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = spawn_serve(&path);
    wait_serve_ready(
        child.stderr.take().expect("stderr"),
        Duration::from_secs(10),
    );

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    // One bad 4-byte prefix (not CREQ/CNXT), then graceful shutdown.
    stdin.write_all(b"XXXX").expect("bad magic");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    match read_ipc_frame(&mut stdout).expect("read frame") {
        IpcFrame::Err { code, msg } => {
            assert_eq!(code, 400, "msg={msg}");
            assert!(
                msg.contains("CREQ") || msg.contains("CNXT"),
                "unexpected message: {msg}"
            );
        }
        IpcFrame::Ok(v) => panic!("expected CREr, got CROk len {}", v.len()),
    }

    assert!(child.wait().expect("wait").success());
}

#[test]
fn serve_ipc_wrong_element_count_returns_cre_422() {
    let (cnox, input_elems) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = spawn_serve(&path);
    wait_serve_ready(
        child.stderr.take().expect("stderr"),
        Duration::from_secs(10),
    );

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    // Claim payload length matches one float, but manifest expects `input_elems`.
    let bad = f32_payload(&[1.0f32]);
    assert_ne!(bad.len(), input_elems * 4);
    stdin.write_all(b"CREQ").expect("CREQ");
    stdin
        .write_all(&(bad.len() as u32).to_le_bytes())
        .expect("len");
    stdin.write_all(&bad).expect("payload");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    match read_ipc_frame(&mut stdout).expect("read frame") {
        IpcFrame::Err { code, msg } => {
            assert_eq!(code, 422, "msg={msg}");
            assert!(msg.contains("manifest") || msg.contains("expects"), "{msg}");
        }
        IpcFrame::Ok(v) => panic!("expected CREr, got ok len {}", v.len()),
    }

    assert!(child.wait().expect("wait").success());
}

#[test]
fn serve_ipc_odd_payload_length_returns_cre_400() {
    let (cnox, _) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = spawn_serve(&path);
    wait_serve_ready(
        child.stderr.take().expect("stderr"),
        Duration::from_secs(10),
    );

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    let odd = [0u8; 7];
    stdin.write_all(b"CREQ").expect("CREQ");
    stdin
        .write_all(&(odd.len() as u32).to_le_bytes())
        .expect("len");
    stdin.write_all(&odd).expect("payload");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    match read_ipc_frame(&mut stdout).expect("read frame") {
        IpcFrame::Err { code, msg } => {
            assert_eq!(code, 400, "msg={msg}");
            assert!(
                msg.contains("multiple") || msg.contains("4"),
                "unexpected: {msg}"
            );
        }
        IpcFrame::Ok(v) => panic!("expected CREr, got ok len {}", v.len()),
    }

    assert!(child.wait().expect("wait").success());
}

#[test]
fn serve_ipc_two_sequential_creq_same_session_both_crok() {
    let (cnox, input_elems) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = spawn_serve(&path);
    wait_serve_ready(
        child.stderr.take().expect("stderr"),
        Duration::from_secs(10),
    );

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    for k in 0..2u32 {
        let input: Vec<f32> = (0..input_elems)
            .map(|i| i as f32 + 1.0 + k as f32 * 0.01)
            .collect();
        let payload = f32_payload(&input);
        stdin.write_all(b"CREQ").expect("CREQ");
        stdin
            .write_all(&(payload.len() as u32).to_le_bytes())
            .expect("len");
        stdin.write_all(&payload).expect("payload");
        let out = read_ipc_ok_only(&mut stdout).expect("CROk");
        assert_eq!(out.len(), input_elems, "iteration {k}");
    }

    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();
    assert!(child.wait().expect("wait").success());
}

#[test]
fn serve_ipc_stdin_close_without_cnxt_exits_cleanly() {
    let (cnox, _) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap().to_owned();

    let mut child = spawn_serve(&path);
    wait_serve_ready(
        child.stderr.take().expect("stderr"),
        Duration::from_secs(10),
    );

    // Drop stdin → EOF; serve loop exits without CNXT.
    drop(child.stdin.take());

    let status = child.wait().expect("wait");
    assert!(status.success(), "serve exit {:?}", status.code());
}
