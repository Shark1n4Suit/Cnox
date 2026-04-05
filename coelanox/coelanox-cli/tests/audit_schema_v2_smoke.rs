//! Regression: audit JSONL schema v2 (`run` + `serve`) on a tiny in-memory container.
//! Does not require `bert_base_uncased_scalar.cnox`.

mod common;

use common::{f32_payload, minimal_relu_container, read_ipc_ok_only, wait_serve_ready};
use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::tempdir;

fn coelanox_bin() -> &'static str {
    option_env!("CARGO_BIN_EXE_coelanox").expect("run with `cargo test -p coelanox-cli`")
}

fn first_jsonl_object(path: &std::path::Path) -> Value {
    let s = std::fs::read_to_string(path).expect("read audit file");
    let line = s.lines().next().expect("at least one JSONL line");
    serde_json::from_str(line).expect("parse JSONL line")
}

#[test]
fn audit_schema_v2_run_and_serve_minimal_relu() {
    let bin = coelanox_bin();
    let (cnox, input_elems) = minimal_relu_container().expect("container");
    let path = cnox.path().to_str().unwrap();

    let dir = tempdir().expect("tempdir");
    let audit_run = dir.path().join("audit_run.json");
    let out_json = dir.path().join("out.json");
    let audit_arg = format!("--audit-output={}", audit_run.display());

    let st = Command::new(bin)
        .args([
            "run",
            "--file",
            path,
            "--output",
            out_json.to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
            "--audit",
            &audit_arg,
            "--audit-sample",
            "6",
            "--audit-tail",
            "2",
            "--audit-stats",
            "--audit-hash",
        ])
        .status()
        .expect("spawn run");
    assert!(st.success(), "run with audit exit {:?}", st.code());

    let v = first_jsonl_object(&audit_run);
    assert_eq!(v["schema"].as_u64(), Some(2), "schema: {:?}", v["schema"]);
    assert!(v.get("run").is_some(), "missing run: {}", v);
    assert!(v.get("input").is_some(), "missing input: {}", v);
    let ops = v["ops"].as_array().expect("ops array");
    assert!(!ops.is_empty(), "ops empty");
    let op0 = &ops[0];
    assert!(op0.get("out_sample_head").is_some() || op0.get("stats").is_some());

    // serve: same audit flags, one CREQ
    let audit_serve = dir.path().join("audit_serve.json");
    let audit_arg2 = format!("--audit-output={}", audit_serve.display());
    let mut child = Command::new(bin)
        .args([
            "serve",
            "-f",
            path,
            "-b",
            "scalar",
            "--no-verify",
            "--audit",
            &audit_arg2,
            "--audit-sample",
            "4",
            "--audit-hash",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");

    let stderr = child.stderr.take().expect("stderr");
    wait_serve_ready(stderr, Duration::from_secs(30));

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    let input: Vec<f32> = vec![1.0f32; input_elems];
    let payload = f32_payload(&input);
    stdin.write_all(b"CREQ").unwrap();
    stdin
        .write_all(&(payload.len() as u32).to_le_bytes())
        .unwrap();
    stdin.write_all(&payload).unwrap();
    stdin.write_all(b"CNXT").unwrap();
    stdin.flush().ok();

    let _ = read_ipc_ok_only(&mut stdout).expect("CROk");
    let st = child.wait().expect("wait");
    assert!(st.success(), "serve exit {:?}", st.code());

    let v2 = first_jsonl_object(&audit_serve);
    assert_eq!(v2["schema"].as_u64(), Some(2));
    assert!(v2.get("input").is_some());
}
