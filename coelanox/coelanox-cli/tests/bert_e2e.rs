//! Full BERT scalar container smoke: `validate`, `info`, `env`, `run`, `benchmark` (JSON),
//! `run --audit`, and `serve` (IPC). Compares **run** vs **serve** outputs within float tolerance.
//!
//! **Not run in CI** — this test is `#[ignore]` (default `cargo test` skips ignored tests).
//! Local opt-in:
//! ```text
//! cargo test -p coelanox-cli bert_e2e -- --ignored
//! ```
//! Default lookup (after `COELANOX_BERT_CNOX`): repo-root `bert_base_uncased_scalar.cnox`, then
//! `models/output/bert_base_uncased_scalar.cnox`. If none exist, the test **passes** with a skip.

mod common;

use common::{f32_payload, read_ipc_ok_only, wait_serve_ready};
use coelanox_core::format::CoelanoxContainer;
use serde_json::Value;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tempfile::tempdir;

fn coelanox_bin() -> &'static str {
    option_env!("CARGO_BIN_EXE_coelanox")
        .expect("run with `cargo test -p coelanox-cli` so CARGO_BIN_EXE_coelanox is set")
}

/// `coelanox-cli` lives at `coelanox/coelanox-cli`; `../..` is the cnox repo root (parent of `coelanox/`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn resolve_bert_cnox() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("COELANOX_BERT_CNOX") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb.canonicalize().unwrap_or(pb));
        }
    }
    let root = repo_root();
    for rel in [
        "bert_base_uncased_scalar.cnox",
        "models/output/bert_base_uncased_scalar.cnox",
    ] {
        let guess = root.join(rel);
        if guess.is_file() {
            // Runtime rejects `..` in paths; canonicalize for subprocess args.
            return Some(guess.canonicalize().unwrap_or(guess));
        }
    }
    None
}

/// Kahan summation for a stable scalar fingerprint (catches swapped outputs with same length).
fn fingerprint_kahan(v: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for x in v {
        let xf = *x as f64;
        let y = xf - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

fn fingerprint_sumsq(v: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    for x in v {
        let xf = (*x as f64) * (*x as f64);
        let y = xf - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Run vs serve should match for the same backend; allow tiny float drift across processes.
const ABS_TOL: f32 = 5e-4;

fn assert_close_f32(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs = 0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= ABS_TOL,
            "{label}: index {i}: {x} vs {y} (abs diff {d}, max allowed {ABS_TOL})"
        );
    }
    eprintln!("{label}: max abs diff {max_abs:.6}");
}

#[test]
#[ignore = "BERT E2E: heavy/opt-in; not for CI. Run: cargo test -p coelanox-cli bert_e2e -- --ignored"]
fn bert_scalar_e2e_run_benchmark_audit_serve_match() {
    let Some(cnox_path) = resolve_bert_cnox() else {
        eprintln!(
            "bert_e2e: skip — no BERT .cnox found. Place bert_base_uncased_scalar.cnox at repo root, or models/output/, or set COELANOX_BERT_CNOX"
        );
        return;
    };
    let cnox = cnox_path.to_string_lossy().into_owned();
    let bin = coelanox_bin();

    let container = CoelanoxContainer::read_binary_file(&cnox_path).expect("read container");
    let expected_in: usize = container
        .manifest
        .input_shapes
        .iter()
        .map(|s| s.iter().product::<usize>())
        .sum();
    let expected_out: usize = container
        .manifest
        .output_shapes
        .iter()
        .map(|s| s.iter().product::<usize>())
        .sum();

    let dir = tempdir().expect("tempdir");
    let run_out = dir.path().join("bert_run.json");
    let run_repeat = dir.path().join("bert_run_repeat.json");
    let audit_out = dir.path().join("audit_bert.json");

    // validate + info + env (sanity that CLI wiring works for a real container)
    let st = Command::new(bin)
        .args(["validate", "--file", &cnox])
        .status()
        .expect("spawn validate");
    assert!(st.success(), "validate exit {:?}", st.code());

    let st = Command::new(bin)
        .args(["info", "--file", &cnox])
        .status()
        .expect("spawn info");
    assert!(st.success(), "info exit {:?}", st.code());

    let st = Command::new(bin).arg("env").status().expect("spawn env");
    assert!(st.success(), "env exit {:?}", st.code());

    // run — synthetic all-ones input (default), scalar backend
    let st = Command::new(bin)
        .args([
            "run",
            "--file",
            &cnox,
            "--output",
            run_out.to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
        ])
        .status()
        .expect("spawn run");
    assert!(st.success(), "run exit {:?}", st.code());

    let run_json = std::fs::read_to_string(&run_out).expect("read run output");
    let run_vec: Vec<f32> = serde_json::from_str(&run_json).expect("parse run JSON as f32[]");
    assert_eq!(
        run_vec.len(),
        expected_out,
        "manifest output element count vs run JSON length"
    );

    // Second identical run — same process, same inputs; must be bitwise-identical (no flaky tolerance).
    let st = Command::new(bin)
        .args([
            "run",
            "--file",
            &cnox,
            "--output",
            run_repeat.to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
        ])
        .status()
        .expect("spawn run repeat");
    assert!(st.success(), "repeat run exit {:?}", st.code());
    let run_json2 = std::fs::read_to_string(&run_repeat).expect("read repeat");
    let run_vec2: Vec<f32> = serde_json::from_str(&run_json2).expect("parse repeat JSON");
    assert_eq!(
        run_vec, run_vec2,
        "determinism: two scalar runs with same inputs must match exactly"
    );

    // Redundant fingerprints (guards wrong tensor swapped in with same length).
    assert_eq!(
        fingerprint_kahan(&run_vec),
        fingerprint_kahan(&run_vec2),
        "kahan sum fingerprint"
    );
    assert_eq!(
        fingerprint_sumsq(&run_vec),
        fingerprint_sumsq(&run_vec2),
        "sum-of-squares fingerprint"
    );

    // benchmark JSON — numeric fields are timing/memory stats (not logits); sanity-check structure
    let bench_out = Command::new(bin)
        .args([
            "benchmark",
            "--file",
            &cnox,
            "--iterations",
            "2",
            "--backend",
            "scalar",
            "--output",
            "json",
            "--no-verify",
        ])
        .output()
        .expect("spawn benchmark");
    assert!(
        bench_out.status.success(),
        "benchmark stderr: {}",
        String::from_utf8_lossy(&bench_out.stderr)
    );
    let bench: Value =
        serde_json::from_slice(&bench_out.stdout).expect("benchmark stdout should be JSON");
    assert!(
        bench["avg_time"].as_u64().unwrap_or(0) > 0,
        "benchmark avg_time: {:?}",
        bench["avg_time"]
    );
    assert_eq!(bench["iterations"].as_u64(), Some(2));
    assert_eq!(
        bench["input_elements"].as_u64(),
        Some(expected_in as u64),
        "benchmark input_elements vs manifest"
    );
    // Scalar-only / fallback graphs may report workspace_elements == 0; weights still load.
    assert!(
        bench["weights_bytes"].as_u64().unwrap_or(0) > 0,
        "benchmark weights_bytes"
    );

    // run with audit — file must be written and non-trivial
    let audit_arg = format!("--audit-output={}", audit_out.display());
    let st = Command::new(bin)
        .args([
            "run",
            "--file",
            &cnox,
            "--output",
            dir.path().join("bert_run2.json").to_str().unwrap(),
            "--backend",
            "scalar",
            "--no-verify",
            "--audit",
            &audit_arg,
        ])
        .status()
        .expect("spawn run audit");
    assert!(st.success(), "run --audit exit {:?}", st.code());
    let meta = std::fs::metadata(&audit_out).expect("audit file");
    assert!(meta.len() > 8, "audit JSON should be non-trivial");

    // serve — same synthetic input as CLI run (all ones, manifest-shaped)
    let input: Vec<f32> = vec![1.0f32; expected_in];
    let payload = f32_payload(&input);

    let mut child = Command::new(bin)
        .args([
            "serve",
            "-f",
            &cnox,
            "-b",
            "scalar",
            "--no-verify",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");

    let stderr = child.stderr.take().expect("stderr");
    wait_serve_ready(stderr, Duration::from_secs(120));

    let mut stdin = child.stdin.take().expect("stdin");
    let mut stdout = child.stdout.take().expect("stdout");

    stdin.write_all(b"CREQ").expect("CREQ");
    stdin
        .write_all(&(payload.len() as u32).to_le_bytes())
        .expect("len");
    stdin.write_all(&payload).expect("payload");
    stdin.write_all(b"CNXT").expect("CNXT");
    stdin.flush().ok();

    let serve_vec = read_ipc_ok_only(&mut stdout).expect("IPC CROk");
    let status = child.wait().expect("wait");
    assert!(status.success(), "serve exit {:?}", status.code());

    assert_eq!(
        serve_vec.len(),
        expected_out,
        "serve output length vs manifest"
    );
    assert_close_f32("run vs serve (scalar)", &run_vec, &serve_vec);
}
