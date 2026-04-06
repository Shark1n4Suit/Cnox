//! Audit mode: log after each op during inference for traceability and debugging.
//! Enabled via CLI `run --audit` / `serve --audit`; state is process-global so
//! worker threads in all backend paths can contribute to a single run audit.
//!
//! **Strategies** ([`AuditStrategy`]):
//! - **`forget`** (default): after each flush, clear RAM; `.json` paths use **JSONL** (one JSON object per line per batch, schema v2). Plain paths append text blocks per batch.
//! - **`last`**: overwrite the audit file with **only the last** flushed batch; clears RAM after each flush.
//! - **`long`**: legacy session mode — entries accumulate in RAM and each flush rewrites the full snapshot (unbounded memory on long-lived `serve`).
//!
//! **Session options** ([`AuditSessionConfig`]): sample head/tail sizes, optional stats and SHA-256
//! over tensor bytes, optional restriction to graph-output nodes, optional run metadata, and
//! optional **pretty JSON** for append mode (`forget` + `.json`).

use crate::audit_ir::graph_output_node_indices;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Mutex;

/// Current on-disk JSON line schema for `.json` audit paths.
pub const AUDIT_SCHEMA_VERSION: u32 = 2;

/// Hard cap for `sample_head` + `sample_tail` (total preview floats per tensor).
pub const MAX_AUDIT_SAMPLE_TOTAL: usize = 4096;

/// Do not scan more than this many floats for stats or SHA-256 (per tensor); larger tensors log elision.
pub const MAX_AUDIT_FULL_SCAN_ELEMENTS: usize = 256 * 1024;

/// How audit entries are retained in memory and on disk (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuditStrategy {
    /// Bounded RAM: flush clears buffer; disk appends each batch (JSONL for `.json`).
    #[default]
    Forget,
    /// Bounded RAM: flush clears buffer; disk overwrites with last batch only.
    Last,
    /// Session cumulative in RAM; disk rewrites full snapshot each flush (legacy).
    Long,
}

impl std::str::FromStr for AuditStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "forget" | "short" => Ok(Self::Forget),
            "last" => Ok(Self::Last),
            "long" => Ok(Self::Long),
            _ => Err(format!(
                "unknown audit strategy '{s}' (expected forget|short|last|long)"
            )),
        }
    }
}

impl std::fmt::Display for AuditStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AuditStrategy::Forget => "forget",
            AuditStrategy::Last => "last",
            AuditStrategy::Long => "long",
        };
        f.write_str(s)
    }
}

/// Metadata attached once per audit session (embedded in each JSONL batch line).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditRunContext {
    pub run_id: String,
    pub started_at_unix_ms: u64,
    pub container_path: String,
    /// SHA-256 of the container file bytes (hex), when the CLI computed it.
    pub container_sha256_hex: Option<String>,
    pub backend: String,
}

/// Visibility and processing options for an audit session (first enable wins for nested calls).
#[derive(Debug, Clone)]
pub struct AuditSessionConfig {
    pub strategy: AuditStrategy,
    /// First N floats from each op output (default 4). Capped by [`MAX_AUDIT_SAMPLE_TOTAL`] with tail.
    pub sample_head: usize,
    /// Last M floats from each op output (default 0).
    pub sample_tail: usize,
    pub include_stats: bool,
    pub include_hash: bool,
    /// If true, only ops whose node index is in `output_node_indices` are logged (empty = no filter).
    pub final_output_only: bool,
    pub output_node_indices: Vec<usize>,
    pub run: Option<AuditRunContext>,
    /// When true and the audit path ends in `.json` with strategy [`AuditStrategy::Forget`], each
    /// flushed batch is written as **pretty-printed JSON** (multi-line per batch). Default is
    /// **compact single-line JSONL** for machine streaming. Strategies `last` / `long` still use
    /// pretty JSON on overwrite (unchanged).
    pub pretty_json: bool,
}

impl Default for AuditSessionConfig {
    fn default() -> Self {
        Self {
            strategy: AuditStrategy::Forget,
            sample_head: 4,
            sample_tail: 0,
            include_stats: false,
            include_hash: false,
            final_output_only: false,
            output_node_indices: Vec::new(),
            run: None,
            pretty_json: false,
        }
    }
}

impl AuditSessionConfig {
    pub fn from_strategy(strategy: AuditStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }
}

#[derive(Debug, serde::Serialize, Clone, Default)]
pub struct AuditTensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}

#[derive(serde::Serialize, Clone)]
pub struct AuditInputSnapshot {
    pub len: usize,
    pub sha256_hex: Option<String>,
    pub sample_head: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sample_tail: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elided: Option<String>,
}

#[derive(serde::Serialize, Clone)]
pub struct AuditOpEntry {
    pub op_index: usize,
    pub op_type: String,
    pub out_len: usize,
    pub out_sample_head: Vec<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub out_sample_tail: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<AuditTensorStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_sha256_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_elided: Option<String>,
}

#[derive(serde::Serialize)]
struct AuditBatchV2 {
    schema: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    run: Option<AuditRunContext>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<AuditInputSnapshot>,
    ops: Vec<AuditOpEntry>,
}

struct AuditState {
    enabled: bool,
    enable_count: usize,
    log_path: Option<std::path::PathBuf>,
    entries: Vec<AuditOpEntry>,
    strategy: AuditStrategy,
    session: AuditSessionConfig,
    /// Last input recorded via [`record_input_snapshot`] for the next flush batch.
    pending_input: Option<AuditInputSnapshot>,
}

static AUDIT_STATE: Mutex<AuditState> = Mutex::new(AuditState {
    enabled: false,
    enable_count: 0,
    log_path: None,
    entries: Vec::new(),
    strategy: AuditStrategy::Forget,
    session: AuditSessionConfig {
        strategy: AuditStrategy::Forget,
        sample_head: 4,
        sample_tail: 0,
        include_stats: false,
        include_hash: false,
        final_output_only: false,
        output_node_indices: Vec::new(),
        run: None,
        pretty_json: false,
    },
    pending_input: None,
});

#[inline]
fn lock_audit_state() -> std::sync::MutexGuard<'static, AuditState> {
    AUDIT_STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn clamp_sample_sizes(head: usize, tail: usize) -> (usize, usize) {
    let head = head.min(MAX_AUDIT_SAMPLE_TOTAL);
    let tail = tail.min(MAX_AUDIT_SAMPLE_TOTAL);
    let total = head.saturating_add(tail);
    if total <= MAX_AUDIT_SAMPLE_TOTAL {
        return (head, tail);
    }
    // Prefer head, then give remainder to tail
    let head = head.min(MAX_AUDIT_SAMPLE_TOTAL);
    let rest = MAX_AUDIT_SAMPLE_TOTAL.saturating_sub(head);
    (head, tail.min(rest))
}

fn take_head_tail(data: &[f32], out_len: usize, head: usize, tail: usize) -> (Vec<f32>, Vec<f32>) {
    let n = data.len().min(out_len);
    let slice = &data[..n];
    let (head, tail_n) = clamp_sample_sizes(head, tail);
    let head_v: Vec<f32> = slice.iter().take(head).copied().collect();
    if tail_n == 0 || slice.len() <= head {
        return (head_v, Vec::new());
    }
    // Non-overlapping tail: last `tail_n` elements unless that would overlap head window
    let tail_start = slice.len().saturating_sub(tail_n);
    let tail_start = tail_start.max(head);
    let tail_v: Vec<f32> = slice[tail_start..].to_vec();
    (head_v, tail_v)
}

fn compute_stats(data: &[f32]) -> AuditTensorStats {
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut min_v = f32::NAN;
    let mut max_v = f32::NAN;
    for &x in data {
        if x.is_nan() {
            nan_count += 1;
            continue;
        }
        if x.is_infinite() {
            inf_count += 1;
        }
        sum += x as f64;
        count += 1;
        if min_v.is_nan() || x < min_v {
            min_v = x;
        }
        if max_v.is_nan() || x > max_v {
            max_v = x;
        }
    }
    let mean = if count > 0 {
        (sum / count as f64) as f32
    } else {
        f32::NAN
    };
    if min_v.is_nan() {
        min_v = 0.0;
    }
    if max_v.is_nan() {
        max_v = 0.0;
    }
    AuditTensorStats {
        min: min_v,
        max: max_v,
        mean,
        nan_count,
        inf_count,
    }
}

/// Hex encode — simple nibbles (no `hex` crate).
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0xf) as usize] as char);
        }
        s
    }
}

fn f32_slice_bytes(data: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(data.len() * 4);
    for x in data {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn sha256_f32_hex(data: &[f32]) -> String {
    let bytes = f32_slice_bytes(data);
    let digest = Sha256::digest(&bytes);
    hex::encode(&digest)
}

fn hash_tensor_maybe(data: &[f32], want: bool) -> Option<String> {
    if !want {
        return None;
    }
    if data.len() > MAX_AUDIT_FULL_SCAN_ELEMENTS {
        return None;
    }
    Some(sha256_f32_hex(data))
}

fn stats_maybe(data: &[f32], want: bool) -> Option<AuditTensorStats> {
    if !want {
        return None;
    }
    if data.len() > MAX_AUDIT_FULL_SCAN_ELEMENTS {
        return None;
    }
    Some(compute_stats(data))
}

fn should_log_node(state: &AuditState, node_index: usize) -> bool {
    if !state.session.final_output_only {
        return true;
    }
    if state.session.output_node_indices.is_empty() {
        return false;
    }
    state
        .session
        .output_node_indices
        .binary_search(&node_index)
        .is_ok()
}

fn build_op_entry(
    session: &AuditSessionConfig,
    node_index: usize,
    op_type: &str,
    output_len: usize,
    data: &[f32],
) -> AuditOpEntry {
    let (head, tail) = clamp_sample_sizes(session.sample_head, session.sample_tail);
    let (out_sample_head, out_sample_tail) = take_head_tail(data, output_len, head, tail);

    let mut audit_elided: Option<String> = None;
    let scan = data.len().min(output_len);
    let scan_slice = &data[..scan];

    let stats = stats_maybe(scan_slice, session.include_stats);
    let out_sha256_hex = hash_tensor_maybe(scan_slice, session.include_hash);
    if session.include_stats && stats.is_none() && scan > MAX_AUDIT_FULL_SCAN_ELEMENTS {
        audit_elided = Some(format!(
            "stats_skipped: len {} > max {}",
            scan, MAX_AUDIT_FULL_SCAN_ELEMENTS
        ));
    }
    if session.include_hash && out_sha256_hex.is_none() && scan > MAX_AUDIT_FULL_SCAN_ELEMENTS {
        let msg = format!(
            "hash_skipped: len {} > max {}",
            scan, MAX_AUDIT_FULL_SCAN_ELEMENTS
        );
        audit_elided = Some(match audit_elided {
            Some(a) => format!("{}; {}", a, msg),
            None => msg,
        });
    }

    AuditOpEntry {
        op_index: node_index,
        op_type: op_type.to_string(),
        out_len: output_len,
        out_sample_head,
        out_sample_tail,
        stats,
        out_sha256_hex,
        audit_elided,
    }
}

fn build_input_snapshot(session: &AuditSessionConfig, input: &[f32]) -> AuditInputSnapshot {
    let (head, tail) = clamp_sample_sizes(session.sample_head, session.sample_tail);
    let (sample_head, sample_tail) = take_head_tail(input, input.len(), head, tail);
    let sha256_hex = hash_tensor_maybe(input, session.include_hash);
    let mut elided = None;
    if input.len() > MAX_AUDIT_FULL_SCAN_ELEMENTS && sha256_hex.is_none() {
        elided = Some(format!(
            "input_hash_skipped: len {} > max {}",
            input.len(),
            MAX_AUDIT_FULL_SCAN_ELEMENTS
        ));
    }
    AuditInputSnapshot {
        len: input.len(),
        sha256_hex,
        sample_head,
        sample_tail,
        elided,
    }
}

fn write_entries_overwrite(path: &Path, batch: &AuditBatchV2) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let is_json = path.extension().is_some_and(|e| e == "json");
    if is_json {
        let s = serde_json::to_string_pretty(batch)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, s)
    } else {
        let lines = format_plain_batch(batch);
        std::fs::write(path, lines)
    }
}

fn format_stats_plain(s: &AuditTensorStats) -> String {
    format!(
        "min={:.6} max={:.6} mean={:.6} nan_count={} inf_count={}",
        s.min, s.max, s.mean, s.nan_count, s.inf_count
    )
}

fn format_plain_batch(batch: &AuditBatchV2) -> String {
    let mut out = String::new();
    out.push_str("--- coelanox_audit ---\n");
    out.push_str(&format!("schema: {}\n", batch.schema));
    if let Some(ref run) = batch.run {
        out.push_str("[run]\n");
        out.push_str(&format!("  run_id: {}\n", run.run_id));
        out.push_str(&format!("  started_at_unix_ms: {}\n", run.started_at_unix_ms));
        out.push_str(&format!("  container_path: {}\n", run.container_path));
        out.push_str(&format!("  backend: {}\n", run.backend));
        out.push_str(&format!(
            "  container_sha256_hex: {}\n",
            run.container_sha256_hex.as_deref().unwrap_or("-")
        ));
    }
    if let Some(ref inp) = batch.input {
        out.push_str("[input]\n");
        out.push_str(&format!("  len: {}\n", inp.len));
        out.push_str(&format!(
            "  sha256_hex: {}\n",
            inp.sha256_hex.as_deref().unwrap_or("-")
        ));
        out.push_str(&format!("  sample_head: {:?}\n", inp.sample_head));
        if !inp.sample_tail.is_empty() {
            out.push_str(&format!("  sample_tail: {:?}\n", inp.sample_tail));
        }
        if let Some(ref e) = inp.elided {
            out.push_str(&format!("  elided: {}\n", e));
        }
    }
    out.push_str("[ops]\n");
    for e in &batch.ops {
        out.push_str(&format!("  - op_index: {}\n", e.op_index));
        out.push_str(&format!("    op_type: {}\n", e.op_type));
        out.push_str(&format!("    out_len: {}\n", e.out_len));
        out.push_str(&format!("    out_sample_head: {:?}\n", e.out_sample_head));
        if !e.out_sample_tail.is_empty() {
            out.push_str(&format!("    out_sample_tail: {:?}\n", e.out_sample_tail));
        }
        if let Some(ref st) = e.stats {
            out.push_str(&format!("    stats: {}\n", format_stats_plain(st)));
        }
        out.push_str(&format!(
            "    out_sha256_hex: {}\n",
            e.out_sha256_hex.as_deref().unwrap_or("-")
        ));
        if let Some(ref el) = e.audit_elided {
            out.push_str(&format!("    elided: {}\n", el));
        }
    }
    out.push('\n');
    out
}

/// Append one batch: for `.json` paths, one JSON object per flush (compact JSONL by default).
/// With `pretty_json`, writes indented JSON (multi-line per batch; not strict single-line JSONL).
fn append_entries_batch(path: &Path, batch: &AuditBatchV2, pretty_json: bool) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let is_json = path.extension().is_some_and(|e| e == "json");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    if is_json {
        let line = if pretty_json {
            serde_json::to_string_pretty(batch).map_err(|e| std::io::Error::other(e.to_string()))?
        } else {
            serde_json::to_string(batch).map_err(|e| std::io::Error::other(e.to_string()))?
        };
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    } else {
        f.write_all(format_plain_batch(batch).as_bytes())?;
    }
    Ok(())
}

fn ordered_entries_for_flush(state: &AuditState) -> Vec<AuditOpEntry> {
    match state.strategy {
        AuditStrategy::Long => state.entries.clone(),
        AuditStrategy::Forget | AuditStrategy::Last => {
            let mut sorted: Vec<&AuditOpEntry> = state.entries.iter().collect();
            sorted.sort_by_key(|e| e.op_index);
            sorted.iter().map(|e| (*e).clone()).collect()
        }
    }
}

fn flush_locked(state: &mut AuditState) -> std::io::Result<()> {
    let Some(path) = state.log_path.as_deref() else {
        return Ok(());
    };
    if state.entries.is_empty() && state.pending_input.is_none() {
        return Ok(());
    }

    let ordered = ordered_entries_for_flush(state);
    let input = state.pending_input.take();
    let batch = AuditBatchV2 {
        schema: AUDIT_SCHEMA_VERSION,
        run: state.session.run.clone(),
        input,
        ops: ordered,
    };

    match state.strategy {
        AuditStrategy::Forget => {
            append_entries_batch(path, &batch, state.session.pretty_json)?;
            state.entries.clear();
        }
        AuditStrategy::Last => {
            write_entries_overwrite(path, &batch)?;
            state.entries.clear();
        }
        AuditStrategy::Long => {
            write_entries_overwrite(path, &batch)?;
        }
    }
    Ok(())
}

/// Enable or disable audit with full session options. On first `enabled=true`, stores path and config.
pub fn set_audit_session(enabled: bool, log_path: Option<&Path>, config: AuditSessionConfig) {
    let mut state = lock_audit_state();
    if enabled {
        if state.enable_count == 0 {
            state.entries.clear();
            state.pending_input = None;
            state.log_path = log_path.map(|p| p.to_path_buf());
            state.strategy = config.strategy;
            let mut config = config;
            config.output_node_indices.sort_unstable();
            state.session = config;
        } else if let Some(p) = log_path {
            let requested = p.to_path_buf();
            if state.log_path.as_ref() != Some(&requested) {
                tracing::warn!(
                    "audit already enabled for {}; ignoring nested request for {}",
                    state
                        .log_path
                        .as_ref()
                        .map(|x| x.display().to_string())
                        .unwrap_or_else(|| "<none>".to_string()),
                    requested.display()
                );
            }
        }
        state.enable_count = state.enable_count.saturating_add(1);
        state.enabled = true;
    } else {
        if state.enable_count > 0 {
            state.enable_count -= 1;
        }
        if state.enable_count == 0 {
            state.enabled = false;
            if let Err(e) = flush_locked(&mut state) {
                tracing::error!("audit flush failed on disable: {}", e);
            }
            state.log_path = None;
            state.entries.clear();
            state.pending_input = None;
        }
    }
}

/// Enable or disable audit mode (legacy: default session options except strategy).
pub fn set_audit_mode(enabled: bool, log_path: Option<&Path>, strategy: AuditStrategy) {
    set_audit_session(
        enabled,
        log_path,
        AuditSessionConfig::from_strategy(strategy),
    );
}

/// Flush current buffered audit entries immediately to the configured audit path.
pub fn flush_now() -> std::io::Result<()> {
    let mut state = lock_audit_state();
    if !state.enabled {
        return Ok(());
    }
    flush_locked(&mut state)
}

/// Returns true if audit mode is enabled (call from execution path).
pub fn is_enabled() -> bool {
    lock_audit_state().enabled
}

/// Record input tensor snapshot for the next flush (call once per inference before ops run).
pub fn record_input_snapshot(input: &[f32]) {
    let mut state = lock_audit_state();
    if !state.enabled {
        return;
    }
    let snap = build_input_snapshot(&state.session, input);
    state.pending_input = Some(snap);
}

/// Resolve graph-output node indices from container IR (for `--audit-final-only`).
pub fn output_node_indices_from_ir_data(
    ir_data: &[u8],
) -> coelanox_core::CoelanoxResult<Vec<usize>> {
    let ir = coelanox_core::ir::deserialize_optimized_ir(ir_data)?;
    Ok(graph_output_node_indices(&ir))
}

/// Log one op: node index, op type string, output size, and **full** output buffer (or representative slice).
pub fn log_after_op(
    node_index: usize,
    op_type: &str,
    output_len: usize,
    output_data: &[f32],
) {
    let mut state = lock_audit_state();
    if !state.enabled {
        return;
    }
    if !should_log_node(&state, node_index) {
        return;
    }
    let entry = build_op_entry(
        &state.session,
        node_index,
        op_type,
        output_len,
        output_data,
    );
    tracing::info!(
        "audit op {} {} out_len={} head={:?}",
        entry.op_index,
        entry.op_type,
        entry.out_len,
        entry.out_sample_head
    );
    state.entries.push(entry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static AUDIT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_audit() {
        for _ in 0..8 {
            set_audit_mode(false, None, AuditStrategy::Forget);
        }
    }

    #[test]
    fn forget_flush_json_appends_jsonl_line() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("audit.json");
        set_audit_mode(true, Some(&path), AuditStrategy::Forget);
        log_after_op(0, "Add", 4, &[1.0, 2.0, 3.0, 4.0]);
        flush_now().expect("flush");
        let s = std::fs::read_to_string(&path).expect("read audit");
        assert!(s.contains("\"schema\":2"));
        assert!(s.contains("Add"));
        assert!(s.lines().count() >= 1);
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn forget_two_flushes_two_jsonl_lines() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("audit.json");
        set_audit_mode(true, Some(&path), AuditStrategy::Forget);
        log_after_op(0, "A", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(0, "B", 1, &[2.0]);
        flush_now().expect("flush2");
        let s = std::fs::read_to_string(&path).expect("read");
        assert_eq!(s.lines().count(), 2);
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn last_overwrites_file_each_flush() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("audit.json");
        set_audit_mode(true, Some(&path), AuditStrategy::Last);
        log_after_op(0, "First", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(0, "Second", 1, &[2.0]);
        flush_now().expect("flush2");
        let s = std::fs::read_to_string(&path).expect("read");
        assert!(s.contains("Second"));
        assert!(!s.contains("First"));
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn long_flush_retains_entries_in_ram() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("audit.json");
        set_audit_mode(true, Some(&path), AuditStrategy::Long);
        log_after_op(0, "A", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(1, "B", 1, &[2.0]);
        flush_now().expect("flush2");
        let s = std::fs::read_to_string(&path).expect("read");
        assert!(s.contains("A"));
        assert!(s.contains("B"));
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn flush_now_writes_plain_lines_for_non_json_path_forget() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("audit.log");
        set_audit_mode(true, Some(&path), AuditStrategy::Forget);
        log_after_op(1, "Mul", 2, &[3.0, 4.0]);
        flush_now().expect("flush");
        let s = std::fs::read_to_string(&path).expect("read audit");
        assert!(s.contains("coelanox_audit"));
        assert!(s.contains("op_type: Mul"));
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn disable_flushes_and_clears_state() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("out.log");
        set_audit_mode(true, Some(&path), AuditStrategy::Forget);
        log_after_op(0, "Sub", 1, &[1.0]);
        set_audit_mode(false, None, AuditStrategy::Forget);
        let meta = std::fs::metadata(&path).expect("file exists");
        assert!(meta.len() > 0);
        reset_audit();
    }

    #[test]
    fn flush_now_noop_when_audit_disabled() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.log");
        std::fs::write(&path, b"").ok();
        flush_now().expect("noop");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn nested_enable_disable_keeps_first_path_until_final_disable() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let p1 = dir.path().join("one.log");
        let p2 = dir.path().join("two.log");
        set_audit_mode(true, Some(&p1), AuditStrategy::Forget);
        set_audit_mode(true, Some(&p2), AuditStrategy::Forget);
        log_after_op(0, "Relu", 1, &[1.0]);
        set_audit_mode(false, None, AuditStrategy::Forget);
        assert!(flush_now().is_ok());
        set_audit_mode(false, None, AuditStrategy::Forget);
        assert!(std::fs::metadata(&p1).is_ok());
        assert!(std::fs::metadata(&p2).is_err());
    }

    #[test]
    fn extended_head_matches_default_prefix() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let cfg = AuditSessionConfig {
            sample_head: 8,
            ..Default::default()
        };
        let e = build_op_entry(&cfg, 0, "T", data.len(), &data);
        assert_eq!(e.out_sample_head.len(), 8);
        assert_eq!(e.out_sample_head[0], 0.0);

        let cfg2 = AuditSessionConfig {
            sample_head: 4,
            ..Default::default()
        };
        let e2 = build_op_entry(&cfg2, 0, "T", data.len(), &data);
        assert_eq!(e.out_sample_head[..4], e2.out_sample_head[..]);
    }
}
