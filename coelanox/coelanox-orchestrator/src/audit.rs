//! Audit mode: log after each op during inference for traceability and debugging.
//! Enabled via CLI `run --audit` / `serve --audit`; state is process-global so
//! worker threads in all backend paths can contribute to a single run audit.
//!
//! **Strategies** ([`AuditStrategy`]):
//! - **`forget`** (default): after each flush, clear RAM; `.json` paths use **JSONL** (one JSON array per line per request). Plain paths append text lines per batch.
//! - **`last`**: overwrite the audit file with **only the last** flushed batch; clears RAM after each flush.
//! - **`long`**: legacy session mode — entries accumulate in RAM and each flush rewrites the full snapshot (unbounded memory on long-lived `serve`).

use std::path::Path;
use std::sync::Mutex;

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

#[derive(serde::Serialize, Clone)]
struct AuditEntry {
    op_index: usize,
    op_type: String,
    out_len: usize,
    out_sample: Vec<f32>,
}

struct AuditState {
    enabled: bool,
    enable_count: usize,
    log_path: Option<std::path::PathBuf>,
    entries: Vec<AuditEntry>,
    strategy: AuditStrategy,
}

static AUDIT_STATE: Mutex<AuditState> = Mutex::new(AuditState {
    enabled: false,
    enable_count: 0,
    log_path: None,
    entries: Vec::new(),
    strategy: AuditStrategy::Forget,
});

#[inline]
fn lock_audit_state() -> std::sync::MutexGuard<'static, AuditState> {
    AUDIT_STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn write_entries_overwrite(path: &Path, entries: &[AuditEntry]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let is_json = path.extension().map_or(false, |e| e == "json");
    if is_json {
        let s = serde_json::to_string_pretty(entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, s)
    } else {
        let lines: Vec<String> = entries
            .iter()
            .map(|e| {
                let sample: String = e
                    .out_sample
                    .iter()
                    .map(|v| format!("{:.4}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                let suffix = if e.out_len > 4 { "..." } else { "" };
                format!(
                    "audit op {} {} out_len={} out_sample=[{}{}]",
                    e.op_index, e.op_type, e.out_len, sample, suffix
                )
            })
            .collect();
        std::fs::write(path, lines.join("\n") + "\n")
    }
}

/// Append one batch: for `.json` paths, one **JSON array** per line (JSONL). For plain paths, append line-oriented audit rows.
fn append_entries_batch(path: &Path, entries: &[AuditEntry]) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let is_json = path.extension().map_or(false, |e| e == "json");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    if is_json {
        let line = serde_json::to_string(entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
    } else {
        for e in entries {
            let sample: String = e
                .out_sample
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ");
            let suffix = if e.out_len > 4 { "..." } else { "" };
            let line = format!(
                "audit op {} {} out_len={} out_sample=[{}{}]\n",
                e.op_index, e.op_type, e.out_len, sample, suffix
            );
            f.write_all(line.as_bytes())?;
        }
    }
    Ok(())
}

fn ordered_entries_for_flush(state: &AuditState) -> Vec<AuditEntry> {
    match state.strategy {
        AuditStrategy::Long => state.entries.clone(),
        AuditStrategy::Forget | AuditStrategy::Last => {
            let mut sorted: Vec<&AuditEntry> = state.entries.iter().collect();
            sorted.sort_by_key(|e| e.op_index);
            sorted
                .into_iter()
                .map(|e| AuditEntry {
                    op_index: e.op_index,
                    op_type: e.op_type.clone(),
                    out_len: e.out_len,
                    out_sample: e.out_sample.clone(),
                })
                .collect()
        }
    }
}

fn flush_locked(state: &mut AuditState) -> std::io::Result<()> {
    let Some(path) = state.log_path.as_deref() else {
        return Ok(());
    };
    if state.entries.is_empty() {
        return Ok(());
    }

    let ordered = ordered_entries_for_flush(state);

    match state.strategy {
        AuditStrategy::Forget => {
            append_entries_batch(path, &ordered)?;
            state.entries.clear();
        }
        AuditStrategy::Last => {
            write_entries_overwrite(path, &ordered)?;
            state.entries.clear();
        }
        AuditStrategy::Long => {
            write_entries_overwrite(path, &ordered)?;
        }
    }
    Ok(())
}

/// Enable or disable audit mode. Optionally set a file path to write logs.
/// `strategy` applies on the first `enabled=true` in a session (nested enables reuse it).
pub fn set_audit_mode(enabled: bool, log_path: Option<&Path>, strategy: AuditStrategy) {
    let mut state = lock_audit_state();
    if enabled {
        if state.enable_count == 0 {
            state.entries.clear();
            state.log_path = log_path.map(|p| p.to_path_buf());
            state.strategy = strategy;
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
        }
    }
}

/// Flush current buffered audit entries immediately to the configured audit path.
/// This is useful for long-lived `serve` sessions where callers need on-disk
/// evidence during process lifetime, not only on shutdown.
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

/// Log one op: node index, op type, output size, and output sample. Buffered until flush/disable.
/// Called by the scalar fallback after each node.
pub fn log_after_op(
    node_index: usize,
    op_type: &impl std::fmt::Debug,
    output_len: usize,
    output_sample: &[f32],
) {
    let mut state = lock_audit_state();
    if !state.enabled {
        return;
    }
    let sample: Vec<f32> = output_sample.iter().take(4).copied().collect();
    let entry = AuditEntry {
        op_index: node_index,
        op_type: format!("{:?}", op_type),
        out_len: output_len,
        out_sample: sample,
    };
    tracing::info!(
        "audit op {} {} out_len={} out_sample=[{:?}]",
        entry.op_index, entry.op_type, entry.out_len, entry.out_sample
    );
    state.entries.push(entry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    /// Global audit state is process-wide; serialize tests that toggle it.
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
        log_after_op(0, &"Add", 4, &[1.0, 2.0, 3.0, 4.0]);
        flush_now().expect("flush");
        let s = std::fs::read_to_string(&path).expect("read audit");
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
        log_after_op(0, &"A", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(0, &"B", 1, &[2.0]);
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
        log_after_op(0, &"First", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(0, &"Second", 1, &[2.0]);
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
        log_after_op(0, &"A", 1, &[1.0]);
        flush_now().expect("flush1");
        log_after_op(1, &"B", 1, &[2.0]);
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
        log_after_op(1, &"Mul", 2, &[3.0, 4.0]);
        flush_now().expect("flush");
        let s = std::fs::read_to_string(&path).expect("read audit");
        assert!(s.contains("audit op"));
        assert!(s.contains("Mul"));
        set_audit_mode(false, None, AuditStrategy::Forget);
    }

    #[test]
    fn disable_flushes_and_clears_state() {
        let _guard = AUDIT_TEST_LOCK.lock().expect("audit test lock");
        reset_audit();
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("out.log");
        set_audit_mode(true, Some(&path), AuditStrategy::Forget);
        log_after_op(0, &"Sub", 1, &[1.0]);
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
        set_audit_mode(true, Some(&p2), AuditStrategy::Forget); // ignored while session active
        log_after_op(0, &"Relu", 1, &[1.0]);
        set_audit_mode(false, None, AuditStrategy::Forget); // still enabled due to nested count
        assert!(flush_now().is_ok());
        set_audit_mode(false, None, AuditStrategy::Forget); // final disable flushes + clears
        assert!(std::fs::metadata(&p1).is_ok());
        assert!(std::fs::metadata(&p2).is_err());
    }
}
