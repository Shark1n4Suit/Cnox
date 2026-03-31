//! Audit mode: log after each op during inference for traceability and debugging.
//! Enabled via CLI `run --audit` / `serve --audit`; state is process-global so
//! worker threads in all backend paths can contribute to a single run audit.
//! When audit_output path has extension .json, writes a JSON array of op entries; otherwise plain text lines.

use std::path::Path;
use std::sync::Mutex;

#[derive(serde::Serialize)]
struct AuditEntry {
    op_index: usize,
    op_type: String,
    out_len: usize,
    out_sample: Vec<f32>,
}

struct AuditState {
    enabled: bool,
    log_path: Option<std::path::PathBuf>,
    entries: Vec<AuditEntry>,
}

static AUDIT_STATE: Mutex<AuditState> = Mutex::new(AuditState {
    enabled: false,
    log_path: None,
    entries: Vec::new(),
});

/// Enable or disable audit mode. Optionally set a file path to write logs.
/// When disabling, if path was set and extension is .json, writes a JSON array; otherwise plain text lines.
pub fn set_audit_mode(enabled: bool, log_path: Option<&Path>) {
    let mut state = AUDIT_STATE.lock().expect("audit state mutex poisoned");
    if enabled {
        state.entries.clear();
        state.log_path = log_path.map(|p| p.to_path_buf());
        state.enabled = true;
    } else {
        state.enabled = false;
        let path = state.log_path.take();
        let entries = std::mem::take(&mut state.entries);
        if let Some(path) = path {
            if !entries.is_empty() {
                let is_json = path.extension().map_or(false, |e| e == "json");
                if is_json {
                    let mut entries = entries;
                    entries.sort_by_key(|e| e.op_index);
                    if let Ok(s) = serde_json::to_string_pretty(&entries) {
                        let _ = std::fs::write(&path, s);
                    }
                } else {
                    // Sort by op_index so scalar (multi-thread) audit matches plan order and is comparable to SIMD audit
                    let mut entries = entries;
                    entries.sort_by_key(|e| e.op_index);
                    let lines: Vec<String> = entries.iter()
                        .map(|e| {
                            let sample: String = e.out_sample.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(", ");
                            let suffix = if e.out_len > 4 { "..." } else { "" };
                            format!("audit op {} {} out_len={} out_sample=[{}{}]", e.op_index, e.op_type, e.out_len, sample, suffix)
                        })
                        .collect();
                    let _ = std::fs::write(&path, lines.join("\n") + "\n");
                }
            }
        }
    }
}

/// Returns true if audit mode is enabled (call from execution path).
pub fn is_enabled() -> bool {
    AUDIT_STATE
        .lock()
        .expect("audit state mutex poisoned")
        .enabled
}

/// Log one op: node index, op type, output size, and output sample. Buffered until set_audit_mode(false).
/// Called by the scalar fallback after each node.
pub fn log_after_op(
    node_index: usize,
    op_type: &impl std::fmt::Debug,
    output_len: usize,
    output_sample: &[f32],
) {
    let mut state = AUDIT_STATE.lock().expect("audit state mutex poisoned");
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
