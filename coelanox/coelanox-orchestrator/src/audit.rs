//! Audit mode: log after each op during inference for traceability and debugging.
//! Set via CLI `run --audit`; state is thread-local so only the current run is audited.
//! When audit_output path has extension .json, writes a JSON array of op entries; otherwise plain text lines.

use std::cell::RefCell;
use std::path::Path;

#[derive(serde::Serialize)]
struct AuditEntry {
    op_index: usize,
    op_type: String,
    out_len: usize,
    out_sample: Vec<f32>,
}

thread_local! {
    static AUDIT_ENABLED: RefCell<bool> = RefCell::new(false);
    static AUDIT_LOG_PATH: RefCell<Option<std::path::PathBuf>> = RefCell::new(None);
    static AUDIT_ENTRIES: RefCell<Vec<AuditEntry>> = RefCell::new(Vec::new());
}

/// Enable or disable audit mode for the current thread. Optionally set a file path to write logs.
/// When disabling, if path was set and extension is .json, writes a JSON array; otherwise plain text lines.
pub fn set_audit_mode(enabled: bool, log_path: Option<&Path>) {
    if enabled {
        AUDIT_ENTRIES.with(|c| c.borrow_mut().clear());
        AUDIT_LOG_PATH.with(|c| *c.borrow_mut() = log_path.map(|p| p.to_path_buf()));
        AUDIT_ENABLED.with(|c| *c.borrow_mut() = true);
    } else {
        AUDIT_ENABLED.with(|c| *c.borrow_mut() = false);
        let path = AUDIT_LOG_PATH.with(|c| c.take());
        let entries = AUDIT_ENTRIES.with(|e| e.take());
        if let Some(path) = path {
            if !entries.is_empty() {
                let is_json = path.extension().map_or(false, |e| e == "json");
                if is_json {
                    if let Ok(s) = serde_json::to_string_pretty(&entries) {
                        let _ = std::fs::write(&path, s);
                    }
                } else {
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
    AUDIT_ENABLED.with(|c| *c.borrow())
}

/// Log one op: node index, op type, output size, and output sample. Buffered until set_audit_mode(false).
/// Called by the scalar fallback after each node.
pub fn log_after_op(
    node_index: usize,
    op_type: &impl std::fmt::Debug,
    output_len: usize,
    output_sample: &[f32],
) {
    if !is_enabled() {
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
    AUDIT_ENTRIES.with(|c| c.borrow_mut().push(entry));
}
