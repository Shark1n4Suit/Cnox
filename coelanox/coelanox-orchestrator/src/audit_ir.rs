//! Helpers to relate audit logging to IR structure (e.g. which nodes feed graph outputs).

use coelanox_core::ir::OptimizedIR;
use std::collections::HashSet;

/// Node indices that produce intermediate tensors not consumed as inputs by any other node
/// (typical graph outputs). Order is sorted ascending. Empty means "could not classify" (caller
/// may treat as "log all ops" for audit).
pub fn graph_output_node_indices(ir: &OptimizedIR) -> Vec<usize> {
    let nodes = &ir.optimized_nodes;
    let mut used = HashSet::new();
    for n in nodes {
        for tid in &n.input_tensors {
            if tid.is_intermediate() {
                used.insert(*tid);
            }
        }
    }

    let mut out = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        let mut is_sink = false;
        for tid in &n.output_tensors {
            if tid.is_intermediate() && !used.contains(tid) {
                is_sink = true;
                break;
            }
        }
        if is_sink {
            out.push(i);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}
