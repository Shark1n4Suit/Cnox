//! CLF mini-executor: walks an execution plan and dispatches to SIMD kernels.
//!
//! Dispatches by op_id to Rust SIMD kernels (clf-kernels-simd) so each op uses its
//! correct ABI instead of a single uniform signature. This avoids SIGSEGV when
//! the container was built from a CLFC whose blobs have per-op ABIs (Add/ReLU/Conv etc.).

use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::{HashMap, HashSet};

#[cfg(target_arch = "x86_64")]
use clf_kernels_simd::{
    clf_simd_add, clf_simd_batchnorm, clf_simd_conv, clf_simd_global_avg_pool, clf_simd_matmul,
    clf_simd_relu,
};

/// Execution plan layout (binary, little-endian):
/// - workspace_elements: u32  (workspace size in f32 elements)
/// - num_nodes: u32
/// - For each node (7 * u32): op_id, in_off, out_off, w_off, in_len, out_len, w_len (all element counts/offsets)
/// - num_kernels: u32
/// - kernel_offset[0..num_kernels]: u32 (byte offset of each blob from code_base; node i uses kernel i)
pub const PLAN_HEADER_SIZE: usize = 4 + 4; // workspace_elements + num_nodes
pub const PLAN_NODE_SIZE: usize = 7 * 4;  // 7 u32 per node
pub const PLAN_KERNEL_HEADER: usize = 4;  // num_kernels

/// Parse execution plan and return (workspace_elements, node steps, kernel_offsets).
/// Plan must be at least PLAN_HEADER_SIZE bytes.
pub fn parse_plan(plan: &[u8]) -> CoelanoxResult<(u32, Vec<PlanStep>, Vec<u32>)> {
    if plan.len() < PLAN_HEADER_SIZE {
        return Err(CoelanoxError::Runtime(format!(
            "Execution plan too short: {} bytes",
            plan.len()
        )));
    }
    let workspace_elements = u32::from_le_bytes([plan[0], plan[1], plan[2], plan[3]]);
    let num_nodes = u32::from_le_bytes([plan[4], plan[5], plan[6], plan[7]]);
    let mut offset = 8usize;
    let mut steps = Vec::with_capacity(num_nodes as usize);
    for _ in 0..num_nodes {
        if offset + PLAN_NODE_SIZE > plan.len() {
            return Err(CoelanoxError::Runtime("Execution plan truncated at node".to_string()));
        }
        steps.push(PlanStep {
            op_id: u32::from_le_bytes(plan[offset..offset + 4].try_into().unwrap()),
            in_off: u32::from_le_bytes(plan[offset + 4..offset + 8].try_into().unwrap()),
            out_off: u32::from_le_bytes(plan[offset + 8..offset + 12].try_into().unwrap()),
            w_off: u32::from_le_bytes(plan[offset + 12..offset + 16].try_into().unwrap()),
            in_len: u32::from_le_bytes(plan[offset + 16..offset + 20].try_into().unwrap()),
            out_len: u32::from_le_bytes(plan[offset + 20..offset + 24].try_into().unwrap()),
            w_len: u32::from_le_bytes(plan[offset + 24..offset + 28].try_into().unwrap()),
        });
        offset += PLAN_NODE_SIZE;
    }
    if offset + 4 > plan.len() {
        return Err(CoelanoxError::Runtime("Execution plan truncated at kernel count".to_string()));
    }
    let num_kernels = u32::from_le_bytes(plan[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut kernel_offsets = Vec::with_capacity(num_kernels);
    for _ in 0..num_kernels {
        if offset + 4 > plan.len() {
            return Err(CoelanoxError::Runtime("Execution plan truncated at kernel offsets".to_string()));
        }
        kernel_offsets.push(u32::from_le_bytes(plan[offset..offset + 4].try_into().unwrap()));
        offset += 4;
    }
    Ok((workspace_elements, steps, kernel_offsets))
}

/// Maximum conv scratch elements needed for any conv in the plan (for allocating scratch buffer).
fn conv_scratch_from_lens(in_len: usize, out_len: usize, w_len: usize) -> usize {
    let mut best = 0usize;
    for c_in in 1..=in_len {
        if in_len % c_in != 0 {
            continue;
        }
        let spatial = in_len / c_in;
        if out_len % spatial != 0 {
            continue;
        }
        let c_out = out_len / spatial;
        let weight_el = c_in * c_out;
        if weight_el == 0 || w_len % weight_el != 0 {
            continue;
        }
        let k_prod = w_len / weight_el;
        let k_h = (k_prod as f64).sqrt() as usize;
        let k_w = if k_h > 0 { k_prod / k_h } else { 0 };
        if k_h == 0 || k_w == 0 || k_h * k_w != k_prod {
            continue;
        }
        let h = (spatial as f64).sqrt() as usize;
        let w = if h > 0 { spatial / h } else { 0 };
        if h * w != spatial {
            continue;
        }
        let pad_h = (k_h - 1) / 2;
        let pad_w = (k_w - 1) / 2;
        let h_out = (h + 2 * pad_h - k_h) / 1 + 1;
        let w_out = (w + 2 * pad_w - k_w) / 1 + 1;
        let m = h_out * w_out;
        let k = c_in * k_h * k_w;
        let need = m * k + m * c_out;
        if need > best {
            best = need;
        }
        break; // first valid is enough for this step; we take max over steps
    }
    best
}

pub fn max_conv_scratch_elements(plan: &[u8]) -> CoelanoxResult<usize> {
    let (_, steps, _) = parse_plan(plan)?;
    Ok(steps
        .iter()
        .filter(|s| s.op_id == 30)
        .map(|s| conv_scratch_from_lens(s.in_len as usize, s.out_len as usize, s.w_len as usize))
        .max()
        .unwrap_or(0))
}

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub op_id: u32,
    pub in_off: u32,
    pub out_off: u32,
    pub w_off: u32,
    pub in_len: u32,
    pub out_len: u32,
    pub w_len: u32,
}

/// Run inference using the execution plan: for each node, dispatch by op_id to Rust SIMD kernels.
///
/// - `code_section`: unused (kept for API compatibility; kernels are Rust, not blobs).
/// - `plan`: binary plan from parse_plan / packager.
/// - `input`: model input (will be copied to workspace start).
/// - `output`: model output buffer (executor writes final node output here).
/// - `weights`: decompressed weights (flat f32).
/// - `workspace`: pre-allocated buffer of size (workspace_elements from plan) * 4 bytes.
/// - `audit_cb`: when Some, called after each step with (step_index, op_id, output_slice) for audit logging.
///
/// Non-x86: we cannot call SIMD kernels; return an error so runtime falls back to scalar.
#[cfg(not(target_arch = "x86_64"))]
pub fn run_with_plan(
    _code_section: &[u8],
    _plan: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _weights: &[f32],
    _workspace: &mut [f32],
    _audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
) -> CoelanoxResult<()> {
    Err(CoelanoxError::Runtime(
        "CLF executor is only implemented for x86_64; use scalar fallback on this platform".to_string(),
    ))
}

#[cfg(target_arch = "x86_64")]
pub fn run_with_plan(
    _code_section: &[u8],
    plan: &[u8],
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    workspace: &mut [f32],
    mut audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
) -> CoelanoxResult<()> {
    let (workspace_elements, steps, _kernel_offsets) = parse_plan(plan)?;
    if workspace.len() < workspace_elements as usize {
        return Err(CoelanoxError::Runtime(format!(
            "Workspace too small: need {} elements, got {}",
            workspace_elements,
            workspace.len()
        )));
    }
    // Copy input into workspace at offset 0 (plan assumes input at start)
    let input_len = input.len().min(workspace.len());
    workspace[..input_len].copy_from_slice(&input[..input_len]);

    for (step_index, step) in steps.iter().enumerate() {
        let in_ptr = workspace.as_ptr().wrapping_add(step.in_off as usize);
        let out_ptr = workspace.as_mut_ptr().wrapping_add(step.out_off as usize);
        let w_ptr = weights.as_ptr().wrapping_add(step.w_off as usize);
        let in_len = step.in_len as usize;
        let out_len = step.out_len as usize;
        let w_len = step.w_len as usize;

        unsafe {
            match step.op_id {
                1 => {
                    // Add: out[i] = a[i] + b[i]. Plan: in=a, w=b, out=out, count=in_len
                    clf_simd_add(out_ptr, in_ptr, w_ptr, in_len);
                }
                10 => {
                    // ReLU: out[i] = max(0, x[i])
                    clf_simd_relu(out_ptr, in_ptr, in_len);
                }
                30 => {
                    // Convolution: w_len==0 is a plan bug (re-package with fixed plan builder). No runtime fallback to avoid heap corruption.
                    dispatch_conv(in_ptr, out_ptr, w_ptr, in_len, out_len, w_len)
                        .map_err(|e| CoelanoxError::Runtime(format!("Step {} (op_id Conv): {} (re-package the model to fix the plan)", step_index, e)))?;
                }
                34 => {
                    // GlobalAvgPool: out (n*c), inp (n*c*h*w), n,c,h,w. in_len=n*c*h*w, out_len=n*c => c=out_len, h*w=in_len/out_len
                    let n = 1usize;
                    let c = out_len;
                    if c == 0 {
                        return Err(CoelanoxError::Runtime("GlobalAvgPool: out_len 0".to_string()));
                    }
                    let hw = in_len / c;
                    let h = (hw as f64).sqrt() as usize;
                    let w = if h > 0 { hw / h } else { 0 };
                    clf_simd_global_avg_pool(out_ptr, in_ptr, n, c, h, w);
                }
                35 => {
                    // BatchNorm: scale,bias,mean,var each c elements => w_len=4*c. If w_len==0, plan has no weights -> identity (copy in -> out).
                    if w_len == 0 {
                        if in_ptr != out_ptr && in_len > 0 {
                            std::ptr::copy_nonoverlapping(in_ptr, out_ptr, in_len);
                        }
                    } else {
                        let c = w_len / 4;
                        if c == 0 {
                            return Err(CoelanoxError::Runtime("BatchNorm: w_len must be 0 or >= 4".to_string()));
                        }
                        let nhw = in_len / c;
                        let n = 1usize;
                        let hw = nhw;
                        let h = (hw as f64).sqrt() as usize;
                        let w = if h > 0 { hw / h } else { 0 };
                        let scale = w_ptr;
                        let bias = w_ptr.add(c);
                        let mean = w_ptr.add(2 * c);
                        let var = w_ptr.add(3 * c);
                        clf_simd_batchnorm(out_ptr, in_ptr, scale, bias, mean, var, n, c, h, w);
                    }
                }
                50 => {
                    // MatMul: in_len=m*k, w_len=k*n, out_len=m*n => k^2=in_len*w_len/out_len, k=sqrt(...), m=in_len/k, n=out_len/m
                    if out_len == 0 {
                        return Err(CoelanoxError::Runtime("MatMul: out_len 0".to_string()));
                    }
                    let k_sq = in_len * w_len / out_len;
                    let k = (k_sq as f64).sqrt() as usize;
                    if k == 0 {
                        return Err(CoelanoxError::Runtime("MatMul: derived k=0".to_string()));
                    }
                    let m = in_len / k;
                    let n = out_len / m;
                    clf_simd_matmul(out_ptr, in_ptr, w_ptr, m, k, n);
                }
                _ => {
                    return Err(CoelanoxError::Runtime(format!(
                        "Unsupported op_id {} in execution plan (use scalar fallback)",
                        step.op_id
                    )));
                }
            }
        }
        // Audit: log after this step when callback provided
        if let Some(cb) = &mut audit_cb {
            let out_start = step.out_off as usize;
            let out_len = step.out_len as usize;
            let end = (out_start + out_len).min(workspace.len());
            if out_start < end {
                cb(step_index, step.op_id, &workspace[out_start..end]);
            }
        }
    }
    // Last node's output is the model output; copy to output buffer
    if let Some(last) = steps.last() {
        let out_start = last.out_off as usize;
        let out_len = last.out_len as usize;
        let copy_len = out_len.min(output.len()).min(workspace.len().saturating_sub(out_start));
        if copy_len > 0 {
            output[..copy_len].copy_from_slice(&workspace[out_start..out_start + copy_len]);
        }
    }
    Ok(())
}

/// Derive conv params from element counts and call clf_simd_conv. Assumes n=1, same padding when possible.
/// in_len = c_in*h*w, out_len = c_out*h_out*w_out, w_len = c_out*c_in*k_h*k_w; we search c_in so spatial = in_len/c_in divides out_len.
/// Zero-length: skip (no-op) so we don't error; w_len==0 still errors and falls back to scalar.
#[cfg(target_arch = "x86_64")]
unsafe fn dispatch_conv(
    in_ptr: *const f32,
    out_ptr: *mut f32,
    w_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    w_len: usize,
) -> CoelanoxResult<()> {
    if in_len == 0 || out_len == 0 {
        return Ok(());
    }
    if w_len == 0 {
        return Err(CoelanoxError::Runtime("Conv: zero weight length".to_string()));
    }
    let n = 1usize;
    let (c_in, c_out, h, w, k_h, k_w) = {
        let mut found = None;
        for c_in_candidate in 1..=in_len {
            if in_len % c_in_candidate != 0 {
                continue;
            }
            let spatial = in_len / c_in_candidate; // h*w
            if out_len % spatial != 0 {
                continue;
            }
            let c_out_candidate = out_len / spatial;
            let weight_elements = c_in_candidate * c_out_candidate;
            if weight_elements == 0 || w_len % weight_elements != 0 {
                continue;
            }
            let k_prod = w_len / weight_elements; // k_h * k_w
            let k_h_cand = (k_prod as f64).sqrt() as usize;
            let k_w_cand = if k_h_cand > 0 { k_prod / k_h_cand } else { 0 };
            if k_h_cand == 0 || k_w_cand == 0 {
                continue;
            }
            if k_h_cand * k_w_cand != k_prod {
                continue;
            }
            let h_cand = (spatial as f64).sqrt() as usize;
            let w_cand = if h_cand > 0 { spatial / h_cand } else { 0 };
            if h_cand * w_cand != spatial {
                continue;
            }
            found = Some((
                c_in_candidate,
                c_out_candidate,
                h_cand,
                w_cand,
                k_h_cand,
                k_w_cand,
            ));
            break;
        }
        found.ok_or_else(|| {
            CoelanoxError::Runtime(format!(
                "Conv: cannot derive shapes in_len={} out_len={} w_len={}",
                in_len, out_len, w_len
            ))
        })?
    };
    let pad_h = (k_h - 1) / 2;
    let pad_w = (k_w - 1) / 2;
    let stride_h = 1usize;
    let stride_w = 1usize;
    clf_simd_conv(
        out_ptr, in_ptr, w_ptr, std::ptr::null(), n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w,
        pad_h, pad_w,
    );
    Ok(())
}

/// Build execution plan from optimized IR (topological order, workspace layout, kernel offsets).
/// Used by the packager. Returns (plan_bytes, workspace_elements).
/// - node_order: topological execution order (indices into optimized_nodes).
/// - op_ids: op_id per node in same order as node_order (op_ids[i] = CLF op_id for node_order[i]).
/// - kernel_byte_offsets: byte offset of each kernel blob in code section (same order as node_order).
pub fn build_plan_from_ir(
    optimized_nodes: &[coelanox_core::ir::GraphNode],
    intermediate_tensors: &[coelanox_core::ir::IntermediateTensor],
    weight_offsets: &[usize],
    input_elements: usize,
    _output_elements: usize,
    node_order: &[usize],
    op_ids: &[u32],
    kernel_byte_offsets: &[u32],
) -> CoelanoxResult<(Vec<u8>, u32)> {
    use coelanox_core::ir::{OpType, TensorId};

    if node_order.len() != op_ids.len() || node_order.len() != kernel_byte_offsets.len() {
        return Err(CoelanoxError::Runtime(format!(
            "build_plan: node_order len {} op_ids len {} kernel_offsets len {}",
            node_order.len(),
            op_ids.len(),
            kernel_byte_offsets.len()
        )));
    }

    let mut tensor_to_offset: HashMap<TensorId, (u32, u32)> = HashMap::new();
    let produced: HashSet<TensorId> = optimized_nodes
        .iter()
        .flat_map(|n| n.output_tensors.iter().copied())
        .collect();
    let graph_inputs: Vec<TensorId> = optimized_nodes
        .iter()
        .flat_map(|n| n.input_tensors.iter().copied())
        .filter(|tid| tid.is_intermediate() && !produced.contains(tid))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut graph_inputs = graph_inputs;
    graph_inputs.sort_by_key(|t| t.0);
    for (i, &tid) in graph_inputs.iter().enumerate() {
        let per_input = input_elements / graph_inputs.len().max(1);
        let start = i * per_input;
        let len = if i == graph_inputs.len() - 1 {
            input_elements - start
        } else {
            per_input
        };
        tensor_to_offset.insert(tid, (start as u32, len as u32));
    }

    /// Resolve (w_off, w_len) for a node. For Conv/MatMul use the weight input with the
    /// largest element count (main kernel/matrix); otherwise use the first weight.
    /// Avoids zero weight length when the first input is not the main weight or index is wrong.
    fn weight_range_for_node(
        node: &coelanox_core::ir::GraphNode,
        weight_offsets: &[usize],
    ) -> (u32, u32) {
        let weight_inputs: Vec<_> = node.input_tensors.iter().filter(|t| t.is_weight()).collect();
        if weight_inputs.is_empty() {
            return (0, 0);
        }
        let use_largest = matches!(node.op_type, OpType::Convolution | OpType::MatMul);
        let (w_off, w_len) = if use_largest {
            weight_inputs
                .iter()
                .filter_map(|tid| {
                    let idx = tid.index();
                    let start = weight_offsets.get(idx).copied()?;
                    let end = weight_offsets.get(idx + 1).copied()?;
                    Some((start as u32, (end - start) as u32))
                })
                .max_by_key(|&(_, len)| len)
                .unwrap_or((0, 0))
        } else {
            weight_inputs
                .first()
                .and_then(|tid| {
                    let idx = tid.index();
                    let start = weight_offsets.get(idx).copied()?;
                    let end = weight_offsets.get(idx + 1).copied()?;
                    Some((start as u32, (end - start) as u32))
                })
                .unwrap_or((0, 0))
        };
        (w_off, w_len)
    }

    // Workspace layout: one slot per unique intermediate tensor that is produced.
    // If multiple nodes output the same tensor (in-place reuse), we reuse the same slot
    // instead of counting it again — otherwise we'd blow workspace to GB scale.
    let mut workspace_offset: u32 = input_elements as u32;
    let mut steps = Vec::with_capacity(node_order.len());
    for (step_index, &node_index) in node_order.iter().enumerate() {
        let node = &optimized_nodes[node_index];
        let (in_off, in_len) = node.input_tensors.iter()
            .find(|tid| tid.is_intermediate())
            .and_then(|tid| tensor_to_offset.get(tid).copied())
            .unwrap_or((0, input_elements as u32));
        let out_len = node.output_tensors.first().and_then(|tid| {
            if tid.is_intermediate() {
                intermediate_tensors.get(tid.index()).map(|t| t.elements() as u32)
            } else {
                None
            }
        }).unwrap_or(0);
        // Reuse existing slot if this output tensor was already assigned (in-place / shared output)
        let (out_off, out_len_actual) = node.output_tensors.first()
            .and_then(|tid| tid.is_intermediate().then_some(tid))
            .and_then(|tid| tensor_to_offset.get(tid).copied())
            .map(|(off, len)| (off, len))
            .unwrap_or_else(|| {
                let off = workspace_offset;
                workspace_offset += out_len;
                (off, out_len)
            });

        let (w_off, w_len) = weight_range_for_node(node, weight_offsets);
        if w_len == 0 && matches!(node.op_type, OpType::Convolution | OpType::MatMul) {
            return Err(CoelanoxError::Runtime(format!(
                "Plan build: node index {} (op {:?}) has no valid weight range (weight_offsets len {}); \
                 ensure every Conv/MatMul has a weight input indexed in 0..weight_tensors.len()",
                node_index,
                node.op_type,
                weight_offsets.len().saturating_sub(1),
            )));
        }

        steps.push((
            op_ids[step_index],
            in_off,
            out_off,
            w_off,
            in_len,
            out_len_actual,
            w_len,
        ));
        for &tid in &node.output_tensors {
            if tid.is_intermediate() {
                let len = intermediate_tensors.get(tid.index()).map(|t| t.elements() as u32).unwrap_or(out_len_actual);
                tensor_to_offset.insert(tid, (out_off, len));
            }
        }
    }

    let workspace_elements = workspace_offset;

    // Sanity check: workspace should not be huge for small models (e.g. 400M+ elements = ~1.6 GB)
    const MAX_WORKSPACE_ELEMENTS: u32 = 100_000_000; // 400 MB at 4 bytes/element
    if workspace_elements > MAX_WORKSPACE_ELEMENTS {
        let (worst_step, worst_out_len, worst_node_idx) = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.5, node_order[i])) // (step_index, out_len, node_index); tuple is (op_id, in_off, out_off, w_off, in_len, out_len, w_len)
            .max_by_key(|(_, out_len, _)| *out_len)
            .unwrap_or((0, 0, 0));
        let shape_info = optimized_nodes
            .get(worst_node_idx)
            .and_then(|n| n.output_tensors.first().copied())
            .filter(|tid| tid.is_intermediate())
            .and_then(|tid| {
                intermediate_tensors.get(tid.index()).map(|t| {
                    format!("tensor idx {} shape {:?} ({} elements)", tid.index(), t.shape.dimensions, t.elements())
                })
            })
            .unwrap_or_else(|| "unknown tensor".to_string());
        return Err(CoelanoxError::Runtime(format!(
            "Workspace would be {} elements (~{:.1} GB) - likely a bug in IR shapes. \
             Largest contributor: step {} (node index {}), output size {} elements. \
             {} — fix this shape in the translator or optimizer.",
            workspace_elements,
            (workspace_elements as f64) * 4.0 / 1e9,
            worst_step,
            worst_node_idx,
            worst_out_len,
            shape_info,
        )));
    }

    let mut plan = Vec::new();
    plan.extend_from_slice(&workspace_elements.to_le_bytes());
    plan.extend_from_slice(&(steps.len() as u32).to_le_bytes());
    for (op_id, in_off, out_off, w_off, in_len, out_len, w_len) in &steps {
        plan.extend_from_slice(&op_id.to_le_bytes());
        plan.extend_from_slice(&in_off.to_le_bytes());
        plan.extend_from_slice(&out_off.to_le_bytes());
        plan.extend_from_slice(&w_off.to_le_bytes());
        plan.extend_from_slice(&in_len.to_le_bytes());
        plan.extend_from_slice(&out_len.to_le_bytes());
        plan.extend_from_slice(&w_len.to_le_bytes());
    }
    plan.extend_from_slice(&(kernel_byte_offsets.len() as u32).to_le_bytes());
    for &k in kernel_byte_offsets {
        plan.extend_from_slice(&k.to_le_bytes());
    }
    Ok((plan, workspace_elements))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plan_empty() {
        let plan = [8u8, 0, 0, 0, 0u8, 0, 0, 0, 0u8, 0, 0, 0]; // workspace=8, nodes=0, num_kernels=0
        let (ws, steps, offsets) = parse_plan(&plan).unwrap();
        assert_eq!(ws, 8);
        assert_eq!(steps.len(), 0);
        assert_eq!(offsets.len(), 0);
    }
}
