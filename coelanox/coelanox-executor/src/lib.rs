//! CLF mini-executor: walks an execution plan and dispatches to kernel blobs.
//!
//! Two execution paths:
//! - **Raw blobs** (`run_with_plan`): Calls (code_base + kernel_offset) as a 6-arg function. Used when
//!   the container embeds CLFC blobs (from packager's `ClfReader::build_code_section`).
//! - **Symbol dispatch** (`run_with_plan_symbols`): Dispatches by op_id to Rust SIMD kernels (clf-kernels-simd).
//!   Used when blobs are not embedded but simd symbols are linked; each op uses its correct ABI.

use coelanox_core::{CoelanoxError, CoelanoxResult};
use std::collections::{HashMap, HashSet};

#[cfg(target_arch = "x86_64")]
use clf_kernels_simd::{
    clf_simd_add, clf_simd_avg_pool, clf_simd_batchnorm, clf_simd_conv, clf_simd_cos, clf_simd_divide,
    clf_simd_dropout, clf_simd_equal, clf_simd_exp, clf_simd_gather, clf_simd_gemm, clf_simd_gelu,
    clf_simd_global_avg_pool, clf_simd_global_max_pool, clf_simd_greater, clf_simd_greater_equal,
    clf_simd_layernorm, clf_simd_less, clf_simd_less_equal, clf_simd_log, clf_simd_log_softmax,
    clf_simd_matmul, clf_simd_max_pool, clf_simd_multiply, clf_simd_not, clf_simd_not_equal,
    clf_simd_or, clf_simd_and, clf_simd_permute, clf_simd_concatenate, clf_simd_split, clf_simd_slice,
    clf_simd_scatter, clf_simd_pow, clf_simd_reduce_mean, clf_simd_reduce_max, clf_simd_reduce_min,
    clf_simd_reduce_prod, clf_simd_reduce_sum, clf_simd_relu, clf_simd_reshape, clf_simd_sigmoid,
    clf_simd_sin, clf_simd_softmax, clf_simd_abs, clf_simd_sqrt, clf_simd_subtract, clf_simd_swish, clf_simd_tanh,
    clf_simd_transpose,
};

/// CLFE 6-arg kernel signature (x86-64 SysV: input, output, weights, in_len, out_len, w_len).
/// All CLFC blobs packed for the mini-executor must conform to this ABI.
#[cfg(target_arch = "x86_64")]
type ClfeKernelFn = unsafe extern "C" fn(
    *const f32,
    *mut f32,
    *const f32,
    usize,
    usize,
    usize,
);

/// Plan format version 1: version(4) + workspace(4) + scratch(4) + num_nodes(4) + nodes + num_kernels(4) + kernel_offsets.
/// Legacy (no version): workspace(4) + num_nodes(4) + nodes + num_kernels(4) + kernel_offsets; scratch computed from conv steps.
pub const PLAN_VERSION_V1: u32 = 1;

/// Execution plan layout (binary, little-endian):
/// - V1: version=1 (4), workspace_elements (4), scratch_elements (4), num_nodes (4), ...
/// - Legacy: workspace_elements (4), num_nodes (4), ...
/// - For each node (7 * u32): op_id, in_off, out_off, w_off, in_len, out_len, w_len
/// - num_kernels: u32, kernel_offset[0..num_kernels]: u32
pub const PLAN_HEADER_SIZE: usize = 4 + 4; // workspace_elements + num_nodes (legacy)
pub const PLAN_HEADER_V1_SIZE: usize = 4 + 4 + 4 + 4; // version + workspace + scratch + num_nodes
pub const PLAN_NODE_SIZE: usize = 7 * 4;  // 7 u32 per node
pub const PLAN_KERNEL_HEADER: usize = 4;  // num_kernels

/// Read u32 little-endian from plan at offset; returns error if truncated.
#[inline]
fn read_u32(plan: &[u8], at: usize) -> CoelanoxResult<u32> {
    let end = at + 4;
    if end > plan.len() {
        return Err(CoelanoxError::Runtime(format!(
            "Execution plan truncated at offset {} (need 4 bytes, have {})",
            at,
            plan.len().saturating_sub(at)
        )));
    }
    Ok(u32::from_le_bytes(
        plan[at..end]
            .try_into()
            .map_err(|_| CoelanoxError::Runtime("Invalid plan layout".to_string()))?,
    ))
}

/// Returns scratch elements for the plan: from V1 header if present, else computed from conv steps (legacy).
pub fn scratch_elements_from_plan(plan: &[u8]) -> CoelanoxResult<usize> {
    if plan.len() >= 12 {
        let version = read_u32(plan, 0)?;
        if version == PLAN_VERSION_V1 {
            return Ok(read_u32(plan, 8)? as usize);
        }
    }
    max_conv_scratch_elements(plan)
}

/// Parse execution plan and return (workspace_elements, node steps, kernel_offsets).
/// Supports V1 (version 1: version, workspace, scratch, num_nodes) and legacy (workspace, num_nodes).
pub fn parse_plan(plan: &[u8]) -> CoelanoxResult<(u32, Vec<PlanStep>, Vec<u32>)> {
    if plan.len() < PLAN_HEADER_SIZE {
        return Err(CoelanoxError::Runtime(format!(
            "Execution plan too short: {} bytes",
            plan.len()
        )));
    }
    let (workspace_elements, num_nodes, mut offset) = if plan.len() >= 4 {
        let version = read_u32(plan, 0)?;
        if version == PLAN_VERSION_V1 && plan.len() >= PLAN_HEADER_V1_SIZE {
            let workspace = read_u32(plan, 4)?;
            let num_nodes = read_u32(plan, 12)?;
            (workspace, num_nodes, 16usize)
        } else {
            let workspace = read_u32(plan, 0)?;
            let num_nodes = read_u32(plan, 4)?;
            (workspace, num_nodes, 8usize)
        }
    } else {
        let workspace_elements = u32::from_le_bytes([plan[0], plan[1], plan[2], plan[3]]);
        let num_nodes = u32::from_le_bytes([plan[4], plan[5], plan[6], plan[7]]);
        (workspace_elements, num_nodes, 8usize)
    };
    let mut steps = Vec::with_capacity(num_nodes as usize);
    for _ in 0..num_nodes {
        if offset + PLAN_NODE_SIZE > plan.len() {
            return Err(CoelanoxError::Runtime("Execution plan truncated at node".to_string()));
        }
        steps.push(PlanStep {
            op_id: read_u32(plan, offset)?,
            in_off: read_u32(plan, offset + 4)?,
            out_off: read_u32(plan, offset + 8)?,
            w_off: read_u32(plan, offset + 12)?,
            in_len: read_u32(plan, offset + 16)?,
            out_len: read_u32(plan, offset + 20)?,
            w_len: read_u32(plan, offset + 24)?,
        });
        offset += PLAN_NODE_SIZE;
    }
    if offset + 4 > plan.len() {
        return Err(CoelanoxError::Runtime("Execution plan truncated at kernel count".to_string()));
    }
    let num_kernels = read_u32(plan, offset)? as usize;
    offset += 4;
    let mut kernel_offsets = Vec::with_capacity(num_kernels);
    for _ in 0..num_kernels {
        if offset + 4 > plan.len() {
            return Err(CoelanoxError::Runtime("Execution plan truncated at kernel offsets".to_string()));
        }
        kernel_offsets.push(read_u32(plan, offset)?);
        offset += 4;
    }
    Ok((workspace_elements, steps, kernel_offsets))
}

/// Raw conv scratch from lengths (no caps). Used for legacy plan scratch and diagnostics.
/// Multiple (c_in, spatial) decompositions can match the same in_len/out_len/w_len;
/// we take the **minimum** scratch over all valid decompositions so allocation matches
/// the actual NCHW layout the IR describes (e.g. 32 channels @ 14x14) instead of
/// a degenerate interpretation that would require GB of scratch.
pub fn conv_scratch_from_lens_raw(in_len: usize, out_len: usize, w_len: usize) -> usize {
    let mut best = usize::MAX;
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
        if need < best {
            best = need;
        }
    }
    if best == usize::MAX {
        0
    } else {
        best
    }
}

/// Maximum conv scratch elements needed for any conv in the plan (for allocating scratch buffer).
/// Returns 0 if any length exceeds a reasonable conv size (plan/IR bug guard).
fn conv_scratch_from_lens(in_len: usize, out_len: usize, w_len: usize) -> usize {
    const REASONABLE_CONV_MAX: usize = 16 * 1024 * 1024; // 16M elements
    if in_len > REASONABLE_CONV_MAX || out_len > REASONABLE_CONV_MAX || w_len > REASONABLE_CONV_MAX {
        return 0;
    }
    conv_scratch_from_lens_raw(in_len, out_len, w_len)
}

/// Dump all conv steps with raw scratch (for finding bad step that blows allocation).
/// Returns (step_index, in_len, out_len, w_len, scratch_elements_raw) for each step with op_id == 30.
pub fn dump_plan_conv_steps(plan: &[u8]) -> CoelanoxResult<Vec<(usize, u32, u32, u32, usize)>> {
    let (_, steps, _) = parse_plan(plan)?;
    let mut out = Vec::new();
    for (step_index, s) in steps.iter().enumerate() {
        if s.op_id == 30 {
            let scratch = conv_scratch_from_lens_raw(
                s.in_len as usize,
                s.out_len as usize,
                s.w_len as usize,
            );
            out.push((step_index, s.in_len, s.out_len, s.w_len, scratch));
        }
    }
    Ok(out)
}

pub fn max_conv_scratch_elements(plan: &[u8]) -> CoelanoxResult<usize> {
    let (_, steps, _) = parse_plan(plan)?;
    let raw = steps
        .iter()
        .filter(|s| s.op_id == 30)
        .map(|s| conv_scratch_from_lens(s.in_len as usize, s.out_len as usize, s.w_len as usize))
        .max()
        .unwrap_or(0);
    Ok(raw)
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
///
/// When `weight_provider` is Some, the executor uses it to get the weight slice per step (for double
/// ring buffer: provider returns a slice from the current buffer; caller overlaps prefetch with compute).
/// The closure receives (step_index, w_off, w_len) and must return a slice of length >= w_len.
#[cfg(not(target_arch = "x86_64"))]
pub fn run_with_plan(
    _code_section: &[u8],
    _plan: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _weights: &[f32],
    _workspace: &mut [f32],
    _audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    _abort_check: Option<&(dyn Fn() -> bool + '_)>,
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
    abort_check: Option<&(dyn Fn() -> bool + '_)>,
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
        // Enforce execution timeout: abort between steps when deadline exceeded
        if let Some(check) = abort_check {
            if check() {
                return Err(CoelanoxError::Runtime(
                    "Execution timeout exceeded (COELANOX_EXECUTION_TIMEOUT_MS)".to_string(),
                ));
            }
        }
        let in_ptr = workspace.as_ptr().wrapping_add(step.in_off as usize);
        let out_ptr = workspace.as_mut_ptr().wrapping_add(step.out_off as usize);
        let w_ptr = if (step.w_off & W_OFF_IN_WORKSPACE_MASK) != 0 {
            workspace.as_ptr().wrapping_add((step.w_off & !W_OFF_IN_WORKSPACE_MASK) as usize)
        } else {
            weights.as_ptr().wrapping_add(step.w_off as usize)
        };
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
                13 => {
                    // Softmax: over last dim. Plan: in_len=out_len; treat as 1D (rows=1, cols=in_len) or use w_len as cols if set.
                    if in_len == 0 {
                        return Err(CoelanoxError::Runtime("Softmax: in_len 0".to_string()));
                    }
                    let cols = if w_len > 0 && in_len % w_len == 0 { w_len } else { in_len };
                    let rows = in_len / cols;
                    clf_simd_softmax(out_ptr, in_ptr, rows, cols);
                }
                15 => {
                    // Gelu: elementwise. out = gelu(x).
                    clf_simd_gelu(out_ptr, in_ptr, in_len);
                }
                36 => {
                    // LayerNorm: w = [gamma (C), beta (C)], w_len=2*C. in_len=out_len. C = w_len/2.
                    if w_len == 0 || w_len % 2 != 0 {
                        return Err(CoelanoxError::Runtime("LayerNorm: w_len must be 2*C (gamma, beta)".to_string()));
                    }
                    let c = w_len / 2;
                    if in_len % c != 0 {
                        return Err(CoelanoxError::Runtime("LayerNorm: in_len not divisible by C".to_string()));
                    }
                    let scale = w_ptr;
                    let bias = w_ptr.add(c);
                    clf_simd_layernorm(out_ptr, in_ptr, scale, bias, in_len, c);
                }
                40 => {
                    // Reshape: copy in -> out (same element count).
                    if in_len != out_len {
                        return Err(CoelanoxError::Runtime("Reshape: in_len must equal out_len".to_string()));
                    }
                    clf_simd_reshape(out_ptr, in_ptr, in_len);
                }
                41 => {
                    // Transpose 2D: out[col*rows+row] = in[row*cols+col]. w_len=0 => square (rows=sqrt(in_len)); else rows=w_len, cols=in_len/rows.
                    if in_len != out_len {
                        return Err(CoelanoxError::Runtime("Transpose: in_len must equal out_len".to_string()));
                    }
                    let (rows, cols) = if w_len > 0 && in_len % w_len == 0 {
                        (w_len, in_len / w_len)
                    } else {
                        let r = (in_len as f64).sqrt() as usize;
                        let c = if r > 0 { in_len / r } else { 0 };
                        (r, c)
                    };
                    if rows == 0 || cols == 0 || rows * cols != in_len {
                        return Err(CoelanoxError::Runtime("Transpose: cannot derive rows*cols".to_string()));
                    }
                    clf_simd_transpose(out_ptr, in_ptr, rows, cols);
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
                51 => {
                    // Gemm: same as MatMul for SIMD (alpha handled in scalar or separate scale).
                    if out_len == 0 {
                        return Err(CoelanoxError::Runtime("Gemm: out_len 0".to_string()));
                    }
                    let k_sq = in_len * w_len / out_len;
                    let k = (k_sq as f64).sqrt() as usize;
                    if k == 0 {
                        return Err(CoelanoxError::Runtime("Gemm: derived k=0".to_string()));
                    }
                    let m = in_len / k;
                    let n = out_len / m;
                    clf_simd_gemm(out_ptr, in_ptr, w_ptr, m, k, n);
                }
                // Arithmetic 2–4
                2 => clf_simd_subtract(out_ptr, in_ptr, w_ptr, in_len),
                3 => clf_simd_multiply(out_ptr, in_ptr, w_ptr, in_len),
                4 => clf_simd_divide(out_ptr, in_ptr, w_ptr, in_len),
                // Activations 11–16
                11 => clf_simd_sigmoid(out_ptr, in_ptr, in_len),
                12 => clf_simd_tanh(out_ptr, in_ptr, in_len),
                14 => {
                    let cols = if w_len > 0 && in_len % w_len == 0 { w_len } else { in_len };
                    let rows = in_len / cols;
                    clf_simd_log_softmax(out_ptr, in_ptr, rows, cols);
                }
                16 => clf_simd_swish(out_ptr, in_ptr, in_len),
                // Math 19–25 (Pow uses first weight as exponent if w_len >= 1)
                19 => clf_simd_abs(out_ptr, in_ptr, in_len),
                20 => clf_simd_sqrt(out_ptr, in_ptr, in_len),
                21 => {
                    let exp = if w_len > 0 { *w_ptr } else { 2.0 };
                    clf_simd_pow(out_ptr, in_ptr, in_len, exp);
                }
                22 => clf_simd_cos(out_ptr, in_ptr, in_len),
                23 => clf_simd_sin(out_ptr, in_ptr, in_len),
                24 => clf_simd_exp(out_ptr, in_ptr, in_len),
                25 => clf_simd_log(out_ptr, in_ptr, in_len),
                // Pool 31–33, Dropout 37
                31 | 32 => {
                    // 2x2 pool: in_len = n*c*h*w, out_len = n*c*(h/2)*(w/2) => in_len == 4*out_len.
                    if out_len == 0 || in_len != 4 * out_len {
                        return Err(CoelanoxError::Runtime("MaxPool/AvgPool: need in_len == 4*out_len".to_string()));
                    }
                    let n = 1usize;
                    let c = 1usize;
                    let hw = in_len;
                    let h = (hw as f64).sqrt() as usize;
                    let w = if h > 0 { hw / h } else { 0 };
                    if h < 2 || w < 2 {
                        return Err(CoelanoxError::Runtime("MaxPool/AvgPool: spatial dims must be >= 2".to_string()));
                    }
                    if step.op_id == 31 {
                        clf_simd_max_pool(out_ptr, in_ptr, n, c, h, w);
                    } else {
                        clf_simd_avg_pool(out_ptr, in_ptr, n, c, h, w);
                    }
                }
                33 => {
                    let n = 1usize;
                    let c = out_len;
                    if c == 0 {
                        return Err(CoelanoxError::Runtime("GlobalMaxPool: out_len 0".to_string()));
                    }
                    let hw = in_len / c;
                    let h = (hw as f64).sqrt() as usize;
                    let w = if h > 0 { hw / h } else { 0 };
                    clf_simd_global_max_pool(out_ptr, in_ptr, n, c, h, w);
                }
                37 => clf_simd_dropout(out_ptr, in_ptr, in_len),
                // Tensor 42–47
                42 => {
                    let rows = if w_len > 0 && in_len % w_len == 0 { w_len } else { (in_len as f64).sqrt() as usize };
                    clf_simd_permute(out_ptr, in_ptr, in_len, out_len, rows);
                }
                43 => clf_simd_concatenate(out_ptr, in_ptr, in_len),
                44 => clf_simd_split(out_ptr, in_ptr, out_len),
                45 => clf_simd_slice(out_ptr, in_ptr, out_len.min(in_len)),
                46 => clf_simd_gather(out_ptr, in_ptr, w_ptr, out_len, in_len),
                47 => clf_simd_scatter(out_ptr, in_ptr, w_ptr, in_len, out_len),
                // Reductions 60–64
                60 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceSum: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_sum(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                61 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMean: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_mean(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                62 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMax: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_max(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                63 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMin: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_min(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                64 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceProd: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_prod(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                // Comparisons 80–85, Logical 90–92
                80 => clf_simd_equal(out_ptr, in_ptr, w_ptr, in_len),
                81 => clf_simd_not_equal(out_ptr, in_ptr, w_ptr, in_len),
                82 => clf_simd_greater(out_ptr, in_ptr, w_ptr, in_len),
                83 => clf_simd_greater_equal(out_ptr, in_ptr, w_ptr, in_len),
                84 => clf_simd_less(out_ptr, in_ptr, w_ptr, in_len),
                85 => clf_simd_less_equal(out_ptr, in_ptr, w_ptr, in_len),
                90 => clf_simd_and(out_ptr, in_ptr, w_ptr, in_len),
                91 => clf_simd_or(out_ptr, in_ptr, w_ptr, in_len),
                92 => clf_simd_not(out_ptr, in_ptr, in_len),
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

/// Symbol-based execution (Rust SIMD kernels). Exposed for simd-symbols backend; same behavior as run_with_plan on x86_64.
#[cfg(not(target_arch = "x86_64"))]
pub fn run_with_plan_symbols(
    _code_section: &[u8],
    _plan: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _weights: &[f32],
    _workspace: &mut [f32],
    _audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    _abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()> {
    Err(CoelanoxError::Runtime(
        "CLF executor is only implemented for x86_64; use scalar fallback on this platform".to_string(),
    ))
}

#[cfg(target_arch = "x86_64")]
pub fn run_with_plan_symbols(
    code_section: &[u8],
    plan: &[u8],
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    workspace: &mut [f32],
    audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()> {
    run_with_plan(code_section, plan, input, output, weights, workspace, audit_cb, abort_check)
}

/// Run inference with a per-step weight provider (for double ring buffer in the orchestrator).
/// The executor provides a single `weight_buffer`; each step the caller's
/// `weight_provider(step_index, w_off, w_len, weight_buffer)` fills it and returns the length.
#[cfg(not(target_arch = "x86_64"))]
pub fn run_with_plan_with_weight_provider<F>(
    _code_section: &[u8],
    _plan: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _workspace: &mut [f32],
    _weight_buffer: &mut [f32],
    _audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    _weight_provider: &mut F,
    _abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()>
where
    F: FnMut(usize, u32, u32, &mut [f32]) -> usize,
{
    Err(CoelanoxError::Runtime(
        "CLF executor is only implemented for x86_64; use scalar fallback on this platform".to_string(),
    ))
}

#[cfg(target_arch = "x86_64")]
pub fn run_with_plan_with_weight_provider<F>(
    code_section: &[u8],
    plan: &[u8],
    input: &[f32],
    output: &mut [f32],
    workspace: &mut [f32],
    weight_buffer: &mut [f32],
    mut audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    weight_provider: &mut F,
    abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()>
where
    F: FnMut(usize, u32, u32, &mut [f32]) -> usize,
{
    let (workspace_elements, steps, kernel_offsets) = parse_plan(plan)?;
    if workspace.len() < workspace_elements as usize {
        return Err(CoelanoxError::Runtime(format!(
            "Workspace too small: need {} elements, got {}",
            workspace_elements,
            workspace.len()
        )));
    }
    if code_section.is_empty() {
        return Err(CoelanoxError::Runtime(
            "Code section is empty; container must contain CLFC blobs".to_string(),
        ));
    }
    if kernel_offsets.len() < steps.len() {
        return Err(CoelanoxError::Runtime(format!(
            "Plan has {} steps but only {} kernel offsets (re-package with CLFC)",
            steps.len(),
            kernel_offsets.len()
        )));
    }
    let code_base = code_section.as_ptr() as usize;

    let input_len = input.len().min(workspace.len());
    workspace[..input_len].copy_from_slice(&input[..input_len]);

    for (step_index, step) in steps.iter().enumerate() {
        if let Some(check) = abort_check {
            if check() {
                return Err(CoelanoxError::Runtime(
                    "Execution timeout exceeded (COELANOX_EXECUTION_TIMEOUT_MS)".to_string(),
                ));
            }
        }
        let w_len = weight_provider(step_index, step.w_off, step.w_len, weight_buffer);
        if weight_buffer.len() < w_len {
            return Err(CoelanoxError::Runtime(format!(
                "Weight buffer too small for step {} (need {} elements)",
                step_index,
                w_len
            )));
        }
        let w_ptr = weight_buffer.as_ptr();
        let k_off = kernel_offsets[step_index] as usize;
        if k_off >= code_section.len() {
            return Err(CoelanoxError::Runtime(format!(
                "Step {}: kernel offset {} out of bounds (code section len {})",
                step_index,
                k_off,
                code_section.len()
            )));
        }
        let in_ptr = workspace.as_ptr().wrapping_add(step.in_off as usize);
        let out_ptr = workspace.as_mut_ptr().wrapping_add(step.out_off as usize);
        let in_len = step.in_len as usize;
        let out_len = step.out_len as usize;

        let kernel_fn: *const ClfeKernelFn = (code_base + k_off) as *const ClfeKernelFn;
        unsafe {
            (*kernel_fn)(in_ptr, out_ptr, w_ptr, in_len, out_len, w_len);
        }

        if let Some(cb) = &mut audit_cb {
            let out_start = step.out_off as usize;
            let out_len_audit = step.out_len as usize;
            let end = (out_start + out_len_audit).min(workspace.len());
            if out_start < end {
                cb(step_index, step.op_id, &workspace[out_start..end]);
            }
        }
    }
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

/// Symbol-based execution (Rust SIMD kernels) for simd-symbols backend.
#[cfg(not(target_arch = "x86_64"))]
pub fn run_with_plan_with_weight_provider_symbols<F>(
    _code_section: &[u8],
    _plan: &[u8],
    _input: &[f32],
    _output: &mut [f32],
    _workspace: &mut [f32],
    _weight_buffer: &mut [f32],
    _audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    _weight_provider: &mut F,
    _abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()>
where
    F: FnMut(usize, u32, u32, &mut [f32]) -> usize,
{
    Err(CoelanoxError::Runtime(
        "SIMD symbol executor is only implemented for x86_64; use scalar fallback on this platform"
            .to_string(),
    ))
}

#[cfg(target_arch = "x86_64")]
pub fn run_with_plan_with_weight_provider_symbols<F>(
    _code_section: &[u8],
    plan: &[u8],
    input: &[f32],
    output: &mut [f32],
    workspace: &mut [f32],
    weight_buffer: &mut [f32],
    mut audit_cb: Option<&mut (dyn FnMut(usize, u32, &[f32]) + '_)>,
    weight_provider: &mut F,
    abort_check: Option<&(dyn Fn() -> bool + '_)>,
) -> CoelanoxResult<()>
where
    F: FnMut(usize, u32, u32, &mut [f32]) -> usize,
{
    let (workspace_elements, steps, _kernel_offsets) = parse_plan(plan)?;
    if workspace.len() < workspace_elements as usize {
        return Err(CoelanoxError::Runtime(format!(
            "Workspace too small: need {} elements, got {}",
            workspace_elements,
            workspace.len()
        )));
    }
    let input_len = input.len().min(workspace.len());
    workspace[..input_len].copy_from_slice(&input[..input_len]);

    for (step_index, step) in steps.iter().enumerate() {
        if let Some(check) = abort_check {
            if check() {
                return Err(CoelanoxError::Runtime(
                    "Execution timeout exceeded (COELANOX_EXECUTION_TIMEOUT_MS)".to_string(),
                ));
            }
        }
        let w_len = weight_provider(step_index, step.w_off, step.w_len, weight_buffer);
        if weight_buffer.len() < w_len {
            return Err(CoelanoxError::Runtime(format!(
                "Weight buffer too small for step {} (need {} elements)",
                step_index,
                w_len
            )));
        }
        let w_ptr = weight_buffer.as_ptr();
        let in_ptr = workspace.as_ptr().wrapping_add(step.in_off as usize);
        let out_ptr = workspace.as_mut_ptr().wrapping_add(step.out_off as usize);
        let in_len = step.in_len as usize;
        let out_len = step.out_len as usize;

        unsafe {
            match step.op_id {
                1 => {
                    clf_simd_add(out_ptr, in_ptr, w_ptr, in_len);
                }
                10 => {
                    clf_simd_relu(out_ptr, in_ptr, in_len);
                }
                30 => {
                    dispatch_conv(in_ptr, out_ptr, w_ptr, in_len, out_len, w_len)
                        .map_err(|e| CoelanoxError::Runtime(format!("Step {} (op_id Conv): {}", step_index, e)))?;
                }
                34 => {
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
                13 => {
                    if in_len == 0 {
                        return Err(CoelanoxError::Runtime("Softmax: in_len 0".to_string()));
                    }
                    let cols = if w_len > 0 && in_len % w_len == 0 { w_len } else { in_len };
                    let rows = in_len / cols;
                    clf_simd_softmax(out_ptr, in_ptr, rows, cols);
                }
                15 => {
                    clf_simd_gelu(out_ptr, in_ptr, in_len);
                }
                36 => {
                    if w_len == 0 || w_len % 2 != 0 {
                        return Err(CoelanoxError::Runtime("LayerNorm: w_len must be 2*C".to_string()));
                    }
                    let c = w_len / 2;
                    if in_len % c != 0 {
                        return Err(CoelanoxError::Runtime("LayerNorm: in_len not divisible by C".to_string()));
                    }
                    let scale = w_ptr;
                    let bias = w_ptr.add(c);
                    clf_simd_layernorm(out_ptr, in_ptr, scale, bias, in_len, c);
                }
                40 => {
                    if in_len != out_len {
                        return Err(CoelanoxError::Runtime("Reshape: in_len must equal out_len".to_string()));
                    }
                    clf_simd_reshape(out_ptr, in_ptr, in_len);
                }
                41 => {
                    if in_len != out_len {
                        return Err(CoelanoxError::Runtime("Transpose: in_len must equal out_len".to_string()));
                    }
                    let (rows, cols) = if w_len > 0 && in_len % w_len == 0 {
                        (w_len, in_len / w_len)
                    } else {
                        let r = (in_len as f64).sqrt() as usize;
                        (r, if r > 0 { in_len / r } else { 0 })
                    };
                    if rows == 0 || cols == 0 || rows * cols != in_len {
                        return Err(CoelanoxError::Runtime("Transpose: cannot derive rows*cols".to_string()));
                    }
                    clf_simd_transpose(out_ptr, in_ptr, rows, cols);
                }
                50 => {
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
                51 => {
                    if out_len == 0 {
                        return Err(CoelanoxError::Runtime("Gemm: out_len 0".to_string()));
                    }
                    let k_sq = in_len * w_len / out_len;
                    let k = (k_sq as f64).sqrt() as usize;
                    if k == 0 {
                        return Err(CoelanoxError::Runtime("Gemm: derived k=0".to_string()));
                    }
                    let m = in_len / k;
                    let n = out_len / m;
                    clf_simd_gemm(out_ptr, in_ptr, w_ptr, m, k, n);
                }
                2 => clf_simd_subtract(out_ptr, in_ptr, w_ptr, in_len),
                3 => clf_simd_multiply(out_ptr, in_ptr, w_ptr, in_len),
                4 => clf_simd_divide(out_ptr, in_ptr, w_ptr, in_len),
                11 => clf_simd_sigmoid(out_ptr, in_ptr, in_len),
                12 => clf_simd_tanh(out_ptr, in_ptr, in_len),
                14 => {
                    let cols = if w_len > 0 && in_len % w_len == 0 { w_len } else { in_len };
                    let rows = in_len / cols;
                    clf_simd_log_softmax(out_ptr, in_ptr, rows, cols);
                }
                16 => clf_simd_swish(out_ptr, in_ptr, in_len),
                19 => clf_simd_abs(out_ptr, in_ptr, in_len),
                20 => clf_simd_sqrt(out_ptr, in_ptr, in_len),
                21 => {
                    let exp = if w_len > 0 { *w_ptr } else { 2.0 };
                    clf_simd_pow(out_ptr, in_ptr, in_len, exp);
                }
                22 => clf_simd_cos(out_ptr, in_ptr, in_len),
                23 => clf_simd_sin(out_ptr, in_ptr, in_len),
                24 => clf_simd_exp(out_ptr, in_ptr, in_len),
                25 => clf_simd_log(out_ptr, in_ptr, in_len),
                31 | 32 => {
                    if out_len == 0 || in_len != 4 * out_len {
                        return Err(CoelanoxError::Runtime("MaxPool/AvgPool: need in_len == 4*out_len".to_string()));
                    }
                    let n = 1usize;
                    let c = 1usize;
                    let hw = in_len;
                    let h = (hw as f64).sqrt() as usize;
                    let w = if h > 0 { hw / h } else { 0 };
                    if h < 2 || w < 2 {
                        return Err(CoelanoxError::Runtime("MaxPool/AvgPool: spatial dims must be >= 2".to_string()));
                    }
                    if step.op_id == 31 {
                        clf_simd_max_pool(out_ptr, in_ptr, n, c, h, w);
                    } else {
                        clf_simd_avg_pool(out_ptr, in_ptr, n, c, h, w);
                    }
                }
                33 => {
                    let n = 1usize;
                    let c = out_len;
                    if c == 0 {
                        return Err(CoelanoxError::Runtime("GlobalMaxPool: out_len 0".to_string()));
                    }
                    let hw = in_len / c;
                    let h = (hw as f64).sqrt() as usize;
                    let w = if h > 0 { hw / h } else { 0 };
                    clf_simd_global_max_pool(out_ptr, in_ptr, n, c, h, w);
                }
                37 => clf_simd_dropout(out_ptr, in_ptr, in_len),
                42 => {
                    let rows = if w_len > 0 && in_len % w_len == 0 { w_len } else { (in_len as f64).sqrt() as usize };
                    clf_simd_permute(out_ptr, in_ptr, in_len, out_len, rows);
                }
                43 => clf_simd_concatenate(out_ptr, in_ptr, in_len),
                44 => clf_simd_split(out_ptr, in_ptr, out_len),
                45 => clf_simd_slice(out_ptr, in_ptr, out_len.min(in_len)),
                46 => clf_simd_gather(out_ptr, in_ptr, w_ptr, out_len, in_len),
                47 => clf_simd_scatter(out_ptr, in_ptr, w_ptr, in_len, out_len),
                60 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceSum: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_sum(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                61 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMean: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_mean(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                62 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMax: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_max(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                63 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceMin: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_min(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                64 => {
                    if out_len == 0 || in_len % out_len != 0 {
                        return Err(CoelanoxError::Runtime("ReduceProd: out_len must divide in_len".to_string()));
                    }
                    clf_simd_reduce_prod(out_ptr, in_ptr, out_len, in_len / out_len);
                }
                80 => clf_simd_equal(out_ptr, in_ptr, w_ptr, in_len),
                81 => clf_simd_not_equal(out_ptr, in_ptr, w_ptr, in_len),
                82 => clf_simd_greater(out_ptr, in_ptr, w_ptr, in_len),
                83 => clf_simd_greater_equal(out_ptr, in_ptr, w_ptr, in_len),
                84 => clf_simd_less(out_ptr, in_ptr, w_ptr, in_len),
                85 => clf_simd_less_equal(out_ptr, in_ptr, w_ptr, in_len),
                90 => clf_simd_and(out_ptr, in_ptr, w_ptr, in_len),
                91 => clf_simd_or(out_ptr, in_ptr, w_ptr, in_len),
                92 => clf_simd_not(out_ptr, in_ptr, in_len),
                _ => {
                    return Err(CoelanoxError::Runtime(format!(
                        "Unsupported op_id {} at step {}",
                        step.op_id, step_index
                    )));
                }
            }
        }
        if let Some(cb) = &mut audit_cb {
            let out_start = step.out_off as usize;
            let out_len_audit = step.out_len as usize;
            let end = (out_start + out_len_audit).min(workspace.len());
            if out_start < end {
                cb(step_index, step.op_id, &workspace[out_start..end]);
            }
        }
    }
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

/// High bit in w_off means second matrix is in workspace (activation×activation MatMul).
const W_OFF_IN_WORKSPACE_MASK: u32 = 1 << 31;
/// Find weight tensor index from (w_off, w_len) and weight_offsets (element offsets).
fn weight_index_from_offset(w_off: u32, weight_offsets: &[usize]) -> Option<usize> {
    let start = (w_off & !W_OFF_IN_WORKSPACE_MASK) as usize;
    for i in 0..weight_offsets.len().saturating_sub(1) {
        if weight_offsets[i] <= start && start < weight_offsets[i + 1] {
            return Some(i);
        }
    }
    None
}

/// Build execution plan from optimized IR (topological order, workspace layout, kernel offsets).
/// Uses shape-based peak scratch contract for all ops; writes V1 plan with scratch_elements.
/// - weight_shapes: shapes of weight tensors in same order as weight_offsets (one per weight).
/// - max_workspace_mb: if set, fail when (workspace + peak_scratch) * 4 > max_workspace_mb * 1024 * 1024.
pub fn build_plan_from_ir(
    optimized_nodes: &[coelanox_core::ir::GraphNode],
    intermediate_tensors: &[coelanox_core::ir::IntermediateTensor],
    weight_offsets: &[usize],
    weight_shapes: &[coelanox_core::ir::Shape],
    input_elements: usize,
    _output_elements: usize,
    node_order: &[usize],
    op_ids: &[u32],
    kernel_byte_offsets: &[u32],
    max_workspace_mb: Option<u64>,
) -> CoelanoxResult<(Vec<u8>, u32)> {
    use coelanox_core::ir::{OpType, TensorId};
    use coelanox_core::scratch::peak_scratch_elements_for_op;

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
    ) -> CoelanoxResult<(u32, u32)> {
        let weight_inputs: Vec<_> = node.input_tensors.iter().filter(|t| t.is_weight()).collect();
        if weight_inputs.is_empty() {
            return Ok((0, 0));
        }
        let use_largest = matches!(node.op_type, OpType::Convolution | OpType::MatMul);
        let pair = if use_largest {
            let candidates: Vec<(u32, u32)> = weight_inputs
                .iter()
                .filter_map(|tid| {
                    let idx = tid.index();
                    let start = weight_offsets.get(idx).copied()?;
                    let end = weight_offsets.get(idx + 1).copied()?;
                    Some((start as u32, (end - start) as u32))
                })
                .collect();
            if candidates.is_empty() {
                return Err(CoelanoxError::Runtime(format!(
                    "Plan build: no valid weight slice for op {:?} (weight indices {:?} vs weight_offsets len {})",
                    node.op_type,
                    weight_inputs.iter().map(|t| t.index()).collect::<Vec<_>>(),
                    weight_offsets.len()
                )));
            }
            candidates.into_iter().max_by_key(|&(_, len)| len).unwrap()
        } else {
            let tid = weight_inputs[0];
            let idx = tid.index();
            let start = weight_offsets.get(idx).copied();
            let end = weight_offsets.get(idx + 1).copied();
            match (start, end) {
                (Some(s), Some(e)) => (s as u32, (e - s) as u32),
                _ => {
                    return Err(CoelanoxError::Runtime(format!(
                        "Plan build: weight tensor index {} out of range for weight_offsets len {}",
                        idx,
                        weight_offsets.len()
                    )));
                }
            }
        };
        Ok(pair)
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

        let (w_off, w_len) = {
            let from_weights = weight_range_for_node(node, weight_offsets)?;
            if from_weights.1 > 0 {
                from_weights
            } else if node.op_type == OpType::MatMul || node.op_type == OpType::Gemm {
                // MatMul/Gemm can be activation × activation (e.g. BERT: Q@K^T, attn_weights@V).
                // Use second intermediate input as the "B" matrix; mark with W_OFF_IN_WORKSPACE_MASK so executor reads from workspace.
                let intermediate_inputs: Vec<_> = node.input_tensors.iter()
                    .filter(|tid| tid.is_intermediate())
                    .filter_map(|tid| tensor_to_offset.get(tid).copied())
                    .collect();
                if intermediate_inputs.len() >= 2 {
                    let (off, len) = intermediate_inputs[1];
                    (off | W_OFF_IN_WORKSPACE_MASK, len)
                } else {
                    (0, 0)
                }
            } else if node.op_type == OpType::Transpose && from_weights.1 == 0 {
                // Transpose 2D: executor needs rows; pass first dim in w_len so rows=w_len, cols=in_len/rows.
                let rows = node.input_tensors.iter()
                    .find(|tid| tid.is_intermediate())
                    .and_then(|tid| intermediate_tensors.get(tid.index()))
                    .and_then(|t| t.shape.dimensions.first().copied())
                    .map(|d| d as u32)
                    .unwrap_or(0);
                (0, rows)
            } else {
                from_weights
            }
        };
        if w_len == 0 && node.op_type == OpType::Convolution {
            return Err(CoelanoxError::Runtime(format!(
                "Plan build: node index {} (op {:?}) has no valid weight range (weight_offsets len {}); \
                 ensure every Conv has a weight input indexed in 0..weight_tensors.len()",
                node_index,
                node.op_type,
                weight_offsets.len().saturating_sub(1),
            )));
        }
        if w_len == 0 && (node.op_type == OpType::MatMul || node.op_type == OpType::Gemm) {
            return Err(CoelanoxError::Runtime(format!(
                "Plan build: node index {} (op {:?}) has no weight and not two intermediate inputs",
                node_index, node.op_type,
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

    // Shape-based peak scratch for all ops (contract).
    let mut peak_scratch_elements: usize = 0;
    for (step_index, &node_index) in node_order.iter().enumerate() {
        let node = &optimized_nodes[node_index];
        let (_, _, _, w_off, _, _, _) = steps[step_index];
        let weight_idx = weight_index_from_offset(w_off, weight_offsets);
        let input_shapes: Vec<_> = node
            .input_tensors
            .iter()
            .filter(|tid| tid.is_intermediate())
            .filter_map(|tid| intermediate_tensors.get(tid.index()).map(|t| t.shape.clone()))
            .collect();
        let output_shapes: Vec<_> = node
            .output_tensors
            .iter()
            .filter_map(|tid| {
                if tid.is_intermediate() {
                    intermediate_tensors.get(tid.index()).map(|t| t.shape.clone())
                } else {
                    None
                }
            })
            .collect();
        let node_weight_shapes: Vec<_> = weight_idx
            .and_then(|i| weight_shapes.get(i))
            .map(|s| s.clone())
            .into_iter()
            .collect();
        let attributes: Vec<_> = node.attributes.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let scratch = peak_scratch_elements_for_op(
            &node.op_type,
            &input_shapes,
            &output_shapes,
            &node_weight_shapes,
            &attributes,
        );
        if scratch > peak_scratch_elements {
            peak_scratch_elements = scratch;
        }
    }

    if let Some(cap_mb) = max_workspace_mb {
        let total_bytes = (workspace_elements as u64 + peak_scratch_elements as u64) * 4;
        let cap_bytes = cap_mb * 1024 * 1024;
        if total_bytes > cap_bytes {
            return Err(CoelanoxError::Runtime(format!(
                "Build failed: model requires {} MB dynamic memory (workspace + scratch), but --max-workspace is {} MB. Increase --max-workspace or reduce model size.",
                (total_bytes + 1024 * 1024 - 1) / (1024 * 1024),
                cap_mb,
            )));
        }
    }

    let mut plan = Vec::new();
    plan.extend_from_slice(&PLAN_VERSION_V1.to_le_bytes());
    plan.extend_from_slice(&workspace_elements.to_le_bytes());
    plan.extend_from_slice(&(peak_scratch_elements as u32).to_le_bytes());
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
