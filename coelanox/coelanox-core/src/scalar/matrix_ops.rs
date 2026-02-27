//! Matrix operations
//!
//! MatMul/Gemm use matrixmultiply (BLIS-style blocked + SIMD) when dimensions are valid,
//! so scalar fallback can approach BLAS-like speed without CLF kernels.

use crate::ir::Attribute;
use crate::{CoelanoxError, CoelanoxResult};
use std::collections::HashMap;

/// Execute matrix multiplication: output = input @ weights
/// 
/// # Arguments
/// * `input` - Input matrix data (flattened)
/// * `output` - Output matrix buffer (must be pre-allocated)
/// * `weights` - Weight matrix data (flattened)
/// * `attributes` - Operation-specific attributes
///   * `input_rows` - Number of rows in input matrix
///   * `input_cols` - Number of columns in input matrix
///   * `weight_rows` - Number of rows in weight matrix (must equal input_cols)
///   * `weight_cols` - Number of columns in weight matrix
/// 
/// # Returns
/// * `Ok(())` if operation succeeds
/// * `Err(CoelanoxError)` if operation fails
pub fn matmul(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    // Infer dimensions from sizes: input (M,K), weights (K,N) -> output (M,N)
    let dims = attributes.get("dimensions")
        .and_then(|a| match a {
            crate::ir::Attribute::IntArray(d) => Some(d.iter().map(|&x| x as usize).collect::<Vec<_>>()),
            _ => None,
        });
    let (m, k, n) = if let Some(d) = dims {
        if d.len() >= 4 {
            (d[0], d[1], d[3])
        } else if d.len() >= 2 {
            (d[0], d[1], output.len() / d[0].max(1))
        } else {
            (1, input.len(), output.len())
        }
    } else {
        // Infer M,K,N from sizes: in_len = M*K, w_len = K*N, out_len = M*N => K^2 = in_len*w_len/out_len
        let in_len = input.len();
        let w_len = weights.len();
        let out_len = output.len();
        if in_len == 0 || w_len == 0 || out_len == 0 {
            (1, in_len, out_len)
        } else if in_len * w_len % out_len != 0 {
            (1, in_len, out_len)
        } else {
            let k_sq = in_len * w_len / out_len;
            let k_cand = (k_sq as f64).sqrt() as usize;
            // Try k_cand and neighbors (sqrt can truncate); K must divide in_len and w_len, and (in_len/K)*(w_len/K) == out_len
            let mut inferred = None;
            for &k_try in &[k_cand, k_cand.saturating_sub(1), k_cand + 1] {
                if k_try == 0 {
                    continue;
                }
                if in_len % k_try != 0 || w_len % k_try != 0 {
                    continue;
                }
                let m = in_len / k_try;
                let n = w_len / k_try;
                if m * n == out_len {
                    inferred = Some((m, k_try, n));
                    break;
                }
            }
            match inferred {
                Some((mm, kk, nn)) => (mm, kk, nn),
                _ => (1, in_len, out_len),
            }
        }
    };
    let out_len = (m * n).min(output.len());
    if input.len() < m * k || weights.len() < k * n {
        return Err(CoelanoxError::Internal(format!(
            "MatMul: input len {} or weights len {} too small for M={} K={} N={}",
            input.len(),
            weights.len(),
            m,
            k,
            n
        )));
    }

    // Use matrixmultiply (blocked + SIMD) for normal sizes; naive loop for tiny/degenerate.
    if m > 0 && k > 0 && n > 0 && out_len >= m * n {
        output[..out_len].fill(0.0f32);
        // Row-major: A(m,k) rsa=k csa=1, B(k,n) rsb=n csb=1, C(m,n) rsc=n csc=1. C = 1*A*B + 0*C.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0f32,
                input.as_ptr(),
                k as isize,
                1isize,
                weights.as_ptr(),
                n as isize,
                1isize,
                0.0f32,
                output.as_mut_ptr(),
                n as isize,
                1isize,
            );
        }
    } else {
        output[..out_len].fill(0.0f32);
        for i in 0..m {
            let out_row = i * n;
            let out_end = (out_row + n).min(output.len());
            for p in 0..k {
                let in_val = input[i * k + p];
                if in_val == 0.0 {
                    continue;
                }
                let w_base = p * n;
                for j in out_row..out_end {
                    output[j] += in_val * weights[w_base + j - out_row];
                }
            }
        }
    }
    Ok(())
}

/// Execute general matrix multiplication: output = alpha * (input @ weights) + beta * output
/// 
/// # Attributes
/// * `alpha` - Scaling factor for matrix product (default: 1.0)
/// * `beta` - Scaling factor for output (default: 0.0)
/// * `input_rows`, `input_cols`, `weight_rows`, `weight_cols` - Matrix dimensions
pub fn gemm(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let alpha = attributes.get("alpha")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(1.0);
    let _beta = attributes.get("beta")
        .and_then(|a| match a { crate::ir::Attribute::Float(f) => Some(*f), _ => None })
        .unwrap_or(0.0);
    matmul(input, output, weights, attributes)?;
    for x in output.iter_mut() {
        *x *= alpha;
    }
    Ok(())
}

/// Helper: parse IntArray attribute as vec of usize (e.g. [kernel_h, kernel_w]).
fn int_array_2(attributes: &HashMap<String, Attribute>, key: &str, default: (usize, usize)) -> (usize, usize) {
    match attributes.get(key) {
        Some(crate::ir::Attribute::IntArray(a)) if a.len() >= 2 => {
            (a[0] as usize, a[1] as usize)
        }
        Some(crate::ir::Attribute::Int(i)) => (*i as usize, *i as usize),
        _ => default,
    }
}

/// Helper: parse optional input_shape [N,C,H,W] from attributes.
fn input_shape_nchw(attributes: &HashMap<String, Attribute>) -> Option<(usize, usize, usize, usize)> {
    let a = attributes.get("input_shape").and_then(|x| match x {
        crate::ir::Attribute::IntArray(v) if v.len() >= 4 => Some(v),
        _ => None,
    })?;
    Some((a[0] as usize, a[1] as usize, a[2] as usize, a[3] as usize))
}

/// 2D NCHW convolution via im2col + sgemm (same fast path as MatMul). Uses matrixmultiply for the heavy work.
fn conv2d_nchw_im2col(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> CoelanoxResult<()> {
    let h_out = (h + 2 * pad_h).saturating_sub(k_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w).saturating_sub(k_w) / stride_w + 1;
    let m = n * h_out * w_out;
    let k = c_in * k_h * k_w;
    if weights.len() < c_out * k {
        return Err(CoelanoxError::Internal(format!(
            "Conv2d: weights len {} < c_out*k {}",
            weights.len(),
            c_out * k,
        )));
    }
    if input.len() < n * c_in * h * w {
        return Err(CoelanoxError::Internal("Conv2d: input too small".to_string()));
    }
    if output.len() < n * c_out * h_out * w_out {
        return Err(CoelanoxError::Internal("Conv2d: output too small".to_string()));
    }
    // im2col buffer: (M, K) row-major. Each row = one output spatial position, flattened patch.
    let mut im2col = vec![0.0f32; m * k];
    for ni in 0..n {
        for ho in 0..h_out {
            for wo in 0..w_out {
                let row = ni * h_out * w_out + ho * w_out + wo;
                for c in 0..c_in {
                    for kh in 0..k_h {
                        for kw in 0..k_w {
                            let hi = (ho * stride_h).wrapping_add(kh).wrapping_sub(pad_h);
                            let wi = (wo * stride_w).wrapping_add(kw).wrapping_sub(pad_w);
                            let val = if hi < h && wi < w {
                                input[ni * c_in * h * w + c * h * w + hi * w + wi]
                            } else {
                                0.0f32
                            };
                            im2col[row * k + (c * k_h * k_w + kh * k_w + kw)] = val;
                        }
                    }
                }
            }
        }
    }
    // Weights (C_out, C_in, k_h, k_w) -> need (K, C_out) for gemm: im2col (M,K) @ W (K,N) -> (M,N).
    let mut w_mat = vec![0.0f32; k * c_out];
    for co in 0..c_out {
        for ci in 0..c_in {
            for kh in 0..k_h {
                for kw in 0..k_w {
                    let src = co * c_in * k_h * k_w + ci * k_h * k_w + kh * k_w + kw;
                    let dst = (ci * k_h * k_w + kh * k_w + kw) * c_out + co;
                    w_mat[dst] = weights[src];
                }
            }
        }
    }
    // gemm: (M,K) @ (K,N) -> (M,N) with N = c_out. Result (M, c_out) row-major.
    let mut gemm_out = vec![0.0f32; m * c_out];
    if m > 0 && k > 0 && c_out > 0 {
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                c_out,
                1.0f32,
                im2col.as_ptr(),
                k as isize,
                1isize,
                w_mat.as_ptr(),
                c_out as isize,
                1isize,
                0.0f32,
                gemm_out.as_mut_ptr(),
                c_out as isize,
                1isize,
            );
        }
    }
    // Copy (M, c_out) to output NCHW: output[n,c,ho,wo] = gemm_out[m, c] with m = n*H_out*W_out + ho*W_out + wo.
    for m in 0..m {
        let ni = m / (h_out * w_out);
        let spatial = m % (h_out * w_out);
        let ho = spatial / w_out;
        let wo = spatial % w_out;
        for c in 0..c_out {
            output[ni * c_out * h_out * w_out + c * h_out * w_out + ho * w_out + wo] = gemm_out[m * c_out + c];
        }
    }
    Ok(())
}

/// Infer (n, c_in, h, w) and (c_out, h_out, w_out) from lengths and kernel/stride/pad when input_shape is missing.
fn infer_conv2d_shapes(
    in_len: usize,
    out_len: usize,
    w_len: usize,
    k_h: usize,
    k_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Option<((usize, usize, usize, usize), (usize, usize, usize))> {
    if k_h == 0 || k_w == 0 {
        return None;
    }
    let k_prod = k_h * k_w;
    if w_len % k_prod != 0 {
        return None;
    }
    let c_out_c_in = w_len / k_prod;
    // Try each c_out that divides out_len (out_len = n * c_out * h_out * w_out, n=1).
    for c_out in 1..=out_len {
        if out_len % c_out != 0 {
            continue;
        }
        let hw_out = out_len / c_out;
        if c_out_c_in % c_out != 0 {
            continue;
        }
        let c_in = c_out_c_in / c_out;
        if in_len % c_in != 0 {
            continue;
        }
        let hw_in = in_len / c_in;
        let h_out = (hw_out as f64).sqrt() as usize;
        let w_out = if h_out > 0 { hw_out / h_out } else { 0 };
        if h_out * w_out != hw_out {
            continue;
        }
        let h = (h_out.saturating_sub(1) * stride_h + k_h).saturating_sub(2 * pad_h);
        let w = (w_out.saturating_sub(1) * stride_w + k_w).saturating_sub(2 * pad_w);
        if h * w != hw_in {
            continue;
        }
        return Some(((1, c_in, h, w), (c_out, h_out, w_out)));
    }
    None
}

/// Execute convolution operation: output = conv(input, weights)
/// 
/// Supports 2D NCHW via im2col + sgemm (fast, like MatMul). Falls back to 1D for legacy or when shapes cannot be inferred.
/// 
/// # Attributes
///   * `kernel_size` - Int or IntArray [k_h, k_w] (default 3)
///   * `stride` - Int or IntArray [s_h, s_w] (default 1)
///   * `padding` - Int or IntArray [p_h, p_w] (default 0)
///   * `input_shape` - Optional IntArray [N,C,H,W] (from fallback or inferred)
pub fn convolution(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    attributes: &HashMap<String, Attribute>,
) -> CoelanoxResult<()> {
    let (k_h, k_w) = int_array_2(attributes, "kernel_size", (3, 3));
    let (stride_h, stride_w) = int_array_2(attributes, "stride", (1, 1));
    let (pad_h, pad_w) = int_array_2(attributes, "padding", (0, 0));

    let in_len = input.len();
    let out_len = output.len();
    let w_len = weights.len();

    // 2D path: require kernel 2D and enough weights for at least one filter.
    if k_h > 0 && k_w > 0 && w_len >= k_h * k_w {
        let (n, c_in, h, w, c_out) = if let Some((nn, ci, hi, wi)) = input_shape_nchw(attributes) {
            let h_out = (hi + 2 * pad_h).saturating_sub(k_h) / stride_h + 1;
            let w_out = (wi + 2 * pad_w).saturating_sub(k_w) / stride_w + 1;
            let spatial_out = (h_out * w_out).max(1);
            let co = out_len / (nn * spatial_out);
            (nn, ci, hi, wi, co)
        } else if let Some(((nn, c_in, h, w), (c_out, _h_out, _w_out))) = infer_conv2d_shapes(
            in_len, out_len, w_len, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
        ) {
            (nn, c_in, h, w, c_out)
        } else {
            (1, 0, 0, 0, 0)
        };
        if n > 0 && c_in > 0 && h > 0 && w > 0 && c_out > 0
            && in_len >= n * c_in * h * w
            && out_len >= n * c_out * ((h + 2 * pad_h).saturating_sub(k_h) / stride_h + 1) * ((w + 2 * pad_w).saturating_sub(k_w) / stride_w + 1)
            && w_len >= c_out * c_in * k_h * k_w
        {
            return conv2d_nchw_im2col(
                input,
                output,
                weights,
                n,
                c_in,
                h,
                w,
                c_out,
                k_h,
                k_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
            );
        }
    }

    // Fallback: 1D conv (legacy behavior for non-2D or when inference fails).
    let kernel_size = k_h.max(k_w).min(weights.len());
    let len = in_len.saturating_sub(kernel_size).saturating_add(1).min(output.len());
    if kernel_size == 0 {
        output[..len].fill(0.0);
        return Ok(());
    }
    for (i, win) in input.windows(kernel_size).enumerate().take(len) {
        output[i] = win.iter().zip(weights[..kernel_size].iter()).map(|(a, b)| a * b).sum();
    }
    Ok(())
}

