//! SIMD kernels for neural-network ops (BERT + CNN). Fixed C ABI for use from C/Rust or a minimal runtime (e.g. CLFC).
//!
//! **Canonical op_id → op name (each CLFC blob is standalone, keyed by op_id):**
//!
//! | op_id | Op         | op_id | Op            | op_id | Op            |
//! |-------|------------|-------|---------------|-------|---------------|
//! | 0     | (unknown)  | 20    | Sqrt          | 40    | Reshape       |
//! | 1     | Add        | 21    | Pow           | 41    | Transpose     |
//! | 2     | Subtract   | 22    | Cos           | 42    | Permute       |
//! | 3     | Multiply   | 23    | Sin           | 43    | Concatenate   |
//! | 4     | Divide     | 24    | Exp           | 44    | Split         |
//! | 10    | Relu       | 25    | Log           | 45    | Slice         |
//! | 11    | Sigmoid    | 30    | Convolution   | 46    | Gather        |
//! | 12    | Tanh       | 31    | MaxPool       | 47    | Scatter       |
//! | 13    | Softmax    | 32    | AvgPool       | 50    | MatMul        |
//! | 14    | LogSoftmax | 33    | GlobalMaxPool | 51    | Gemm          |
//! | 15    | Gelu       | 34    | GlobalAvgPool | 60    | ReduceSum     |
//! | 16    | Swish      | 35    | BatchNorm     | 61    | ReduceMean    |
//! |       | 17–19 reserved | 36 | LayerNorm    | 62    | ReduceMax     |
//! |       |            | 37    | Dropout       | 63    | ReduceMin     |
//! |       | 26–29 reserved | 38–39 reserved | 64    | ReduceProd    |
//! |       |            |       |               | 70–71 | Broadcast, Expand |
//! |       |            |       |               | 80–85 | Equal,NotEqual,Greater,GreaterEqual,Less,LessEqual |
//! |       |            |       |               | 90    | And           |
//! |       |            |       |               | 91    | Or            |
//! |       |            |       |               | 92    | Not           |
//! |       |            |       |               | 93–94 | Min, Max (element-wise) |

//! For a true no_std / no-OS build: use nightly, `rustup component add rust-src`, and
//! `cargo build -Z build-std=core,panic_abort --release` with a panic handler.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use libm::sqrtf;
#[cfg(feature = "std")]
fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const LANE_F32: usize = 8; // AVX2 = 256 bits = 8 x f32

// --------------- Add ---------------
// ABI: (out, a, b, count). out[i] = a[i] + b[i]. count in elements. All ptrs non-null, count consistent.

#[inline(always)]
fn add_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        unsafe { *out.add(i) = *a.add(i) + *b.add(i) };
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn add_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(va, vb));
        i += LANE_F32;
    }
    add_scalar(out.add(i), a.add(i), b.add(i), count - i);
}

/// Add: out[i] = a[i] + b[i]. Same shape, f32.
/// ABI: out (mut), a (in), b (in), count (elements). No alloc; caller guarantees non-null and consistent lengths.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_add(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        add_avx2(out, a, b, count);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        add_scalar(out, a, b, count);
    }
}

// --------------- ReLU ---------------
// ABI: (out, x, count). out[i] = max(0, x[i]). count in elements.

#[inline(always)]
fn relu_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        let v = unsafe { *x.add(i) };
        unsafe { *out.add(i) = if v > 0.0 { v } else { 0.0 } };
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn relu_avx2(out: *mut f32, x: *const f32, count: usize) {
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let v = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_max_ps(v, zero));
        i += LANE_F32;
    }
    relu_scalar(out.add(i), x.add(i), count - i);
}

/// ReLU: out[i] = max(0, x[i]). f32.
/// ABI: out (mut), x (in), count (elements).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_relu(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        relu_avx2(out, x, count);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        relu_scalar(out, x, count);
    }
}

// --------------- BatchNorm ---------------
// out[i] = scale[c] * (x[i] - mean[c]) / sqrt(var[c] + eps) + bias[c]. NCHW; channels are the second dim (C).
// ABI: out, x, scale, bias, mean, var (each length C), n, c, h, w. x/out layout: n*C*H*W, index = n*C*H*W + c*H*W + h*W + w.

const BN_EPS: f32 = 1e-5;

#[inline(always)]
#[allow(dead_code)] // used when avx2 disabled or non-x86_64
fn batchnorm_scalar(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    mean: *const f32,
    var: *const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let hw = h * w;
    let chw = c * hw;
    let _nchw = n * c * hw;
    for ni in 0..n {
        for ci in 0..c {
            let sc = unsafe { *scale.add(ci) };
            let bi = unsafe { *bias.add(ci) };
            let me = unsafe { *mean.add(ci) };
            let va = unsafe { *var.add(ci) };
            let inv_std = 1.0 / sqrtf(va + BN_EPS);
            let base = ni * chw + ci * hw;
            for j in 0..hw {
                let idx = base + j;
                let v = unsafe { *x.add(idx) };
                unsafe { *out.add(idx) = sc * (v - me) * inv_std + bi };
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn batchnorm_avx2(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    mean: *const f32,
    var: *const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let hw = h * w;
    let chw = c * hw;
    for ni in 0..n {
        for ci in 0..c {
            let sc = _mm256_set1_ps(*scale.add(ci));
            let bi = _mm256_set1_ps(*bias.add(ci));
            let me = _mm256_set1_ps(*mean.add(ci));
            let va = *var.add(ci) + BN_EPS;
            let inv_std = _mm256_set1_ps(1.0 / sqrtf(va));
            let base = ni * chw + ci * hw;
            let mut j = 0usize;
            while j + LANE_F32 <= hw {
                let idx = base + j;
                let v = _mm256_loadu_ps(x.add(idx));
                let t = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(v, me), inv_std), sc);
                _mm256_storeu_ps(out.add(idx), _mm256_add_ps(t, bi));
                j += LANE_F32;
            }
            for jj in j..hw {
                let idx = base + jj;
                let v = *x.add(idx);
                *out.add(idx) = (*scale.add(ci)) * (v - *mean.add(ci)) / sqrtf(va) + *bias.add(ci);
            }
        }
    }
}

/// BatchNorm: out = scale * (x - mean) / sqrt(var + eps) + bias. One scale, bias, mean, var per channel (NCHW).
/// ABI: out, x, scale, bias, mean, var (C elements each), n, c, h, w.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_batchnorm(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    mean: *const f32,
    var: *const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    if n == 0 || c == 0 || h == 0 || w == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        batchnorm_avx2(out, x, scale, bias, mean, var, n, c, h, w);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        batchnorm_scalar(out, x, scale, bias, mean, var, n, c, h, w);
    }
}

// --------------- GlobalAvgPool ---------------
// out[n,c] = mean over H,W of inp[n,c,:,:]. Input NCHW (n*c*h*w), output (n*c).
// ABI: out, inp, n, c, h, w. All sizes in elements.

#[allow(dead_code)]
fn global_avg_pool_scalar(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    let hw = h * w;
    let chw = c * hw;
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * chw + ci * hw;
            let mut sum = 0.0f32;
            for j in 0..hw {
                sum += unsafe { *inp.add(base + j) };
            }
            unsafe { *out.add(ni * c + ci) = sum / hw as f32 };
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn global_avg_pool_avx2(
    out: *mut f32,
    inp: *const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    let hw = h * w;
    let chw = c * hw;
    let inv_hw = 1.0 / hw as f32;
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * chw + ci * hw;
            let mut acc = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + LANE_F32 <= hw {
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(inp.add(base + j)));
                j += LANE_F32;
            }
            // horizontal sum of acc
            let shuf = _mm256_shuffle_ps(acc, acc, 0b10_11_00_01);
            let sum1 = _mm256_add_ps(acc, shuf);
            let shuf2 = _mm256_permute2f128_ps(sum1, sum1, 1);
            let sum2 = _mm256_add_ps(sum1, shuf2);
            let shuf3 = _mm256_shuffle_ps(sum2, sum2, 0b01_00_11_10);
            let sum3 = _mm256_add_ps(sum2, shuf3);
            let s = _mm256_cvtss_f32(sum3);
            let mut scalar_sum = s;
            for jj in j..hw {
                scalar_sum += *inp.add(base + jj);
            }
            *out.add(ni * c + ci) = scalar_sum * inv_hw;
        }
    }
}

/// GlobalAvgPool: out[n,c] = mean over H,W of inp. Input [N,C,H,W], output [N,C]. f32.
/// ABI: out (n*c), inp (n*c*h*w), n, c, h, w (all elements).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_global_avg_pool(
    out: *mut f32,
    inp: *const f32,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) {
    if n == 0 || c == 0 || h == 0 || w == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        global_avg_pool_avx2(out, inp, n, c, h, w);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        global_avg_pool_scalar(out, inp, n, c, h, w);
    }
}

// --------------- Convolution ---------------
// 2D NCHW: inp [N, c_in, H, W], weights [c_out, c_in, k_h, k_w], out [N, c_out, H_out, W_out].
// H_out = (H + 2*pad_h - k_h) / stride_h + 1; same for W.
// ABI: out, inp, weights, bias (may be null), n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w.
// Weights layout: index (co, ci, kh, kw) = co*(c_in*k_h*k_w) + ci*(k_h*k_w) + kh*k_w + kw.

unsafe fn conv_scalar(
    out: *mut f32,
    inp: *const f32,
    weights: *const f32,
    bias: *const f32,
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
) {
    let h_out = (h + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w - k_w) / stride_w + 1;
    let in_hw = h * w;
    let in_chw = c_in * in_hw;
    let w_k = k_h * k_w;
    let w_ck = c_in * w_k;
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut sum = if bias.is_null() {
                        0.0
                    } else {
                        *bias.add(co)
                    };
                    for ci in 0..c_in {
                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let hi = (ho * stride_h).wrapping_add(kh).wrapping_sub(pad_h);
                                let wi = (wo * stride_w).wrapping_add(kw).wrapping_sub(pad_w);
                                if hi < h && wi < w {
                                    let in_idx = ni * in_chw + ci * in_hw + hi * w + wi;
                                    let w_idx = co * w_ck + ci * w_k + kh * k_w + kw;
                                    sum += *inp.add(in_idx) * *weights.add(w_idx);
                                }
                            }
                        }
                    }
                    let out_idx = ni * c_out * h_out * w_out + co * h_out * w_out + ho * w_out + wo;
                    *out.add(out_idx) = sum;
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn conv_avx2(
    out: *mut f32,
    inp: *const f32,
    weights: *const f32,
    bias: *const f32,
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
) {
    // Inner dot product has non-contiguous indices (per output pixel); use scalar. Still benefit from AVX2 build.
    conv_scalar(
        out, inp, weights, bias, n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
    );
}

/// Convolution 2D NCHW. Supports 1x1 and 3x3 (and any k_h, k_w). Optional bias (null = no bias).
/// ABI: out, inp, weights, bias (or null), n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_conv(
    out: *mut f32,
    inp: *const f32,
    weights: *const f32,
    bias: *const f32,
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
) {
    if n == 0 || c_out == 0 || c_in == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        conv_avx2(
            out, inp, weights, bias, n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
        );
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        conv_scalar(
            out, inp, weights, bias, n, c_in, h, w, c_out, k_h, k_w, stride_h, stride_w, pad_h, pad_w,
        );
    }
}

// --------------- MatMul ---------------
// out = A @ B. A [M,K], B [K,N], out [M,N]. Row-major: A[i,k], B[k,j], out[i,j].
// ABI: out, a, b, m, k, n (all sizes in elements).

#[allow(dead_code)]
unsafe fn matmul_scalar(out: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a.add(i * k + p) * *b.add(p * n + j);
            }
            *out.add(i * n + j) = sum;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn matmul_avx2(out: *mut f32, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) {
    for i in 0..m {
        let mut j = 0usize;
        while j + LANE_F32 <= n {
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let av = _mm256_set1_ps(*a.add(i * k + p));
                let b_idx = p * n + j;
                let bv = _mm256_loadu_ps(b.add(b_idx));
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            _mm256_storeu_ps(out.add(i * n + j), acc);
            j += LANE_F32;
        }
        for j in j..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a.add(i * k + p) * *b.add(p * n + j);
            }
            *out.add(i * n + j) = sum;
        }
    }
}

/// MatMul: out = A @ B. A [M,K], B [K,N], out [M,N]. f32, row-major.
/// ABI: out, a, b, m, k, n (elements).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_matmul(
    out: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        matmul_avx2(out, a, b, m, k, n);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        matmul_scalar(out, a, b, m, k, n);
    }
}

// --------------- LayerNorm (op_id 36) ---------------
// out = (x - mean) / sqrt(var + eps) * gamma + beta. Normalized over last C elements; gamma/beta length C.
// Weights: gamma (C), beta (C) => w_len = 2*C. in_len = out_len. C = w_len/2. Rows = in_len/C.
const LN_EPS: f32 = 1e-5;

#[allow(dead_code)]
#[inline(always)]
unsafe fn layernorm_scalar(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    count: usize,
    c: usize,
) {
    let n_rows = count / c;
    for r in 0..n_rows {
        let base = r * c;
        let mut mean = 0.0f32;
        for i in 0..c {
            mean += *x.add(base + i);
        }
        mean /= c as f32;
        let mut var = 0.0f32;
        for i in 0..c {
            let d = *x.add(base + i) - mean;
            var += d * d;
        }
        var = (var / c as f32) + LN_EPS;
        let inv_std = 1.0 / sqrtf(var);
        for i in 0..c {
            let v = (*x.add(base + i) - mean) * inv_std * *scale.add(i) + *bias.add(i);
            *out.add(base + i) = v;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn layernorm_avx2(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    count: usize,
    c: usize,
) {
    let n_rows = count / c;
    for r in 0..n_rows {
        let base = r * c;
        let (mean, var) = {
            let mut mean = 0.0f32;
            for i in 0..c {
                mean += *x.add(base + i);
            }
            mean /= c as f32;
            let mut var = 0.0f32;
            for i in 0..c {
                let d = *x.add(base + i) - mean;
                var += d * d;
            }
            (mean, (var / c as f32) + LN_EPS)
        };
        let inv_std = 1.0 / sqrtf(var);
        let mean_v = _mm256_set1_ps(mean);
        let inv_std_v = _mm256_set1_ps(inv_std);
        let mut i = 0usize;
        while i + LANE_F32 <= c {
            let v = _mm256_loadu_ps(x.add(base + i));
            let sc = _mm256_loadu_ps(scale.add(i));
            let bi = _mm256_loadu_ps(bias.add(i));
            let n = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(v, mean_v), inv_std_v), sc);
            _mm256_storeu_ps(out.add(base + i), _mm256_add_ps(n, bi));
            i += LANE_F32;
        }
        for ii in i..c {
            *out.add(base + ii) = (*x.add(base + ii) - mean) * inv_std * *scale.add(ii) + *bias.add(ii);
        }
    }
}

/// LayerNorm: normalize over last C elements; scale by gamma, add beta. w = [gamma (C), beta (C)], w_len=2*C.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_layernorm(
    out: *mut f32,
    x: *const f32,
    scale: *const f32,
    bias: *const f32,
    count: usize,
    c: usize,
) {
    if count == 0 || c == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    {
        layernorm_avx2(out, x, scale, bias, count, c);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    {
        layernorm_scalar(out, x, scale, bias, count, c);
    }
}


// --------------- Softmax (op_id 13) ---------------
// out = exp(x - max) / sum(exp(x - max)). Per row; row_len = cols. in_len = rows*cols.
#[inline(always)]
unsafe fn softmax_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut max_x = *x.add(base);
        for c in 1..cols {
            let v = *x.add(base + c);
            if v > max_x {
                max_x = v;
            }
        }
        let mut sum = 0.0f32;
        for c in 0..cols {
            let v = *x.add(base + c);
            let e = expf_approx(v - max_x);
            *out.add(base + c) = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for c in 0..cols {
            *out.add(base + c) *= inv_sum;
        }
    }
}

#[cfg(not(feature = "std"))]
fn expf_approx(x: f32) -> f32 {
    libm::expf(x)
}
#[cfg(feature = "std")]
fn expf_approx(x: f32) -> f32 {
    x.exp()
}

/// Softmax over last dimension. rows = in_len/cols, cols = row length (or in_len if 1D).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_softmax(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    softmax_scalar(out, x, rows, cols);
}


// --------------- Gelu (op_id 15) ---------------
// GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
#[inline(always)]
unsafe fn gelu_scalar(out: *mut f32, x: *const f32, count: usize) {
    const SQRTPI: f32 = 0.7978845608; // sqrt(2/pi)
    for i in 0..count {
        let v = *x.add(i);
        let inner = v + 0.044715 * v * v * v;
        #[cfg(not(feature = "std"))]
        let t = libm::tanhf(SQRTPI * inner);
        #[cfg(feature = "std")]
        let t = (SQRTPI * inner).tanh();
        *out.add(i) = 0.5 * v * (1.0 + t);
    }
}

/// Gelu: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_gelu(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 {
        return;
    }
    gelu_scalar(out, x, count);
}

// --------------- Reshape (op_id 40) ---------------
// Copy in -> out (same element count). No shape change in data layout.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reshape(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 {
        return;
    }
    if out != x as *mut f32 {
        core::ptr::copy_nonoverlapping(x, out, count);
    }
}

// --------------- Transpose (op_id 41) ---------------
// 2D transpose: out[col*rows+row] = in[row*cols+col]. in_len = out_len = rows*cols.
#[inline(always)]
unsafe fn transpose_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            *out.add(col * rows + row) = *x.add(row * cols + col);
        }
    }
}

/// Transpose 2D matrix. rows and cols from dimensions (rows*cols = in_len).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_transpose(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    transpose_scalar(out, x, rows, cols);
}


// --------------- Gemm (op_id 51) ---------------
// Same as MatMul for SIMD (alpha applied in scalar path or via separate scale). out = A @ B.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_gemm(
    out: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    clf_simd_matmul(out, a, b, m, k, n);
}

// =============================================================================
// Remaining canonical op_ids (2–4, 11–16, 20–25, 31–33, 37, 42–47, 60–64, 80–85, 90–92)
// =============================================================================

// --------------- Subtract (2) ---------------
#[inline(always)]
fn subtract_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        unsafe { *out.add(i) = *a.add(i) - *b.add(i) };
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn subtract_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_sub_ps(va, vb));
        i += LANE_F32;
    }
    subtract_scalar(out.add(i), a.add(i), b.add(i), count - i);
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_subtract(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { subtract_avx2(out, a, b, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { subtract_scalar(out, a, b, count); }
}

// --------------- Multiply (3) ---------------
#[inline(always)]
fn multiply_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count { unsafe { *out.add(i) = *a.add(i) * *b.add(i) }; }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn multiply_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i))));
        i += LANE_F32;
    }
    multiply_scalar(out.add(i), a.add(i), b.add(i), count - i);
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_multiply(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { multiply_avx2(out, a, b, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { multiply_scalar(out, a, b, count); }
}

// --------------- Divide (4) ---------------
#[inline(always)]
fn divide_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        let bv = unsafe { *b.add(i) };
        unsafe { *out.add(i) = if bv == 0.0 { 0.0 } else { *a.add(i) / bv } };
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn divide_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        _mm256_storeu_ps(out.add(i), _mm256_div_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i))));
        i += LANE_F32;
    }
    divide_scalar(out.add(i), a.add(i), b.add(i), count - i);
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_divide(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { divide_avx2(out, a, b, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { divide_scalar(out, a, b, count); }
}

// --------------- Min (93), Max (94): element-wise ---------------
#[inline(always)]
fn min_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        let va = unsafe { *a.add(i) };
        let vb = unsafe { *b.add(i) };
        unsafe { *out.add(i) = va.min(vb) };
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn min_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_min_ps(va, vb));
        i += LANE_F32;
    }
    min_scalar(out.add(i), a.add(i), b.add(i), count - i);
}
#[link_section = ".text.clf_simd_min"]
#[no_mangle]
pub unsafe extern "C" fn clf_simd_min(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { min_avx2(out, a, b, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { min_scalar(out, a, b, count); }
}

#[inline(always)]
fn max_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        let va = unsafe { *a.add(i) };
        let vb = unsafe { *b.add(i) };
        unsafe { *out.add(i) = va.max(vb) };
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn max_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_max_ps(va, vb));
        i += LANE_F32;
    }
    max_scalar(out.add(i), a.add(i), b.add(i), count - i);
}
#[link_section = ".text.clf_simd_max"]
#[no_mangle]
pub unsafe extern "C" fn clf_simd_max(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { max_avx2(out, a, b, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { max_scalar(out, a, b, count); }
}

// --------------- Sigmoid (11) ---------------
#[inline(always)]
fn sigmoid_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        let v = unsafe { *x.add(i) };
        #[cfg(not(feature = "std"))]
        let e = libm::expf(-v);
        #[cfg(feature = "std")]
        let e = (-v).exp();
        unsafe { *out.add(i) = 1.0 / (1.0 + e) };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_sigmoid(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    sigmoid_scalar(out, x, count);
}

// --------------- Tanh (12) ---------------
#[inline(always)]
fn tanh_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        #[cfg(not(feature = "std"))]
        let t = libm::tanhf(unsafe { *x.add(i) });
        #[cfg(feature = "std")]
        let t = (unsafe { *x.add(i) }).tanh();
        unsafe { *out.add(i) = t };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_tanh(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    tanh_scalar(out, x, count);
}

// --------------- LogSoftmax (14): log(softmax(x)) = x - max - log(sum(exp(x - max))) ---------------
#[inline(always)]
unsafe fn log_softmax_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut max_x = *x.add(base);
        for c in 1..cols { let v = *x.add(base + c); if v > max_x { max_x = v; } }
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = expf_approx(*x.add(base + c) - max_x);
            *out.add(base + c) = e;
            sum += e;
        }
        #[cfg(not(feature = "std"))]
        let log_sum = if sum > 0.0 { libm::logf(sum) } else { -1e10 };
        #[cfg(feature = "std")]
        let log_sum = if sum > 0.0 { sum.ln() } else { -1e10 };
        for c in 0..cols {
            *out.add(base + c) = *x.add(base + c) - max_x - log_sum;
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_log_softmax(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    log_softmax_scalar(out, x, rows, cols);
}

// --------------- Swish (16): x * sigmoid(x) ---------------
#[inline(always)]
fn swish_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        let v = unsafe { *x.add(i) };
        #[cfg(not(feature = "std"))]
        let s = 1.0 / (1.0 + libm::expf(-v));
        #[cfg(feature = "std")]
        let s = 1.0 / (1.0 + (-v).exp());
        unsafe { *out.add(i) = v * s };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_swish(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    swish_scalar(out, x, count);
}

// --------------- Sqrt (20), Pow (21), Cos (22), Sin (23), Exp (24), Log (25) ---------------
#[inline(always)]
fn sqrt_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        let v = unsafe { *x.add(i) };
        unsafe { *out.add(i) = sqrtf(v.max(0.0)); }
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn sqrt_avx2(out: *mut f32, x: *const f32, count: usize) {
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let v = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_sqrt_ps(_mm256_max_ps(v, zero)));
        i += LANE_F32;
    }
    sqrt_scalar(out.add(i), x.add(i), count - i);
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_sqrt(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { sqrt_avx2(out, x, count); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { sqrt_scalar(out, x, count); }
}
#[inline(always)]
fn pow_scalar(out: *mut f32, x: *const f32, count: usize, exp: f32) {
    for i in 0..count {
        #[cfg(not(feature = "std"))]
        let v = libm::powf(unsafe { *x.add(i) }, exp);
        #[cfg(feature = "std")]
        let v = (unsafe { *x.add(i) }).powf(exp);
        unsafe { *out.add(i) = v };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_pow(out: *mut f32, x: *const f32, count: usize, exp: f32) {
    if count == 0 { return; }
    pow_scalar(out, x, count, exp);
}
#[inline(always)]
fn cos_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        #[cfg(not(feature = "std"))]
        let v = libm::cosf(unsafe { *x.add(i) });
        #[cfg(feature = "std")]
        let v = (unsafe { *x.add(i) }).cos();
        unsafe { *out.add(i) = v };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_cos(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    cos_scalar(out, x, count);
}
#[inline(always)]
fn sin_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        #[cfg(not(feature = "std"))]
        let v = libm::sinf(unsafe { *x.add(i) });
        #[cfg(feature = "std")]
        let v = (unsafe { *x.add(i) }).sin();
        unsafe { *out.add(i) = v };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_sin(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    sin_scalar(out, x, count);
}
#[inline(always)]
fn exp_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count { unsafe { *out.add(i) = expf_approx(*x.add(i)); }; }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_exp(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    exp_scalar(out, x, count);
}
#[inline(always)]
fn log_scalar(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        let v = unsafe { *x.add(i) };
        #[cfg(not(feature = "std"))]
        let l = if v > 0.0 { libm::logf(v) } else { -1e10 };
        #[cfg(feature = "std")]
        let l = if v > 0.0 { v.ln() } else { -1e10 };
        unsafe { *out.add(i) = l };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_log(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    log_scalar(out, x, count);
}

// --------------- MaxPool (31), AvgPool (32), GlobalMaxPool (33) ---------------
// NCHW. kernel 2x2, stride 2, pad 0. in_len=n*c*h*w, out_len=n*c*(h/2)*(w/2). h,w from spatial.
fn max_pool_2x2_scalar(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    let h_out = h / 2;
    let w_out = w / 2;
    let in_hw = h * w;
    let out_hw = h_out * w_out;
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut m = unsafe { *inp.add(ni * c * in_hw + ci * in_hw + (2 * ho) * w + (2 * wo)) };
                    for dh in 0..2usize {
                        for dw in 0..2usize {
                            let v = unsafe { *inp.add(ni * c * in_hw + ci * in_hw + (2 * ho + dh) * w + (2 * wo + dw)) };
                            if v > m { m = v; }
                        }
                    }
                    unsafe { *out.add(ni * c * out_hw + ci * out_hw + ho * w_out + wo) = m };
                }
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_max_pool(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    if n == 0 || c == 0 || h < 2 || w < 2 { return; }
    max_pool_2x2_scalar(out, inp, n, c, h, w);
}
unsafe fn avg_pool_2x2_scalar(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    let h_out = h / 2;
    let w_out = w / 2;
    let in_hw = h * w;
    let out_hw = h_out * w_out;
    let scale = 1.0 / 4.0;
    for ni in 0..n {
        for ci in 0..c {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut sum = 0.0f32;
                    for dh in 0..2 { for dw in 0..2 {
                        sum += *inp.add(ni * c * in_hw + ci * in_hw + (2 * ho + dh) * w + (2 * wo + dw));
                    }}
                    *out.add(ni * c * out_hw + ci * out_hw + ho * w_out + wo) = sum * scale;
                }
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_avg_pool(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    if n == 0 || c == 0 || h < 2 || w < 2 { return; }
    avg_pool_2x2_scalar(out, inp, n, c, h, w);
}
#[inline(always)]
unsafe fn global_max_pool_scalar(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    let hw = h * w;
    let chw = c * hw;
    for ni in 0..n {
        for ci in 0..c {
            let base = ni * chw + ci * hw;
            let mut m = *inp.add(base);
            for j in 1..hw { let v = *inp.add(base + j); if v > m { m = v; } }
            *out.add(ni * c + ci) = m;
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_global_max_pool(out: *mut f32, inp: *const f32, n: usize, c: usize, h: usize, w: usize) {
    if n == 0 || c == 0 || h == 0 || w == 0 { return; }
    unsafe { global_max_pool_scalar(out, inp, n, c, h, w); }
}

// --------------- Dropout (37): inference = identity copy ---------------
#[no_mangle]
pub unsafe extern "C" fn clf_simd_dropout(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    if out != x as *mut f32 { core::ptr::copy_nonoverlapping(x, out, count); }
}

// --------------- Permute (42), Concatenate (43), Split (44), Slice (45), Gather (46), Scatter (47) ---------------
// Plan: single in/out. Permute: 2D transpose when rows=w_len; else copy. Concat/Split: copy. Slice: copy first out_len. Gather: w = indices (f32 as u32).
#[no_mangle]
pub unsafe extern "C" fn clf_simd_permute(out: *mut f32, x: *const f32, in_len: usize, out_len: usize, rows: usize) {
    if in_len == 0 || out_len != in_len { if out != x as *mut f32 && out_len > 0 { core::ptr::copy_nonoverlapping(x, out, out_len.min(in_len)); } return; }
    if rows > 0 && in_len % rows == 0 { let cols = in_len / rows; transpose_scalar(out, x, rows, cols); } else { core::ptr::copy_nonoverlapping(x, out, in_len); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_concatenate(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    if out != x as *mut f32 { core::ptr::copy_nonoverlapping(x, out, count); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_split(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    if out != x as *mut f32 { core::ptr::copy_nonoverlapping(x, out, count); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_slice(out: *mut f32, x: *const f32, count: usize) {
    if count == 0 { return; }
    if out != x as *mut f32 { core::ptr::copy_nonoverlapping(x, out, count); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_gather(out: *mut f32, data: *const f32, indices: *const f32, out_len: usize, data_len: usize) {
    for i in 0..out_len {
        let idx = unsafe { *indices.add(i) } as i32 as usize;
        if idx < data_len { unsafe { *out.add(i) = *data.add(idx) }; }
    }
}
/// Scatter: output is zeroed, then output[indices[i]] = data[i]. Matches scalar semantics.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_scatter(out: *mut f32, data: *const f32, indices: *const f32, in_len: usize, out_len: usize) {
    for j in 0..out_len {
        *out.add(j) = 0.0;
    }
    for i in 0..in_len {
        let idx = (*indices.add(i) as i32).clamp(0, out_len as i32 - 1) as usize;
        *out.add(idx) = *data.add(i);
    }
}

// --------------- ReduceSum (60), ReduceMean (61), ReduceMax (62), ReduceMin (63), ReduceProd (64) ---------------
// Reduce over last dim: in_len = rows*cols, out_len = rows. Plan gives in_len, out_len => cols = in_len/out_len.

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let shuf = _mm256_shuffle_ps(v, v, 0b10_11_00_01);
    let sum1 = _mm256_add_ps(v, shuf);
    let shuf2 = _mm256_permute2f128_ps(sum1, sum1, 1);
    let sum2 = _mm256_add_ps(sum1, shuf2);
    let shuf3 = _mm256_shuffle_ps(sum2, sum2, 0b01_00_11_10);
    let sum3 = _mm256_add_ps(sum2, shuf3);
    _mm256_cvtss_f32(sum3)
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn horizontal_max_avx2(v: __m256) -> f32 {
    let shuf = _mm256_shuffle_ps(v, v, 0b10_11_00_01);
    let max1 = _mm256_max_ps(v, shuf);
    let shuf2 = _mm256_permute2f128_ps(max1, max1, 1);
    let max2 = _mm256_max_ps(max1, shuf2);
    let shuf3 = _mm256_shuffle_ps(max2, max2, 0b01_00_11_10);
    let max3 = _mm256_max_ps(max2, shuf3);
    _mm256_cvtss_f32(max3)
}

#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn horizontal_min_avx2(v: __m256) -> f32 {
    let shuf = _mm256_shuffle_ps(v, v, 0b10_11_00_01);
    let min1 = _mm256_min_ps(v, shuf);
    let shuf2 = _mm256_permute2f128_ps(min1, min1, 1);
    let min2 = _mm256_min_ps(min1, shuf2);
    let shuf3 = _mm256_shuffle_ps(min2, min2, 0b01_00_11_10);
    let min3 = _mm256_min_ps(min2, shuf3);
    _mm256_cvtss_f32(min3)
}

#[allow(dead_code)]
unsafe fn reduce_sum_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut s = 0.0f32;
        for c in 0..cols { s += *x.add(base + c); }
        *out.add(r) = s;
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn reduce_sum_avx2(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut acc = _mm256_setzero_ps();
        let mut c = 0usize;
        while c + LANE_F32 <= cols {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.add(base + c)));
            c += LANE_F32;
        }
        let mut s = horizontal_sum_avx2(acc);
        for cc in c..cols {
            s += *x.add(base + cc);
        }
        *out.add(r) = s;
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reduce_sum(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { reduce_sum_avx2(out, x, rows, cols); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { reduce_sum_scalar(out, x, rows, cols); }
}
#[allow(dead_code)]
unsafe fn reduce_mean_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    let inv = 1.0 / cols as f32;
    for r in 0..rows {
        let base = r * cols;
        let mut s = 0.0f32;
        for c in 0..cols { s += *x.add(base + c); }
        *out.add(r) = s * inv;
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn reduce_mean_avx2(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    let inv = 1.0 / cols as f32;
    reduce_sum_avx2(out, x, rows, cols);
    let mut r = 0usize;
    while r + LANE_F32 <= rows {
        let v = _mm256_loadu_ps(out.add(r));
        _mm256_storeu_ps(out.add(r), _mm256_mul_ps(v, _mm256_set1_ps(inv)));
        r += LANE_F32;
    }
    for rr in r..rows {
        *out.add(rr) *= inv;
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reduce_mean(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { reduce_mean_avx2(out, x, rows, cols); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { unsafe { reduce_mean_scalar(out, x, rows, cols); } }
}
#[allow(dead_code)]
unsafe fn reduce_max_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut m = *x.add(base);
        for c in 1..cols { let v = *x.add(base + c); if v > m { m = v; } }
        *out.add(r) = m;
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn reduce_max_avx2(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut acc = _mm256_loadu_ps(x.add(base));
        let mut c = LANE_F32;
        while c + LANE_F32 <= cols {
            acc = _mm256_max_ps(acc, _mm256_loadu_ps(x.add(base + c)));
            c += LANE_F32;
        }
        let mut m = horizontal_max_avx2(acc);
        for cc in c..cols {
            let v = *x.add(base + cc);
            if v > m { m = v; }
        }
        *out.add(r) = m;
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reduce_max(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { reduce_max_avx2(out, x, rows, cols); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { unsafe { reduce_max_scalar(out, x, rows, cols); } }
}
#[allow(dead_code)]
unsafe fn reduce_min_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut m = *x.add(base);
        for c in 1..cols { let v = *x.add(base + c); if v < m { m = v; } }
        *out.add(r) = m;
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe fn reduce_min_avx2(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut acc = _mm256_loadu_ps(x.add(base));
        let mut c = LANE_F32;
        while c + LANE_F32 <= cols {
            acc = _mm256_min_ps(acc, _mm256_loadu_ps(x.add(base + c)));
            c += LANE_F32;
        }
        let mut m = horizontal_min_avx2(acc);
        for cc in c..cols {
            let v = *x.add(base + cc);
            if v < m { m = v; }
        }
        *out.add(r) = m;
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reduce_min(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { reduce_min_avx2(out, x, rows, cols); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { unsafe { reduce_min_scalar(out, x, rows, cols); } }
}
unsafe fn reduce_prod_scalar(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    for r in 0..rows {
        let base = r * cols;
        let mut p = 1.0f32;
        for c in 0..cols { p *= *x.add(base + c); }
        *out.add(r) = p;
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_reduce_prod(out: *mut f32, x: *const f32, rows: usize, cols: usize) {
    if rows == 0 || cols == 0 { return; }
    unsafe { reduce_prod_scalar(out, x, rows, cols); }
}

// --------------- Equal (80), NotEqual (81), Greater (82), GreaterEqual (83), Less (84), LessEqual (85) ---------------
// out[i] = (a[i] op b[i]) ? 1.0 : 0.0. Plan: in=a, w=b, count=in_len.
fn compare_scalar(out: *mut f32, a: *const f32, b: *const f32, count: usize, op: u8) {
    for i in 0..count {
        let va = unsafe { *a.add(i) };
        let vb = unsafe { *b.add(i) };
        let v = match op {
            80 => if va == vb { 1.0 } else { 0.0 },
            81 => if va != vb { 1.0 } else { 0.0 },
            82 => if va > vb { 1.0 } else { 0.0 },
            83 => if va >= vb { 1.0 } else { 0.0 },
            84 => if va < vb { 1.0 } else { 0.0 },
            85 => if va <= vb { 1.0 } else { 0.0 },
            _ => 0.0,
        };
        unsafe { *out.add(i) = v };
    }
}
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
#[inline(always)]
unsafe fn compare_avx2(out: *mut f32, a: *const f32, b: *const f32, count: usize, op: u8) {
    let ones = _mm256_set1_ps(1.0f32);
    let mut i = 0usize;
    while i + LANE_F32 <= count {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let mask = match op {
            80 => _mm256_cmp_ps(va, vb, _CMP_EQ_OQ),
            81 => _mm256_cmp_ps(va, vb, _CMP_NEQ_OQ),
            82 => _mm256_cmp_ps(va, vb, _CMP_GT_OQ),
            83 => _mm256_cmp_ps(va, vb, _CMP_GE_OQ),
            84 => _mm256_cmp_ps(va, vb, _CMP_LT_OQ),
            85 => _mm256_cmp_ps(va, vb, _CMP_LE_OQ),
            _ => _mm256_setzero_ps(),
        };
        _mm256_storeu_ps(out.add(i), _mm256_and_ps(mask, ones));
        i += LANE_F32;
    }
    compare_scalar(out.add(i), a.add(i), b.add(i), count - i, op);
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_equal(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 80); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 80); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_not_equal(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 81); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 81); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_greater(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 82); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 82); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_greater_equal(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 83); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 83); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_less(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 84); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 84); }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_less_equal(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    if count == 0 { return; }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))] { compare_avx2(out, a, b, count, 85); }
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))] { compare_scalar(out, a, b, count, 85); }
}

// --------------- And (90), Or (91), Not (92) ---------------
// Treat !=0 as true; output 1.0 or 0.0.
#[no_mangle]
pub unsafe extern "C" fn clf_simd_and(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        let va = unsafe { *a.add(i) };
        let vb = unsafe { *b.add(i) };
        unsafe { *out.add(i) = if va != 0.0 && vb != 0.0 { 1.0 } else { 0.0 } };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_or(out: *mut f32, a: *const f32, b: *const f32, count: usize) {
    for i in 0..count {
        let va = unsafe { *a.add(i) };
        let vb = unsafe { *b.add(i) };
        unsafe { *out.add(i) = if va != 0.0 || vb != 0.0 { 1.0 } else { 0.0 } };
    }
}
#[no_mangle]
pub unsafe extern "C" fn clf_simd_not(out: *mut f32, x: *const f32, count: usize) {
    for i in 0..count {
        unsafe { *out.add(i) = if *x.add(i) != 0.0 { 0.0 } else { 1.0 } };
    }
}

// --------------- Broadcast (70), Expand (71): repeat input to fill output ---------------
// ABI: out, in, in_len, out_len. out[i] = in[i % in_len]. in_len > 0; out_len can be >= in_len.
#[inline(always)]
unsafe fn broadcast_expand_scalar(out: *mut f32, in_ptr: *const f32, in_len: usize, out_len: usize) {
    if in_len == 0 {
        return;
    }
    for i in 0..out_len {
        *out.add(i) = *in_ptr.add(i % in_len);
    }
}
#[link_section = ".text.clf_simd_broadcast"]
#[no_mangle]
pub unsafe extern "C" fn clf_simd_broadcast(out: *mut f32, in_ptr: *const f32, in_len: usize, out_len: usize) {
    broadcast_expand_scalar(out, in_ptr, in_len, out_len);
}
#[link_section = ".text.clf_simd_expand"]
#[no_mangle]
pub unsafe extern "C" fn clf_simd_expand(out: *mut f32, in_ptr: *const f32, in_len: usize, out_len: usize) {
    broadcast_expand_scalar(out, in_ptr, in_len, out_len);
}

// =============================================================================
// CLFE uniform 6-arg ABI wrappers (input_ptr, output_ptr, weights_ptr, in_len, out_len, w_len).
// These are the entry points packed into .clfc; each blob is self-contained (kernels inlined).
// x86-64 SysV: rdi, rsi, rdx, rcx, r8, r9.
// =============================================================================

/// CLFE 6-arg entry for Add. Plan: in=input_ptr, w=weights_ptr (second addend), out=output_ptr, count=in_len.
/// Calls inner add_* so blob is self-contained (no external symbol).
#[link_section = ".text.clf_abi6_add"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_add(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    _out_len: usize,
    _w_len: usize,
) {
    if in_len == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    add_avx2(output_ptr, input_ptr, weights_ptr, in_len);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    add_scalar(output_ptr, input_ptr, weights_ptr, in_len);
}

/// CLFE 6-arg entry for ReLU. Plan: in=input_ptr, out=output_ptr, count=in_len. Weights unused.
#[link_section = ".text.clf_abi6_relu"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_relu(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    _weights_ptr: *const f32,
    in_len: usize,
    _out_len: usize,
    _w_len: usize,
) {
    if in_len == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    relu_avx2(output_ptr, input_ptr, in_len);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    relu_scalar(output_ptr, input_ptr, in_len);
}

/// CLFE 6-arg entry for BatchNorm. Weights: scale,bias,mean,var each c elements => w_len=4*c. If w_len==0, identity copy.
#[link_section = ".text.clf_abi6_batchnorm"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_batchnorm(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    _out_len: usize,
    w_len: usize,
) {
    if in_len == 0 {
        return;
    }
    if w_len == 0 {
        if input_ptr != output_ptr {
            core::ptr::copy_nonoverlapping(input_ptr, output_ptr, in_len);
        }
        return;
    }
    let c = w_len / 4;
    if c == 0 {
        return;
    }
    let n = 1usize;
    let hw = in_len / c;
    let h = (hw as f64).sqrt() as usize;
    let w = if h > 0 { hw / h } else { 0 };
    let scale = weights_ptr;
    let bias = weights_ptr.add(c);
    let mean = weights_ptr.add(2 * c);
    let var = weights_ptr.add(3 * c);
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    batchnorm_avx2(output_ptr, input_ptr, scale, bias, mean, var, n, c, h, w);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    batchnorm_scalar(output_ptr, input_ptr, scale, bias, mean, var, n, c, h, w);
}

/// CLFE 6-arg entry for GlobalAvgPool. in_len=n*c*h*w, out_len=n*c => n=1, c=out_len, h*w=in_len/c.
#[link_section = ".text.clf_abi6_global_avg_pool"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_global_avg_pool(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    _weights_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    _w_len: usize,
) {
    if out_len == 0 {
        return;
    }
    let n = 1usize;
    let c = out_len;
    let hw = in_len / c;
    let h = (hw as f64).sqrt() as usize;
    let w = if h > 0 { hw / h } else { 0 };
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    global_avg_pool_avx2(output_ptr, input_ptr, n, c, h, w);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    global_avg_pool_scalar(output_ptr, input_ptr, n, c, h, w);
}

/// CLFE 6-arg entry for Convolution. Derive c_in,c_out,h,w,k_h,k_w from in_len,out_len,w_len; same logic as executor.
#[link_section = ".text.clf_abi6_conv"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_conv(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    w_len: usize,
) {
    if in_len == 0 || out_len == 0 || w_len == 0 {
        return;
    }
    let n = 1usize;
    let (c_in, c_out, h, w, k_h, k_w) = match derive_conv_shapes(in_len, out_len, w_len) {
        Some(t) => t,
        None => return,
    };
    let pad_h = (k_h - 1) / 2;
    let pad_w = (k_w - 1) / 2;
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    conv_avx2(
        output_ptr, input_ptr, weights_ptr, core::ptr::null(),
        n, c_in, h, w, c_out, k_h, k_w, 1, 1, pad_h, pad_w,
    );
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    conv_scalar(
        output_ptr, input_ptr, weights_ptr, core::ptr::null(),
        n, c_in, h, w, c_out, k_h, k_w, 1, 1, pad_h, pad_w,
    );
}

/// Derive (c_in, c_out, h, w, k_h, k_w) from in_len, out_len, w_len. Returns None if invalid.
fn derive_conv_shapes(
    in_len: usize,
    out_len: usize,
    w_len: usize,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    for c_in_candidate in 1..=in_len {
        if in_len % c_in_candidate != 0 {
            continue;
        }
        let spatial = in_len / c_in_candidate;
        if out_len % spatial != 0 {
            continue;
        }
        let c_out_candidate = out_len / spatial;
        let weight_elements = c_in_candidate * c_out_candidate;
        if weight_elements == 0 || w_len % weight_elements != 0 {
            continue;
        }
        let k_prod = w_len / weight_elements;
        let k_h_cand = (k_prod as f64).sqrt() as usize;
        let k_w_cand = if k_h_cand > 0 { k_prod / k_h_cand } else { 0 };
        if k_h_cand == 0 || k_w_cand == 0 || k_h_cand * k_w_cand != k_prod {
            continue;
        }
        let h_cand = (spatial as f64).sqrt() as usize;
        let w_cand = if h_cand > 0 { spatial / h_cand } else { 0 };
        if h_cand * w_cand != spatial {
            continue;
        }
        return Some((
            c_in_candidate,
            c_out_candidate,
            h_cand,
            w_cand,
            k_h_cand,
            k_w_cand,
        ));
    }
    None
}

/// CLFE 6-arg entry for MatMul. in_len=m*k, w_len=k*n, out_len=m*n => k=sqrt(in_len*w_len/out_len), m=in_len/k, n=out_len/m.
#[link_section = ".text.clf_abi6_matmul"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_matmul(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    w_len: usize,
) {
    if out_len == 0 {
        return;
    }
    let k_sq = in_len * w_len / out_len;
    let k = (k_sq as f64).sqrt() as usize;
    if k == 0 {
        return;
    }
    let m = in_len / k;
    let n = out_len / m;
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx2"))]
    matmul_avx2(output_ptr, input_ptr, weights_ptr, m, k, n);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
    matmul_scalar(output_ptr, input_ptr, weights_ptr, m, k, n);
}

/// CLFE 6-arg entry for Broadcast (op_id 70). Plan: in=input_ptr, out=output_ptr, in_len, out_len; weights unused.
#[link_section = ".text.clf_abi6_broadcast"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_broadcast(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    _weights_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    _w_len: usize,
) {
    if out_len == 0 {
        return;
    }
    clf_simd_broadcast(output_ptr, input_ptr, in_len, out_len);
}

/// CLFE 6-arg entry for Expand (op_id 71). Plan: in=input_ptr, out=output_ptr, in_len, out_len; weights unused.
#[link_section = ".text.clf_abi6_expand"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_expand(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    _weights_ptr: *const f32,
    in_len: usize,
    out_len: usize,
    _w_len: usize,
) {
    if out_len == 0 {
        return;
    }
    clf_simd_expand(output_ptr, input_ptr, in_len, out_len);
}

/// CLFE 6-arg entry for Min (op_id 93). Plan: first operand = input_ptr, second = weights_ptr, count = in_len.
#[link_section = ".text.clf_abi6_min"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_min(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    _out_len: usize,
    _w_len: usize,
) {
    if in_len == 0 {
        return;
    }
    clf_simd_min(output_ptr, input_ptr, weights_ptr, in_len);
}

/// CLFE 6-arg entry for Max (op_id 94). Plan: first operand = input_ptr, second = weights_ptr, count = in_len.
#[link_section = ".text.clf_abi6_max"]
#[no_mangle]
pub unsafe extern "C" fn clf_abi6_max(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    weights_ptr: *const f32,
    in_len: usize,
    _out_len: usize,
    _w_len: usize,
) {
    if in_len == 0 {
        return;
    }
    clf_simd_max(output_ptr, input_ptr, weights_ptr, in_len);
}
