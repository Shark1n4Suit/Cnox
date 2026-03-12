# SIMD kernels — C ABI for callable ops

All functions use the **C calling convention** (`extern "C"`). Sizes are in **elements** (f32 count), not bytes. Caller guarantees non-null pointers and consistent lengths; no internal allocations.

---

## 1. Add

```c
void clf_simd_add(float *out, const float *a, const float *b, size_t count);
```

- **out**: output buffer (write-only), `count` elements.
- **a**, **b**: input buffers (read-only), each `count` elements.
- **count**: number of f32 elements (same for out, a, b).
- **Semantics**: `out[i] = a[i] + b[i]` for `i = 0..count-1`.

---

## 2. ReLU

```c
void clf_simd_relu(float *out, const float *x, size_t count);
```

- **out**: output buffer, `count` elements.
- **x**: input buffer, `count` elements.
- **count**: number of elements.
- **Semantics**: `out[i] = max(0, x[i])`.

---

## 3. BatchNorm

```c
void clf_simd_batchnorm(
    float *out,
    const float *x,
    const float *scale,
    const float *bias,
    const float *mean,
    const float *var,
    size_t n, size_t c, size_t h, size_t w
);
```

- **out**: output buffer, shape NCHW, `n * c * h * w` elements.
- **x**: input buffer, same shape.
- **scale**, **bias**, **mean**, **var**: per-channel arrays, each **c** elements (channel index = second dimension in NCHW).
- **n, c, h, w**: batch, channels, height, width (elements).
- **Semantics**: `out[i] = scale[c] * (x[i] - mean[c]) / sqrt(var[c] + 1e-5) + bias[c]`, where channel `c` is derived from the linear index (NCHW layout).

---

## 4. GlobalAvgPool

```c
void clf_simd_global_avg_pool(
    float *out,
    const float *inp,
    size_t n, size_t c, size_t h, size_t w
);
```

- **out**: output buffer, **n * c** elements (one value per (n, c)).
- **inp**: input buffer, NCHW, **n * c * h * w** elements.
- **n, c, h, w**: batch, channels, height, width.
- **Semantics**: `out[n*c + c] = mean over (h,w) of inp[n,c,:,:]`.

---

## 5. Convolution

```c
void clf_simd_conv(
    float *out,
    const float *inp,
    const float *weights,
    const float *bias,
    size_t n, size_t c_in, size_t h, size_t w,
    size_t c_out, size_t k_h, size_t k_w,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w
);
```

- **out**: output buffer. Layout NCHW; size `n * c_out * h_out * w_out` where  
  `h_out = (h + 2*pad_h - k_h) / stride_h + 1`,  
  `w_out = (w + 2*pad_w - k_w) / stride_w + 1`.
- **inp**: input NCHW, `n * c_in * h * w` elements.
- **weights**: shape [c_out, c_in, k_h, k_w], row-major:  
  index `(co, ci, kh, kw)` = `co*(c_in*k_h*k_w) + ci*(k_h*k_w) + kh*k_w + kw`.
- **bias**: optional; **c_out** elements. Pass **NULL** if no bias.
- **n, c_in, h, w**: batch, input channels, height, width.
- **c_out, k_h, k_w**: output channels, kernel height/width (e.g. 1x1 or 3x3).
- **stride_h, stride_w, pad_h, pad_w**: stride and padding (elements).

---

## 6. MatMul

```c
void clf_simd_matmul(
    float *out,
    const float *a,
    const float *b,
    size_t m, size_t k, size_t n
);
```

- **out**: output buffer, **m * n** elements, row-major [M, N].
- **a**: matrix [M, K], **m * k** elements; row-major.
- **b**: matrix [K, N], **k * n** elements; row-major.
- **m, k, n**: matrix dimensions (elements).
- **Semantics**: `out = A @ B`; `out[i,j] = sum_p a[i,p] * b[p,j]`.

---

## CLFE uniform 6-arg ABI (packed blobs)

The **.clfc** blobs use a **single uniform signature** so the runtime can call every kernel the same way (CLFE dispatch contract):

```c
void clf_abi6_*(const float *input_ptr, float *output_ptr, const float *weights_ptr,
                size_t in_len, size_t out_len, size_t w_len);
```

- **input_ptr**: input buffer (read-only), `in_len` elements.
- **output_ptr**: output buffer (write-only), `out_len` elements.
- **weights_ptr**: weights buffer (read-only), `w_len` elements (unused for ReLU).
- **in_len, out_len, w_len**: element counts from the execution plan.

x86-64 SysV: arguments in **rdi, rsi, rdx, rcx, r8, r9**. Each blob is self-contained and derives op-specific parameters from these six values. The per-op ABIs above are used by Rust callers (`clf_simd_*`); the packed blobs are the 6-arg entry points only.

---

## Package via CLFC

To build the SIMD kernels and pack them into a single **.clfc** (CLF Compute) file for the Coelanox packager:

```bash
cd /path/to/CLF/kernels/simd
./pack_clfc.sh
```

This will:

1. Build the crate with AVX2 (`libclf_kernels_simd.a`).
2. Extract each **CLFE 6-arg wrapper** (`.text.clf_abi6_*`) into `blobs/*.bin` (one blob per op_id; each uses the uniform 6-arg ABI).
3. Run `coelanox-packer` to produce **`resnet_tiny_mnist_simd.clfc`** with op_ids: 1 (Add), 10 (Relu), 30 (Convolution), 34 (GlobalAvgPool), 35 (BatchNorm), 50 (MatMul).

Point the Coelanox backend at this `.clfc` when building a ResNet-tiny container for the **x86_64-AVX2** target. The runtime loads each blob and calls it with **(input_ptr, output_ptr, weights_ptr, in_len, out_len, w_len)** (CLFE uniform 6-arg ABI).

---

## Building for a minimal runtime (no OS)

- **With std** (default):  
  `RUSTFLAGS="-C target-feature=+avx2" cargo build --release`  
  Produces `target/release/libclf_kernels_simd.a` (or `.so` for cdylib). Link from C/Rust.

- **no_std** (e.g. sealed CLFC): Use nightly, add `rust-src`, build with  
  `-Z build-std=core,panic_abort`, disable feature `std`, and provide a panic handler (see crate docs).

## SIMD

- **AVX2** (default): 8 f32 per step (256-bit). Build with `RUSTFLAGS="-C target-feature=+avx2"`.
- **AVX-512**: reserved (feature `avx512`); 16 f32 per step when implemented.
