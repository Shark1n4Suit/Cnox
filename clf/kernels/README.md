# CLF kernels (in-tree)

SIMD kernels for the Coelanox CLF compute backend. Built and packed here so `clf/clfc/simd.clfc` is produced from this directory.

## Layout

- **simd/** — x86_64 AVX2 kernels (Add, ReLU, BatchNorm, GlobalAvgPool, Conv, MatMul, Broadcast, Expand, Min, Max). Each blob uses the CLFE 6-arg ABI.

## Build and pack

From `clf/kernels/simd/`:

```bash
./pack_clfc.sh
```

Requires: `cargo`, `ar`, `objcopy`. To create `simd.clfc` you need the CLF packer: set `CLF_REPO` to the CLF repo root (e.g. sibling `CLF` of cnox), or have `coelanox-packer` on your PATH.

Output: `clf/clfc/simd.clfc`.

## Op IDs packed (6-arg only)

| op_id | Op            |
|-------|----------------|
| 1     | Add            |
| 10    | Relu           |
| 30    | Convolution    |
| 34    | GlobalAvgPool  |
| 35    | BatchNorm      |
| 50    | MatMul         |
| 70    | Broadcast      |
| 71    | Expand         |
| 93    | Min            |
| 94    | Max            |
