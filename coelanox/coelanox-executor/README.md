# coelanox-executor

CLF **mini-executor**: minimal runtime that runs multi-blob containers via an execution plan (graph walk + kernel dispatch).

## Role

When a container has multiple CLF kernel blobs (one per op) instead of a single monolithic blob, something must walk the graph and call each kernel with the right pointers. The executor does that:

- **At package time:** The packager builds an **execution plan** (binary: workspace size, node steps with op_id and buffer offsets, kernel byte offsets) and stores it in the container.
- **At run time:** The runtime allocates workspace, copies the code section into executable memory, and calls `run_with_plan(code_section, plan, input, output, weights, workspace)`. The executor parses the plan and, for each node, calls the corresponding kernel blob with the correct input/output/weight pointers.

## Plan format (little-endian)

- `workspace_elements`: u32  
- `num_nodes`: u32  
- For each node (7 × u32): `op_id`, `in_off`, `out_off`, `w_off`, `in_len`, `out_len`, `w_len`  
- `num_kernels`: u32  
- `kernel_offset[0..num_kernels]`: u32 (byte offset of each blob from code base)

## Kernel ABI (x86-64 SysV)

Each kernel is called as:

```c
void kernel(const float* input, float* output, const float* weights,
            size_t input_len, size_t output_len, size_t weight_len);
```

Registers: RDI, RSI, RDX, RCX, R8, R9 (and stack if needed).

## Platform

- **x86_64:** Full support; `run_with_plan` dispatches to kernel blobs.
- **Other arches:** `run_with_plan` returns an error so the runtime falls back to scalar.

## Usage

- **Library:** Used by `coelanox-orchestrator` (runtime) and `coelanox-packager` (build plan).
- **No standalone binary:** The executor is not yet built as a blob to be injected at the start of the code section; that would allow a true “load container + call one entry” shim. Current flow: runtime loads container, allocates workspace, calls `run_with_plan` in-process.
