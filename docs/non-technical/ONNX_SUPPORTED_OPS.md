# COELANOX — ONNX support for customers (opset 13)

**Purpose:** Human-readable summary of what the **in-tree ONNX translator** can turn into **Universal IR** so **`coelanox package`** succeeds. It is **not** the internal CLF `op_id` registry (see [sdk/clf.md](../sdk/clf.md) for developers).

**Scope:** **ONNX opset 13** import path. Other opsets are not documented here as supported.

---

## 1. How to read this document

- **1:1 mapping** — One ONNX operator becomes one IR operator (direct support).  
- **Decomposed** — One ONNX operator becomes a **small subgraph** of IR primitives (still valid).  
- **Rejected** — Packaging **fails** with a clear message (you must change the model or app).  
- **Custom / unsupported** — Translator emits **`Custom("OpName")`**; **`package` fails** until resolved.

---

## 2. Directly supported ONNX ops (1:1 examples)

The following categories are representative; the **canonical list** is maintained in the repository:

**Arithmetic:** Add, Sub, Mul, Div  
**Activations:** Relu, Sigmoid, Tanh, Softmax, LogSoftmax, Gelu  
**Matrix:** MatMul, Gemm  
**Conv / pool / norm:** Conv, MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool, BatchNormalization, LayerNormalization, Dropout  
**Tensor:** Reshape, Transpose, Concat, Split, Slice, Gather, Scatter (variants), Expand, Broadcast  
**Reductions:** ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd  
**Math:** Sqrt, Pow, Exp, Log, Cos, Sin, Min, Max  
**Comparisons / logical:** Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual, And, Or, Not  
**Other:** Identity (rewired)

**Full machine-readable list:** [ONNX decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) §1.

---

## 3. Decomposed ONNX ops (still supported)

These ONNX ops are **not** missing—they are **lowered** to supported primitives:

| ONNX | Lowering idea |
|------|----------------|
| Clip | Max / Min chain |
| Neg | Multiply by −1 |
| Reciprocal | Divide 1 / x |
| Flatten, Unsqueeze, Squeeze | Reshape with attributes |
| Abs | Pow(2) + Sqrt (or equivalent chain) |
| Mean | Add chain + Divide |
| MeanVarianceNormalization | ReduceMean, Subtract, Pow, Sqrt, Divide |
| Constant | Float32 constant materialized as weight |
| LSTM | Forward-only subgraph (see limitations) |

Details: [ONNX decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) §3–5.

---

## 4. Explicitly rejected at translation (not “unsupported silently”)

| ONNX | Behaviour |
|------|-----------|
| **If**, **Loop** | **Error:** control flow must live in **your application**; export a model with a **fixed** execution path. |

---

## 5. Unsupported ONNX ops (what happens)

Any ONNX operator **not** covered by §2–4 is typically translated as **`Custom("OnnxOpName")`**. **`coelanox package` will fail** with an error indicating unresolved **`Custom`** nodes.

**Examples of ops that fall into “Custom” today** (non-exhaustive):

- Many **Reduce\*** variants not mapped (e.g. ReduceL1, ReduceL2, …)  
- **Resize**, **Pad**, **Cast**, **Shape**, **Erf**, **GridSample**, **RNN**, **GRU** (unless specifically handled elsewhere)  
- **QuantizeLinear** / **DequantizeLinear** as first-class paths  
- **Einsum**, **ArgMax**, **ArgMin**, and many others  

See **§4 Custom** in the [full decomposition tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md).

---

## 6. What to do if your op does not compile

1. **Open the error** — It should name **`Custom("OpName")`** or a translator error.  
2. **Check the full tree** — [ONNX_DECOMPOSITION_TREE.md](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) for a known decomposition.  
3. **Change the export** — Simplify graph (remove unsupported ops), or fuse ops in the framework before export.  
4. **Move control flow to the app** — If **If/Loop**, restructure.  
5. **Extend the translator** — For vendors, this means Rust changes in `coelanox-packager` (see [frontend SDK](../sdk/frontend.md)); not a runtime configuration toggle.

---

## 7. LSTM caveat

**LSTM** is decomposed with **constraints** (e.g. forward direction, documented assumptions). Read the **LSTM** section in the [full tree](../../coelanox/coelanox-packager/docs/ONNX_DECOMPOSITION_TREE.md) before relying on it for production.

---

← [Non-technical hub](README.md)
