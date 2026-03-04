# ONNX → UniversalIR decomposition tree (opset 13)

How each ONNX op is translated: **1:1** (single IR node) or **decomposed** (subgraph of IR primitives).

---

## 1. Direct 1:1 mapping (single IR node)

```
ONNX op                    →  IR OpType
────────────────────────────────────────────────────────────
Arithmetic
  Add                      →  Add
  Sub                      →  Subtract
  Mul                      →  Multiply
  Div                      →  Divide

Activations
  Relu                     →  Relu
  Sigmoid                  →  Sigmoid
  Tanh                     →  Tanh
  Softmax                  →  Softmax
  LogSoftmax               →  LogSoftmax
  Gelu                     →  Gelu

Matrix
  MatMul                   →  MatMul
  Gemm                     →  Gemm

Conv / pool / norm
  Conv                     →  Convolution
  MaxPool                  →  MaxPool
  AveragePool              →  AvgPool
  GlobalAveragePool        →  GlobalAvgPool
  GlobalMaxPool            →  GlobalMaxPool
  BatchNormalization       →  BatchNorm
  LayerNormalization       →  LayerNorm
  Dropout                  →  Dropout

Tensor manipulation
  Reshape                  →  Reshape
  Transpose                →  Transpose
  Concat                   →  Concatenate
  Split                    →  Split
  Slice                    →  Slice
  Gather                   →  Gather
  GatherElements           →  Gather
  ScatterElements          →  Scatter
  ScatterND                →  Scatter
  Expand                   →  Expand
  Broadcast                →  Broadcast

Reductions
  ReduceSum                →  ReduceSum
  ReduceMean               →  ReduceMean
  ReduceMax                →  ReduceMax
  ReduceMin                →  ReduceMin
  ReduceProd               →  ReduceProd

Math (element-wise)
  Sqrt                     →  Sqrt
  Pow                      →  Pow
  Exp                      →  Exp
  Log                      →  Log
  Cos                      →  Cos
  Sin                      →  Sin
  Min                      →  Min
  Max                      →  Max

Comparisons
  Equal                    →  Equal
  NotEqual                 →  NotEqual
  Greater                  →  Greater
  GreaterEqual             →  GreaterEqual
  Less                     →  Less
  LessEqual                →  LessEqual

Logical
  And                      →  And
  Or                       →  Or
  Not                      →  Not

Other
  Identity                 →  Identity
```

---

## 2. Control flow → error (app layer)

These ops are **rejected** with a clear error so control flow stays in the app:

- **If** → `Err("ONNX control flow (If/Loop) is not supported in the model. Move control flow to the application layer and export a model with a fixed execution path.")`
- **Loop** → same error

---

## 3. Decomposed (subgraph of IR nodes)

### Clip

```
Clip(x [, min] [, max])
        │
        ▼
   ┌─────────┐
   │  const  │  min (attr or f32::MIN)
   └────┬────┘
        │
   ┌────▼────┐     x
   │   Max   │◄─────
   └────┬────┘
        │ mid
   ┌────▼────┐
   │  const  │  max (attr or f32::MAX)
   └────┬────┘
        │
   ┌────▼────┐
   │   Min   │
   └────┬────┘
        ▼
      output
```

**IR:** `Max(x, min_const)` → intermediate → `Min(mid, max_const)` → output.

---

### Neg

```
Neg(x)
   │
   ▼
   x ──────┬──────► Multiply ──► output
           │
           ▼
      const (-1.0)
```

**IR:** `Multiply(x, const_-1)` → output.

---

### Reciprocal

```
Reciprocal(x)
      │
      ▼
 const (1.0) ──┬──► Divide ──► output
               │
               x
```

**IR:** `Divide(1_const, x)` → output.

---

### Flatten

```
Flatten(x, axis?)
   │
   ▼
   x ──────► Reshape(axis, shape=[0,-1]) ──► output
```

**IR:** Single `Reshape` with attributes `axis` (default 1) and `shape: [0, -1]` for backend to interpret.

---

### Abs

```
Abs(x)
   │
   ▼
   x ──┬──► Pow(2) ──► Sqrt ──► output
       │
       └── const 2
```

**IR:** `Pow(x, 2)` → intermediate → `Sqrt` → output.

---

### Mean

```
Mean(in0, in1, ..., inN)
   │
   ▼
   in0 + in1 + ... + inN  (chain Add)
   │
   ▼
   Divide(..., n) ──► output   (const n = N)
```

**IR:** N−1 `Add` nodes (running sum), then one `Divide(sum, n_const)`.

---

### MeanVarianceNormalization

```
MVN(x, axes?, epsilon?)
   │
   ▼
   mean = ReduceMean(x, axes)
   xc   = Subtract(x, mean)
   sq   = Pow(xc, 2)
   var  = ReduceMean(sq, axes)
   std  = Sqrt(Add(var, eps))
   output = Divide(xc, std)
```

**IR:** ReduceMean → Subtract → Pow(2) → ReduceMean → Add(eps) → Sqrt → Divide.

---

### Unsqueeze

```
Unsqueeze(x, axis? | axes?)
   │
   ▼
   x ──────► Reshape(unsqueeze_axes) ──► output
```

**IR:** Single `Reshape` with attribute `unsqueeze_axes` for backend to interpret.

---

### Squeeze

```
Squeeze(x, axes?)
   │
   ▼
   x ──────► Reshape(squeeze_axes) ──► output
```

**IR:** Single `Reshape` with attribute `squeeze_axes` for backend to interpret.

---

### Constant

```
Constant(value: TensorProto)
   │
   ▼
   weight = materialize(value)
   weight ──────► Identity ──► output
```

**IR:** One weight tensor from the `value` attribute, then one `Identity(weight)` → output. Only float32 value supported.

---

## 4. Custom (no IR primitive, kept as Custom(name))

Any ONNX op that is not in **§1 (1:1)**, **§2 (control-flow error)**, or **§3 (decomposed)** is translated as `OpType::Custom("OpName")`.

Examples:

- **ReduceL1**, **ReduceL2**, **ReduceLogSum**, **ReduceLogSumExp**, **ReduceSumSquare**
- **Ceil**, **Floor**, **Sign**, **Erf**, **Hardmax**, **LRN**, **Resize**, **Pad**, **Cast**, **Shape**, **Size**, **NonZero**
- **DequantizeLinear**, **QuantizeLinear**, **NegativeLogLikelihoodLoss**, **SoftmaxCrossEntropyLoss**
- **DepthToSpace**, **SpaceToDepth**, **ArgMax**, **ArgMin**, **Einsum**, **Mod**, **Compress**, **OneHot**, **ConstantOfShape**, **IsNaN**, **GRU**, **LSTM**, **RNN**, etc.

---

## Summary

| Category        | Count (approx) | Notes                                                                 |
|----------------|----------------|-----------------------------------------------------------------------|
| 1:1 mapping    | 50+            | One ONNX node → one IR node                                           |
| Control flow   | 2              | If, Loop → **error**: move to app layer                               |
| Decomposed     | 10             | Clip, Neg, Reciprocal, Flatten, Abs, Mean, MeanVarianceNormalization, Unsqueeze, Squeeze, Constant |
| Custom         | rest           | Unsupported or composite → Custom(name)                               |

**Gemm** is intentionally kept as a single op (no decomposition to MatMul + Add).
