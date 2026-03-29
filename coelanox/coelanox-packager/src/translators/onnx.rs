//! ONNX translator: converts ONNX models (**target ONNX opset 13** semantics) to Coelanox UniversalIR.
//!
//! - **Implemented:** a large subset of ops via direct [`OpType`] mapping, [`decompose_node`], or graph rewiring.
//! - **Not implemented:** operators that have no lowering yet become [`OpType::Custom`]; packaging then **fails**
//!   with an explicit message (we do not silently run `Custom` in the scalar backend).
//! - **Application-layer / export:** dynamic control flow (`If` / `Loop` / `Scan`), `Cast`, sequence ops, etc. are
//!   **rejected at package time** with actionable text (normalize types and graphs before `coelanox package`).
//! - **Recent additions:** **`Where`** (mask select), **`Square`**, **`Clip`** (opset 11+ optional min/max tensors), variadic **`Sum`**, **`Celu`**, **`LpNormalization`** (p=2), `Sign`, `ReduceL1`/`L2`/`SumSquare`/`LogSum`/`LogSumExp`, `Selu`, `Mish`;
//!   `forward_onnx_reduce_attrs` copies `axes` / `keepdims` / `noop_with_empty_axes` onto lowered `ReduceSum` nodes.

#[cfg(feature = "onnx-protobuf")]
mod impl_ {
    use coelanox_core::ir::{
        Attribute, DataType, NodeId, OpType, Shape, TensorId, UniversalIR,
    };
    use coelanox_core::plugins::CoelanoxTranslator;
    use coelanox_core::format::ModelFormat;
    use coelanox_core::{CoelanoxError, CoelanoxResult};
    use rten_onnx::onnx::{ModelProto, DataType as OnnxDataType};
    use std::collections::HashMap;
    use std::path::Path;

    /// ONNX translator plugin (opset 13 semantics).
    pub struct OnnxTranslator;

    impl OnnxTranslator {
        pub fn new() -> Self {
            Self
        }

        fn parse_model(data: &[u8]) -> CoelanoxResult<ModelProto> {
            ModelProto::parse_buf(data)
                .map_err(|e| CoelanoxError::format(&format!("ONNX parse error: {}", e)))
        }

        /// Ops that must not appear in the graph (or must be fixed in export). Returns a full user-facing message.
        fn onnx_op_rejected_at_package_time(op_type: &str) -> Option<&'static str> {
            match op_type {
                "Cast" => Some(
                    "ONNX Cast is not supported: the Coelanox runtime executes a single numeric type (float32). \
                     Remove Cast nodes by exporting the model in float32, or fold casts in your training/export pipeline \
                     (this is an application / export concern, not something the container can fix at runtime).",
                ),
                "If" | "Loop" | "Scan" => Some(
                    "ONNX control flow (If / Loop / Scan) is not supported inside the packaged model. \
                     Unroll loops, specialize branches, or move control flow to your application; then export a single static graph.",
                ),
                "SequenceAt" | "SequenceConstruct" | "SequenceEmpty" | "SequenceErase"
                | "SequenceInsert" | "SequenceLength" => Some(
                    "ONNX Sequence_* operators are not supported. Use tensors only (no sequence-of-tensor graphs); \
                     reshape or batch your data in the application and re-export.",
                ),
                "Optional" | "OptionalGetElement" | "OptionalHasElement" => Some(
                    "ONNX Optional* operators are not supported. Export a model without optional types.",
                ),
                "MatMulInteger" | "QLinearMatMul" | "ConvInteger" | "QLinearConv" => Some(
                    "Integer / quantized MatMul and Conv are not supported. Export float32 MatMul/Gemm/Conv or dequantize outside the graph.",
                ),
                "QuantizeLinear" | "DequantizeLinear" => Some(
                    "ONNX QuantizeLinear / DequantizeLinear are not supported. Export a float32 graph or run dequantization outside the packaged model.",
                ),
                _ => None,
            }
        }

        /// Explains why a Custom ONNX op name cannot be packaged yet (fallback after translation).
        fn unimplemented_custom_onnx_op_message(name: &str) -> String {
            match name {
                "Cast" => Self::onnx_op_rejected_at_package_time("Cast")
                    .expect("Cast rejection message")
                    .to_string(),
                "Einsum" => "ONNX Einsum is not implemented. Expand to MatMul/Transpose in export, or use a simpler expression.".to_string(),
                "Resize" | "Upsample" => "ONNX Resize/Upsample is not implemented for this runtime. Use fixed spatial sizes or pre-process images outside the model.".to_string(),
                "NonMaxSuppression" | "RoiAlign" | "NonZero" => format!(
                    "ONNX operator '{}' is not implemented in the Coelanox ONNX translator (opset 13 subset).",
                    name
                ),
                "Pad" => "ONNX Pad is not implemented. Use explicit padding in the graph export or conv with padding, if applicable.".to_string(),
                "InstanceNormalization" | "GroupNormalization" => format!(
                    "ONNX {} is not implemented in the translator. Use BatchNorm/LayerNorm in export or extend the translator.",
                    name
                ),
                "Tile" => "ONNX Tile is not implemented. Use Expand or repeat via Concat in export, or extend the translator.".to_string(),
                "TopK" | "ArgMax" | "ArgMin" => format!(
                    "ONNX operator '{}' is not implemented in the translator. Avoid top-k / arg reductions in the packaged path, or extend the translator.",
                    name
                ),
                "GRU" | "RNN" => format!(
                    "ONNX {} is not implemented (LSTM has a partial decomposition). Export without recurrent cells or extend the translator.",
                    name
                ),
                other => format!(
                    "ONNX operator '{}' is not implemented in the Coelanox ONNX translator (opset 13 target; growing subset). \
                     Simplify or replace it in your export, or add a mapping/decomposition in coelanox-packager/src/translators/onnx.rs.",
                    other
                ),
            }
        }

        /// Maps ONNX op names (opset 13) to IR OpType. Ops that require decomposition
        /// are not mapped here; they are handled in build_ir.
        fn map_op_type(op_type: &str) -> OpType {
            match op_type {
                // Arithmetic
                "Add" => OpType::Add,
                "Sub" => OpType::Subtract,
                "Mul" => OpType::Multiply,
                "Div" => OpType::Divide,
                // Activations
                "Relu" => OpType::Relu,
                "Sigmoid" => OpType::Sigmoid,
                "Tanh" => OpType::Tanh,
                "Softmax" => OpType::Softmax,
                "LogSoftmax" => OpType::LogSoftmax,
                "Gelu" => OpType::Gelu,
                // Swish = x * sigmoid(x); native IR op (common in PyTorch exports)
                "Swish" => OpType::Swish,
                // Matrix
                "MatMul" => OpType::MatMul,
                "Gemm" => OpType::Gemm,
                // Conv / pool / norm
                "Conv" => OpType::Convolution,
                "MaxPool" => OpType::MaxPool,
                "AveragePool" => OpType::AvgPool,
                "GlobalAveragePool" => OpType::GlobalAvgPool,
                "GlobalMaxPool" => OpType::GlobalMaxPool,
                "BatchNormalization" => OpType::BatchNorm,
                "LayerNormalization" => OpType::LayerNorm,
                "Dropout" => OpType::Dropout,
                // Tensor manipulation
                "Reshape" => OpType::Reshape,
                "Transpose" => OpType::Transpose,
                "Concat" => OpType::Concatenate,
                "Split" => OpType::Split,
                "Slice" => OpType::Slice,
                "Gather" => OpType::Gather,
                "GatherElements" => OpType::Gather,
                "ScatterElements" => OpType::Scatter,
                "ScatterND" => OpType::Scatter,
                "Expand" => OpType::Expand,
                // Reductions
                "ReduceSum" => OpType::ReduceSum,
                "ReduceMean" => OpType::ReduceMean,
                "ReduceMax" => OpType::ReduceMax,
                "ReduceMin" => OpType::ReduceMin,
                "ReduceProd" => OpType::ReduceProd,
                // Math
                "Sqrt" => OpType::Sqrt,
                "Abs" => OpType::Abs,
                "Pow" => OpType::Pow,
                "Exp" => OpType::Exp,
                "Log" => OpType::Log,
                "Cos" => OpType::Cos,
                "Sin" => OpType::Sin,
                // Element-wise min/max (used by decomposition; also direct from ONNX Max op)
                "Min" => OpType::Min,
                "Max" => OpType::Max,
                // Comparisons
                "Equal" => OpType::Equal,
                "Greater" => OpType::Greater,
                "Less" => OpType::Less,
                "Not" => OpType::Not,
                // Opset 13 comparison variants
                "NotEqual" => OpType::NotEqual,
                "GreaterEqual" => OpType::GreaterEqual,
                "LessEqual" => OpType::LessEqual,
                // Logical (opset 13)
                "And" => OpType::And,
                "Or" => OpType::Or,
                // Identity: handled by rewiring in node loop (no IR node)
                "Identity" => unreachable!("Identity handled by rewire in build_ir"),
                // Broadcast: ONNX uses Expand or Mul with broadcast; map if present
                "Broadcast" => OpType::Broadcast,
                other => OpType::Custom(other.to_string()),
            }
        }

        fn tensor_shape(dims: &[i64]) -> Shape {
            Shape::new(dims.iter().map(|&d| d.max(0) as usize).collect())
        }

        /// Convert IEEE 754 binary16 to f32 (no external crate).
        fn f16_to_f32(u: u16) -> f32 {
            let sign = (u >> 15) & 1;
            let exp = (u >> 10) & 0x1F;
            let frac = u & 0x3FF;
            if exp == 0 {
                let v = (frac as f32) * 2.0_f32.powi(-24);
                if sign != 0 {
                    -v
                } else {
                    v
                }
            } else if exp == 31 {
                if frac == 0 {
                    if sign != 0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                } else {
                    f32::NAN
                }
            } else {
                let bits = (sign as u32) << 31
                    | ((exp as u32 + 112) << 23)
                    | ((frac as u32) << 13);
                f32::from_bits(bits)
            }
        }

        fn tensor_float_data(t: &rten_onnx::onnx::TensorProto) -> CoelanoxResult<Vec<f32>> {
            if !t.float_data.is_empty() {
                return Ok(t.float_data.clone());
            }
            if let Some(ref raw) = t.raw_data {
                let data = raw.borrow();
                let n = data.len() / 4;
                let mut out = Vec::with_capacity(n);
                for chunk in data.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                return Ok(out);
            }
            Err(CoelanoxError::format(
                "Tensor has no float_data or raw_data",
            ))
        }

        /// Read int64/int32 initializer data (for shape constants). Embedded only; not in external weights blob.
        fn initializer_int64_data(init: &rten_onnx::onnx::TensorProto) -> CoelanoxResult<Vec<i64>> {
            if !init.int64_data.is_empty() {
                return Ok(init.int64_data.clone());
            }
            if !init.int32_data.is_empty() {
                return Ok(init.int32_data.iter().map(|&x| x as i64).collect());
            }
            if let Some(ref raw) = init.raw_data {
                let data = raw.borrow();
                let dtype = init.data_type.as_ref().copied().unwrap_or(OnnxDataType::INT64);
                if dtype == OnnxDataType::INT32 {
                    let n = data.len() / 4;
                    let mut out = Vec::with_capacity(n);
                    for chunk in data.chunks_exact(4) {
                        out.push(i32::from_le_bytes(chunk.try_into().unwrap()) as i64);
                    }
                    return Ok(out);
                }
                let n = data.len() / 8;
                let mut out = Vec::with_capacity(n);
                for chunk in data.chunks_exact(8) {
                    out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                }
                return Ok(out);
            }
            Err(CoelanoxError::format(
                "Int64/Int32 initializer has no int64_data, int32_data, or raw_data",
            ))
        }

        /// Read initializer data: from embedded (float_data/raw_data) or from external buffer at running offset.
        /// Supports FLOAT (4 bytes) and FLOAT16 (2 bytes, converted to f32). Advances *external_offset when using external.
        fn initializer_float_data(
            init: &rten_onnx::onnx::TensorProto,
            external_bytes: Option<&[u8]>,
            external_offset: &mut usize,
        ) -> CoelanoxResult<Vec<f32>> {
            if !init.float_data.is_empty() {
                return Ok(init.float_data.clone());
            }
            if let Some(ref raw) = init.raw_data {
                let data = raw.borrow();
                let dtype = init.data_type.as_ref().copied().unwrap_or(OnnxDataType::FLOAT);
                if dtype == OnnxDataType::FLOAT16 {
                    let n = data.len() / 2;
                    let mut out = Vec::with_capacity(n);
                    for chunk in data.chunks_exact(2) {
                        let u = u16::from_le_bytes(chunk.try_into().unwrap());
                        out.push(Self::f16_to_f32(u));
                    }
                    return Ok(out);
                }
                let n = data.len() / 4;
                let mut out = Vec::with_capacity(n);
                for chunk in data.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                return Ok(out);
            }
            // External data: require external_bytes and use running offset (contiguous layout in initializer order).
            let external = match external_bytes {
                Some(b) => b,
                None => {
                    return Err(CoelanoxError::format(
                        "Tensor has external data but no external weights file was provided (use --weights)",
                    ));
                }
            };
            let dims = &init.dims;
            let num_elements: usize = dims.iter().map(|&d| d.max(0) as usize).product();
            let dtype = init.data_type.as_ref().copied().unwrap_or(OnnxDataType::FLOAT);
            let elem_size = if dtype == OnnxDataType::FLOAT16 { 2 } else { 4 };
            let byte_len = num_elements * elem_size;
            if *external_offset + byte_len > external.len() {
                return Err(CoelanoxError::format(&format!(
                    "External weights file too small for initializer (need {} bytes, have {} remaining)",
                    byte_len,
                    external.len().saturating_sub(*external_offset)
                )));
            }
            let slice = &external[*external_offset..*external_offset + byte_len];
            *external_offset += byte_len;
            let out = if dtype == OnnxDataType::FLOAT16 {
                let mut v = Vec::with_capacity(num_elements);
                for chunk in slice.chunks_exact(2) {
                    let u = u16::from_le_bytes(chunk.try_into().unwrap());
                    v.push(Self::f16_to_f32(u));
                }
                v
            } else {
                let mut v = Vec::with_capacity(num_elements);
                for chunk in slice.chunks_exact(4) {
                    v.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                v
            };
            Ok(out)
        }

        /// Get a node attribute by name (from ONNX AttributeProto list).
        fn get_attr<'a>(
            node: &'a rten_onnx::onnx::NodeProto,
            name: &str,
        ) -> Option<Attribute> {
            node.attribute
                .iter()
                .find(|a| a.name.as_deref() == Some(name))
                .and_then(|a| convert_attr(a).map(|(_, v)| v))
        }

        /// Add a scalar float constant as a weight tensor; returns its TensorId.
        fn add_scalar_constant(
            ir: &mut UniversalIR,
            name: String,
            value: f32,
        ) -> TensorId {
            let data = value.to_le_bytes().to_vec();
            ir.add_weight_tensor_with_data(name, Shape::new(vec![1]), DataType::Float32, data)
        }

        /// Decompose ONNX ops that have no single OpType into subgraphs. Returns true if the node was decomposed.
        fn decompose_node(
            ir: &mut UniversalIR,
            name_to_id: &mut HashMap<String, TensorId>,
            node: &rten_onnx::onnx::NodeProto,
            op_type_str: &str,
        ) -> CoelanoxResult<bool> {
            let input_ids: Vec<TensorId> = node
                .input
                .iter()
                .filter_map(|n| name_to_id.get(n).copied())
                .collect();
            let output_names: Vec<&str> = node
                .output
                .iter()
                .map(|s| s.as_str())
                .collect();
            if output_names.is_empty() {
                return Ok(false);
            }
            let out_id = name_to_id.get(output_names[0]).copied().ok_or_else(|| {
                CoelanoxError::format("Decompose: output tensor not in name_to_id")
            })?;
            let node_name = output_names[0].to_string();

            match op_type_str {
                "Clip" => {
                    // Clip(x, min?, max?) -> Min(Max(x, min), max). Opset 11+: optional tensor min/max; else attributes / defaults.
                    let x_id = node
                        .input
                        .first()
                        .filter(|s| !s.is_empty())
                        .and_then(|n| name_to_id.get(n).copied())
                        .ok_or_else(|| CoelanoxError::format("Clip requires data input"))?;

                    let min_tid = if node.input.len() > 1 && !node.input[1].is_empty() {
                        name_to_id
                            .get(&node.input[1])
                            .copied()
                            .ok_or_else(|| CoelanoxError::format("Clip: min tensor not found"))?
                    } else {
                        let min_val = Self::get_attr(node, "min")
                            .and_then(|a| a.as_float())
                            .unwrap_or(f32::MIN);
                        Self::add_scalar_constant(ir, format!("{}_clip_min", node_name), min_val)
                    };

                    let max_tid = if node.input.len() > 2 && !node.input[2].is_empty() {
                        name_to_id
                            .get(&node.input[2])
                            .copied()
                            .ok_or_else(|| CoelanoxError::format("Clip: max tensor not found"))?
                    } else {
                        let max_val = Self::get_attr(node, "max")
                            .and_then(|a| a.as_float())
                            .unwrap_or(f32::MAX);
                        Self::add_scalar_constant(ir, format!("{}_clip_max", node_name), max_val)
                    };

                    let mid_name = format!("{}_clip_mid", node_name);
                    let mid_tid = ir.add_intermediate_tensor(
                        mid_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(mid_name, mid_tid);

                    let n_max = ir.add_node(OpType::Max);
                    ir.add_node_input(n_max, x_id)?;
                    ir.add_node_input(n_max, min_tid)?;
                    ir.add_node_output(n_max, mid_tid)?;

                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, mid_tid)?;
                    ir.add_node_input(n_min, max_tid)?;
                    ir.add_node_output(n_min, out_id)?;
                    return Ok(true);
                }
                "Where" => {
                    // Where(cond, X, Y) with float mask cond in {0,1}: cond * X + (1 - cond) * Y (opset 9+).
                    if input_ids.len() != 3 {
                        return Err(CoelanoxError::format(
                            "Where requires three inputs (condition, X, Y); optional ONNX inputs must be omitted, not unknown tensors.",
                        ));
                    }
                    let cond = input_ids[0];
                    let x_t = input_ids[1];
                    let y_t = input_ids[2];
                    let one_c = Self::add_scalar_constant(ir, format!("{}_where_1", node_name), 1.0);
                    let om_name = format!("{}_where_1mc", node_name);
                    let om_tid = ir.add_intermediate_tensor(om_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(om_name, om_tid);
                    let n_sub = ir.add_node(OpType::Subtract);
                    ir.add_node_input(n_sub, one_c)?;
                    ir.add_node_input(n_sub, cond)?;
                    ir.add_node_output(n_sub, om_tid)?;
                    let t1_name = format!("{}_where_cx", node_name);
                    let t1_tid = ir.add_intermediate_tensor(t1_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t1_name, t1_tid);
                    let n_mx = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mx, cond)?;
                    ir.add_node_input(n_mx, x_t)?;
                    ir.add_node_output(n_mx, t1_tid)?;
                    let t2_name = format!("{}_where_cy", node_name);
                    let t2_tid = ir.add_intermediate_tensor(t2_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t2_name, t2_tid);
                    let n_my = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_my, om_tid)?;
                    ir.add_node_input(n_my, y_t)?;
                    ir.add_node_output(n_my, t2_tid)?;
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, t1_tid)?;
                    ir.add_node_input(n_add, t2_tid)?;
                    ir.add_node_output(n_add, out_id)?;
                    return Ok(true);
                }
                "Square" => {
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Square requires one input")
                    })?;
                    let two_c = Self::add_scalar_constant(ir, format!("{}_sq2", node_name), 2.0);
                    let n = ir.add_node(OpType::Pow);
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_input(n, two_c)?;
                    ir.add_node_output(n, out_id)?;
                    return Ok(true);
                }
                "Neg" => {
                    // Neg(x) = Mul(x, -1)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Neg requires one input")
                    })?;
                    let neg_one_name = format!("{}_neg_const", node_name);
                    let neg_one = Self::add_scalar_constant(ir, neg_one_name, -1.0);
                    let n = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_input(n, neg_one)?;
                    ir.add_node_output(n, out_id)?;
                    return Ok(true);
                }
                "Reciprocal" => {
                    // Reciprocal(x) = Div(1, x)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Reciprocal requires one input")
                    })?;
                    let one_name = format!("{}_recip_const", node_name);
                    let one = Self::add_scalar_constant(ir, one_name, 1.0);
                    let n = ir.add_node(OpType::Divide);
                    ir.add_node_input(n, one)?;
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_output(n, out_id)?;
                    return Ok(true);
                }
                "LeakyRelu" => {
                    // LeakyRelu(x) = Relu(x) - alpha * Relu(-x); default alpha = 0.01 (ONNX opset 13).
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("LeakyRelu requires one input")
                    })?;
                    let alpha = Self::get_attr(node, "alpha")
                        .and_then(|a| a.as_float())
                        .unwrap_or(0.01f32);
                    let neg_x_name = format!("{}_leaky_negx", node_name);
                    let neg_x_tid = ir.add_intermediate_tensor(
                        neg_x_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(neg_x_name, neg_x_tid);
                    let m_neg1 = ir.add_node(OpType::Multiply);
                    let neg_one = Self::add_scalar_constant(ir, format!("{}_leaky_m1", node_name), -1.0);
                    ir.add_node_input(m_neg1, *x_id)?;
                    ir.add_node_input(m_neg1, neg_one)?;
                    ir.add_node_output(m_neg1, neg_x_tid)?;

                    let r_pos_name = format!("{}_leaky_rpos", node_name);
                    let r_pos_tid = ir.add_intermediate_tensor(
                        r_pos_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(r_pos_name, r_pos_tid);
                    let relu1 = ir.add_node(OpType::Relu);
                    ir.add_node_input(relu1, *x_id)?;
                    ir.add_node_output(relu1, r_pos_tid)?;

                    let r_neg_name = format!("{}_leaky_rneg", node_name);
                    let r_neg_tid = ir.add_intermediate_tensor(
                        r_neg_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(r_neg_name, r_neg_tid);
                    let relu2 = ir.add_node(OpType::Relu);
                    ir.add_node_input(relu2, neg_x_tid)?;
                    ir.add_node_output(relu2, r_neg_tid)?;

                    let alpha_tid = Self::add_scalar_constant(ir, format!("{}_leaky_alpha", node_name), alpha);
                    let scaled_name = format!("{}_leaky_scaled", node_name);
                    let scaled_tid = ir.add_intermediate_tensor(
                        scaled_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(scaled_name, scaled_tid);
                    let mul_s = ir.add_node(OpType::Multiply);
                    ir.add_node_input(mul_s, r_neg_tid)?;
                    ir.add_node_input(mul_s, alpha_tid)?;
                    ir.add_node_output(mul_s, scaled_tid)?;

                    let sub_n = ir.add_node(OpType::Subtract);
                    ir.add_node_input(sub_n, r_pos_tid)?;
                    ir.add_node_input(sub_n, scaled_tid)?;
                    ir.add_node_output(sub_n, out_id)?;
                    return Ok(true);
                }
                "Elu" => {
                    // Elu(x) = x if x>0 else alpha*(exp(x)-1); alpha default 1.0
                    // = Relu(x) + alpha * (Exp(Min(x,0)) - 1)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Elu requires one input")
                    })?;
                    let alpha = Self::get_attr(node, "alpha")
                        .and_then(|a| a.as_float())
                        .unwrap_or(1.0f32);
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_elu_z", node_name), 0.0);
                    let one_c = Self::add_scalar_constant(ir, format!("{}_elu_one", node_name), 1.0);
                    let alpha_c = Self::add_scalar_constant(ir, format!("{}_elu_alpha", node_name), alpha);
                    let m_name = format!("{}_elu_min", node_name);
                    let m_tid = ir.add_intermediate_tensor(m_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(m_name, m_tid);
                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, *x_id)?;
                    ir.add_node_input(n_min, zero_c)?;
                    ir.add_node_output(n_min, m_tid)?;
                    let e_name = format!("{}_elu_exp", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, m_tid)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let t_name = format!("{}_elu_em1", node_name);
                    let t_tid = ir.add_intermediate_tensor(t_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t_name, t_tid);
                    let n_sub = ir.add_node(OpType::Subtract);
                    ir.add_node_input(n_sub, e_tid)?;
                    ir.add_node_input(n_sub, one_c)?;
                    ir.add_node_output(n_sub, t_tid)?;
                    let neg_b_name = format!("{}_elu_negb", node_name);
                    let neg_b_tid = ir.add_intermediate_tensor(neg_b_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(neg_b_name, neg_b_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, t_tid)?;
                    ir.add_node_input(n_mul, alpha_c)?;
                    ir.add_node_output(n_mul, neg_b_tid)?;
                    let pos_name = format!("{}_elu_pos", node_name);
                    let pos_tid = ir.add_intermediate_tensor(pos_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(pos_name, pos_tid);
                    let n_relu = ir.add_node(OpType::Relu);
                    ir.add_node_input(n_relu, *x_id)?;
                    ir.add_node_output(n_relu, pos_tid)?;
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, pos_tid)?;
                    ir.add_node_input(n_add, neg_b_tid)?;
                    ir.add_node_output(n_add, out_id)?;
                    return Ok(true);
                }
                "PRelu" => {
                    // PRelu(x, slope): x if x>=0 else slope*x  =>  Relu(x) + slope * Min(x, 0)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("PRelu requires X input")
                    })?;
                    let slope_id = *input_ids.get(1).ok_or_else(|| {
                        CoelanoxError::format("PRelu requires slope input")
                    })?;
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_prelu_z", node_name), 0.0);
                    let neg_name = format!("{}_prelu_neg", node_name);
                    let neg_tid = ir.add_intermediate_tensor(neg_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(neg_name, neg_tid);
                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, *x_id)?;
                    ir.add_node_input(n_min, zero_c)?;
                    ir.add_node_output(n_min, neg_tid)?;
                    let sn_name = format!("{}_prelu_sn", node_name);
                    let sn_tid = ir.add_intermediate_tensor(sn_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sn_name, sn_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, slope_id)?;
                    ir.add_node_input(n_mul, neg_tid)?;
                    ir.add_node_output(n_mul, sn_tid)?;
                    let pos_name = format!("{}_prelu_pos", node_name);
                    let pos_tid = ir.add_intermediate_tensor(pos_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(pos_name, pos_tid);
                    let n_relu = ir.add_node(OpType::Relu);
                    ir.add_node_input(n_relu, *x_id)?;
                    ir.add_node_output(n_relu, pos_tid)?;
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, pos_tid)?;
                    ir.add_node_input(n_add, sn_tid)?;
                    ir.add_node_output(n_add, out_id)?;
                    return Ok(true);
                }
                "HardSigmoid" => {
                    // max(0, min(1, alpha*x + beta)); defaults alpha=0.2, beta=0.5 (ONNX)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("HardSigmoid requires one input")
                    })?;
                    let alpha = Self::get_attr(node, "alpha")
                        .and_then(|a| a.as_float())
                        .unwrap_or(0.2f32);
                    let beta = Self::get_attr(node, "beta")
                        .and_then(|a| a.as_float())
                        .unwrap_or(0.5f32);
                    let alpha_c = Self::add_scalar_constant(ir, format!("{}_hs_a", node_name), alpha);
                    let beta_c = Self::add_scalar_constant(ir, format!("{}_hs_b", node_name), beta);
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_hs_z", node_name), 0.0);
                    let one_c = Self::add_scalar_constant(ir, format!("{}_hs_o", node_name), 1.0);
                    let t1_name = format!("{}_hs_ax", node_name);
                    let t1_tid = ir.add_intermediate_tensor(t1_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t1_name, t1_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, *x_id)?;
                    ir.add_node_input(n_mul, alpha_c)?;
                    ir.add_node_output(n_mul, t1_tid)?;
                    let t2_name = format!("{}_hs_lin", node_name);
                    let t2_tid = ir.add_intermediate_tensor(t2_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t2_name, t2_tid);
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, t1_tid)?;
                    ir.add_node_input(n_add, beta_c)?;
                    ir.add_node_output(n_add, t2_tid)?;
                    let t3_name = format!("{}_hs_lo", node_name);
                    let t3_tid = ir.add_intermediate_tensor(t3_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t3_name, t3_tid);
                    let n_mx = ir.add_node(OpType::Max);
                    ir.add_node_input(n_mx, t2_tid)?;
                    ir.add_node_input(n_mx, zero_c)?;
                    ir.add_node_output(n_mx, t3_tid)?;
                    let n_mn = ir.add_node(OpType::Min);
                    ir.add_node_input(n_mn, t3_tid)?;
                    ir.add_node_input(n_mn, one_c)?;
                    ir.add_node_output(n_mn, out_id)?;
                    return Ok(true);
                }
                "Softplus" => {
                    // Softplus(x) = log(exp(x) + 1)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Softplus requires one input")
                    })?;
                    let one_c = Self::add_scalar_constant(ir, format!("{}_sp1", node_name), 1.0);
                    let e_name = format!("{}_spex", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, *x_id)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let s_name = format!("{}_sp_s", node_name);
                    let s_tid = ir.add_intermediate_tensor(s_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(s_name, s_tid);
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, e_tid)?;
                    ir.add_node_input(n_add, one_c)?;
                    ir.add_node_output(n_add, s_tid)?;
                    let n_log = ir.add_node(OpType::Log);
                    ir.add_node_input(n_log, s_tid)?;
                    ir.add_node_output(n_log, out_id)?;
                    return Ok(true);
                }
                "HardSwish" => {
                    // x * clip(x+3, 0, 6) / 6  (MobileNet-style; ONNX op may be opset 14+ but we accept the name)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("HardSwish requires one input")
                    })?;
                    let three = Self::add_scalar_constant(ir, format!("{}_hsw3", node_name), 3.0);
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_hswz", node_name), 0.0);
                    let six_c = Self::add_scalar_constant(ir, format!("{}_hsw6", node_name), 6.0);
                    let six_inv = Self::add_scalar_constant(ir, format!("{}_hsw_inv6", node_name), 1.0 / 6.0);
                    let t0_name = format!("{}_hsw_x3", node_name);
                    let t0_tid = ir.add_intermediate_tensor(t0_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t0_name, t0_tid);
                    let n_a = ir.add_node(OpType::Add);
                    ir.add_node_input(n_a, *x_id)?;
                    ir.add_node_input(n_a, three)?;
                    ir.add_node_output(n_a, t0_tid)?;
                    let t1_name = format!("{}_hsw_cl1", node_name);
                    let t1_tid = ir.add_intermediate_tensor(t1_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t1_name, t1_tid);
                    let n_mx = ir.add_node(OpType::Max);
                    ir.add_node_input(n_mx, t0_tid)?;
                    ir.add_node_input(n_mx, zero_c)?;
                    ir.add_node_output(n_mx, t1_tid)?;
                    let t2_name = format!("{}_hsw_cl2", node_name);
                    let t2_tid = ir.add_intermediate_tensor(t2_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t2_name, t2_tid);
                    let n_mn = ir.add_node(OpType::Min);
                    ir.add_node_input(n_mn, t1_tid)?;
                    ir.add_node_input(n_mn, six_c)?;
                    ir.add_node_output(n_mn, t2_tid)?;
                    let t3_name = format!("{}_hsw_mul", node_name);
                    let t3_tid = ir.add_intermediate_tensor(t3_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t3_name, t3_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, *x_id)?;
                    ir.add_node_input(n_mul, t2_tid)?;
                    ir.add_node_output(n_mul, t3_tid)?;
                    let n_out = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_out, t3_tid)?;
                    ir.add_node_input(n_out, six_inv)?;
                    ir.add_node_output(n_out, out_id)?;
                    return Ok(true);
                }
                "Softsign" => {
                    // Softsign(x) = x / (1 + |x|); |x| = Sqrt(Pow(x,2))
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Softsign requires one input")
                    })?;
                    let one_c = Self::add_scalar_constant(ir, format!("{}_ss1", node_name), 1.0);
                    let two_c = Self::add_scalar_constant(ir, format!("{}_ss2", node_name), 2.0);
                    let sq_name = format!("{}_ss_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let ax_name = format!("{}_ss_abs", node_name);
                    let ax_tid = ir.add_intermediate_tensor(ax_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(ax_name, ax_tid);
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, sq_tid)?;
                    ir.add_node_output(n_sqrt, ax_tid)?;
                    let den_name = format!("{}_ss_den", node_name);
                    let den_tid = ir.add_intermediate_tensor(den_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(den_name, den_tid);
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, one_c)?;
                    ir.add_node_input(n_add, ax_tid)?;
                    ir.add_node_output(n_add, den_tid)?;
                    let n_div = ir.add_node(OpType::Divide);
                    ir.add_node_input(n_div, *x_id)?;
                    ir.add_node_input(n_div, den_tid)?;
                    ir.add_node_output(n_div, out_id)?;
                    return Ok(true);
                }
                "Sign" => {
                    // sign(x) ≈ x / (|x| + eps) for float (ONNX Sign)
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Sign requires one input")
                    })?;
                    let eps_c = Self::add_scalar_constant(ir, format!("{}_sg_eps", node_name), 1e-12f32);
                    let two_c = Self::add_scalar_constant(ir, format!("{}_sg2", node_name), 2.0);
                    let sq_name = format!("{}_sg_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let ax_name = format!("{}_sg_abs", node_name);
                    let ax_tid = ir.add_intermediate_tensor(ax_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(ax_name, ax_tid);
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, sq_tid)?;
                    ir.add_node_output(n_sqrt, ax_tid)?;
                    let den_name = format!("{}_sg_den", node_name);
                    let den_tid = ir.add_intermediate_tensor(den_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(den_name, den_tid);
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, ax_tid)?;
                    ir.add_node_input(n_add, eps_c)?;
                    ir.add_node_output(n_add, den_tid)?;
                    let n_div = ir.add_node(OpType::Divide);
                    ir.add_node_input(n_div, *x_id)?;
                    ir.add_node_input(n_div, den_tid)?;
                    ir.add_node_output(n_div, out_id)?;
                    return Ok(true);
                }
                "Sum" => {
                    // Variadic Sum = chained Add (opset 13).
                    let n_in = input_ids.len();
                    if n_in == 0 {
                        return Err(CoelanoxError::format("Sum requires at least one input"));
                    }
                    if n_in == 1 {
                        let z = Self::add_scalar_constant(ir, format!("{}_sum_z", node_name), 0.0);
                        let add_n = ir.add_node(OpType::Add);
                        ir.add_node_input(add_n, input_ids[0])?;
                        ir.add_node_input(add_n, z)?;
                        ir.add_node_output(add_n, out_id)?;
                        return Ok(true);
                    }
                    let mut cur = input_ids[0];
                    for i in 1..n_in {
                        let add_n = ir.add_node(OpType::Add);
                        if i + 1 == n_in {
                            ir.add_node_input(add_n, cur)?;
                            ir.add_node_input(add_n, input_ids[i])?;
                            ir.add_node_output(add_n, out_id)?;
                        } else {
                            let tmp_name = format!("{}_sum_{}", node_name, i);
                            let tmp_tid = ir.add_intermediate_tensor(
                                tmp_name.clone(),
                                Shape::new(vec![1]),
                                DataType::Float32,
                            );
                            name_to_id.insert(tmp_name, tmp_tid);
                            ir.add_node_input(add_n, cur)?;
                            ir.add_node_input(add_n, input_ids[i])?;
                            ir.add_node_output(add_n, tmp_tid)?;
                            cur = tmp_tid;
                        }
                    }
                    return Ok(true);
                }
                "ReduceL2" => {
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("ReduceL2 requires input")
                    })?;
                    let two_c = Self::add_scalar_constant(ir, format!("{}_rl2_2", node_name), 2.0);
                    let sq_name = format!("{}_rl2_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let rs_name = format!("{}_rl2_rs", node_name);
                    let rs_tid = ir.add_intermediate_tensor(rs_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(rs_name, rs_tid);
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, sq_tid)?;
                    ir.add_node_output(n_rs, rs_tid)?;
                    forward_onnx_reduce_attrs(ir, n_rs, node);
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, rs_tid)?;
                    ir.add_node_output(n_sqrt, out_id)?;
                    return Ok(true);
                }
                "ReduceL1" => {
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("ReduceL1 requires input")
                    })?;
                    let two_c = Self::add_scalar_constant(ir, format!("{}_rl1_2", node_name), 2.0);
                    let sq_name = format!("{}_rl1_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let ax_name = format!("{}_rl1_abs", node_name);
                    let ax_tid = ir.add_intermediate_tensor(ax_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(ax_name, ax_tid);
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, sq_tid)?;
                    ir.add_node_output(n_sqrt, ax_tid)?;
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, ax_tid)?;
                    ir.add_node_output(n_rs, out_id)?;
                    forward_onnx_reduce_attrs(ir, n_rs, node);
                    return Ok(true);
                }
                "ReduceSumSquare" => {
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("ReduceSumSquare requires input")
                    })?;
                    let two_c = Self::add_scalar_constant(ir, format!("{}_rss_2", node_name), 2.0);
                    let sq_name = format!("{}_rss_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, sq_tid)?;
                    ir.add_node_output(n_rs, out_id)?;
                    forward_onnx_reduce_attrs(ir, n_rs, node);
                    return Ok(true);
                }
                "LpNormalization" => {
                    // L2 norm along axis: x / sqrt(reduce_sum(x^2, axis, keepdims=1)). Only p=2.
                    let p = Self::get_attr(node, "p").and_then(|a| a.as_int()).unwrap_or(2);
                    if p != 2 {
                        return Err(CoelanoxError::format(
                            "LpNormalization: only p=2 is implemented; use another export or extend the translator.",
                        ));
                    }
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("LpNormalization requires input")
                    })?;
                    let axis = Self::get_attr(node, "axis").and_then(|a| a.as_int()).unwrap_or(-1);
                    let two_c = Self::add_scalar_constant(ir, format!("{}_lpn2", node_name), 2.0);
                    let sq_name = format!("{}_lpn_sq", node_name);
                    let sq_tid = ir.add_intermediate_tensor(sq_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_name, sq_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two_c)?;
                    ir.add_node_output(n_pow, sq_tid)?;
                    let ss_name = format!("{}_lpn_ss", node_name);
                    let ss_tid = ir.add_intermediate_tensor(ss_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(ss_name, ss_tid);
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, sq_tid)?;
                    ir.add_node_output(n_rs, ss_tid)?;
                    if let Some(n) = ir.get_node_mut(n_rs) {
                        n.add_attribute("axes".to_string(), Attribute::IntArray(vec![axis]));
                        n.add_attribute("keepdims".to_string(), Attribute::Int(1));
                    }
                    let norm_name = format!("{}_lpn_n", node_name);
                    let norm_tid = ir.add_intermediate_tensor(norm_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(norm_name, norm_tid);
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, ss_tid)?;
                    ir.add_node_output(n_sqrt, norm_tid)?;
                    let n_div = ir.add_node(OpType::Divide);
                    ir.add_node_input(n_div, *x_id)?;
                    ir.add_node_input(n_div, norm_tid)?;
                    ir.add_node_output(n_div, out_id)?;
                    return Ok(true);
                }
                "ReduceLogSum" => {
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("ReduceLogSum requires input")
                    })?;
                    let e_name = format!("{}_rls_e", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, *x_id)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let s_name = format!("{}_rls_s", node_name);
                    let s_tid = ir.add_intermediate_tensor(s_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(s_name, s_tid);
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, e_tid)?;
                    ir.add_node_output(n_rs, s_tid)?;
                    forward_onnx_reduce_attrs(ir, n_rs, node);
                    let n_log = ir.add_node(OpType::Log);
                    ir.add_node_input(n_log, s_tid)?;
                    ir.add_node_output(n_log, out_id)?;
                    return Ok(true);
                }
                "ReduceLogSumExp" => {
                    // Log(ReduceSum(Exp(x))) — naive; numerically unstable vs two-pass log-sum-exp.
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("ReduceLogSumExp requires input")
                    })?;
                    let e_name = format!("{}_rlse_e", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, *x_id)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let s_name = format!("{}_rlse_s", node_name);
                    let s_tid = ir.add_intermediate_tensor(s_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(s_name, s_tid);
                    let n_rs = ir.add_node(OpType::ReduceSum);
                    ir.add_node_input(n_rs, e_tid)?;
                    ir.add_node_output(n_rs, s_tid)?;
                    forward_onnx_reduce_attrs(ir, n_rs, node);
                    let n_log = ir.add_node(OpType::Log);
                    ir.add_node_input(n_log, s_tid)?;
                    ir.add_node_output(n_log, out_id)?;
                    return Ok(true);
                }
                "Selu" => {
                    // Selu(x) = gamma * Elu(x) with ONNX defaults alpha=1.67326..., gamma=1.0507...
                    let x_id = input_ids.first().ok_or_else(|| CoelanoxError::format("Selu requires one input"))?;
                    let alpha = Self::get_attr(node, "alpha")
                        .and_then(|a| a.as_float())
                        .unwrap_or(1.6732632f32);
                    let gamma = Self::get_attr(node, "gamma")
                        .and_then(|a| a.as_float())
                        .unwrap_or(1.050701f32);
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_selu_z", node_name), 0.0);
                    let one_c = Self::add_scalar_constant(ir, format!("{}_selu_one", node_name), 1.0);
                    let alpha_c = Self::add_scalar_constant(ir, format!("{}_selu_alpha", node_name), alpha);
                    let m_name = format!("{}_selu_min", node_name);
                    let m_tid = ir.add_intermediate_tensor(m_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(m_name, m_tid);
                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, *x_id)?;
                    ir.add_node_input(n_min, zero_c)?;
                    ir.add_node_output(n_min, m_tid)?;
                    let e_name = format!("{}_selu_exp", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, m_tid)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let t_name = format!("{}_selu_em1", node_name);
                    let t_tid = ir.add_intermediate_tensor(t_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t_name, t_tid);
                    let n_sub = ir.add_node(OpType::Subtract);
                    ir.add_node_input(n_sub, e_tid)?;
                    ir.add_node_input(n_sub, one_c)?;
                    ir.add_node_output(n_sub, t_tid)?;
                    let neg_b_name = format!("{}_selu_negb", node_name);
                    let neg_b_tid = ir.add_intermediate_tensor(neg_b_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(neg_b_name, neg_b_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, t_tid)?;
                    ir.add_node_input(n_mul, alpha_c)?;
                    ir.add_node_output(n_mul, neg_b_tid)?;
                    let pos_name = format!("{}_selu_pos", node_name);
                    let pos_tid = ir.add_intermediate_tensor(pos_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(pos_name, pos_tid);
                    let n_relu = ir.add_node(OpType::Relu);
                    ir.add_node_input(n_relu, *x_id)?;
                    ir.add_node_output(n_relu, pos_tid)?;
                    let elu_name = format!("{}_selu_elu", node_name);
                    let elu_tid = ir.add_intermediate_tensor(elu_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(elu_name, elu_tid);
                    let n_add_elu = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add_elu, pos_tid)?;
                    ir.add_node_input(n_add_elu, neg_b_tid)?;
                    ir.add_node_output(n_add_elu, elu_tid)?;
                    let gamma_c = Self::add_scalar_constant(ir, format!("{}_selu_g", node_name), gamma);
                    let n_g = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_g, elu_tid)?;
                    ir.add_node_input(n_g, gamma_c)?;
                    ir.add_node_output(n_g, out_id)?;
                    return Ok(true);
                }
                "Celu" => {
                    // Celu(x) = max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
                    let x_id = input_ids.first().ok_or_else(|| CoelanoxError::format("Celu requires one input"))?;
                    let alpha = Self::get_attr(node, "alpha")
                        .and_then(|a| a.as_float())
                        .unwrap_or(1.0f32);
                    let zero_c = Self::add_scalar_constant(ir, format!("{}_celu_z", node_name), 0.0);
                    let one_c = Self::add_scalar_constant(ir, format!("{}_celu_one", node_name), 1.0);
                    let alpha_c = Self::add_scalar_constant(ir, format!("{}_celu_a", node_name), alpha);
                    let m_name = format!("{}_celu_m", node_name);
                    let m_tid = ir.add_intermediate_tensor(m_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(m_name, m_tid);
                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, *x_id)?;
                    ir.add_node_input(n_min, zero_c)?;
                    ir.add_node_output(n_min, m_tid)?;
                    let t_name = format!("{}_celu_t", node_name);
                    let t_tid = ir.add_intermediate_tensor(t_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(t_name, t_tid);
                    let n_div = ir.add_node(OpType::Divide);
                    ir.add_node_input(n_div, m_tid)?;
                    ir.add_node_input(n_div, alpha_c)?;
                    ir.add_node_output(n_div, t_tid)?;
                    let e_name = format!("{}_celu_e", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, t_tid)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let u_name = format!("{}_celu_u", node_name);
                    let u_tid = ir.add_intermediate_tensor(u_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(u_name, u_tid);
                    let n_sub = ir.add_node(OpType::Subtract);
                    ir.add_node_input(n_sub, e_tid)?;
                    ir.add_node_input(n_sub, one_c)?;
                    ir.add_node_output(n_sub, u_tid)?;
                    let v_name = format!("{}_celu_v", node_name);
                    let v_tid = ir.add_intermediate_tensor(v_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(v_name, v_tid);
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, alpha_c)?;
                    ir.add_node_input(n_mul, u_tid)?;
                    ir.add_node_output(n_mul, v_tid)?;
                    let w_name = format!("{}_celu_w", node_name);
                    let w_tid = ir.add_intermediate_tensor(w_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(w_name, w_tid);
                    let n_min2 = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min2, zero_c)?;
                    ir.add_node_input(n_min2, v_tid)?;
                    ir.add_node_output(n_min2, w_tid)?;
                    let pos_name = format!("{}_celu_pos", node_name);
                    let pos_tid = ir.add_intermediate_tensor(pos_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(pos_name, pos_tid);
                    let n_relu = ir.add_node(OpType::Relu);
                    ir.add_node_input(n_relu, *x_id)?;
                    ir.add_node_output(n_relu, pos_tid)?;
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, pos_tid)?;
                    ir.add_node_input(n_add, w_tid)?;
                    ir.add_node_output(n_add, out_id)?;
                    return Ok(true);
                }
                "Mish" => {
                    // Mish(x) = x * Tanh(Softplus(x))
                    let x_id = input_ids.first().ok_or_else(|| CoelanoxError::format("Mish requires one input"))?;
                    let one_c = Self::add_scalar_constant(ir, format!("{}_m1", node_name), 1.0);
                    let e_name = format!("{}_m_e", node_name);
                    let e_tid = ir.add_intermediate_tensor(e_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(e_name, e_tid);
                    let n_exp = ir.add_node(OpType::Exp);
                    ir.add_node_input(n_exp, *x_id)?;
                    ir.add_node_output(n_exp, e_tid)?;
                    let s_name = format!("{}_m_s", node_name);
                    let s_tid = ir.add_intermediate_tensor(s_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(s_name, s_tid);
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, e_tid)?;
                    ir.add_node_input(n_add, one_c)?;
                    ir.add_node_output(n_add, s_tid)?;
                    let sp_name = format!("{}_m_sp", node_name);
                    let sp_tid = ir.add_intermediate_tensor(sp_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sp_name, sp_tid);
                    let n_log = ir.add_node(OpType::Log);
                    ir.add_node_input(n_log, s_tid)?;
                    ir.add_node_output(n_log, sp_tid)?;
                    let th_name = format!("{}_m_th", node_name);
                    let th_tid = ir.add_intermediate_tensor(th_name.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(th_name, th_tid);
                    let n_tanh = ir.add_node(OpType::Tanh);
                    ir.add_node_input(n_tanh, sp_tid)?;
                    ir.add_node_output(n_tanh, th_tid)?;
                    let n_mul = ir.add_node(OpType::Multiply);
                    ir.add_node_input(n_mul, *x_id)?;
                    ir.add_node_input(n_mul, th_tid)?;
                    ir.add_node_output(n_mul, out_id)?;
                    return Ok(true);
                }
                "Flatten" => {
                    // Flatten(axis) -> Reshape with shape [dim0*..*dim(axis-1), dim(axis)*..*dim(n)].
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Flatten requires one input")
                    })?;
                    let axis = Self::get_attr(node, "axis").and_then(|a| a.as_int()).unwrap_or(1);
                    let n = ir.add_node(OpType::Reshape);
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_output(n, out_id)?;
                    if let Some(nd) = ir.get_node_mut(n) {
                        nd.add_attribute("axis".to_string(), Attribute::Int(axis));
                        nd.add_attribute("shape".to_string(), Attribute::IntArray(vec![0, -1]));
                    }
                    return Ok(true);
                }
                "Mean" => {
                    // Mean(in0, in1, ...) = (in0 + in1 + ...) / n. Chain Add then Divide by n.
                    let n_inputs = input_ids.len();
                    if n_inputs == 0 {
                        return Err(CoelanoxError::format("Mean requires at least one input"));
                    }
                    let n_f = n_inputs as f32;
                    let count_tid = Self::add_scalar_constant(
                        ir,
                        format!("{}_mean_n", node_name),
                        n_f,
                    );
                    let mut running_tid = input_ids[0];
                    for &in_id in input_ids.iter().skip(1) {
                        let sum_name = format!("{}_mean_sum_{}", node_name, name_to_id.len());
                        let sum_tid = ir.add_intermediate_tensor(
                            sum_name.clone(),
                            Shape::new(vec![1]),
                            DataType::Float32,
                        );
                        name_to_id.insert(sum_name, sum_tid);
                        let add_n = ir.add_node(OpType::Add);
                        ir.add_node_input(add_n, running_tid)?;
                        ir.add_node_input(add_n, in_id)?;
                        ir.add_node_output(add_n, sum_tid)?;
                        running_tid = sum_tid;
                    }
                    let div_n = ir.add_node(OpType::Divide);
                    ir.add_node_input(div_n, running_tid)?;
                    ir.add_node_input(div_n, count_tid)?;
                    ir.add_node_output(div_n, out_id)?;
                    return Ok(true);
                }
                "MeanVarianceNormalization" => {
                    // y = (x - mean) / sqrt(var + eps). axes attribute (e.g. [0,1,2] for NCHW).
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("MeanVarianceNormalization requires one input")
                    })?;
                    let eps = Self::get_attr(node, "epsilon")
                        .and_then(|a| a.as_float())
                        .unwrap_or(1e-5);
                    let axes: Vec<i64> = Self::get_attr(node, "axes")
                        .and_then(|a| a.as_int_array().map(|a| a.to_vec()))
                        .unwrap_or_else(|| vec![0, 1, 2]);
                    let (mean_t, xc_t, sq_t, var_t, var_eps_t, std_t) = (
                        format!("{}_mvn_mean", node_name),
                        format!("{}_mvn_xc", node_name),
                        format!("{}_mvn_sq", node_name),
                        format!("{}_mvn_var", node_name),
                        format!("{}_mvn_var_eps", node_name),
                        format!("{}_mvn_std", node_name),
                    );
                    let mean_id = ir.add_intermediate_tensor(mean_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(mean_t, mean_id);
                    let xc_id = ir.add_intermediate_tensor(xc_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(xc_t, xc_id);
                    let sq_id = ir.add_intermediate_tensor(sq_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(sq_t, sq_id);
                    let var_id = ir.add_intermediate_tensor(var_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(var_t, var_id);
                    let var_eps_id = ir.add_intermediate_tensor(var_eps_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(var_eps_t, var_eps_id);
                    let std_id = ir.add_intermediate_tensor(std_t.clone(), Shape::new(vec![1]), DataType::Float32);
                    name_to_id.insert(std_t, std_id);
                    let eps_tid = Self::add_scalar_constant(ir, format!("{}_mvn_eps", node_name), eps);
                    let two_tid = Self::add_scalar_constant(ir, format!("{}_mvn_two", node_name), 2.0);

                    let n_mean = ir.add_node(OpType::ReduceMean);
                    ir.add_node_input(n_mean, *x_id)?;
                    ir.add_node_output(n_mean, mean_id)?;
                    if let Some(nd) = ir.get_node_mut(n_mean) {
                        nd.add_attribute("axes".to_string(), Attribute::IntArray(axes.clone()));
                    }
                    let n_sub = ir.add_node(OpType::Subtract);
                    ir.add_node_input(n_sub, *x_id)?;
                    ir.add_node_input(n_sub, mean_id)?;
                    ir.add_node_output(n_sub, xc_id)?;
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, xc_id)?;
                    ir.add_node_input(n_pow, two_tid)?;
                    ir.add_node_output(n_pow, sq_id)?;
                    let n_var = ir.add_node(OpType::ReduceMean);
                    ir.add_node_input(n_var, sq_id)?;
                    ir.add_node_output(n_var, var_id)?;
                    if let Some(nd) = ir.get_node_mut(n_var) {
                        nd.add_attribute("axes".to_string(), Attribute::IntArray(axes));
                    }
                    let n_add = ir.add_node(OpType::Add);
                    ir.add_node_input(n_add, var_id)?;
                    ir.add_node_input(n_add, eps_tid)?;
                    ir.add_node_output(n_add, var_eps_id)?;
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, var_eps_id)?;
                    ir.add_node_output(n_sqrt, std_id)?;
                    let n_div = ir.add_node(OpType::Divide);
                    ir.add_node_input(n_div, xc_id)?;
                    ir.add_node_input(n_div, std_id)?;
                    ir.add_node_output(n_div, out_id)?;
                    return Ok(true);
                }
                "Unsqueeze" => {
                    // Insert dimension 1 at axis. Single Reshape with axis for backend to interpret.
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Unsqueeze requires one input")
                    })?;
                    let axis = Self::get_attr(node, "axis").and_then(|a| a.as_int()).unwrap_or(0);
                    let axes: Vec<i64> = Self::get_attr(node, "axes")
                        .and_then(|a| a.as_int_array().map(|a| a.to_vec()))
                        .unwrap_or_else(|| vec![axis]);
                    let n = ir.add_node(OpType::Reshape);
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_output(n, out_id)?;
                    if let Some(nd) = ir.get_node_mut(n) {
                        nd.add_attribute("unsqueeze_axes".to_string(), Attribute::IntArray(axes));
                    }
                    return Ok(true);
                }
                "Squeeze" => {
                    // Remove dimensions of size 1. Reshape with axes for backend to interpret.
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Squeeze requires one input")
                    })?;
                    let axes: Vec<i64> = Self::get_attr(node, "axes")
                        .and_then(|a| a.as_int_array().map(|a| a.to_vec()))
                        .unwrap_or_default();
                    let n = ir.add_node(OpType::Reshape);
                    ir.add_node_input(n, *x_id)?;
                    ir.add_node_output(n, out_id)?;
                    if let Some(nd) = ir.get_node_mut(n) {
                        nd.add_attribute("squeeze_axes".to_string(), Attribute::IntArray(axes));
                    }
                    return Ok(true);
                }
                "Constant" => {
                    // Constant node: value in attribute "value" (TensorProto). Materialize as weight; wire output name to it (no Identity node).
                    let value_attr = node
                        .attribute
                        .iter()
                        .find(|a| a.name.as_deref() == Some("value"));
                    let tensor_proto = value_attr
                        .and_then(|a| a.t.as_ref())
                        .ok_or_else(|| CoelanoxError::format("Constant node missing 'value' tensor attribute"))?;
                    let is_float = tensor_proto
                        .data_type
                        .as_ref()
                        .map(|dt| *dt == OnnxDataType::FLOAT)
                        .unwrap_or(true);
                    if !is_float {
                        return Err(CoelanoxError::format(
                            "Constant node: only float32 value is supported",
                        ));
                    }
                    let shape = Self::tensor_shape(&tensor_proto.dims);
                    let data = Self::tensor_float_data(tensor_proto)?
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<u8>>();
                    let const_name = format!("{}_const", node_name);
                    let weight_tid = ir.add_weight_tensor_with_data(
                        const_name,
                        shape,
                        DataType::Float32,
                        data,
                    );
                    if let Some(out_name) = node.output.first() {
                        name_to_id.insert(out_name.clone(), weight_tid);
                    }
                    return Ok(true);
                }
                "LSTM" => {
                    // Decompose ONNX LSTM into MatMul, Add, Sigmoid, Tanh, Slice, Split, Multiply.
                    // Supports forward direction only, no peephole. Equations:
                    //   gates = X_t @ W^T + H_prev @ R^T + B
                    //   i,f,g,o = split(gates); i=Sigmoid(i), f=Sigmoid(f), g=Tanh(g), o=Sigmoid(o)
                    //   C_t = f*C_prev + i*g; H_t = o*Tanh(C_t)
                    if input_ids.len() < 3 {
                        return Err(CoelanoxError::format(
                            "LSTM requires at least X, W, R inputs",
                        ));
                    }
                    let x_id = input_ids[0];
                    let w_id = input_ids[1];
                    let r_id = input_ids[2];
                    let b_id = input_ids.get(3).copied();

                    let hidden_size: usize = Self::get_attr(node, "hidden_size")
                        .and_then(|a| a.as_int())
                        .ok_or_else(|| CoelanoxError::format("LSTM requires hidden_size attribute"))? as usize;

                    let direction = Self::get_attr(node, "direction")
                        .and_then(|a| a.as_string().map(|s| s.as_str().to_string()))
                        .unwrap_or_else(|| "forward".to_string());
                    if direction != "forward" {
                        return Err(CoelanoxError::format(format!(
                            "LSTM decomposition supports forward direction only; got '{}'",
                            direction,
                        )));
                    }

                    let x_shape = match ir.get_tensor(x_id) {
                        Some(coelanox_core::ir::TensorRef::Weight(t)) => t.shape.dimensions.clone(),
                        Some(coelanox_core::ir::TensorRef::Intermediate(t)) => t.shape.dimensions.clone(),
                        None => {
                            return Err(CoelanoxError::format("LSTM: cannot get shape for X"));
                        }
                    };
                    let w_shape = match ir.get_tensor(w_id) {
                        Some(coelanox_core::ir::TensorRef::Weight(t)) => t.shape.dimensions.clone(),
                        Some(coelanox_core::ir::TensorRef::Intermediate(t)) => t.shape.dimensions.clone(),
                        None => {
                            return Err(CoelanoxError::format("LSTM: cannot get shape for W"));
                        }
                    };
                    let _r_shape = match ir.get_tensor(r_id) {
                        Some(coelanox_core::ir::TensorRef::Weight(t)) => t.shape.dimensions.clone(),
                        Some(coelanox_core::ir::TensorRef::Intermediate(t)) => t.shape.dimensions.clone(),
                        None => {
                            return Err(CoelanoxError::format("LSTM: cannot get shape for R"));
                        }
                    };

                    // X: [seq_length, batch_size, input_size]
                    // W: [num_directions, 4*hidden_size, input_size]
                    // R: [num_directions, 4*hidden_size, hidden_size]
                    let seq_len = *x_shape.get(0).ok_or_else(|| CoelanoxError::format("LSTM: X needs 3 dims"))?;
                    let batch = *x_shape.get(1).ok_or_else(|| CoelanoxError::format("LSTM: X needs 3 dims"))?;
                    let input_size = *x_shape.get(2).ok_or_else(|| CoelanoxError::format("LSTM: X needs 3 dims"))?;
                    let four_h = *w_shape.get(1).ok_or_else(|| CoelanoxError::format("LSTM: W needs 3 dims"))?;
                    if four_h != 4 * hidden_size {
                        return Err(CoelanoxError::format(&format!(
                            "LSTM: W dim 1 {} != 4*hidden_size {}",
                            four_h,
                            4 * hidden_size
                        )));
                    }

                    // Output IDs: Y (concat of H_t), Y_h (last H), Y_c (last C)
                    let y_id = name_to_id.get(output_names[0]).copied();
                    let y_h_id = output_names.get(1).and_then(|n| name_to_id.get(*n).copied());
                    let y_c_id = output_names.get(2).and_then(|n| name_to_id.get(*n).copied());

                    // Initial H and C (zeros) - shared weight tensor
                    let zero_shape = Shape::new(vec![batch, hidden_size]);
                    let zero_data: Vec<u8> = vec![0u8; batch * hidden_size * 4];
                    let zero_name = format!("{}_lstm_zero", node_name);
                    let zero_tid = ir.add_weight_tensor_with_data(
                        zero_name,
                        zero_shape.clone(),
                        DataType::Float32,
                        zero_data,
                    );

                    let mut h_prev = zero_tid;
                    let mut c_prev = zero_tid;
                    let mut h_outputs: Vec<TensorId> = Vec::with_capacity(seq_len);

                    let x_t_shape = Shape::new(vec![batch, input_size]);
                    let gates_shape = Shape::new(vec![batch, 4 * hidden_size]);

                    // Transpose W and R once (outside loop). W: [4*hidden, input] -> [input, 4*hidden]; R: [4*hidden, hidden] -> [hidden, 4*hidden]
                    let w_t_name = format!("{}_lstm_wt", node_name);
                    let r_t_name = format!("{}_lstm_rt", node_name);
                    let w_t_id = ir.add_intermediate_tensor(
                        w_t_name.clone(),
                        Shape::new(vec![input_size, 4 * hidden_size]),
                        DataType::Float32,
                    );
                    let r_t_id = ir.add_intermediate_tensor(
                        r_t_name.clone(),
                        Shape::new(vec![hidden_size, 4 * hidden_size]),
                        DataType::Float32,
                    );
                    name_to_id.insert(w_t_name, w_t_id);
                    name_to_id.insert(r_t_name, r_t_id);

                    let transp_w = ir.add_node(OpType::Transpose);
                    ir.add_node_input(transp_w, w_id)?;
                    ir.add_node_output(transp_w, w_t_id)?;
                    if let Some(nd) = ir.get_node_mut(transp_w) {
                        nd.add_attribute("perm".to_string(), Attribute::IntArray(vec![1, 0]));
                    }
                    let transp_r = ir.add_node(OpType::Transpose);
                    ir.add_node_input(transp_r, r_id)?;
                    ir.add_node_output(transp_r, r_t_id)?;
                    if let Some(nd) = ir.get_node_mut(transp_r) {
                        nd.add_attribute("perm".to_string(), Attribute::IntArray(vec![1, 0]));
                    }

                    for t in 0..seq_len {
                        let prefix = format!("{}_lstm_t{}", node_name, t);

                        // Slice X[t,:,:] -> X_t [batch, input_size]
                        let x_t_name = format!("{}_xt", prefix);
                        let x_t_id = ir.add_intermediate_tensor(x_t_name.clone(), x_t_shape.clone(), DataType::Float32);
                        name_to_id.insert(x_t_name, x_t_id);

                        let slice_n = ir.add_node(OpType::Slice);
                        ir.add_node_input(slice_n, x_id)?;
                        ir.add_node_output(slice_n, x_t_id)?;
                        if let Some(nd) = ir.get_node_mut(slice_n) {
                            nd.add_attribute("starts".to_string(), Attribute::IntArray(vec![t as i64, 0, 0]));
                            nd.add_attribute("ends".to_string(), Attribute::IntArray(vec![(t + 1) as i64, batch as i64, input_size as i64]));
                            nd.add_attribute("axes".to_string(), Attribute::IntArray(vec![0, 1, 2]));
                            nd.add_attribute("input_shape".to_string(), Attribute::IntArray(vec![seq_len as i64, batch as i64, input_size as i64]));
                        }

                        // gates_xt = X_t @ W^T (W^T is w_t_id)
                        let gates_xt_name = format!("{}_gates_xt", prefix);
                        let gates_xt_id = ir.add_intermediate_tensor(gates_xt_name.clone(), gates_shape.clone(), DataType::Float32);
                        name_to_id.insert(gates_xt_name, gates_xt_id);

                        let mm1 = ir.add_node(OpType::MatMul);
                        ir.add_node_input(mm1, x_t_id)?;
                        ir.add_node_input(mm1, w_t_id)?;
                        ir.add_node_output(mm1, gates_xt_id)?;

                        // gates_h = H_prev @ R^T (R^T is r_t_id)
                        let gates_h_name = format!("{}_gates_h", prefix);
                        let gates_h_id = ir.add_intermediate_tensor(gates_h_name.clone(), gates_shape.clone(), DataType::Float32);
                        name_to_id.insert(gates_h_name, gates_h_id);

                        let mm2 = ir.add_node(OpType::MatMul);
                        ir.add_node_input(mm2, h_prev)?;
                        ir.add_node_input(mm2, r_t_id)?;
                        ir.add_node_output(mm2, gates_h_id)?;

                        // gates = gates_xt + gates_h + B
                        let gates_pre_name = format!("{}_gates_pre", prefix);
                        let gates_pre_id = ir.add_intermediate_tensor(gates_pre_name.clone(), gates_shape.clone(), DataType::Float32);
                        name_to_id.insert(gates_pre_name, gates_pre_id);

                        let add1 = ir.add_node(OpType::Add);
                        ir.add_node_input(add1, gates_xt_id)?;
                        ir.add_node_input(add1, gates_h_id)?;
                        ir.add_node_output(add1, gates_pre_id)?;

                        let gates_id = if let Some(bid) = b_id {
                            let gates_name = format!("{}_gates", prefix);
                            let gid = ir.add_intermediate_tensor(gates_name.clone(), gates_shape.clone(), DataType::Float32);
                            name_to_id.insert(gates_name, gid);
                            let add_b = ir.add_node(OpType::Add);
                            ir.add_node_input(add_b, gates_pre_id)?;
                            ir.add_node_input(add_b, bid)?;
                            ir.add_node_output(add_b, gid)?;
                            gid
                        } else {
                            gates_pre_id
                        };

                        // Split gates into i, f, g, o
                        let i_name = format!("{}_i", prefix);
                        let f_name = format!("{}_f", prefix);
                        let g_name = format!("{}_g", prefix);
                        let o_name = format!("{}_o", prefix);
                        let part_shape = Shape::new(vec![batch, hidden_size]);
                        let i_id = ir.add_intermediate_tensor(i_name.clone(), part_shape.clone(), DataType::Float32);
                        let f_id = ir.add_intermediate_tensor(f_name.clone(), part_shape.clone(), DataType::Float32);
                        let g_id = ir.add_intermediate_tensor(g_name.clone(), part_shape.clone(), DataType::Float32);
                        let o_id = ir.add_intermediate_tensor(o_name.clone(), part_shape.clone(), DataType::Float32);
                        name_to_id.insert(i_name, i_id);
                        name_to_id.insert(f_name, f_id);
                        name_to_id.insert(g_name, g_id);
                        name_to_id.insert(o_name, o_id);

                        let split_n = ir.add_node(OpType::Split);
                        ir.add_node_input(split_n, gates_id)?;
                        ir.add_node_output(split_n, i_id)?;
                        ir.add_node_output(split_n, f_id)?;
                        ir.add_node_output(split_n, g_id)?;
                        ir.add_node_output(split_n, o_id)?;
                        if let Some(nd) = ir.get_node_mut(split_n) {
                            nd.add_attribute("split".to_string(), Attribute::IntArray(vec![
                                hidden_size as i64,
                                hidden_size as i64,
                                hidden_size as i64,
                                hidden_size as i64,
                            ]));
                        }

                        // i_act = Sigmoid(i), f_act = Sigmoid(f), g_act = Tanh(g), o_act = Sigmoid(o)
                        let i_act_name = format!("{}_i_act", prefix);
                        let f_act_name = format!("{}_f_act", prefix);
                        let g_act_name = format!("{}_g_act", prefix);
                        let o_act_name = format!("{}_o_act", prefix);
                        let i_act_id = ir.add_intermediate_tensor(i_act_name.clone(), part_shape.clone(), DataType::Float32);
                        let f_act_id = ir.add_intermediate_tensor(f_act_name.clone(), part_shape.clone(), DataType::Float32);
                        let g_act_id = ir.add_intermediate_tensor(g_act_name.clone(), part_shape.clone(), DataType::Float32);
                        let o_act_id = ir.add_intermediate_tensor(o_act_name.clone(), part_shape.clone(), DataType::Float32);
                        name_to_id.insert(i_act_name, i_act_id);
                        name_to_id.insert(f_act_name, f_act_id);
                        name_to_id.insert(g_act_name, g_act_id);
                        name_to_id.insert(o_act_name, o_act_id);

                        let sig_i = ir.add_node(OpType::Sigmoid);
                        ir.add_node_input(sig_i, i_id)?;
                        ir.add_node_output(sig_i, i_act_id)?;
                        let sig_f = ir.add_node(OpType::Sigmoid);
                        ir.add_node_input(sig_f, f_id)?;
                        ir.add_node_output(sig_f, f_act_id)?;
                        let tanh_g = ir.add_node(OpType::Tanh);
                        ir.add_node_input(tanh_g, g_id)?;
                        ir.add_node_output(tanh_g, g_act_id)?;
                        let sig_o = ir.add_node(OpType::Sigmoid);
                        ir.add_node_input(sig_o, o_id)?;
                        ir.add_node_output(sig_o, o_act_id)?;

                        // C_t = f*C_prev + i*g
                        let f_c_name = format!("{}_f_c", prefix);
                        let i_g_name = format!("{}_i_g", prefix);
                        let f_c_id = ir.add_intermediate_tensor(f_c_name.clone(), part_shape.clone(), DataType::Float32);
                        let i_g_id = ir.add_intermediate_tensor(i_g_name.clone(), part_shape.clone(), DataType::Float32);
                        name_to_id.insert(f_c_name, f_c_id);
                        name_to_id.insert(i_g_name, i_g_id);

                        let mul_fc = ir.add_node(OpType::Multiply);
                        ir.add_node_input(mul_fc, f_act_id)?;
                        ir.add_node_input(mul_fc, c_prev)?;
                        ir.add_node_output(mul_fc, f_c_id)?;
                        let mul_ig = ir.add_node(OpType::Multiply);
                        ir.add_node_input(mul_ig, i_act_id)?;
                        ir.add_node_input(mul_ig, g_act_id)?;
                        ir.add_node_output(mul_ig, i_g_id)?;

                        let use_y_h = (t == seq_len - 1) && y_h_id.is_some();
                        let use_y_c = (t == seq_len - 1) && y_c_id.is_some();
                        let use_y = (t == seq_len - 1) && seq_len == 1 && y_id.is_some();

                        let c_t_name = format!("{}_ct", prefix);
                        let c_out_id = if use_y_c {
                            y_c_id.unwrap()
                        } else {
                            let cid = ir.add_intermediate_tensor(c_t_name.clone(), part_shape.clone(), DataType::Float32);
                            name_to_id.insert(c_t_name, cid);
                            cid
                        };

                        let add_ct = ir.add_node(OpType::Add);
                        ir.add_node_input(add_ct, f_c_id)?;
                        ir.add_node_input(add_ct, i_g_id)?;
                        ir.add_node_output(add_ct, c_out_id)?;

                        // H_t = o * Tanh(C_t)
                        let tanh_ct_name = format!("{}_tanh_ct", prefix);
                        let tanh_ct_id = ir.add_intermediate_tensor(tanh_ct_name.clone(), part_shape.clone(), DataType::Float32);
                        name_to_id.insert(tanh_ct_name, tanh_ct_id);

                        let tanh_ct = ir.add_node(OpType::Tanh);
                        ir.add_node_input(tanh_ct, c_out_id)?;
                        ir.add_node_output(tanh_ct, tanh_ct_id)?;

                        let h_out_id = if use_y { y_id.unwrap() } else if use_y_h { y_h_id.unwrap() } else {
                            let h_t_name = format!("{}_ht", prefix);
                            let hid = ir.add_intermediate_tensor(h_t_name.clone(), part_shape.clone(), DataType::Float32);
                            name_to_id.insert(h_t_name, hid);
                            hid
                        };

                        let mul_ht = ir.add_node(OpType::Multiply);
                        ir.add_node_input(mul_ht, o_act_id)?;
                        ir.add_node_input(mul_ht, tanh_ct_id)?;
                        ir.add_node_output(mul_ht, h_out_id)?;

                        h_outputs.push(h_out_id);
                        h_prev = h_out_id;
                        c_prev = c_out_id;
                    }

                    // Wire Y output: concat of H_t when seq_len > 1
                    if let Some(yid) = y_id {
                        if seq_len > 1 {
                            // Unsqueeze each h_t to [1, batch, hidden], then Concat along axis 0 -> [seq_len, batch, hidden]
                            let unsqueezed: Vec<TensorId> = h_outputs
                                .iter()
                                .enumerate()
                                .map(|(i, &hid)| {
                                    let u_name = format!("{}_lstm_y_u{}", node_name, i);
                                    let u_shape = Shape::new(vec![1, batch, hidden_size]);
                                    let u_id = ir.add_intermediate_tensor(u_name.clone(), u_shape, DataType::Float32);
                                    name_to_id.insert(u_name, u_id);
                                    let n = ir.add_node(OpType::Reshape);
                                    ir.add_node_input(n, hid).unwrap();
                                    ir.add_node_output(n, u_id).unwrap();
                                    if let Some(nd) = ir.get_node_mut(n) {
                                        nd.add_attribute("unsqueeze_axes".to_string(), Attribute::IntArray(vec![0]));
                                    }
                                    u_id
                                })
                                .collect();
                            let concat_n = ir.add_node(OpType::Concatenate);
                            for &uid in &unsqueezed {
                                ir.add_node_input(concat_n, uid)?;
                            }
                            ir.add_node_output(concat_n, yid)?;
                            if let Some(nd) = ir.get_node_mut(concat_n) {
                                nd.add_attribute("axis".to_string(), Attribute::Int(0));
                            }
                        }
                    }

                    return Ok(true);
                }
                _ => {}
            }
            Ok(false)
        }

        fn build_ir(model: ModelProto, external_weights: Option<&[u8]>) -> CoelanoxResult<UniversalIR> {
            let graph = model
                .graph
                .as_ref()
                .ok_or_else(|| CoelanoxError::format("ONNX model has no graph"))?;

            let name = model
                .producer_name
                .as_deref()
                .unwrap_or("onnx_model")
                .to_string();

            let mut ir = UniversalIR::new(name);
            let mut name_to_id: HashMap<String, TensorId> = HashMap::new();
            let mut shape_constants: HashMap<String, Vec<i64>> = HashMap::new();
            let mut external_offset = 0usize;

            // 1) Initializers: float/f16 as weight tensors; int64/int32 as shape constants (for Reshape etc.).
            for init in &graph.initializer {
                let onnx_name = init
                    .name
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("")
                    .to_string();
                if onnx_name.is_empty() || name_to_id.contains_key(&onnx_name) {
                    continue;
                }
                let dtype = init.data_type.as_ref().copied().unwrap_or(OnnxDataType::FLOAT);

                // Non-float initializers (int64, int32): shape constants for Reshape/Slice etc. Skip as weights.
                if dtype == OnnxDataType::INT64 || dtype == OnnxDataType::INT32 {
                    let vals = Self::initializer_int64_data(init)?;
                    shape_constants.insert(onnx_name, vals);
                    continue;
                }

                if dtype != OnnxDataType::FLOAT && dtype != OnnxDataType::FLOAT16 {
                    return Err(CoelanoxError::format(&format!(
                        "Initializer '{}': only float32, float16, int32, and int64 initializers supported",
                        onnx_name
                    )));
                }

                let shape = Self::tensor_shape(&init.dims);
                let floats = Self::initializer_float_data(init, external_weights, &mut external_offset)?;
                let data = floats
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<u8>>();
                let tid = ir.add_weight_tensor_with_data(
                    onnx_name.clone(),
                    shape,
                    DataType::Float32,
                    data,
                );
                name_to_id.insert(onnx_name, tid);
            }

            // 2) Graph inputs not in initializers -> intermediate placeholders.
            for input in &graph.input {
                let onnx_name = input.name.as_deref().unwrap_or("").to_string();
                if onnx_name.is_empty() || name_to_id.contains_key(&onnx_name) {
                    continue;
                }
                let shape = input
                    .r#type
                    .as_ref()
                    .and_then(|tp| tp.tensor_type.as_ref())
                    .and_then(|tt| tt.shape.as_ref())
                    .map(|s| {
                        let dims = s
                            .dim
                            .iter()
                            .map(|d| d.dim_value.unwrap_or(1).max(0) as usize)
                            .collect();
                        Shape::new(dims)
                    })
                    .unwrap_or_else(|| Shape::new(vec![1]));
                let tid = ir.add_intermediate_tensor(
                    onnx_name.clone(),
                    shape.clone(),
                    DataType::Float32,
                );
                ir.metadata.add_input_shape(shape);
                name_to_id.insert(onnx_name, tid);
            }

            // 3) One intermediate per node output.
            for node in &graph.node {
                for out_name in &node.output {
                    let s = out_name.clone();
                    if s.is_empty() || name_to_id.contains_key(&s) {
                        continue;
                    }
                    let tid = ir.add_intermediate_tensor(
                        s.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(s, tid);
                }
            }

            // Output shapes in metadata.
            for out in &graph.output {
                let shape = out
                    .r#type
                    .as_ref()
                    .and_then(|tp| tp.tensor_type.as_ref())
                    .and_then(|tt| tt.shape.as_ref())
                    .map(|s| {
                        let dims = s
                            .dim
                            .iter()
                            .map(|d| d.dim_value.unwrap_or(1).max(0) as usize)
                            .collect();
                        Shape::new(dims)
                    })
                    .unwrap_or_else(|| Shape::new(vec![1]));
                ir.metadata.add_output_shape(shape);
            }

            // 4) Nodes: map op type, wire inputs/outputs (or decompose into subgraph).
            for node in &graph.node {
                let op_type_str = node.op_type.as_deref().unwrap_or("Unknown");

                if let Some(msg) = Self::onnx_op_rejected_at_package_time(op_type_str) {
                    return Err(CoelanoxError::format(msg));
                }

                // Identity: rewire output name to input tensor (no node; avoids dead code).
                if op_type_str == "Identity" {
                    if node.input.len() >= 1 && node.output.len() >= 1 {
                        if let Some(&tid) = name_to_id.get(&node.input[0]) {
                            name_to_id.insert(node.output[0].clone(), tid);
                        }
                    }
                    continue;
                }

                // Decompose composite ops into IR primitives.
                if Self::decompose_node(&mut ir, &mut name_to_id, node, op_type_str)? {
                    continue;
                }

                let op_type = Self::map_op_type(op_type_str);
                let node_id = ir.add_node(op_type);

                // Wire inputs. For Reshape, skip shape input (int64 constant) and add as attribute.
                let is_reshape = op_type_str == "Reshape";
                for (ix, input_name) in node.input.iter().enumerate() {
                    if is_reshape && ix == 1 {
                        if let Some(shape_vals) = shape_constants.get(input_name) {
                            if let Some(n) = ir.get_node_mut(node_id) {
                                n.add_attribute("shape".to_string(), Attribute::IntArray(shape_vals.clone()));
                            }
                        }
                        continue;
                    }
                    if let Some(&tid) = name_to_id.get(input_name) {
                        let _ = ir.add_node_input(node_id, tid);
                    }
                }
                for output_name in &node.output {
                    if let Some(&tid) = name_to_id.get(output_name) {
                        let _ = ir.add_node_output(node_id, tid);
                    }
                }

                for attr in &node.attribute {
                    if let Some((k, v)) = convert_attr(attr) {
                        if let Some(n) = ir.get_node_mut(node_id) {
                            n.add_attribute(k, v);
                        }
                    }
                }
            }

            // 5) Any remaining Custom op means the translator has no lowering — fail at package time
            // with a specific or generic message (never rely on the runtime scalar backend for Custom).
            for node in &ir.nodes {
                if let OpType::Custom(name) = &node.op_type {
                    return Err(CoelanoxError::format(Self::unimplemented_custom_onnx_op_message(name)));
                }
            }

            ir.compute_topological_order()?;
            Ok(ir)
        }
    }

    fn convert_attr(attr: &rten_onnx::onnx::AttributeProto) -> Option<(String, Attribute)> {
        use rten_onnx::onnx::AttributeType;
        let name = attr.name.as_ref()?.clone();
        let value = match attr.r#type.as_ref()? {
            t if *t == AttributeType::FLOAT => Attribute::Float(attr.f.unwrap_or(0.0)),
            t if *t == AttributeType::INT => Attribute::Int(attr.i.unwrap_or(0)),
            t if *t == AttributeType::STRING => {
                Attribute::String(attr.s.as_ref().cloned().unwrap_or_default())
            }
            t if *t == AttributeType::FLOATS => Attribute::FloatArray(attr.floats.clone()),
            t if *t == AttributeType::INTS => Attribute::IntArray(attr.ints.clone()),
            _ => return None,
        };
        Some((name, value))
    }

    /// Forwards ONNX Reduce* attributes onto an IR [`ReduceSum`] / reduction node.
    fn forward_onnx_reduce_attrs(
        ir: &mut UniversalIR,
        node_id: NodeId,
        onnx_node: &rten_onnx::onnx::NodeProto,
    ) {
        for attr in &onnx_node.attribute {
            if let Some((k, v)) = convert_attr(attr) {
                match k.as_str() {
                    "axes" | "keepdims" | "noop_with_empty_axes" => {
                        if let Some(n) = ir.get_node_mut(node_id) {
                            n.add_attribute(k, v);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    impl Default for OnnxTranslator {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CoelanoxTranslator for OnnxTranslator {
        fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
            let model = Self::parse_model(model_data)?;
            Self::build_ir(model, None)
        }

        fn translate_from_path_with_weights(&self, input_path: &Path, external_weights_path: Option<&Path>) -> CoelanoxResult<UniversalIR> {
            let model_data = std::fs::read(input_path)
                .map_err(|e| CoelanoxError::format(&format!("Failed to read model file {}: {}", input_path.display(), e)))?;
            let model = Self::parse_model(&model_data)?;
            let external_bytes = match external_weights_path {
                None => None,
                Some(p) => {
                    let bytes = std::fs::read(p)
                        .map_err(|e| CoelanoxError::format(&format!("Failed to read weights file {}: {}", p.display(), e)))?;
                    Some(bytes)
                }
            };
            // build_ir needs Option<&[u8]>; we own the Vec in external_bytes so pass a reference.
            let ext_ref = external_bytes.as_deref();
            Self::build_ir(model, ext_ref)
        }

        fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
            self.extract_weights_with_external(model_data, None)
        }

        fn extract_weights_with_external(&self, model_data: &[u8], external_weights_path: Option<&Path>) -> CoelanoxResult<Vec<f32>> {
            let model = Self::parse_model(model_data)?;
            let graph = model
                .graph
                .as_ref()
                .ok_or_else(|| CoelanoxError::format("ONNX model has no graph"))?;
            let external_bytes = match external_weights_path {
                None => None,
                Some(p) => {
                    let bytes = std::fs::read(p)
                        .map_err(|e| CoelanoxError::format(&format!("Failed to read weights file {}: {}", p.display(), e)))?;
                    Some(bytes)
                }
            };
            let ext_ref = external_bytes.as_deref();
            let mut external_offset = 0usize;
            let mut out = Vec::new();
            for init in &graph.initializer {
                let dtype = init.data_type.as_ref().copied().unwrap_or(OnnxDataType::FLOAT);
                if dtype == OnnxDataType::INT64 || dtype == OnnxDataType::INT32 {
                    continue;
                }
                if dtype != OnnxDataType::FLOAT && dtype != OnnxDataType::FLOAT16 {
                    continue;
                }
                let floats = Self::initializer_float_data(init, ext_ref, &mut external_offset)?;
                out.extend(floats);
            }
            Ok(out)
        }

        fn supported_formats(&self) -> Vec<ModelFormat> {
            vec![ModelFormat::Onnx]
        }

        fn name(&self) -> &str {
            "onnx"
        }

        fn version(&self) -> &str {
            "0.1.0"
        }

        fn author(&self) -> &str {
            "Coelanox"
        }

        fn description(&self) -> &str {
            "ONNX translator (opset 13); maps ONNX ops to UniversalIR."
        }
    }
}

#[cfg(not(feature = "onnx-protobuf"))]
mod impl_ {
    use coelanox_core::ir::UniversalIR;
    use coelanox_core::plugins::CoelanoxTranslator;
    use coelanox_core::format::ModelFormat;
    use coelanox_core::{CoelanoxError, CoelanoxResult};

    pub struct OnnxTranslator;

    impl OnnxTranslator {
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for OnnxTranslator {
        fn default() -> Self {
            Self::new()
        }
    }

    impl CoelanoxTranslator for OnnxTranslator {
        fn translate(&self, _model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
            Err(CoelanoxError::format(
                "ONNX translator requires the 'onnx-protobuf' feature to be enabled",
            ))
        }

        fn extract_weights(&self, _model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
            Err(CoelanoxError::format(
                "ONNX translator requires the 'onnx-protobuf' feature to be enabled",
            ))
        }

        fn supported_formats(&self) -> Vec<ModelFormat> {
            vec![ModelFormat::Onnx]
        }

        fn name(&self) -> &str {
            "onnx"
        }

        fn version(&self) -> &str {
            "0.1.0"
        }

        fn author(&self) -> &str {
            "Coelanox"
        }
    }
}

pub use impl_::OnnxTranslator;
