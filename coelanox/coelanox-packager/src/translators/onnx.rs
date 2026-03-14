//! ONNX translator: converts ONNX models (opset 13) to Coelanox UniversalIR.
//!
//! Uses rten-onnx for parsing. Supports a subset of ONNX ops that map to existing
//! IR OpTypes. Unsupported ops are translated as Custom(name).

#[cfg(feature = "onnx-protobuf")]
mod impl_ {
    use coelanox_core::ir::{
        Attribute, DataType, OpType, Shape, TensorId, UniversalIR,
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
                "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" | "ReduceSumSquare" => {
                    OpType::Custom(op_type.to_string())
                }
                // Math
                "Sqrt" => OpType::Sqrt,
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
                    // Clip(x, min?, max?) -> Min(Max(x, min), max). Use attributes or default bounds.
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Clip requires at least one input")
                    })?;
                    let min_val = Self::get_attr(node, "min")
                        .and_then(|a| a.as_float())
                        .unwrap_or(f32::MIN);
                    let max_val = Self::get_attr(node, "max")
                        .and_then(|a| a.as_float())
                        .unwrap_or(f32::MAX);
                    let min_name = format!("{}_clip_min", node_name);
                    let max_name = format!("{}_clip_max", node_name);
                    let min_tid = Self::add_scalar_constant(ir, min_name, min_val);
                    let max_tid = Self::add_scalar_constant(ir, max_name, max_val);
                    let mid_name = format!("{}_clip_mid", node_name);
                    let mid_tid = ir.add_intermediate_tensor(
                        mid_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(mid_name, mid_tid);

                    let n_max = ir.add_node(OpType::Max);
                    ir.add_node_input(n_max, *x_id)?;
                    ir.add_node_input(n_max, min_tid)?;
                    ir.add_node_output(n_max, mid_tid)?;

                    let n_min = ir.add_node(OpType::Min);
                    ir.add_node_input(n_min, mid_tid)?;
                    ir.add_node_input(n_min, max_tid)?;
                    ir.add_node_output(n_min, out_id)?;
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
                "Abs" => {
                    // Abs(x) = Sqrt(Pow(x, 2))
                    let x_id = input_ids.first().ok_or_else(|| {
                        CoelanoxError::format("Abs requires one input")
                    })?;
                    let two = Self::add_scalar_constant(
                        ir,
                        format!("{}_abs_two", node_name),
                        2.0,
                    );
                    let mid_name = format!("{}_abs_sq", node_name);
                    let mid_tid = ir.add_intermediate_tensor(
                        mid_name.clone(),
                        Shape::new(vec![1]),
                        DataType::Float32,
                    );
                    name_to_id.insert(mid_name, mid_tid);
                    let n_pow = ir.add_node(OpType::Pow);
                    ir.add_node_input(n_pow, *x_id)?;
                    ir.add_node_input(n_pow, two)?;
                    ir.add_node_output(n_pow, mid_tid)?;
                    let n_sqrt = ir.add_node(OpType::Sqrt);
                    ir.add_node_input(n_sqrt, mid_tid)?;
                    ir.add_node_output(n_sqrt, out_id)?;
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

                // Control flow belongs in the application layer, not in the model IR.
                if op_type_str == "If" || op_type_str == "Loop" {
                    return Err(CoelanoxError::format(
                        "ONNX control flow (If/Loop) is not supported in the model. \
                         Move control flow to the application layer and export a model with a fixed execution path.",
                    ));
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

            // 5) Policy checks on the constructed IR.
            //
            // Treat ONNX Cast as a preprocessing/export concern for now. We don't support
            // mixed-precision execution in the Coelanox runtime: everything runs as f32.
            // If a model still contains Cast, fail fast with a clear, actionable error
            // instead of letting it reach the scalar backend as Custom("Cast").
            for node in &ir.nodes {
                if let OpType::Custom(name) = &node.op_type {
                    if name == "Cast" {
                        return Err(CoelanoxError::format(
                            "ONNX Cast is not supported in the Coelanox runtime. \
                             This beta build executes models in a single numeric type (f32). \
                             Please normalize your model to f32 (e.g., convert weights and activations) \
                             or remove Cast in your preprocessing/export pipeline before packaging.",
                        ));
                    }
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
