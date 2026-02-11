//! ResNet-tiny-MNIST translator: extracts ops from the HuggingFace-style model
//! (config + PyTorch state_dict) and builds Coelanox UniversalIR.
//!
//! Expects a single-file bundle produced by the Python export script:
//! magic "RTM1" | config_len (u32 LE) | config_json | num_tensors (u32) |
//! for each tensor: ndim (u32), dims[ndim] (u32 each), data (f32 LE).

use coelanox_core::ir::{
    Attribute, DataType, GraphNode, OpType, Shape, UniversalIR,
};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use coelanox_core::plugins::{PluginConfig, PluginMetadata, PluginCapabilities, TranslatorPlugin};
use std::collections::HashMap;
use std::path::Path;

/// Magic bytes for ResNet-tiny-MNIST bundle (v1).
const BUNDLE_MAGIC: &[u8; 4] = b"RTM1";

/// ResNet-tiny-MNIST translator plugin.
pub struct ResNetTinyMnistTranslator;

impl ResNetTinyMnistTranslator {
    pub fn new() -> Self {
        Self
    }

    /// Parse bundle: after magic, [config_len (u32), config_bytes, num_tensors, tensors...].
    fn parse_bundle(data: &[u8]) -> CoelanoxResult<(ResNetConfig, Vec<WeightBlob>)> {
        if data.len() < 4 + 4 {
            return Err(CoelanoxError::format("Bundle too short"));
        }
        let config_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let config_start = 8;
        let config_end = config_start + config_len;
        if data.len() < config_end + 4 {
            return Err(CoelanoxError::format("Bundle truncated at config"));
        }
        let config_json: &[u8] = &data[config_start..config_end];
        let config: ResNetConfig = serde_json::from_slice(config_json)
            .map_err(|e| CoelanoxError::format(&format!("Invalid config JSON: {}", e)))?;

        let mut pos = config_end;
        let num_tensors = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
        ]) as usize;
        pos += 4;

        let mut weights = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            if pos + 4 > data.len() {
                return Err(CoelanoxError::format("Bundle truncated at tensor header"));
            }
            let ndim = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            ]) as usize;
            pos += 4;
            let dims_len = ndim * 4;
            if pos + dims_len > data.len() {
                return Err(CoelanoxError::format("Bundle truncated at tensor dims"));
            }
            let mut dims = Vec::with_capacity(ndim);
            for d in 0..ndim {
                let d_start = pos + d * 4;
                dims.push(u32::from_le_bytes([
                    data[d_start], data[d_start + 1], data[d_start + 2], data[d_start + 3],
                ]) as usize);
            }
            pos += dims_len;
            let elements: usize = dims.iter().product();
            let bytes = elements * 4;
            if pos + bytes > data.len() {
                return Err(CoelanoxError::format("Bundle truncated at tensor data"));
            }
            let mut blob = Vec::with_capacity(elements);
            for i in 0..elements {
                let o = pos + i * 4;
                blob.push(f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]));
            }
            pos += bytes;
            weights.push(WeightBlob { shape: dims, data: blob });
        }

        Ok((config, weights))
    }

    /// Build UniversalIR from parsed config and weight blobs.
    /// Infers dimensions from the first and key weight shapes so bundles with different configs (e.g. 32 vs 64 stem) work.
    fn build_ir(config: &ResNetConfig, weights: Vec<WeightBlob>) -> CoelanoxResult<UniversalIR> {
        let depths = &config.depths;
        if depths.len() < 2 {
            return Err(CoelanoxError::format("Config must have depths len >= 2"));
        }
        // Infer from bundle weights so export can use any compatible ResNet-tiny layout
        let (c0, num_channels) = {
            let w = weights.get(0).ok_or_else(|| CoelanoxError::format("Bundle has no weights"))?;
            if w.shape.len() != 4 {
                return Err(CoelanoxError::format("Stem conv weight must be 4D [out_ch, in_ch, 7, 7]"));
            }
            (w.shape[0], w.shape[1])
        };
        // Stage1 block0 conv1 index: with downsample (62 tensors) or without (56 tensors)
        let has_downsample = weights.len() == 6 + 12 * depths[0] as usize + 6 + 12 * depths[1] as usize + 2;
        let stage1_conv1_idx = 6 + 12 * depths[0] as usize + if has_downsample { 6 } else { 0 };
        let c1 = weights.get(stage1_conv1_idx)
            .ok_or_else(|| CoelanoxError::format("Bundle missing stage1 block0 conv1 weight"))?
            .shape[0];
        let (num_labels, embedding_size) = {
            let w = weights.get(weights.len().saturating_sub(2))
                .ok_or_else(|| CoelanoxError::format("Bundle missing classifier weight"))?;
            if w.shape.len() != 2 {
                return Err(CoelanoxError::format("Classifier weight must be 2D [num_labels, embedding_size]"));
            }
            (w.shape[0], w.shape[1])
        };

        let mut ir = UniversalIR::new("resnet_tiny_mnist".to_string());
        ir.metadata.add_input_shape(Shape::new(vec![1, num_channels as usize, 28, 28]));
        ir.metadata.add_output_shape(Shape::new(vec![1, num_labels]));

        let mut weight_index = 0usize;

        // Input tensor (intermediate).
        let input_id = ir.add_intermediate_tensor(
            "input".to_string(),
            Shape::new(vec![1, num_channels as usize, 28, 28]),
            DataType::Float32,
        );

        // --- Stem: Conv -> BN -> ReLU ---
        let stem_conv_w = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.conv.weight")?;
        let stem_conv_b = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.conv.bias")?;
        let stem_conv = ir.add_node(OpType::Convolution);
        ir.add_node_input(stem_conv, input_id)?;
        ir.add_node_input(stem_conv, stem_conv_w)?;
        ir.add_node_input(stem_conv, stem_conv_b)?;
        let stem_conv_attr = ir.get_node_mut(stem_conv).unwrap();
        stem_conv_attr.add_attribute("kernel_size".to_string(), Attribute::IntArray(vec![7, 7]));
        stem_conv_attr.add_attribute("stride".to_string(), Attribute::IntArray(vec![2, 2]));
        stem_conv_attr.add_attribute("padding".to_string(), Attribute::IntArray(vec![3, 3]));

        let stem_conv_out = ir.add_intermediate_tensor(
            "stem.conv.out".to_string(),
            Shape::new(vec![1, c0, 14, 14]),
            DataType::Float32,
        );
        ir.add_node_output(stem_conv, stem_conv_out)?;

        let stem_bn_g = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.bn.weight")?;
        let stem_bn_b = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.bn.bias")?;
        let stem_bn_mean = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.bn.running_mean")?;
        let stem_bn_var = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "stem.bn.running_var")?;
        let stem_bn = ir.add_node(OpType::BatchNorm);
        ir.add_node_input(stem_bn, stem_conv_out)?;
        ir.add_node_input(stem_bn, stem_bn_g)?;
        ir.add_node_input(stem_bn, stem_bn_b)?;
        ir.add_node_input(stem_bn, stem_bn_mean)?;
        ir.add_node_input(stem_bn, stem_bn_var)?;
        let stem_bn_out = ir.add_intermediate_tensor("stem.bn.out".to_string(), Shape::new(vec![1, c0, 14, 14]), DataType::Float32);
        ir.add_node_output(stem_bn, stem_bn_out)?;
        ir.get_node_mut(stem_bn).unwrap().add_attribute("epsilon".to_string(), Attribute::Float(1e-5));

        let stem_relu = ir.add_node(OpType::Relu);
        ir.add_node_input(stem_relu, stem_bn_out)?;
        let stem_relu_out = ir.add_intermediate_tensor("stem.relu.out".to_string(), Shape::new(vec![1, c0, 14, 14]), DataType::Float32);
        ir.add_node_output(stem_relu, stem_relu_out)?;

        // --- Stage 0: 2 basic blocks ---
        let (s0_out, _) = add_basic_blocks(&mut ir, 2, c0, c0, 14, 14, stem_relu_out, &weights, &mut weight_index, "stage0", false)?;

        // --- Stage 1: 2 basic blocks (optional downsample when c0 != c1) ---
        let (h1, w1) = if c0 != c1 { (7, 7) } else { (14, 14) };
        let (s1_out, _) = add_basic_blocks(&mut ir, 2, c0, c1, 14, h1, s0_out, &weights, &mut weight_index, "stage1", has_downsample)?;

        // --- Global average pool: [1,64,7,7] -> [1,64] ---
        let pool_out = ir.add_intermediate_tensor("pool.out".to_string(), Shape::new(vec![1, embedding_size as usize]), DataType::Float32);
        let pool_node = ir.add_node(OpType::GlobalAvgPool);
        ir.add_node_input(pool_node, s1_out)?;
        ir.add_node_output(pool_node, pool_out)?;

        // --- Classifier: Linear 64 -> 10 (MatMul + Add) ---
        let fc_w = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "classifier.weight")?;
        let fc_b = add_weight_from_blobs(&mut ir, &mut weight_index, &weights, "classifier.bias")?;
        let matmul_node = ir.add_node(OpType::MatMul);
        ir.add_node_input(matmul_node, pool_out)?;
        ir.add_node_input(matmul_node, fc_w)?;
        let fc_hidden = ir.add_intermediate_tensor("classifier.hidden".to_string(), Shape::new(vec![1, num_labels]), DataType::Float32);
        ir.add_node_output(matmul_node, fc_hidden)?;
        let add_node = ir.add_node(OpType::Add);
        ir.add_node_input(add_node, fc_hidden)?;
        ir.add_node_input(add_node, fc_b)?;
        let logits = ir.add_intermediate_tensor("logits".to_string(), Shape::new(vec![1, num_labels]), DataType::Float32);
        ir.add_node_output(add_node, logits)?;

        if weight_index != weights.len() {
            return Err(CoelanoxError::format(&format!(
                "Weight count mismatch: used {}, total {}",
                weight_index, weights.len()
            )));
        }

        ir.compute_topological_order()?;
        Ok(ir)
    }
}

/// Add basic blocks for a stage. Each block: conv1->bn->relu->conv2->bn->add(residual)->relu.
/// When has_downsample is true and first block has in_ch != out_ch, consumes 6 downsample weights first.
fn add_basic_blocks(
    ir: &mut UniversalIR,
    num_blocks: usize,
    in_ch: usize,
    out_ch: usize,
    h_in: usize,
    h_out: usize,
    mut prev: coelanox_core::ir::TensorId,
    weights: &[WeightBlob],
    weight_index: &mut usize,
    stage_name: &str,
    has_downsample: bool,
) -> CoelanoxResult<(coelanox_core::ir::TensorId, usize)> {
    let mut w = *weight_index;

    for b in 0..num_blocks {
        let (cin, cout) = if b == 0 { (in_ch, out_ch) } else { (out_ch, out_ch) };
        let (hin, hout) = if b == 0 && in_ch != out_ch { (h_in, h_out) } else { (h_out, h_out) };

        // conv1
        let conv1_w = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let conv1_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let conv1_w_id = ir.add_weight_tensor_with_data(
            format!("{}.block{}.conv1.weight", stage_name, b),
            Shape::new(conv1_w.shape.clone()),
            DataType::Float32,
            conv1_w.data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        );
        let conv1_b_id = ir.add_weight_tensor_with_data(
            format!("{}.block{}.conv1.bias", stage_name, b),
            Shape::new(conv1_b.shape.clone()),
            DataType::Float32,
            conv1_b.data.iter().flat_map(|f| f.to_le_bytes()).collect(),
        );
        let conv1 = ir.add_node(OpType::Convolution);
        ir.add_node_input(conv1, prev)?;
        ir.add_node_input(conv1, conv1_w_id)?;
        ir.add_node_input(conv1, conv1_b_id)?;
        ir.get_node_mut(conv1).unwrap().add_attribute("kernel_size".to_string(), Attribute::IntArray(vec![3, 3]));
        ir.get_node_mut(conv1).unwrap().add_attribute("stride".to_string(), Attribute::IntArray(vec![if b == 0 && in_ch != out_ch { 2 } else { 1 }, if b == 0 && in_ch != out_ch { 2 } else { 1 }]));
        ir.get_node_mut(conv1).unwrap().add_attribute("padding".to_string(), Attribute::IntArray(vec![1, 1]));
        let conv1_out = ir.add_intermediate_tensor(
            format!("{}.block{}.conv1.out", stage_name, b),
            Shape::new(vec![1, cout, hout, hout]),
            DataType::Float32,
        );
        ir.add_node_output(conv1, conv1_out)?;

        // bn1
        let bn1_g = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn1_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn1_m = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn1_v = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn1_g_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn1.weight", stage_name, b), Shape::new(bn1_g.shape.clone()), DataType::Float32, bn1_g.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn1_b_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn1.bias", stage_name, b), Shape::new(bn1_b.shape.clone()), DataType::Float32, bn1_b.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn1_m_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn1.running_mean", stage_name, b), Shape::new(bn1_m.shape.clone()), DataType::Float32, bn1_m.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn1_v_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn1.running_var", stage_name, b), Shape::new(bn1_v.shape.clone()), DataType::Float32, bn1_v.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn1 = ir.add_node(OpType::BatchNorm);
        ir.add_node_input(bn1, conv1_out)?;
        ir.add_node_input(bn1, bn1_g_id)?;
        ir.add_node_input(bn1, bn1_b_id)?;
        ir.add_node_input(bn1, bn1_m_id)?;
        ir.add_node_input(bn1, bn1_v_id)?;
        let bn1_out = ir.add_intermediate_tensor(format!("{}.block{}.bn1.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(bn1, bn1_out)?;
        ir.get_node_mut(bn1).unwrap().add_attribute("epsilon".to_string(), Attribute::Float(1e-5));

        let relu1 = ir.add_node(OpType::Relu);
        ir.add_node_input(relu1, bn1_out)?;
        let relu1_out = ir.add_intermediate_tensor(format!("{}.block{}.relu1.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(relu1, relu1_out)?;

        // conv2
        let conv2_w = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let conv2_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let conv2_w_id = ir.add_weight_tensor_with_data(format!("{}.block{}.conv2.weight", stage_name, b), Shape::new(conv2_w.shape.clone()), DataType::Float32, conv2_w.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let conv2_b_id = ir.add_weight_tensor_with_data(format!("{}.block{}.conv2.bias", stage_name, b), Shape::new(conv2_b.shape.clone()), DataType::Float32, conv2_b.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let conv2 = ir.add_node(OpType::Convolution);
        ir.add_node_input(conv2, relu1_out)?;
        ir.add_node_input(conv2, conv2_w_id)?;
        ir.add_node_input(conv2, conv2_b_id)?;
        ir.get_node_mut(conv2).unwrap().add_attribute("kernel_size".to_string(), Attribute::IntArray(vec![3, 3]));
        ir.get_node_mut(conv2).unwrap().add_attribute("stride".to_string(), Attribute::IntArray(vec![1, 1]));
        ir.get_node_mut(conv2).unwrap().add_attribute("padding".to_string(), Attribute::IntArray(vec![1, 1]));
        let conv2_out = ir.add_intermediate_tensor(format!("{}.block{}.conv2.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(conv2, conv2_out)?;

        // bn2
        let bn2_g = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn2_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn2_m = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn2_v = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing weight"))?;
        w += 1;
        let bn2_g_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn2.weight", stage_name, b), Shape::new(bn2_g.shape.clone()), DataType::Float32, bn2_g.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn2_b_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn2.bias", stage_name, b), Shape::new(bn2_b.shape.clone()), DataType::Float32, bn2_b.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn2_m_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn2.running_mean", stage_name, b), Shape::new(bn2_m.shape.clone()), DataType::Float32, bn2_m.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn2_v_id = ir.add_weight_tensor_with_data(format!("{}.block{}.bn2.running_var", stage_name, b), Shape::new(bn2_v.shape.clone()), DataType::Float32, bn2_v.data.iter().flat_map(|f| f.to_le_bytes()).collect());
        let bn2 = ir.add_node(OpType::BatchNorm);
        ir.add_node_input(bn2, conv2_out)?;
        ir.add_node_input(bn2, bn2_g_id)?;
        ir.add_node_input(bn2, bn2_b_id)?;
        ir.add_node_input(bn2, bn2_m_id)?;
        ir.add_node_input(bn2, bn2_v_id)?;
        let bn2_out = ir.add_intermediate_tensor(format!("{}.block{}.bn2.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(bn2, bn2_out)?;
        ir.get_node_mut(bn2).unwrap().add_attribute("epsilon".to_string(), Attribute::Float(1e-5));

        // Residual input: prev, or downsample branch when bundle includes it (consume 6 tensors so block weights stay aligned).
        let residual_in = if b == 0 && has_downsample {
            // Downsample: 1x1 conv stride 2 + BN (weights consumed at start of this block below).
            let ds_conv_w = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample weight"))?;
            w += 1;
            let ds_conv_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample bias"))?;
            w += 1;
            let ds_bn_g = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample bn"))?;
            w += 1;
            let ds_bn_b = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample bn"))?;
            w += 1;
            let ds_bn_m = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample bn"))?;
            w += 1;
            let ds_bn_v = weights.get(w).ok_or_else(|| CoelanoxError::format("Missing downsample bn"))?;
            w += 1;
            let ds_conv_w_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.0.weight", stage_name, b), Shape::new(ds_conv_w.shape.clone()), DataType::Float32, ds_conv_w.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_conv_b_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.0.bias", stage_name, b), Shape::new(ds_conv_b.shape.clone()), DataType::Float32, ds_conv_b.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_conv = ir.add_node(OpType::Convolution);
            ir.add_node_input(ds_conv, prev)?;
            ir.add_node_input(ds_conv, ds_conv_w_id)?;
            ir.add_node_input(ds_conv, ds_conv_b_id)?;
            ir.get_node_mut(ds_conv).unwrap().add_attribute("kernel_size".to_string(), Attribute::IntArray(vec![1, 1]));
            let stride = if in_ch != out_ch { 2 } else { 1 };
            ir.get_node_mut(ds_conv).unwrap().add_attribute("stride".to_string(), Attribute::IntArray(vec![stride, stride]));
            ir.get_node_mut(ds_conv).unwrap().add_attribute("padding".to_string(), Attribute::IntArray(vec![0, 0]));
            let ds_conv_out = ir.add_intermediate_tensor(format!("{}.block{}.downsample.conv.out", stage_name, b), Shape::new(vec![1, out_ch, hout, hout]), DataType::Float32);
            ir.add_node_output(ds_conv, ds_conv_out)?;
            let ds_bn_g_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.1.weight", stage_name, b), Shape::new(ds_bn_g.shape.clone()), DataType::Float32, ds_bn_g.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_bn_b_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.1.bias", stage_name, b), Shape::new(ds_bn_b.shape.clone()), DataType::Float32, ds_bn_b.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_bn_m_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.1.running_mean", stage_name, b), Shape::new(ds_bn_m.shape.clone()), DataType::Float32, ds_bn_m.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_bn_v_id = ir.add_weight_tensor_with_data(format!("{}.block{}.downsample.1.running_var", stage_name, b), Shape::new(ds_bn_v.shape.clone()), DataType::Float32, ds_bn_v.data.iter().flat_map(|f| f.to_le_bytes()).collect());
            let ds_bn = ir.add_node(OpType::BatchNorm);
            ir.add_node_input(ds_bn, ds_conv_out)?;
            ir.add_node_input(ds_bn, ds_bn_g_id)?;
            ir.add_node_input(ds_bn, ds_bn_b_id)?;
            ir.add_node_input(ds_bn, ds_bn_m_id)?;
            ir.add_node_input(ds_bn, ds_bn_v_id)?;
            let ds_bn_out = ir.add_intermediate_tensor(format!("{}.block{}.downsample.bn.out", stage_name, b), Shape::new(vec![1, out_ch, hout, hout]), DataType::Float32);
            ir.add_node_output(ds_bn, ds_bn_out)?;
            ir.get_node_mut(ds_bn).unwrap().add_attribute("epsilon".to_string(), Attribute::Float(1e-5));
            ds_bn_out
        } else {
            prev
        };
        let add_node = ir.add_node(OpType::Add);
        ir.add_node_input(add_node, bn2_out)?;
        ir.add_node_input(add_node, residual_in)?;
        let add_out = ir.add_intermediate_tensor(format!("{}.block{}.add.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(add_node, add_out)?;

        let relu2 = ir.add_node(OpType::Relu);
        ir.add_node_input(relu2, add_out)?;
        prev = ir.add_intermediate_tensor(format!("{}.block{}.relu2.out", stage_name, b), Shape::new(vec![1, cout, hout, hout]), DataType::Float32);
        ir.add_node_output(relu2, prev)?;
    }

    *weight_index = w;
    Ok((prev, w))
}

#[derive(Debug, serde::Deserialize)]
struct ResNetConfig {
    #[serde(default)]
    num_channels: Option<u32>,
    #[serde(default)]
    hidden_sizes: Vec<u32>,
    #[serde(default)]
    depths: Vec<u32>,
    #[serde(default)]
    embedding_size: Option<u32>,
    #[serde(default)]
    id2label: Option<HashMap<String, String>>,
}

impl ResNetConfig {
    fn num_labels(&self) -> Option<usize> {
        self.id2label.as_ref().map(|m| m.len())
    }
}

struct WeightBlob {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// Add the next weight tensor from blobs to the IR using the blob's actual shape (bundle is source of truth).
fn add_weight_from_blobs(
    ir: &mut UniversalIR,
    weight_index: &mut usize,
    weights: &[WeightBlob],
    name: &str,
) -> CoelanoxResult<coelanox_core::ir::TensorId> {
    let blob = weights.get(*weight_index)
        .ok_or_else(|| CoelanoxError::format(&format!("Missing weight blob at index {}", weight_index)))?;
    let shape = blob.shape.clone();
    let bytes: Vec<u8> = blob.data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let tensor_id = ir.add_weight_tensor_with_data(
        name.to_string(),
        Shape::new(shape),
        DataType::Float32,
        bytes,
    );
    *weight_index += 1;
    Ok(tensor_id)
}

impl TranslatorPlugin for ResNetTinyMnistTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        if model_data.len() < 4 || &model_data[0..4] != BUNDLE_MAGIC {
            return Err(CoelanoxError::format("Invalid ResNet-tiny-MNIST bundle (bad magic)"));
        }
        let (config, weights) = Self::parse_bundle(model_data)?;
        Self::build_ir(&config, weights)
    }

    fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR> {
        let data = std::fs::read(input_path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read {}: {}", input_path.display(), e)))?;
        self.translate(&data)
    }

    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        if model_data.len() < 4 || &model_data[0..4] != BUNDLE_MAGIC {
            return Err(CoelanoxError::format("Invalid ResNet-tiny-MNIST bundle (bad magic)"));
        }
        let (_, weights) = Self::parse_bundle(model_data)?;
        let flat: Vec<f32> = weights.into_iter().flat_map(|b| b.data).collect();
        Ok(flat)
    }

    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Custom("resnet_tiny_mnist".to_string())]
    }

    fn name(&self) -> &str {
        "resnet_tiny_mnist"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn author(&self) -> &str {
        "Coelanox"
    }

    fn description(&self) -> &str {
        "Translates ResNet-tiny-MNIST (config + PyTorch state_dict bundle) to UniversalIR."
    }

    fn configure(&mut self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }

    fn validate_config(&self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }
}
