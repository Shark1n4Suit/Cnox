//! BERT bundle translator: reads BRT1 bundle (config + weight tensors) and builds Coelanox UniversalIR.
//!
//! Bundle format: magic "BRT1" | config_len (u32 LE) | config_json | num_tensors (u32 LE) |
//! for each tensor: ndim (u32 LE), dims[ndim] (u32 LE each), data (f32 LE, 4 bytes per element).
//!
//! Builds full 12-layer encoder using only generic OpTypes (LayerNorm, MatMul, Add, Gelu, Softmax, Reshape, Transpose, Gemm).

use coelanox_core::ir::{Attribute, DataType, OpType, Shape, UniversalIR};
use coelanox_core::{CoelanoxError, CoelanoxResult};
use coelanox_core::format::ModelFormat;
use coelanox_core::plugins::{PluginConfig, TranslatorPlugin};
use std::path::Path;

/// Magic bytes for BERT bundle (v1).
const BUNDLE_MAGIC: &[u8; 4] = b"BRT1";

/// First 3 = embeddings; then 12 layers × 16 tensors each in bundle.
const EMB_COUNT: usize = 3;
const TENSORS_PER_LAYER: usize = 16;
fn layer_offset(layer: usize) -> usize {
    EMB_COUNT + layer * TENSORS_PER_LAYER
}

/// BERT translator plugin.
pub struct BertTranslator;

#[derive(Debug, serde::Deserialize)]
struct BertConfig {
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    hidden_size: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    num_attention_heads: Option<u32>,
    #[serde(default)]
    type_vocab_size: Option<u32>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
}

struct WeightBlob {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl BertTranslator {
    pub fn new() -> Self {
        Self
    }

    /// Parse BRT1 bundle: after magic, [config_len (u32 LE), config_bytes, num_tensors, tensors...].
    /// Tensor data is f32 (4 bytes per element).
    fn parse_bundle(data: &[u8]) -> CoelanoxResult<(BertConfig, Vec<WeightBlob>)> {
        if data.len() < 4 + 4 {
            return Err(CoelanoxError::format("BERT bundle too short"));
        }
        if &data[0..4] != BUNDLE_MAGIC {
            return Err(CoelanoxError::format("Invalid BERT bundle (bad magic, expect BRT1)"));
        }
        let config_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let config_start = 8;
        let config_end = config_start + config_len;
        if data.len() < config_end + 4 {
            return Err(CoelanoxError::format("BERT bundle truncated at config"));
        }
        let config_json: &[u8] = &data[config_start..config_end];
        let config: BertConfig = serde_json::from_slice(config_json)
            .map_err(|e| CoelanoxError::format(&format!("Invalid BERT config JSON: {}", e)))?;

        let mut pos = config_end;
        let num_tensors = u32::from_le_bytes([
            data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
        ]) as usize;
        pos += 4;

        let mut weights = Vec::with_capacity(num_tensors);
        for _ in 0..num_tensors {
            if pos + 4 > data.len() {
                return Err(CoelanoxError::format("BERT bundle truncated at tensor header"));
            }
            let ndim = u32::from_le_bytes([
                data[pos], data[pos + 1], data[pos + 2], data[pos + 3],
            ]) as usize;
            pos += 4;
            if ndim > 16 {
                return Err(CoelanoxError::format("BERT bundle invalid ndim (corrupt or wrong format)"));
            }
            let dims_len = ndim * 4;
            if pos + dims_len > data.len() {
                return Err(CoelanoxError::format("BERT bundle truncated at tensor dims"));
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
                return Err(CoelanoxError::format("BERT bundle truncated at tensor data"));
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

    /// Add weight tensors to IR in the order expected by extract_weights: 3 emb + 12 layers × 14 tensors
    /// (8 + attn_ln_combined + 4 + out_ln_combined). Combined = gamma ++ beta for LayerNorm.
    fn add_weights_to_ir(ir: &mut UniversalIR, weights: &[WeightBlob], hidden_size: usize, num_layers: usize) -> CoelanoxResult<()> {
        let to_bytes = |d: &[f32]| d.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();

        for i in 0..EMB_COUNT.min(weights.len()) {
            let w = &weights[i];
            ir.add_weight_tensor_with_data(
                format!("bert.emb{}", i),
                Shape::new(w.shape.clone()),
                DataType::Float32,
                to_bytes(&w.data),
            );
        }

        for layer in 0..num_layers {
            let base = layer_offset(layer);
            if base + TENSORS_PER_LAYER > weights.len() {
                return Err(CoelanoxError::format(&format!("BERT bundle has fewer than {} tensors for layer {}", base + TENSORS_PER_LAYER, layer)));
            }
            for j in 0..8 {
                let w = &weights[base + j];
                ir.add_weight_tensor_with_data(
                    format!("bert.L{}.{}", layer, j),
                    Shape::new(w.shape.clone()),
                    DataType::Float32,
                    to_bytes(&w.data),
                );
            }
            let g = &weights[base + 8].data;
            let b = &weights[base + 9].data;
            let mut combined = g.clone();
            combined.extend(b.iter().copied());
            ir.add_weight_tensor_with_data(
                format!("bert.L{}.attn_ln", layer),
                Shape::new(vec![2 * hidden_size]),
                DataType::Float32,
                to_bytes(&combined),
            );
            for j in 10..14 {
                let w = &weights[base + j];
                ir.add_weight_tensor_with_data(
                    format!("bert.L{}.{}", layer, j),
                    Shape::new(w.shape.clone()),
                    DataType::Float32,
                    to_bytes(&w.data),
                );
            }
            let g = &weights[base + 14].data;
            let b = &weights[base + 15].data;
            let mut combined = g.clone();
            combined.extend(b.iter().copied());
            ir.add_weight_tensor_with_data(
                format!("bert.L{}.out_ln", layer),
                Shape::new(vec![2 * hidden_size]),
                DataType::Float32,
                to_bytes(&combined),
            );
        }
        Ok(())
    }

    /// Weight index in IR: 3 + layer*14 + offset. Offsets: 0..8 (q_w..attn_ob), 8 attn_ln, 9..13 (inter_w..out_b), 13 out_ln.
    fn ir_weight_layer(layer: usize, offset: u32) -> usize {
        3 + layer * 14 + offset as usize
    }

    /// Build full 12-layer encoder graph using only generic ops.
    fn build_ir(config: &BertConfig, weights: Vec<WeightBlob>) -> CoelanoxResult<UniversalIR> {
        let hidden_size = config.hidden_size.unwrap_or(768) as usize;
        let num_layers = config.num_hidden_layers.unwrap_or(12) as usize;
        let intermediate_size = config.intermediate_size.unwrap_or(3072) as usize;
        let max_seq_len = config.max_position_embeddings.unwrap_or(512) as usize;
        let layer_norm_eps = config.layer_norm_eps.unwrap_or(1e-5) as f32;

        if weights.len() < layer_offset(0) + TENSORS_PER_LAYER {
            return Err(CoelanoxError::format("BERT bundle has too few tensors for one layer"));
        }

        let mut ir = UniversalIR::new("bert_base_uncased".to_string());
        let batch = 1usize;
        let seq_len = max_seq_len.min(128);
        ir.metadata.add_input_shape(Shape::new(vec![batch, seq_len, hidden_size]));
        ir.metadata.add_output_shape(Shape::new(vec![batch, seq_len, hidden_size]));

        Self::add_weights_to_ir(&mut ir, &weights, hidden_size, num_layers)?;

        let flat_seq = batch * seq_len;
        let m = flat_seq;
        let k = hidden_size;
        let n = hidden_size;

        let inp = ir.add_intermediate_tensor("input".to_string(), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
        let mut prev_id = inp;

        for layer in 0..num_layers {
            let ln_attn_out = ir.add_intermediate_tensor(format!("L{}.ln_attn", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let ln_attn_node = ir.add_node(OpType::LayerNorm);
            ir.add_node_input(ln_attn_node, prev_id)?;
            ir.add_node_input(ln_attn_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 8)))?;
            ir.add_node_output(ln_attn_node, ln_attn_out)?;
            if let Some(nd) = ir.get_node_mut(ln_attn_node) {
                nd.add_attribute("epsilon".to_string(), Attribute::Float(layer_norm_eps));
                nd.add_attribute("normalized_shape".to_string(), Attribute::IntArray(vec![hidden_size as i64]));
            }

            let q_out = ir.add_intermediate_tensor(format!("L{}.Q", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let q_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(q_mm, ln_attn_out)?;
            ir.add_node_input(q_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 0)))?;
            ir.add_node_output(q_mm, q_out)?;
            ir.get_node_mut(q_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m as i64, k as i64, k as i64, n as i64]));
            let q_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, q_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 1)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.Qb", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let k_out = ir.add_intermediate_tensor(format!("L{}.K", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let k_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(k_mm, ln_attn_out)?;
            ir.add_node_input(k_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 2)))?;
            ir.add_node_output(k_mm, k_out)?;
            ir.get_node_mut(k_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m as i64, k as i64, k as i64, n as i64]));
            let k_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, k_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 3)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.Kb", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let v_out = ir.add_intermediate_tensor(format!("L{}.V", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let v_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(v_mm, ln_attn_out)?;
            ir.add_node_input(v_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 4)))?;
            ir.add_node_output(v_mm, v_out)?;
            ir.get_node_mut(v_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m as i64, k as i64, k as i64, n as i64]));
            let v_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, v_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 5)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.Vb", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let q_2d = ir.add_intermediate_tensor(format!("L{}.Q2d", layer), Shape::new(vec![seq_len, hidden_size]), DataType::Float32);
            let rq = ir.add_node(OpType::Reshape);
            ir.add_node_input(rq, q_out)?;
            ir.add_node_output(rq, q_2d)?;

            let k_2d = ir.add_intermediate_tensor(format!("L{}.K2d", layer), Shape::new(vec![seq_len, hidden_size]), DataType::Float32);
            let rk = ir.add_node(OpType::Reshape);
            ir.add_node_input(rk, k_out)?;
            ir.add_node_output(rk, k_2d)?;

            let kt = ir.add_intermediate_tensor(format!("L{}.Kt", layer), Shape::new(vec![hidden_size, seq_len]), DataType::Float32);
            let tr = ir.add_node(OpType::Transpose);
            ir.add_node_input(tr, k_2d)?;
            ir.add_node_output(tr, kt)?;
            if let Some(nd) = ir.get_node_mut(tr) {
                nd.add_attribute("shape".to_string(), Attribute::IntArray(vec![seq_len as i64, hidden_size as i64]));
                nd.add_attribute("perm".to_string(), Attribute::IntArray(vec![1, 0]));
            }

            let scores = ir.add_intermediate_tensor(format!("L{}.scores", layer), Shape::new(vec![seq_len, seq_len]), DataType::Float32);
            let mm_scores = ir.add_node(OpType::Gemm);
            ir.add_node_input(mm_scores, q_2d)?;
            ir.add_node_input(mm_scores, kt)?;
            ir.add_node_output(mm_scores, scores)?;
            let scale = 1.0 / (hidden_size as f32).sqrt();
            if let Some(nd) = ir.get_node_mut(mm_scores) {
                nd.add_attribute("dimensions".to_string(), Attribute::IntArray(vec![seq_len as i64, hidden_size as i64, hidden_size as i64, seq_len as i64]));
                nd.add_attribute("alpha".to_string(), Attribute::Float(scale));
            }

            let attn_weights = ir.add_intermediate_tensor(format!("L{}.attn_w", layer), Shape::new(vec![seq_len, seq_len]), DataType::Float32);
            let sm = ir.add_node(OpType::Softmax);
            ir.add_node_input(sm, scores)?;
            ir.add_node_output(sm, attn_weights)?;
            if let Some(nd) = ir.get_node_mut(sm) {
                nd.add_attribute("shape".to_string(), Attribute::IntArray(vec![seq_len as i64, seq_len as i64]));
                nd.add_attribute("axis".to_string(), Attribute::Int(-1));
            }

            let v_2d = ir.add_intermediate_tensor(format!("L{}.V2d", layer), Shape::new(vec![seq_len, hidden_size]), DataType::Float32);
            let rv = ir.add_node(OpType::Reshape);
            ir.add_node_input(rv, v_out)?;
            ir.add_node_output(rv, v_2d)?;

            let attn_out_2d = ir.add_intermediate_tensor(format!("L{}.attn_o2d", layer), Shape::new(vec![seq_len, hidden_size]), DataType::Float32);
            let mm_attn = ir.add_node(OpType::MatMul);
            ir.add_node_input(mm_attn, attn_weights)?;
            ir.add_node_input(mm_attn, v_2d)?;
            ir.add_node_output(mm_attn, attn_out_2d)?;
            ir.get_node_mut(mm_attn).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![seq_len as i64, seq_len as i64, seq_len as i64, hidden_size as i64]));

            let attn_out_3d = ir.add_intermediate_tensor(format!("L{}.attn_o3d", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let rback = ir.add_node(OpType::Reshape);
            ir.add_node_input(rback, attn_out_2d)?;
            ir.add_node_output(rback, attn_out_3d)?;

            let proj_out = ir.add_intermediate_tensor(format!("L{}.proj", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let proj_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(proj_mm, attn_out_3d)?;
            ir.add_node_input(proj_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 6)))?;
            ir.add_node_output(proj_mm, proj_out)?;
            ir.get_node_mut(proj_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m as i64, k as i64, k as i64, n as i64]));
            let proj_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, proj_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 7)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.projb", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let attn_residual = ir.add_intermediate_tensor(format!("L{}.attn_res", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let add_res = ir.add_node(OpType::Add);
            ir.add_node_input(add_res, prev_id)?;
            ir.add_node_input(add_res, proj_out)?;
            ir.add_node_output(add_res, attn_residual)?;

            let ln_ffn_out = ir.add_intermediate_tensor(format!("L{}.ln_ffn", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let ln_ffn_node = ir.add_node(OpType::LayerNorm);
            ir.add_node_input(ln_ffn_node, attn_residual)?;
            ir.add_node_input(ln_ffn_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 13)))?;
            ir.add_node_output(ln_ffn_node, ln_ffn_out)?;
            if let Some(nd) = ir.get_node_mut(ln_ffn_node) {
                nd.add_attribute("epsilon".to_string(), Attribute::Float(layer_norm_eps));
                nd.add_attribute("normalized_shape".to_string(), Attribute::IntArray(vec![hidden_size as i64]));
            }

            let inter_out = ir.add_intermediate_tensor(format!("L{}.inter", layer), Shape::new(vec![batch, seq_len, intermediate_size]), DataType::Float32);
            let inter_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(inter_mm, ln_ffn_out)?;
            ir.add_node_input(inter_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 9)))?;
            ir.add_node_output(inter_mm, inter_out)?;
            let m_i = flat_seq;
            let n_i = intermediate_size;
            ir.get_node_mut(inter_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m_i as i64, k as i64, k as i64, n_i as i64]));
            let inter_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, inter_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 10)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.interb", layer), Shape::new(vec![batch, seq_len, intermediate_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let gelu_out = ir.add_intermediate_tensor(format!("L{}.gelu", layer), Shape::new(vec![batch, seq_len, intermediate_size]), DataType::Float32);
            let gelu_node = ir.add_node(OpType::Gelu);
            ir.add_node_input(gelu_node, inter_out)?;
            ir.add_node_output(gelu_node, gelu_out)?;

            let ffn_out = ir.add_intermediate_tensor(format!("L{}.ffn", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let out_mm = ir.add_node(OpType::MatMul);
            ir.add_node_input(out_mm, gelu_out)?;
            ir.add_node_input(out_mm, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 11)))?;
            ir.add_node_output(out_mm, ffn_out)?;
            let n_o = hidden_size;
            ir.get_node_mut(out_mm).ok_or_else(|| CoelanoxError::format("Node not found"))?
                .add_attribute("dimensions".to_string(), Attribute::IntArray(vec![m_i as i64, n_i as i64, n_i as i64, n_o as i64]));
            let ffn_out = {
                let add_node = ir.add_node(OpType::Add);
                ir.add_node_input(add_node, ffn_out)?;
                ir.add_node_input(add_node, coelanox_core::ir::TensorId::weight(Self::ir_weight_layer(layer, 12)))?;
                let out = ir.add_intermediate_tensor(format!("L{}.ffnb", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
                ir.add_node_output(add_node, out)?;
                out
            };

            let layer_out = ir.add_intermediate_tensor(format!("L{}.out", layer), Shape::new(vec![batch, seq_len, hidden_size]), DataType::Float32);
            let add_ffn = ir.add_node(OpType::Add);
            ir.add_node_input(add_ffn, attn_residual)?;
            ir.add_node_input(add_ffn, ffn_out)?;
            ir.add_node_output(add_ffn, layer_out)?;

            prev_id = layer_out;
        }

        ir.compute_topological_order()?;
        Ok(ir)
    }
}

impl TranslatorPlugin for BertTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        if model_data.len() < 4 || &model_data[0..4] != BUNDLE_MAGIC {
            return Err(CoelanoxError::format("Invalid BERT bundle (bad magic)"));
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
            return Err(CoelanoxError::format("Invalid BERT bundle (bad magic)"));
        }
        let (config, weights) = Self::parse_bundle(model_data)?;
        let num_layers = config.num_hidden_layers.unwrap_or(12) as usize;
        let mut out = Vec::new();
        for i in 0..EMB_COUNT.min(weights.len()) {
            out.extend(weights[i].data.iter().copied());
        }
        for layer in 0..num_layers {
            let base = layer_offset(layer);
            if base + TENSORS_PER_LAYER > weights.len() {
                break;
            }
            for j in 0..8 {
                out.extend(weights[base + j].data.iter().copied());
            }
            out.extend(weights[base + 8].data.iter().copied());
            out.extend(weights[base + 9].data.iter().copied());
            for j in 10..14 {
                out.extend(weights[base + j].data.iter().copied());
            }
            out.extend(weights[base + 14].data.iter().copied());
            out.extend(weights[base + 15].data.iter().copied());
        }
        Ok(out)
    }

    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Custom("bert_base_uncased".to_string())]
    }

    fn name(&self) -> &str {
        "bert_base_uncased"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn author(&self) -> &str {
        "Coelanox"
    }

    fn description(&self) -> &str {
        "Translates BERT BRT1 bundle to UniversalIR. Full 12-layer encoder (LayerNorm, MatMul, Add, Gelu, Softmax, Reshape, Transpose, Gemm)."
    }

    fn configure(&mut self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }

    fn validate_config(&self, _config: &PluginConfig) -> CoelanoxResult<()> {
        Ok(())
    }
}
