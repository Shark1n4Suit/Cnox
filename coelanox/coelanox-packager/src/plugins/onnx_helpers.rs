//! Helper functions for creating valid ONNX models
use super::onnx::{ModelProto, GraphProto, OperatorSetIdProto, ValueInfoProto, TypeProto, type_proto, TensorShapeProto, tensor_shape_proto, NodeProto, TensorProto};

pub fn build_min_model_with_graph(nodes: Vec<NodeProto>, initializers: Vec<TensorProto>) -> ModelProto {
    ModelProto {
        ir_version: Some(7),
        opset_import: vec![OperatorSetIdProto { domain: Some("".to_string()), version: Some(14), ..Default::default() }],
        producer_name: Some("test".to_string()),
        producer_version: Some("1.0".to_string()),
        domain: Some("test".to_string()),
        model_version: Some(1),
        doc_string: Some("unit-test".to_string()),
        graph: Some(GraphProto {
            name: Some("g".to_string()),
            doc_string: Some("".to_string()),
            node: nodes,
            initializer: initializers,
            input: vec![ValueInfoProto {
                name: Some("in".to_string()),
                r#type: Some(TypeProto {
                    value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                        elem_type: Some(1), // FLOAT
                        shape: Some(TensorShapeProto {
                            dim: vec![tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimValue(1)), ..Default::default() }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    })),
                    ..Default::default()
                }),
                doc_string: Some("".to_string()),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: Some("out".to_string()),
                r#type: Some(TypeProto {
                    value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                        elem_type: Some(1),
                        shape: Some(TensorShapeProto {
                            dim: vec![tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimValue(1)), ..Default::default() }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    })),
                    ..Default::default()
                }),
                doc_string: Some("".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    }
}

pub fn tensor_float(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: Some(name.to_string()),
        data_type: Some(1), // FLOAT
        dims: dims.to_vec(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

pub fn tensor_raw(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
    let mut raw = Vec::with_capacity(data.len() * 4);
    for f in data {
        raw.extend_from_slice(&f.to_le_bytes());
    }
    TensorProto {
        name: Some(name.to_string()),
        data_type: Some(1), // FLOAT
        dims: dims.to_vec(),
        raw_data: Some(raw),
        ..Default::default()
    }
}
