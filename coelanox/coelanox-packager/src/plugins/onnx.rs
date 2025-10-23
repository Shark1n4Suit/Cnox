//! ONNX protobuf definitions
//! 
//! This module contains placeholder ONNX protobuf definitions.

// Placeholder types for now - these would normally be generated from protobuf
pub struct ModelProto;
pub struct GraphProto;
pub struct OperatorSetIdProto;
pub struct ValueInfoProto;
pub struct TypeProto;
pub struct TensorShapeProto;
pub struct NodeProto;
pub struct TensorProto;
pub struct AttributeProto;

pub mod type_proto {
    pub enum Value {
        TensorType(super::TensorShapeProto),
    }
}

pub mod tensor_shape_proto {
    pub struct Dimension {
        pub value: Option<super::dimension::Value>,
    }
    
    pub mod dimension {
        pub enum Value {
            DimValue(i64),
            DimParam(String),
        }
    }
}
