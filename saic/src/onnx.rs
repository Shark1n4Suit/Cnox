//! ONNX protobuf definitions
//! 
//! This module contains the protobuf definitions for ONNX models.

use prost::Message;

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {
    #[prost(int32, tag = "1")]
    pub ir_version: i32,
    #[prost(string, repeated, tag = "2")]
    pub producer_name: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    pub producer_version: Vec<String>,
    #[prost(string, repeated, tag = "4")]
    pub domain: Vec<String>,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, repeated, tag = "6")]
    pub doc_string: Vec<String>,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, repeated, tag = "9")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "3")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = "4")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "5")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "6")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "7")]
    pub value_info: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(string, repeated, tag = "5")]
    pub domain: Vec<String>,
    #[prost(message, repeated, tag = "6")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "7")]
    pub doc_string: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {
    #[prost(enumeration = "TensorProtoDataType", tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeProtoDimension>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProtoDimension {
    #[prost(oneof = "tensor_shape_proto_dimension::Value", tags = "1, 2")]
    pub value: Option<tensor_shape_proto_dimension::Value>,
}

pub mod tensor_shape_proto_dimension {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        #[prost(int64, tag = "1")]
        DimValue(i64),
        #[prost(string, tag = "2")]
        DimParam(String),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    #[prost(int32, repeated, tag = "1")]
    pub dims: Vec<i32>,
    #[prost(enumeration = "TensorProtoDataType", tag = "2")]
    pub data_type: i32,
    #[prost(message, repeated, tag = "3")]
    pub segment: Vec<TensorProtoSegment>,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(bytes, repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes, tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProtoSegment {
    #[prost(int64, tag = "1")]
    pub begin: i64,
    #[prost(int64, tag = "2")]
    pub end: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes, repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
    #[prost(enumeration = "AttributeProtoAttributeType", tag = "12")]
    pub r#type: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum TensorProtoDataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum AttributeProtoAttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    TypeProto = 13,
    FloatS = 6,
    IntS = 7,
    StringS = 8,
    TensorS = 9,
    GraphS = 10,
    SparseTensors = 12,
    TypeProtos = 14,
}
