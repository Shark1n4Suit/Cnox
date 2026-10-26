//! Regression: runtime rejects containers whose manifest IR semantics version is unsupported.

use coelanox_core::format::{
    CoelanoxContainer, CoelanoxManifest, CompressionType, HardwareTarget, ModelFormat,
};
use coelanox_core::ir::{self, DataType, OpType, Shape, UniversalIR};
use coelanox_core::CoelanoxResult;
use coelanox_orchestrator::CoelanoxRuntime;
use coelanox_packager::optimizers::{CpuOptimizer, HardwareOptimizer};
use tempfile::NamedTempFile;

fn container_with_semantics_version(version: Option<String>) -> CoelanoxResult<NamedTempFile> {
    let mut manifest = CoelanoxManifest::new(
        "semantics_test".into(),
        ModelFormat::Onnx,
        vec![vec![1, 10]],
        vec![vec![1, 10]],
        1024,
    );
    manifest.ir_semantics_version = version;

    let mut universal_ir = UniversalIR::new("semantics_test".to_string());
    let shape = Shape::new(vec![1, 10]);
    let tid_in =
        universal_ir.add_intermediate_tensor("in".into(), shape.clone(), DataType::Float32);
    let tid_out = universal_ir.add_intermediate_tensor("out".into(), shape, DataType::Float32);
    let node_id = universal_ir.add_node(OpType::Relu);
    {
        let node = universal_ir.get_node_mut(node_id).expect("Relu node");
        node.add_input_tensor(tid_in);
        node.add_output_tensor(tid_out);
    }
    universal_ir.sort_nodes_for_execution()?;
    let optimized_ir = CpuOptimizer::new().optimize(&universal_ir, HardwareTarget::Cpu)?;
    let ir_data = ir::serialize_optimized_ir_for_container(&optimized_ir)?;

    let container = CoelanoxContainer::new(
        manifest,
        vec![0u8; 1024],
        vec![],
        CompressionType::None,
        ir_data,
    );

    let file = NamedTempFile::new()?;
    container.write_binary_file(file.path())?;
    Ok(file)
}

#[test]
fn load_rejects_unsupported_ir_semantics_version() {
    let file = container_with_semantics_version(Some("999".into())).expect("container");
    let rt = CoelanoxRuntime::new();
    let err = rt
        .load_container(file.path().to_str().unwrap())
        .expect_err("expected semantics mismatch");
    let msg = err.to_string();
    assert!(
        msg.contains("IR semantics") || msg.contains("semantics"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn load_accepts_current_ir_semantics_version() {
    let file = container_with_semantics_version(Some(
        coelanox_core::format::COELANOX_IR_SEMANTICS_VERSION.into(),
    ))
    .expect("container");
    let rt = CoelanoxRuntime::new();
    rt.load_container(file.path().to_str().unwrap())
        .expect("current semantics version should load");
}
