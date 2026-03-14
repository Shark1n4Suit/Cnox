//! Test that the Frontend SDK contract works: a translator can depend on
//! `coelanox-core` only and implement `CoelanoxTranslator` to produce UniversalIR.
//!
//! This validates the "third-party translator" path: no packager dependency
//! required for the trait impl; only coelanox-core.

use coelanox_core::format::ModelFormat;
use coelanox_core::ir::{DataType, OpType, Shape, UniversalIR};
use coelanox_core::{CoelanoxResult, CoelanoxTranslator};

/// Minimal translator that produces a trivial IR (single Add node).
/// Used to verify the Frontend SDK contract.
struct MinimalTestTranslator;

impl CoelanoxTranslator for MinimalTestTranslator {
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR> {
        // Ignore input; produce a minimal valid IR
        let _ = model_data;
        let mut ir = UniversalIR::new("minimal_test".to_string());
        ir.metadata.add_input_shape(Shape::new(vec![1, 4]));
        ir.metadata.add_output_shape(Shape::new(vec![1, 4]));

        let input_id = ir.add_intermediate_tensor(
            "input".to_string(),
            Shape::new(vec![1, 4]),
            DataType::Float32,
        );
        let output_id = ir.add_intermediate_tensor(
            "output".to_string(),
            Shape::new(vec![1, 4]),
            DataType::Float32,
        );

        let node_id = ir.add_node(OpType::Add);
        ir.add_node_input(node_id, input_id)?;
        ir.add_node_input(node_id, input_id)?; // trivial: input + input
        ir.add_node_output(node_id, output_id)?;

        ir.compute_topological_order()?;
        ir.validate()?;
        Ok(ir)
    }

    fn extract_weights(&self, _model_data: &[u8]) -> CoelanoxResult<Vec<f32>> {
        Ok(vec![])
    }

    fn supported_formats(&self) -> Vec<ModelFormat> {
        vec![ModelFormat::Custom("minimal_test".to_string())]
    }

    fn name(&self) -> &str {
        "minimal_test_translator"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn author(&self) -> &str {
        "Frontend SDK test"
    }
}

#[test]
fn test_external_translator_implements_trait() {
    // Verify a translator can be constructed and used with only coelanox-core
    let translator = MinimalTestTranslator;
    assert_eq!(translator.name(), "minimal_test_translator");
    assert_eq!(translator.supported_formats().len(), 1);
}

#[test]
fn test_external_translator_produces_valid_ir() {
    let translator = MinimalTestTranslator;
    let ir = translator
        .translate(b"dummy_model_data")
        .expect("translate should succeed");

    assert_eq!(ir.metadata.name, "minimal_test");
    assert!(!ir.nodes.is_empty());
    ir.validate().expect("IR should be valid");
}
