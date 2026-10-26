//! v1 hardening: IR semantics version gate + scalar determinism smoke.

use coelanox_core::format::ModelFormat;
use coelanox_core::format::{
    validate_ir_semantics_version, CoelanoxManifest, PackagingFingerprint,
    COELANOX_IR_SEMANTICS_VERSION,
};
use coelanox_core::ir::{Attribute, OpType};
use coelanox_core::scalar::execute_scalar_op;
use std::collections::HashMap;

#[test]
fn ir_semantics_version_accepts_current() {
    let mut m = CoelanoxManifest::new(
        "t".into(),
        ModelFormat::Onnx,
        vec![vec![1]],
        vec![vec![1]],
        0,
    );
    m.ir_semantics_version = Some(COELANOX_IR_SEMANTICS_VERSION.to_string());
    validate_ir_semantics_version(&m).expect("current version should load");
}

#[test]
fn ir_semantics_version_rejects_future() {
    let mut m = CoelanoxManifest::new(
        "t".into(),
        ModelFormat::Onnx,
        vec![vec![1]],
        vec![vec![1]],
        0,
    );
    m.ir_semantics_version = Some("999".to_string());
    assert!(validate_ir_semantics_version(&m).is_err());
}

#[test]
fn ir_semantics_version_accepts_legacy_missing_field() {
    let mut m = CoelanoxManifest::new(
        "t".into(),
        ModelFormat::Onnx,
        vec![vec![1]],
        vec![vec![1]],
        0,
    );
    m.ir_semantics_version = None;
    validate_ir_semantics_version(&m).expect("legacy containers without field should load");
}

#[test]
fn ir_semantics_version_rejects_non_matching_version() {
    let mut m = CoelanoxManifest::new(
        "t".into(),
        ModelFormat::Onnx,
        vec![vec![1]],
        vec![vec![1]],
        0,
    );
    m.ir_semantics_version = Some("0".to_string());
    assert!(validate_ir_semantics_version(&m).is_err());
}

#[test]
fn packaging_fingerprint_roundtrips_in_manifest() {
    let mut m = CoelanoxManifest::new(
        "t".into(),
        ModelFormat::Onnx,
        vec![vec![2]],
        vec![vec![2]],
        0,
    );
    m.packaging_fingerprint = Some(PackagingFingerprint {
        translator_name: "onnx".into(),
        translator_version: "0.1".into(),
        use_demo_translator: false,
        fallback_only: true,
        target_str: "cpu".into(),
        optimization: "Advanced".into(),
        packager_crate_version: "0.7.5".into(),
    });
    let json = serde_json::to_vec(&m).expect("serialize");
    let back: CoelanoxManifest = serde_json::from_slice(&json).expect("deserialize");
    assert_eq!(back.packaging_fingerprint, m.packaging_fingerprint);
}

#[test]
fn scalar_add_is_deterministic() {
    let a = [1.0f32, 2.0f32];
    let b = [3.0f32, 4.0f32];
    let mut out1 = [0.0f32; 2];
    let mut out2 = [0.0f32; 2];
    let attrs: HashMap<String, Attribute> = HashMap::new();
    execute_scalar_op(&OpType::Add, &a, Some(&b), &mut out1, None, &attrs).unwrap();
    execute_scalar_op(&OpType::Add, &a, Some(&b), &mut out2, None, &attrs).unwrap();
    assert_eq!(out1, out2);
}
