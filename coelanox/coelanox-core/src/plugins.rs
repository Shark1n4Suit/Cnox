//! Coelanox translator (compiler frontend) trait and types.
//!
//! **Stable API for translator plugins.** This module provides the core trait for
//! model format translators. Translators are compiler frontends: they lower source
//! formats (ONNX, BERT, etc.) to UniversalIR. Implement [`CoelanoxTranslator`] to
//! add support for new model formats.

use std::path::Path;
use crate::{CoelanoxError, CoelanoxResult, format::ModelFormat};
use crate::ir::UniversalIR;

/// Coelanox translator trait (compiler frontend).
///
/// Translators lower model formats (ONNX, BERT, etc.) to UniversalIR.
/// Implement this trait to add support for new model formats.
pub trait CoelanoxTranslator: Send + Sync + 'static {
    /// Translate model data to Universal IR
    fn translate(&self, model_data: &[u8]) -> CoelanoxResult<UniversalIR>;

    /// Translate model data to Universal IR with model type context
    fn translate_with_context(
        &self,
        model_data: &[u8],
        _model_type: &str,
        _custom_input_shapes: Option<&str>,
        _custom_output_shapes: Option<&str>,
    ) -> CoelanoxResult<UniversalIR> {
        self.translate(model_data)
    }

    /// Translate model from file path (supports sharded models)
    fn translate_from_path(&self, input_path: &Path) -> CoelanoxResult<UniversalIR> {
        self.translate_from_path_with_weights(input_path, None)
    }

    /// Translate from path with optional external weights file (e.g. ONNX model + separate weight file).
    /// Default: when None, reads file and calls translate; when Some, returns error.
    fn translate_from_path_with_weights(
        &self,
        input_path: &Path,
        external_weights_path: Option<&Path>,
    ) -> CoelanoxResult<UniversalIR> {
        if external_weights_path.is_some() {
            return Err(CoelanoxError::format(
                "External weights file is not supported by this translator",
            ));
        }
        let data = std::fs::read(input_path)
            .map_err(|e| CoelanoxError::format(&format!("Failed to read {}: {}", input_path.display(), e)))?;
        self.translate(&data)
    }

    /// Extract weights from model data
    fn extract_weights(&self, model_data: &[u8]) -> CoelanoxResult<Vec<f32>>;

    /// Extract weights when model uses optional external weights file.
    fn extract_weights_with_external(
        &self,
        model_data: &[u8],
        external_weights_path: Option<&Path>,
    ) -> CoelanoxResult<Vec<f32>> {
        match external_weights_path {
            None => self.extract_weights(model_data),
            Some(_) => Err(CoelanoxError::format(
                "External weights file is not supported by this translator",
            )),
        }
    }

    /// Get supported model formats
    fn supported_formats(&self) -> Vec<ModelFormat>;

    /// Get translator name
    fn name(&self) -> &str;

    /// Get translator version (for logging)
    fn version(&self) -> &str;

    /// Get translator author (for logging)
    fn author(&self) -> &str;

    /// Get translator description (for logging)
    fn description(&self) -> &str {
        "No description available"
    }

    /// Check if translator supports a specific model format
    fn supports_format(&self, format: ModelFormat) -> bool {
        self.supported_formats().contains(&format)
    }
}
