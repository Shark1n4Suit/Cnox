//! Tests for the SAIC packager
//! 
//! This module contains comprehensive tests for all packager components.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packager::*;
    use crate::format::{HardwareTarget, ModelFormat};
    use std::path::Path;

    #[test]
    fn test_universal_ir_creation() {
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        
        // Test metadata
        assert_eq!(ir.metadata.name, "test_model");
        assert_eq!(ir.metadata.version, "1.0.0");
        assert_eq!(ir.metadata.data_type, ir::DataType::Float32);
        
        // Test adding operations
        let mut op = ir::Operation::new(ir::OpType::Add);
        op.add_input("input1".to_string());
        op.add_input("input2".to_string());
        op.add_output("output".to_string());
        ir.add_operation(op);
        
        assert_eq!(ir.operation_count(), 1);
        
        // Test adding tensors
        let tensor = ir::Tensor::new(
            "weights".to_string(),
            ir::Shape::new(vec![100, 100]),
            ir::DataType::Float32,
        );
        ir.add_tensor(tensor);
        
        assert_eq!(ir.tensor_count(), 1);
        
        // Test validation
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn test_universal_ir_validation() {
        let mut ir = ir::UniversalIR::new("".to_string());
        
        // Test empty name validation
        assert!(ir.validate().is_err());
        
        // Test valid IR
        ir.metadata.name = "valid_model".to_string();
        assert!(ir.validate().is_ok());
        
        // Test operation without inputs/outputs
        let mut op = ir::Operation::new(ir::OpType::Add);
        ir.add_operation(op);
        assert!(ir.validate().is_err());
    }

    #[test]
    fn test_data_types() {
        assert_eq!(ir::DataType::Float32.size(), 4);
        assert_eq!(ir::DataType::Float64.size(), 8);
        assert_eq!(ir::DataType::Int32.size(), 4);
        assert_eq!(ir::DataType::Int64.size(), 8);
        assert_eq!(ir::DataType::UInt8.size(), 1);
        assert_eq!(ir::DataType::Bool.size(), 1);
    }

    #[test]
    fn test_shape_operations() {
        let shape = ir::Shape::new(vec![2, 3, 4]);
        
        assert_eq!(shape.total_elements(), 24);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.dimensions, vec![2, 3, 4]);
    }

    #[test]
    fn test_cpu_optimizer() {
        let optimizer = optimizers::CpuOptimizer::new();
        
        assert_eq!(optimizer.name(), "cpu_optimizer");
        assert_eq!(optimizer.version(), "1.0.0");
        assert_eq!(optimizer.supported_targets(), vec![HardwareTarget::Cpu]);
        
        // Test optimization
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let mut op = ir::Operation::new(ir::OpType::Add);
        op.add_input("input".to_string());
        op.add_output("output".to_string());
        ir.add_operation(op);
        
        let optimized_ir = optimizer.optimize(&ir, HardwareTarget::Cpu).unwrap();
        assert_eq!(optimized_ir.target, HardwareTarget::Cpu);
        assert!(!optimized_ir.optimized_operations.is_empty());
    }

    #[test]
    fn test_gpu_optimizer() {
        let optimizer = optimizers::GpuOptimizer::new();
        
        assert_eq!(optimizer.name(), "gpu_optimizer");
        assert_eq!(optimizer.version(), "1.0.0");
        assert_eq!(optimizer.supported_targets(), vec![HardwareTarget::Gpu]);
        
        // Test optimization
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let mut op = ir::Operation::new(ir::OpType::Convolution);
        op.add_input("input".to_string());
        op.add_output("output".to_string());
        ir.add_operation(op);
        
        let optimized_ir = optimizer.optimize(&ir, HardwareTarget::Gpu).unwrap();
        assert_eq!(optimized_ir.target, HardwareTarget::Gpu);
        assert!(!optimized_ir.optimized_operations.is_empty());
    }

    #[test]
    fn test_edge_optimizer() {
        let optimizer = optimizers::EdgeOptimizer::new();
        
        assert_eq!(optimizer.name(), "edge_optimizer");
        assert_eq!(optimizer.version(), "1.0.0");
        assert_eq!(optimizer.supported_targets(), vec![HardwareTarget::Edge]);
        
        // Test optimization
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let mut op = ir::Operation::new(ir::OpType::Activation);
        op.add_input("input".to_string());
        op.add_output("output".to_string());
        ir.add_operation(op);
        
        let optimized_ir = optimizer.optimize(&ir, HardwareTarget::Edge).unwrap();
        assert_eq!(optimized_ir.target, HardwareTarget::Edge);
        assert!(!optimized_ir.optimized_operations.is_empty());
    }

    #[test]
    fn test_cpu_code_generator() {
        let generator = generators::CpuCodeGenerator::new();
        
        assert_eq!(generator.name(), "cpu_generator");
        assert_eq!(generator.version(), "1.0.0");
        assert_eq!(generator.target_hardware(), HardwareTarget::Cpu);
        
        // Test code generation
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let optimized_ir = ir::OptimizedIR::new(ir, HardwareTarget::Cpu);
        
        let machine_code = generator.generate(&optimized_ir).unwrap();
        assert!(!machine_code.is_empty());
    }

    #[test]
    fn test_gpu_code_generator() {
        let generator = generators::GpuCodeGenerator::new();
        
        assert_eq!(generator.name(), "gpu_generator");
        assert_eq!(generator.version(), "1.0.0");
        assert_eq!(generator.target_hardware(), HardwareTarget::Gpu);
        
        // Test code generation
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let optimized_ir = ir::OptimizedIR::new(ir, HardwareTarget::Gpu);
        
        let machine_code = generator.generate(&optimized_ir).unwrap();
        assert!(!machine_code.is_empty());
    }

    #[test]
    fn test_edge_code_generator() {
        let generator = generators::EdgeCodeGenerator::new();
        
        assert_eq!(generator.name(), "edge_generator");
        assert_eq!(generator.version(), "1.0.0");
        assert_eq!(generator.target_hardware(), HardwareTarget::Edge);
        
        // Test code generation
        let mut ir = ir::UniversalIR::new("test_model".to_string());
        let optimized_ir = ir::OptimizedIR::new(ir, HardwareTarget::Edge);
        
        let machine_code = generator.generate(&optimized_ir).unwrap();
        assert!(!machine_code.is_empty());
    }

    #[test]
    fn test_plugin_manager() {
        let mut manager = plugins::PluginManager::new();
        
        assert_eq!(manager.plugin_count(), 0);
        
        // Test registering plugins
        let dummy_plugin = Box::new(plugins::DummyPlugin::new());
        manager.register_plugin(dummy_plugin);
        
        assert_eq!(manager.plugin_count(), 1);
        
        // Test finding plugin
        let plugin = manager.find_plugin(ModelFormat::Onnx);
        assert!(plugin.is_some());
        
        // Test getting plugin by name
        let plugin = manager.get_plugin("dummy_plugin");
        assert!(plugin.is_some());
        
        // Test removing plugin
        let removed = manager.remove_plugin("dummy_plugin");
        assert!(removed.is_some());
        assert_eq!(manager.plugin_count(), 0);
    }

    #[test]
    fn test_dummy_plugin() {
        let plugin = plugins::DummyPlugin::new();
        
        assert_eq!(plugin.name(), "dummy_plugin");
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.author(), "SAIC Team");
        
        let supported_formats = plugin.supported_formats();
        assert!(supported_formats.contains(&ModelFormat::Onnx));
        // PyTorch and TensorFlow support removed - focusing on ONNX only
        // assert!(supported_formats.contains(&ModelFormat::PyTorch));
        // assert!(supported_formats.contains(&ModelFormat::TensorFlow));
        
        // Test translation
        let dummy_data = vec![0u8; 100];
        let ir = plugin.translate(&dummy_data).unwrap();
        
        assert_eq!(ir.metadata.name, "dummy_model");
        assert_eq!(ir.operation_count(), 1);
        assert_eq!(ir.tensor_count(), 1);
    }

    #[test]
    fn test_onnx_plugin() {
        let plugin = plugins::OnnxPlugin::new();
        
        assert_eq!(plugin.name(), "onnx_plugin");
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.author(), "SAIC Team");
        
        let supported_formats = plugin.supported_formats();
        assert_eq!(supported_formats.len(), 1);
        assert!(supported_formats.contains(&ModelFormat::Onnx));
        
        // Test translation
        let dummy_data = vec![0u8; 100];
        let ir = plugin.translate(&dummy_data).unwrap();
        
        assert_eq!(ir.metadata.name, "onnx_model");
        assert_eq!(ir.operation_count(), 1);
        assert_eq!(ir.tensor_count(), 1);
    }

    #[test]
    fn test_pytorch_plugin() {
        // PyTorch plugin removed - focusing on ONNX only
        assert!(true); // Placeholder test
    }

    #[test]
    fn test_tensorflow_plugin() {
        // TensorFlow plugin removed - focusing on ONNX only
        assert!(true); // Placeholder test
    }

    #[test]
    fn test_format_detection() {
        // Test extension detection
        let path = Path::new("model.onnx");
        let format = utils::format_detection::detect_from_extension(path);
        assert_eq!(format, Some(ModelFormat::Onnx));
        
        // PyTorch and TensorFlow format detection removed - focusing on ONNX only
        // let path = Path::new("model.pt");
        // let format = utils::format_detection::detect_from_extension(path);
        // assert_eq!(format, Some(ModelFormat::PyTorch));
        
        // let path = Path::new("model.pb");
        // let format = utils::format_detection::detect_from_extension(path);
        // assert_eq!(format, Some(ModelFormat::TensorFlow));
        
        // Test unknown extension
        let path = Path::new("model.xyz");
        let format = utils::format_detection::detect_from_extension(path);
        assert!(format.is_none());
    }

    #[test]
    fn test_performance_timer() {
        let timer = utils::performance::Timer::start();
        
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
        
        let elapsed_micros = timer.elapsed_micros();
        assert!(elapsed_micros >= 10000);
        
        let elapsed_millis = timer.elapsed_millis();
        assert!(elapsed_millis >= 10);
    }

    #[test]
    fn test_validation_utils() {
        // Test valid shapes
        assert!(utils::validation::validate_input_shape(&[1, 2, 3]).is_ok());
        assert!(utils::validation::validate_output_shape(&[1, 2, 3]).is_ok());
        
        // Test invalid shapes
        assert!(utils::validation::validate_input_shape(&[]).is_err());
        assert!(utils::validation::validate_output_shape(&[]).is_err());
        assert!(utils::validation::validate_input_shape(&[1, 0, 3]).is_err());
        assert!(utils::validation::validate_output_shape(&[1, 0, 3]).is_err());
        
        // Test element calculation
        assert_eq!(utils::validation::calculate_elements(&[2, 3, 4]), 24);
    }

    #[test]
    fn test_saic_packager() {
        let packager = SaicPackager::new();
        
        // Test that packager has default optimizers and generators
        // (This is tested indirectly through the package method)
        
        // Test packaging with dummy data
        let temp_dir = tempfile::tempdir().unwrap();
        let model_path = temp_dir.path().join("dummy_model.onnx");
        std::fs::write(&model_path, vec![0u8; 100]).unwrap();
        
        let container = packager.package(&model_path, HardwareTarget::Cpu).unwrap();
        
        // Verify container was created
        assert!(container.total_size() > 0);
    }

    #[test]
    fn test_optimization_hints() {
        let mut hint = ir::OptimizationHint::new(ir::OptimizationType::Quantization);
        hint.add_parameter("bits".to_string(), ir::Attribute::Int(8));
        hint.add_parameter("symmetric".to_string(), ir::Attribute::Bool(true));
        
        assert_eq!(hint.hint_type, ir::OptimizationType::Quantization);
        assert_eq!(hint.parameters.len(), 2);
    }

    #[test]
    fn test_tensor_operations() {
        let tensor = ir::Tensor::new(
            "test_tensor".to_string(),
            ir::Shape::new(vec![2, 3, 4]),
            ir::DataType::Float32,
        );
        
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.elements(), 24);
        assert_eq!(tensor.size(), 96); // 24 * 4 bytes
        
        let tensor_with_data = ir::Tensor::with_data(
            "test_tensor".to_string(),
            ir::Shape::new(vec![2, 2]),
            ir::DataType::Float32,
            vec![1.0f32.to_le_bytes().to_vec(), 2.0f32.to_le_bytes().to_vec()].concat(),
        );
        
        assert_eq!(tensor_with_data.size(), 8); // 4 * 2 bytes
    }
}

