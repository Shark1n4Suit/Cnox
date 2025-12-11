//! Test script to verify bit-packed TensorId system

use coelanox_core::ir::TensorId;

fn main() {
    println!("=== Bit-Packed TensorId System Verification ===\n");
    
    // Test weight tensor IDs
    println!("Weight Tensors (Bit 0 = 0):");
    for i in 0..10 {
        let tid = TensorId::weight(i);
        let expected_value = i << 1; // index shifted left by 1
        println!("  Weight({:2}) -> TensorId({:064b}) = {} -> is_weight: {}, index: {}",
                 i, tid.0, tid.0, tid.is_weight(), tid.index());
        assert_eq!(tid.0, expected_value, "Weight({}) encoding mismatch", i);
        assert!(tid.is_weight(), "Weight({}) should be weight", i);
        assert!(!tid.is_intermediate(), "Weight({}) should not be intermediate", i);
        assert_eq!(tid.index(), i, "Weight({}) index extraction", i);
    }
    
    println!("\nIntermediate Tensors (Bit 0 = 1):");
    for i in 0..10 {
        let tid = TensorId::intermediate(i);
        let expected_value = (i << 1) | 1; // index shifted left, then OR with 1
        println!("  Intermediate({:2}) -> TensorId({:064b}) = {} -> is_intermediate: {}, index: {}",
                 i, tid.0, tid.0, tid.is_intermediate(), tid.index());
        assert_eq!(tid.0, expected_value, "Intermediate({}) encoding mismatch", i);
        assert!(tid.is_intermediate(), "Intermediate({}) should be intermediate", i);
        assert!(!tid.is_weight(), "Intermediate({}) should not be weight", i);
        assert_eq!(tid.index(), i, "Intermediate({}) index extraction", i);
    }
    
    println!("\n=== Bit Layout Analysis ===");
    println!("TensorId structure (64-bit usize):");
    println!("  Bit 0:     Type flag (0=weight, 1=intermediate)");
    println!("  Bits 1-63: Index into respective Vec (supports up to 2^63 tensors)");
    
    println!("\n=== Edge Cases ===");
    // Test large indices
    let large_weight = TensorId::weight(1000000);
    let large_intermediate = TensorId::intermediate(1000000);
    println!("  Large Weight(1000000):      TensorId({}) -> index: {}", 
             large_weight.0, large_weight.index());
    println!("  Large Intermediate(1000000): TensorId({}) -> index: {}", 
             large_intermediate.0, large_intermediate.index());
    assert_eq!(large_weight.index(), 1000000);
    assert_eq!(large_intermediate.index(), 1000000);
    
    // Test that weight and intermediate with same index don't collide
    let w5 = TensorId::weight(5);
    let i5 = TensorId::intermediate(5);
    println!("\n  Weight(5) vs Intermediate(5):");
    println!("    Weight(5):      TensorId({:064b}) = {}", w5.0, w5.0);
    println!("    Intermediate(5): TensorId({:064b}) = {}", i5.0, i5.0);
    assert_ne!(w5.0, i5.0, "Weight and Intermediate with same index must differ");
    assert_eq!(w5.index(), i5.index(), "Both should extract same index");
    
    println!("\n=== Verification Summary ===");
    println!("✅ Bit-packing encoding: CORRECT");
    println!("✅ Type detection (is_weight/is_intermediate): CORRECT");
    println!("✅ Index extraction: CORRECT");
    println!("✅ No collisions between weight and intermediate: CORRECT");
    println!("✅ Large indices supported: CORRECT");
    println!("\n🎉 Bit-packed TensorId system is working correctly!");
}

