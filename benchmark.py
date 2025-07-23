#!/usr/bin/env python3
"""
Simple SAIC Benchmark

Measures:
1. Packaging performance
2. Validation performance  
3. File overhead
"""

import subprocess
import time
import os
import statistics

def run_command(cmd, iterations=5):
    """Run command and return timing statistics."""
    times = []
    successes = 0
    
    for i in range(iterations):
        start_time = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            end_time = time.time()
            
            if result.returncode == 0:
                times.append((end_time - start_time) * 1000)  # Convert to ms
                successes += 1
        except Exception as e:
            print(f"   ❌ Iteration {i+1} failed: {e}")
    
    if times:
        return {
            'success': True,
            'avg_time': statistics.mean(times),
            'min_time': min(times),
            'max_time': max(times),
            'successes': successes,
            'iterations': iterations
        }
    else:
        return {
            'success': False,
            'avg_time': 0,
            'min_time': 0,
            'max_time': 0,
            'successes': 0,
            'iterations': iterations
        }

def main():
    print("🏆 SAIC Performance Benchmark")
    print("=" * 40)
    
    # Check if files exist
    if not os.path.exists("mobilenetv2-1.0.onnx"):
        print("❌ MobileNetV2 ONNX file not found")
        return
    
    if not os.path.exists("mobilenetv2.saic"):
        print("❌ SAIC file not found")
        return
    
    print(f"📋 Testing with:")
    print(f"   ONNX: {os.path.getsize('mobilenetv2-1.0.onnx'):,} bytes")
    print(f"   SAIC: {os.path.getsize('mobilenetv2.saic'):,} bytes")
    
    # Test 1: Packaging Performance
    print(f"\n🎯 Test 1: SAIC Packaging Performance")
    print("-" * 40)
    
    packaging_result = run_command("./packager/target/release/packager --model mobilenetv2-1.0.onnx --name test --output test.saic")
    
    if packaging_result['success']:
        print(f"   ✅ Packaging: {packaging_result['avg_time']:.2f}ms avg")
        print(f"   📊 Min: {packaging_result['min_time']:.2f}ms, Max: {packaging_result['max_time']:.2f}ms")
        print(f"   📊 Success rate: {packaging_result['successes']}/{packaging_result['iterations']}")
    else:
        print(f"   ❌ Packaging failed")
    
    # Test 2: Validation Performance
    print(f"\n🎯 Test 2: SAIC Validation Performance")
    print("-" * 40)
    
    validation_result = run_command("./runner/target/release/runner run mobilenetv2.saic", 10)
    
    if validation_result['success']:
        print(f"   ✅ Validation: {validation_result['avg_time']:.2f}ms avg")
        print(f"   📊 Min: {validation_result['min_time']:.2f}ms, Max: {validation_result['max_time']:.2f}ms")
        print(f"   📊 Success rate: {validation_result['successes']}/{validation_result['iterations']}")
    else:
        print(f"   ❌ Validation failed")
    
    # Calculate overhead
    onnx_size = os.path.getsize("mobilenetv2-1.0.onnx")
    saic_size = os.path.getsize("mobilenetv2.saic")
    overhead = ((saic_size - onnx_size) / onnx_size) * 100
    
    # Performance summary
    print(f"\n📊 Performance Summary")
    print("-" * 30)
    
    if packaging_result['success']:
        print(f"📦 Packaging: {packaging_result['avg_time']:.2f}ms")
    
    if validation_result['success']:
        print(f"🔍 Validation: {validation_result['avg_time']:.2f}ms")
    
    print(f"📦 Container overhead: {overhead:.3f}%")
    print(f"📦 Size: {onnx_size:,} → {saic_size:,} bytes")
    
    # Performance claims validation
    print(f"\n🎯 Performance Claims Validation")
    print("-" * 40)
    
    if validation_result['success']:
        if validation_result['avg_time'] < 10:
            print(f"✅ Sub-10ms validation: {validation_result['avg_time']:.2f}ms ✓")
        else:
            print(f"❌ Sub-10ms validation: {validation_result['avg_time']:.2f}ms ✗")
    
    if overhead < 1:
        print(f"✅ Sub-1% overhead: {overhead:.3f}% ✓")
    else:
        print(f"❌ Sub-1% overhead: {overhead:.3f}% ✗")
    
    print(f"\n🏁 Benchmark completed!")

if __name__ == "__main__":
    main() 