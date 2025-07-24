#!/usr/bin/env python3
"""
🚀 SAIC World-Class Performance Benchmark
Testing 1000+ GB/s throughput and world-class performance targets
"""

import time
import json
import struct
import subprocess
import os
import sys
from pathlib import Path
import statistics

class WorldClassBenchmark:
    def __init__(self):
        self.results = {}
        self.test_files = {}
        
    def create_test_saic_file(self, size_mb, name="TestModel"):
        """Create a SAIC file with specified size for testing"""
        manifest = {"model_name": f"{name}-{size_mb}MB"}
        manifest_json = json.dumps(manifest)
        manifest_bytes = manifest_json.encode('utf-8')
        
        # Create model payload of specified size
        model_payload = b'X' * (size_mb * 1024 * 1024)
        
        filename = f"test_{size_mb}mb.saic"
        
        with open(filename, 'wb') as f:
            # Header: magic + version
            f.write(b'SAIC')
            f.write(struct.pack('>4B', 0, 1, 0, 0))
            
            # Manifest length
            f.write(struct.pack('>I', len(manifest_bytes)))
            
            # Manifest
            f.write(manifest_bytes)
            
            # Model payload
            f.write(model_payload)
        
        self.test_files[size_mb] = filename
        print(f"✅ Created {filename}: {len(manifest_bytes)} bytes manifest + {len(model_payload)} bytes payload")
        return filename
    
    def benchmark_throughput(self, filename, size_mb):
        """Benchmark I/O throughput for SAIC file reading"""
        print(f"\n🚀 Benchmarking {filename} ({size_mb}MB)...")
        
        # Test file reading throughput
        start_time = time.time()
        
        try:
            # Run SAIC validation (this tests our memory mapping)
            result = subprocess.run(
                ["./runner/target/release/runner", "run", filename],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                # Calculate throughput
                size_bytes = size_mb * 1024 * 1024
                throughput_gbps = (size_bytes / duration) / (1024**3)
                
                print(f"✅ Success: {throughput_gbps:.2f} GB/s")
                return {
                    'success': True,
                    'duration': duration,
                    'throughput_gbps': throughput_gbps,
                    'size_mb': size_mb
                }
            else:
                print(f"❌ Failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout after 30 seconds")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def benchmark_latency(self, filename, size_mb):
        """Benchmark latency for different model sizes"""
        print(f"\n⚡ Benchmarking latency for {filename}...")
        
        latencies = []
        for i in range(5):  # 5 runs for average
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    ["./runner/target/release/runner", "run", filename],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                
                if result.returncode == 0:
                    latencies.append(latency)
                    print(f"  Run {i+1}: {latency:.2f}ms")
                else:
                    print(f"  Run {i+1}: Failed")
                    
            except Exception as e:
                print(f"  Run {i+1}: Error - {e}")
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            print(f"✅ Average latency: {avg_latency:.2f}ms, Min: {min_latency:.2f}ms")
            return {
                'success': True,
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'latencies': latencies
            }
        else:
            return {'success': False, 'error': 'All runs failed'}
    
    def run_world_class_benchmarks(self):
        """Run comprehensive world-class performance benchmarks"""
        print("🏆 SAIC World-Class Performance Benchmark")
        print("=" * 50)
        
        # Test different model sizes
        test_sizes = [1, 10, 100, 500, 1000]  # MB
        
        for size_mb in test_sizes:
            print(f"\n📊 Testing {size_mb}MB model...")
            
            # Create test file
            filename = self.create_test_saic_file(size_mb)
            
            # Benchmark throughput
            throughput_result = self.benchmark_throughput(filename, size_mb)
            
            # Benchmark latency
            latency_result = self.benchmark_latency(filename, size_mb)
            
            # Store results
            self.results[size_mb] = {
                'throughput': throughput_result,
                'latency': latency_result,
                'filename': filename
            }
            
            # Check against world-class targets
            if throughput_result.get('success'):
                throughput = throughput_result['throughput_gbps']
                target = self.get_world_class_target(size_mb)
                
                if throughput >= target:
                    print(f"🎯 WORLD-CLASS ACHIEVED: {throughput:.2f} GB/s >= {target} GB/s target")
                else:
                    print(f"📈 Good: {throughput:.2f} GB/s (target: {target} GB/s)")
        
        # Generate performance report
        self.generate_performance_report()
    
    def get_world_class_target(self, size_mb):
        """Get world-class performance target for model size"""
        if size_mb < 100:
            return 1000  # 1000+ GB/s for small models
        elif size_mb < 1000:
            return 500   # 500+ GB/s for medium models
        else:
            return 100   # 100+ GB/s for large models
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 60)
        print("🏆 SAIC WORLD-CLASS PERFORMANCE REPORT")
        print("=" * 60)
        
        # Calculate overall statistics
        successful_throughputs = []
        successful_latencies = []
        
        for size_mb, result in self.results.items():
            if result['throughput'].get('success'):
                successful_throughputs.append(result['throughput']['throughput_gbps'])
            
            if result['latency'].get('success'):
                successful_latencies.append(result['latency']['avg_latency_ms'])
        
        if successful_throughputs:
            max_throughput = max(successful_throughputs)
            avg_throughput = statistics.mean(successful_throughputs)
            
            print(f"\n🚀 THROUGHPUT PERFORMANCE:")
            print(f"   Maximum: {max_throughput:.2f} GB/s")
            print(f"   Average: {avg_throughput:.2f} GB/s")
            
            if max_throughput >= 1000:
                print(f"   🎯 WORLD-CLASS: 1000+ GB/s ACHIEVED!")
            elif max_throughput >= 500:
                print(f"   🥈 EXCELLENT: 500+ GB/s achieved")
            elif max_throughput >= 100:
                print(f"   🥉 GOOD: 100+ GB/s achieved")
        
        if successful_latencies:
            min_latency = min(successful_latencies)
            avg_latency = statistics.mean(successful_latencies)
            
            print(f"\n⚡ LATENCY PERFORMANCE:")
            print(f"   Minimum: {min_latency:.2f}ms")
            print(f"   Average: {avg_latency:.2f}ms")
            
            if min_latency < 5:
                print(f"   🎯 WORLD-CLASS: <5ms latency achieved!")
            elif min_latency < 10:
                print(f"   🥈 EXCELLENT: <10ms latency achieved")
            elif min_latency < 50:
                print(f"   🥉 GOOD: <50ms latency achieved")
        
        # Detailed results table
        print(f"\n📊 DETAILED RESULTS:")
        print(f"{'Size (MB)':<10} {'Throughput (GB/s)':<18} {'Latency (ms)':<15} {'Status':<15}")
        print("-" * 60)
        
        for size_mb in sorted(self.results.keys()):
            result = self.results[size_mb]
            
            if result['throughput'].get('success'):
                throughput = f"{result['throughput']['throughput_gbps']:.2f}"
            else:
                throughput = "FAILED"
            
            if result['latency'].get('success'):
                latency = f"{result['latency']['avg_latency_ms']:.2f}"
            else:
                latency = "FAILED"
            
            # Determine status
            if result['throughput'].get('success') and result['latency'].get('success'):
                target = self.get_world_class_target(size_mb)
                if result['throughput']['throughput_gbps'] >= target:
                    status = "🎯 WORLD-CLASS"
                else:
                    status = "✅ GOOD"
            else:
                status = "❌ FAILED"
            
            print(f"{size_mb:<10} {throughput:<18} {latency:<15} {status:<15}")
        
        # World-class achievement summary
        print(f"\n🏆 WORLD-CLASS ACHIEVEMENTS:")
        world_class_count = 0
        total_tests = len(self.results)
        
        for size_mb, result in self.results.items():
            if result['throughput'].get('success'):
                target = self.get_world_class_target(size_mb)
                if result['throughput']['throughput_gbps'] >= target:
                    world_class_count += 1
                    print(f"   ✅ {size_mb}MB: {result['throughput']['throughput_gbps']:.2f} GB/s")
        
        print(f"\n🎯 Overall: {world_class_count}/{total_tests} tests achieved world-class performance")
        
        if world_class_count == total_tests:
            print("🏆 PERFECT SCORE! All tests achieved world-class performance!")
        elif world_class_count >= total_tests * 0.8:
            print("🥇 EXCELLENT! 80%+ tests achieved world-class performance!")
        elif world_class_count >= total_tests * 0.5:
            print("🥈 GOOD! 50%+ tests achieved world-class performance!")
        else:
            print("📈 PROGRESS! Some tests achieved world-class performance!")
    
    def cleanup(self):
        """Clean up test files"""
        print(f"\n🧹 Cleaning up test files...")
        for filename in self.test_files.values():
            try:
                os.remove(filename)
                print(f"   Removed {filename}")
            except:
                pass

if __name__ == "__main__":
    benchmark = WorldClassBenchmark()
    
    try:
        benchmark.run_world_class_benchmarks()
    finally:
        benchmark.cleanup() 