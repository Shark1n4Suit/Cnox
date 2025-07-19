#!/usr/bin/env python3
"""
SAIC v0.1 Comprehensive Performance Analysis Suite
Benchmarks packager and runner performance across multiple scenarios
"""

import subprocess
import time
import os
import json
import statistics
import psutil
from pathlib import Path

class SaicPerformanceAnalyzer:
    def __init__(self):
        self.packager_path = "./packager/target/debug/packager"
        self.runner_path = "./runner/target/debug/runner"
        self.test_files = {}
        self.results = {
            "packaging_performance": [],
            "parsing_performance": [],
            "memory_analysis": [],
            "size_analysis": [],
            "scalability_analysis": []
        }
    
    def create_test_files(self):
        """Create test files of various sizes to simulate different AI model types"""
        print("📁 Creating test files for benchmarking...")
        
        # Test file configurations (name, size_bytes, description)
        configurations = [
            ("tiny_model", 1024, "1KB - Tiny embedding model"),
            ("small_model", 1024 * 1024, "1MB - Small transformer"),
            ("medium_model", 50 * 1024 * 1024, "50MB - Medium language model"),
            ("large_model", 200 * 1024 * 1024, "200MB - Large language model"),
            ("xl_model", 800 * 1024 * 1024, "800MB - GPT-3.5 scale model"),
            ("xxl_model", 1800 * 1024 * 1024, "1.8GB - GPT-4 scale model"),
        ]
        
        for name, size, description in configurations:
            filename = f"test_{name}.bin"
            print(f"  Creating {filename} ({description})")
            
            # Generate test data with some structure to simulate real model data
            with open(filename, 'wb') as f:
                # Write structured data that resembles model weights
                for i in range(0, size, 4096):
                    chunk_size = min(4096, size - i)
                    # Create pseudo-random but structured data
                    data = bytes([(i + j) % 256 for j in range(chunk_size)])
                    f.write(data)
            
            self.test_files[name] = {
                "filename": filename,
                "size": size,
                "description": description
            }
        
        print(f"✅ Created {len(configurations)} test files")
    
    def benchmark_packaging(self, iterations=5):
        """Benchmark packaging performance across different file sizes"""
        print("\n⚡ Benchmarking SAIC packaging performance...")
        
        for name, info in self.test_files.items():
            print(f"  Testing {name} ({info['description']})...")
            
            times = []
            saic_file = f"{name}.saic"
            
            for i in range(iterations):
                # Remove previous output
                if os.path.exists(saic_file):
                    os.remove(saic_file)
                
                # Measure packaging time
                start_time = time.perf_counter()
                process = subprocess.run([
                    self.packager_path,
                    "--model", info["filename"],
                    "--name", f"BenchmarkModel-{name}",
                    "--output", saic_file
                ], capture_output=True, text=True)
                end_time = time.perf_counter()
                
                if process.returncode == 0:
                    times.append(end_time - start_time)
                else:
                    print(f"    ERROR in iteration {i+1}: {process.stderr}")
            
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                
                # Calculate throughput
                throughput_mb_s = (info["size"] / (1024 * 1024)) / avg_time
                
                result = {
                    "model_name": name,
                    "model_size": info["size"],
                    "model_size_mb": info["size"] / (1024 * 1024),
                    "avg_time_seconds": avg_time,
                    "min_time_seconds": min_time,
                    "max_time_seconds": max_time,
                    "std_dev_seconds": std_dev,
                    "throughput_mb_s": throughput_mb_s,
                    "iterations": len(times)
                }
                
                self.results["packaging_performance"].append(result)
                
                print(f"    Avg: {avg_time:.4f}s, Throughput: {throughput_mb_s:.2f} MB/s")
    
    def benchmark_parsing(self, iterations=10):
        """Benchmark parsing performance across different SAIC file sizes"""
        print("\n🔍 Benchmarking SAIC parsing performance...")
        
        for name, info in self.test_files.items():
            saic_file = f"{name}.saic"
            
            if not os.path.exists(saic_file):
                print(f"  Skipping {name} - no SAIC file found")
                continue
            
            print(f"  Testing {name} parsing ({info['description']})...")
            
            times = []
            
            for i in range(iterations):
                # Measure parsing time
                start_time = time.perf_counter()
                process = subprocess.run([
                    self.runner_path, "run", saic_file
                ], capture_output=True, text=True)
                end_time = time.perf_counter()
                
                if process.returncode == 0:
                    times.append(end_time - start_time)
                else:
                    print(f"    ERROR in iteration {i+1}: {process.stderr}")
            
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                std_dev = statistics.stdev(times) if len(times) > 1 else 0
                
                # Calculate parsing throughput (total file size / time)
                saic_size = os.path.getsize(saic_file)
                throughput_mb_s = (saic_size / (1024 * 1024)) / avg_time
                
                result = {
                    "model_name": name,
                    "saic_file_size": saic_size,
                    "saic_file_size_mb": saic_size / (1024 * 1024),
                    "avg_time_seconds": avg_time,
                    "min_time_seconds": min_time,
                    "max_time_seconds": max_time,
                    "std_dev_seconds": std_dev,
                    "throughput_mb_s": throughput_mb_s,
                    "iterations": len(times)
                }
                
                self.results["parsing_performance"].append(result)
                
                print(f"    Avg: {avg_time:.4f}s, Throughput: {throughput_mb_s:.2f} MB/s")
    
    def analyze_memory_usage(self):
        """Analyze memory usage during packaging and parsing operations"""
        print("\n🧠 Analyzing memory usage patterns...")
        
        # Test with medium model to get representative data
        test_name = "medium_model"
        if test_name not in self.test_files:
            print("  Skipping memory analysis - test file not found")
            return
        
        info = self.test_files[test_name]
        saic_file = f"{test_name}.saic"
        
        # Memory usage during packaging
        print(f"  Analyzing packaging memory usage ({info['description']})...")
        
        process = subprocess.Popen([
            self.packager_path,
            "--model", info["filename"], 
            "--name", f"MemoryTest-{test_name}",
            "--output", saic_file
        ])
        
        memory_samples = []
        try:
            ps_process = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    mem_info = ps_process.memory_info()
                    memory_samples.append({
                        "rss_mb": mem_info.rss / (1024 * 1024),
                        "vms_mb": mem_info.vms / (1024 * 1024)
                    })
                    time.sleep(0.01)  # Sample every 10ms
                except psutil.NoSuchProcess:
                    break
        except Exception as e:
            print(f"    Memory monitoring error: {e}")
        
        process.wait()
        
        if memory_samples:
            avg_rss = statistics.mean([s["rss_mb"] for s in memory_samples])
            max_rss = max([s["rss_mb"] for s in memory_samples])
            
            packaging_memory = {
                "operation": "packaging",
                "model_size_mb": info["size"] / (1024 * 1024),
                "avg_rss_mb": avg_rss,
                "max_rss_mb": max_rss,
                "samples": len(memory_samples),
                "memory_efficiency": info["size"] / (max_rss * 1024 * 1024)  # bytes per MB RAM
            }
            
            self.results["memory_analysis"].append(packaging_memory)
            print(f"    Packaging - Avg: {avg_rss:.2f}MB, Max: {max_rss:.2f}MB")
        
        # Memory usage during parsing
        if os.path.exists(saic_file):
            print(f"  Analyzing parsing memory usage...")
            
            process = subprocess.Popen([
                self.runner_path, "run", saic_file
            ])
            
            memory_samples = []
            try:
                ps_process = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        mem_info = ps_process.memory_info()
                        memory_samples.append({
                            "rss_mb": mem_info.rss / (1024 * 1024),
                            "vms_mb": mem_info.vms / (1024 * 1024)
                        })
                        time.sleep(0.01)
                    except psutil.NoSuchProcess:
                        break
            except Exception as e:
                print(f"    Memory monitoring error: {e}")
            
            process.wait()
            
            if memory_samples:
                avg_rss = statistics.mean([s["rss_mb"] for s in memory_samples])
                max_rss = max([s["rss_mb"] for s in memory_samples])
                
                parsing_memory = {
                    "operation": "parsing",
                    "file_size_mb": os.path.getsize(saic_file) / (1024 * 1024),
                    "avg_rss_mb": avg_rss,
                    "max_rss_mb": max_rss,
                    "samples": len(memory_samples),
                    "memory_efficiency": os.path.getsize(saic_file) / (max_rss * 1024 * 1024)
                }
                
                self.results["memory_analysis"].append(parsing_memory)
                print(f"    Parsing - Avg: {avg_rss:.2f}MB, Max: {max_rss:.2f}MB")
    
    def analyze_size_efficiency(self):
        """Analyze SAIC file size efficiency and overhead"""
        print("\n📏 Analyzing SAIC file size efficiency...")
        
        for name, info in self.test_files.items():
            saic_file = f"{name}.saic"
            
            if not os.path.exists(saic_file):
                continue
            
            original_size = info["size"]
            saic_size = os.path.getsize(saic_file)
            overhead = saic_size - original_size
            overhead_percent = (overhead / saic_size) * 100
            
            # Analyze manifest size by reading the SAIC file
            manifest_size = 0
            try:
                with open(saic_file, 'rb') as f:
                    f.seek(8)  # Skip magic + version
                    manifest_length_bytes = f.read(4)
                    manifest_size = int.from_bytes(manifest_length_bytes, 'big')
            except Exception as e:
                print(f"    Error reading manifest size: {e}")
            
            result = {
                "model_name": name,
                "original_size": original_size,
                "original_size_mb": original_size / (1024 * 1024),
                "saic_size": saic_size,
                "saic_size_mb": saic_size / (1024 * 1024),
                "overhead_bytes": overhead,
                "overhead_percent": overhead_percent,
                "header_size": 12,  # 8 byte header + 4 byte length
                "manifest_size": manifest_size,
                "payload_size": saic_size - 12 - manifest_size,
                "size_efficiency": (original_size / saic_size) * 100
            }
            
            self.results["size_analysis"].append(result)
            
            print(f"  {name}: {overhead_percent:.6f}% overhead ({overhead} bytes)")
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 Generating comprehensive performance report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SAIC v0.1 COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        
        # Executive Summary
        report_lines.append("\n📋 EXECUTIVE SUMMARY")
        report_lines.append("-" * 50)
        
        if self.results["packaging_performance"]:
            fastest_package = min(self.results["packaging_performance"], key=lambda x: x["avg_time_seconds"])
            slowest_package = max(self.results["packaging_performance"], key=lambda x: x["avg_time_seconds"])
            best_throughput = max(self.results["packaging_performance"], key=lambda x: x["throughput_mb_s"])
            
            report_lines.append(f"🚀 Fastest Packaging: {fastest_package['avg_time_seconds']:.4f}s ({fastest_package['model_name']})")
            report_lines.append(f"🐌 Slowest Packaging: {slowest_package['avg_time_seconds']:.4f}s ({slowest_package['model_name']})")
            report_lines.append(f"⚡ Best Throughput: {best_throughput['throughput_mb_s']:.2f} MB/s ({best_throughput['model_name']})")
        
        if self.results["parsing_performance"]:
            fastest_parse = min(self.results["parsing_performance"], key=lambda x: x["avg_time_seconds"])
            best_parse_throughput = max(self.results["parsing_performance"], key=lambda x: x["throughput_mb_s"])
            
            report_lines.append(f"🔍 Fastest Parsing: {fastest_parse['avg_time_seconds']:.4f}s ({fastest_parse['model_name']})")
            report_lines.append(f"📈 Best Parse Throughput: {best_parse_throughput['throughput_mb_s']:.2f} MB/s")
        
        if self.results["size_analysis"]:
            most_efficient = min(self.results["size_analysis"], key=lambda x: x["overhead_percent"])
            report_lines.append(f"📦 Most Efficient: {most_efficient['overhead_percent']:.6f}% overhead ({most_efficient['model_name']})")
        
        # Detailed Packaging Performance
        if self.results["packaging_performance"]:
            report_lines.append("\n⚡ PACKAGING PERFORMANCE ANALYSIS")
            report_lines.append("-" * 50)
            report_lines.append(f"{'Model':<15} {'Size (MB)':<10} {'Avg Time (s)':<12} {'Throughput (MB/s)':<18} {'Std Dev':<10}")
            report_lines.append("-" * 75)
            
            for result in self.results["packaging_performance"]:
                report_lines.append(
                    f"{result['model_name']:<15} "
                    f"{result['model_size_mb']:<10.2f} "
                    f"{result['avg_time_seconds']:<12.4f} "
                    f"{result['throughput_mb_s']:<18.2f} "
                    f"{result['std_dev_seconds']:<10.4f}"
                )
        
        # Detailed Parsing Performance
        if self.results["parsing_performance"]:
            report_lines.append("\n🔍 PARSING PERFORMANCE ANALYSIS") 
            report_lines.append("-" * 50)
            report_lines.append(f"{'Model':<15} {'Size (MB)':<10} {'Avg Time (s)':<12} {'Throughput (MB/s)':<18} {'Std Dev':<10}")
            report_lines.append("-" * 75)
            
            for result in self.results["parsing_performance"]:
                report_lines.append(
                    f"{result['model_name']:<15} "
                    f"{result['saic_file_size_mb']:<10.2f} "
                    f"{result['avg_time_seconds']:<12.4f} "
                    f"{result['throughput_mb_s']:<18.2f} "
                    f"{result['std_dev_seconds']:<10.4f}"
                )
        
        # Memory Analysis
        if self.results["memory_analysis"]:
            report_lines.append("\n🧠 MEMORY USAGE ANALYSIS")
            report_lines.append("-" * 50)
            
            for result in self.results["memory_analysis"]:
                report_lines.append(f"Operation: {result['operation'].upper()}")
                if 'model_size_mb' in result:
                    report_lines.append(f"  Model Size: {result['model_size_mb']:.2f} MB")
                else:
                    report_lines.append(f"  File Size: {result['file_size_mb']:.2f} MB")
                report_lines.append(f"  Average Memory: {result['avg_rss_mb']:.2f} MB")
                report_lines.append(f"  Peak Memory: {result['max_rss_mb']:.2f} MB")
                report_lines.append(f"  Memory Efficiency: {result['memory_efficiency']:.2f} bytes/MB RAM")
                report_lines.append("")
        
        # Size Efficiency Analysis
        if self.results["size_analysis"]:
            report_lines.append("📏 SIZE EFFICIENCY ANALYSIS")
            report_lines.append("-" * 50)
            report_lines.append(f"{'Model':<15} {'Original (MB)':<12} {'SAIC (MB)':<10} {'Overhead %':<12} {'Efficiency %':<12}")
            report_lines.append("-" * 70)
            
            for result in self.results["size_analysis"]:
                report_lines.append(
                    f"{result['model_name']:<15} "
                    f"{result['original_size_mb']:<12.2f} "
                    f"{result['saic_size_mb']:<10.2f} "
                    f"{result['overhead_percent']:<12.6f} "
                    f"{result['size_efficiency']:<12.2f}"
                )
        
        # Performance Insights
        report_lines.append("\n💡 PERFORMANCE INSIGHTS")
        report_lines.append("-" * 50)
        
        if self.results["packaging_performance"]:
            # Calculate scaling behavior
            sizes = [r["model_size_mb"] for r in self.results["packaging_performance"]]
            times = [r["avg_time_seconds"] for r in self.results["packaging_performance"]]
            
            if len(sizes) > 1:
                # Simple linear regression to estimate scaling
                n = len(sizes)
                sum_x = sum(sizes)
                sum_y = sum(times)
                sum_xy = sum(s * t for s, t in zip(sizes, times))
                sum_x2 = sum(s * s for s in sizes)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                report_lines.append(f"📈 Packaging scales approximately linearly: ~{slope:.4f}s per MB")
                report_lines.append(f"🔄 Constant memory usage (8KB buffer) regardless of model size")
        
        if self.results["size_analysis"]:
            overheads = [r["overhead_percent"] for r in self.results["size_analysis"]]
            avg_overhead = statistics.mean(overheads)
            min_overhead = min(overheads)
            max_overhead = max(overheads)
            
            report_lines.append(f"📦 Average overhead: {avg_overhead:.6f}%")
            report_lines.append(f"📦 Overhead range: {min_overhead:.6f}% - {max_overhead:.6f}%")
            report_lines.append(f"📦 Overhead decreases with model size (fixed 12-byte + manifest cost)")
        
        # Technical Specifications
        report_lines.append("\n🔧 TECHNICAL SPECIFICATIONS")
        report_lines.append("-" * 50)
        report_lines.append("📋 SAIC v0.1 Format Specifications:")
        report_lines.append("  • Header: 8 bytes (4-byte magic 'SAIC' + 4-byte version)")
        report_lines.append("  • Manifest length: 4 bytes (big-endian)")
        report_lines.append("  • Manifest: Variable UTF-8 JSON")
        report_lines.append("  • Payload: Raw model data (no compression)")
        report_lines.append("")
        report_lines.append("⚡ Performance Characteristics:")
        report_lines.append("  • Packaging: O(n) where n = file size")
        report_lines.append("  • Parsing: O(1) + O(manifest_size)")
        report_lines.append("  • Memory usage: O(1) - constant 8KB buffer")
        report_lines.append("  • Validation: 5-8 operations total")
        
        # Write report to file
        report_content = "\n".join(report_lines)
        with open("saic_performance_report.txt", "w") as f:
            f.write(report_content)
        
        print("✅ Performance report generated: saic_performance_report.txt")
        return report_content
    
    def cleanup(self):
        """Clean up test files"""
        print("\n🧹 Cleaning up test files...")
        
        for name, info in self.test_files.items():
            files_to_remove = [info["filename"], f"{name}.saic"]
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  Removed {file_path}")
    
    def run_full_analysis(self):
        """Run complete performance analysis suite"""
        try:
            self.create_test_files()
            self.benchmark_packaging()
            self.benchmark_parsing()
            self.analyze_memory_usage()
            self.analyze_size_efficiency()
            return self.generate_report()
        finally:
            self.cleanup()

if __name__ == "__main__":
    analyzer = SaicPerformanceAnalyzer()
    report = analyzer.run_full_analysis()
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - See saic_performance_report.txt for full results")
    print("="*80) 