//! SAIC Memory Management
//! 
//! Efficient memory allocation and management for SAIC execution

use crate::{SaicError, SaicResult};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use once_cell::sync::Lazy;

/// Memory pool for efficient tensor allocation
pub struct TensorPool {
    /// Available tensors by size
    available_tensors: HashMap<usize, Vec<Tensor>>,
    /// Maximum pool size per tensor size
    max_pool_size: usize,
    /// Total memory usage
    total_memory_usage: usize,
}

/// Tensor for memory pooling
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data
    pub data: Vec<f32>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub data_type: String,
    /// Whether tensor is currently in use
    pub in_use: bool,
}

impl TensorPool {
    /// Create a new tensor pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available_tensors: HashMap::new(),
            max_pool_size,
            total_memory_usage: 0,
        }
    }

    /// Get a tensor from the pool (reuse existing or create new)
    pub fn get_tensor(&mut self, shape: &[usize], data_type: &str) -> Tensor {
        let size = shape.iter().product::<usize>();
        
        // Try to reuse existing tensor
        if let Some(tensors) = self.available_tensors.get_mut(&size) {
            if let Some(tensor) = tensors.pop() {
                let mut reused_tensor = tensor;
                reused_tensor.shape = shape.to_vec();
                reused_tensor.data_type = data_type.to_string();
                reused_tensor.in_use = true;
                return reused_tensor;
            }
        }
        
        // Create new tensor if none available
        let new_tensor = Tensor {
            data: vec![0.0; size],
            shape: shape.to_vec(),
            data_type: data_type.to_string(),
            in_use: true,
        };
        
        self.total_memory_usage += size * std::mem::size_of::<f32>();
        new_tensor
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, mut tensor: Tensor) {
        let size = tensor.data.len();
        tensor.in_use = false;
        
        // Clear data for security
        tensor.data.fill(0.0);
        
        // Add to available pool if not at capacity
        let pool = self.available_tensors.entry(size).or_insert_with(Vec::new);
        if pool.len() < self.max_pool_size {
            pool.push(tensor);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut total_available = 0;
        let total_in_use = 0;
        
        for tensors in self.available_tensors.values() {
            total_available += tensors.len();
        }
        
        PoolStats {
            total_available,
            total_in_use,
            total_memory_usage: self.total_memory_usage,
        }
    }
}

/// Pool statistics
#[derive(Debug)]
pub struct PoolStats {
    pub total_available: usize,
    pub total_in_use: usize,
    pub total_memory_usage: usize,
}

/// Global memory pool
pub static GLOBAL_TENSOR_POOL: Lazy<Arc<RwLock<TensorPool>>> = Lazy::new(|| {
    Arc::new(RwLock::new(TensorPool::new(100))) // Max 100 tensors per size
});

/// Get tensor from global pool
pub fn get_tensor_from_pool(shape: &[usize], data_type: &str) -> Tensor {
    GLOBAL_TENSOR_POOL.write().get_tensor(shape, data_type)
}

/// Return tensor to global pool
pub fn return_tensor_to_pool(tensor: Tensor) {
    GLOBAL_TENSOR_POOL.write().return_tensor(tensor);
}

/// Get global pool statistics
pub fn get_pool_stats() -> PoolStats {
    GLOBAL_TENSOR_POOL.read().stats()
}

/// Zero-copy memory mapping for maximum performance
pub fn map_file(file_path: &std::path::Path) -> SaicResult<memmap2::Mmap> {
    use std::fs::File;
    
    let file = File::open(file_path)
        .map_err(SaicError::Io)?;
    
    unsafe { 
        memmap2::Mmap::map(&file)
            .map_err(|e| SaicError::MemoryMappingFailed(
                format!("Failed to memory map file: {}", e)
            ))
    }
}

/// Ultra-fast memory mapping for validation performance
#[inline(always)]
pub fn map_file_ultra_fast(file_path: &std::path::Path) -> SaicResult<memmap2::Mmap> {
    use std::fs::File;
    
    let file = File::open(file_path)
        .map_err(|_| SaicError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound, 
            "File not found"
        )))?;
    
    unsafe { 
        memmap2::Mmap::map(&file)
            .map_err(|_| SaicError::MemoryMappingFailed(
                "Memory mapping failed".to_string()
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_pool() {
        let mut pool = TensorPool::new(5);
        
        // Get a tensor
        let tensor1 = pool.get_tensor(&[10, 10], "f32");
        assert_eq!(tensor1.shape, vec![10, 10]);
        assert_eq!(tensor1.data_type, "f32");
        assert!(tensor1.in_use);
        
        // Get another tensor
        let tensor2 = pool.get_tensor(&[5, 5], "f32");
        assert_eq!(tensor2.shape, vec![5, 5]);
        assert_eq!(tensor2.data_type, "f32");
        assert!(tensor2.in_use);
        
        // Return one tensor
        pool.return_tensor(tensor1);
        assert!(!tensor1.in_use);
        
        // Get the same tensor again
        let reused_tensor = pool.get_tensor(&[10, 10], "f32");
        assert_eq!(reused_tensor.shape, vec![10, 10]);
        assert_eq!(reused_tensor.data_type, "f32");
        assert!(reused_tensor.in_use);
        assert_eq!(reused_tensor.data, vec![0.0; 100]); // Data should be cleared
    }

    #[test]
    fn test_global_pool() {
        let mut pool = TensorPool::new(5);
        let tensor1 = pool.get_tensor(&[10, 10], "f32");
        let tensor2 = pool.get_tensor(&[5, 5], "f32");
        
        return_tensor_to_pool(tensor1);
        return_tensor_to_pool(tensor2);
        
        let stats = get_pool_stats();
        assert_eq!(stats.total_available, 2);
        assert_eq!(stats.total_in_use, 0);
        assert_eq!(stats.total_memory_usage, 100 * 4 + 25 * 4); // 100 * 4 for 10x10, 25 * 4 for 5x5
    }

    #[test]
    fn test_memory_mapper() {
        // Test slice mapping
        let test_data = b"Hello, World!";
        let mmap = map_file(&std::path::Path::new("test_file")).unwrap();
        
        assert_eq!(mmap.len(), test_data.len());
        assert_eq!(&mmap[..], test_data);
    }
} 