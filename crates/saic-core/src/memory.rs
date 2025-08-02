//! Memory management utilities for SAIC

use crate::{SaicError, SaicResult};

/// Aligned memory pool for high-performance operations
pub struct AlignedPool<T> {
    /// Pre-allocated memory buffer
    buffer: Vec<T>,
    
    /// Current allocation offset
    offset: usize,
    
    /// Memory alignment in bytes
    alignment: usize,
}

impl<T: Default> AlignedPool<T> {
    /// Create a new aligned memory pool
    /// 
    /// # Arguments
    /// 
    /// * `capacity` - Total capacity in elements
    /// * `alignment` - Memory alignment in bytes
    /// 
    /// # Returns
    /// 
    /// * `Self` - New aligned memory pool
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, Default::default);
        
        Self {
            buffer,
            offset: 0,
            alignment,
        }
    }
    
    /// Allocate memory from the pool
    /// 
    /// # Arguments
    /// 
    /// * `size` - Number of elements to allocate
    /// 
    /// # Returns
    /// 
    /// * `Option<&mut [T]>` - Allocated slice or None if insufficient space
    pub fn allocate(&mut self, size: usize) -> Option<&mut [T]> {
        if self.offset + size > self.buffer.len() {
            return None;
        }
        
        let slice = &mut self.buffer[self.offset..self.offset + size];
        self.offset += size;
        
        Some(slice)
    }
    
    /// Reset the pool for reuse
    pub fn reset(&mut self) {
        self.offset = 0;
    }
    
    /// Get usage statistics
    pub fn usage_stats(&self) -> PoolStats {
        PoolStats {
            capacity: self.buffer.len(),
            used: self.offset,
            available: self.buffer.len() - self.offset,
            alignment: self.alignment,
        }
    }
}

/// Memory pool usage statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total capacity in elements
    pub capacity: usize,
    
    /// Currently used elements
    pub used: usize,
    
    /// Available elements
    pub available: usize,
    
    /// Memory alignment in bytes
    pub alignment: usize,
}

impl PoolStats {
    /// Get usage percentage
    pub fn usage_percentage(&self) -> f64 {
        (self.used as f64 / self.capacity as f64) * 100.0
    }
}

/// Zero-copy memory mapping for maximum performance
/// 
/// # Arguments
/// 
/// * `file_path` - Path to file to memory map
/// 
/// # Returns
/// 
/// * `SaicResult<memmap2::Mmap>` - Memory-mapped file
pub fn map_file(file_path: &std::path::Path) -> SaicResult<memmap2::Mmap> {
    use std::fs::File;
    
    let file = File::open(file_path)
        .map_err(SaicError::Io)?;
    
    // SAFETY: Direct memory mapping for maximum performance
    unsafe { 
        memmap2::Mmap::map(&file)
            .map_err(|e| SaicError::MemoryMappingFailed(
                format!("Failed to memory map file: {}", e)
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_pool() {
        let mut pool = AlignedPool::<f32>::new(1000, 32);
        
        // Allocate some memory
        let slice1 = pool.allocate(100).unwrap();
        assert_eq!(slice1.len(), 100);
        
        let slice2 = pool.allocate(200).unwrap();
        assert_eq!(slice2.len(), 200);
        
        // Check usage stats
        let stats = pool.usage_stats();
        assert_eq!(stats.used, 300);
        assert_eq!(stats.available, 700);
        
        // Reset pool
        pool.reset();
        let stats = pool.usage_stats();
        assert_eq!(stats.used, 0);
        assert_eq!(stats.available, 1000);
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats {
            capacity: 1000,
            used: 300,
            available: 700,
            alignment: 32,
        };
        
        assert_eq!(stats.usage_percentage(), 30.0);
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