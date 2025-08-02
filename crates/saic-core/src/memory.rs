//! Memory management utilities for SAIC

use std::alloc::{alloc, dealloc, Layout};
use crate::{SaicError, SaicResult};

/// Aligned memory pool for high-performance operations
/// 
/// Provides pre-allocated, aligned memory pools for zero-copy operations
/// and SIMD-optimized memory access patterns.
pub struct AlignedPool<T> {
    /// Pre-allocated memory buffer
    buffer: Vec<T>,
    
    /// Current allocation offset
    offset: usize,
    
    /// Memory alignment in bytes
    alignment: usize,
}

impl<T> AlignedPool<T> {
    /// Create a new aligned memory pool
    /// 
    /// # Arguments
    /// 
    /// * `capacity` - Maximum number of elements
    /// * `alignment` - Memory alignment in bytes
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        
        // Ensure proper alignment
        if alignment > 1 {
            let ptr = buffer.as_mut_ptr();
            let addr = ptr as usize;
            if addr % alignment != 0 {
                // Reallocate with proper alignment
                let layout = Layout::from_size_align(
                    std::mem::size_of::<T>() * capacity,
                    alignment
                ).unwrap();
                
                unsafe {
                    let new_ptr = alloc(layout);
                    if !new_ptr.is_null() {
                        buffer = Vec::from_raw_parts(
                            new_ptr as *mut T,
                            0,
                            capacity
                        );
                    }
                }
            }
        }
        
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
    /// * `Option<&mut [T]>` - Allocated memory slice, or None if insufficient space
    pub fn allocate(&mut self, size: usize) -> Option<&mut [T]> {
        if self.offset + size <= self.buffer.capacity() {
            let slice = &mut self.buffer[self.offset..self.offset + size];
            self.offset += size;
            Some(slice)
        } else {
            None
        }
    }
    
    /// Reset the pool for reuse
    pub fn reset(&mut self) {
        self.offset = 0;
    }
    
    /// Get current usage statistics
    pub fn usage_stats(&self) -> PoolStats {
        PoolStats {
            capacity: self.buffer.capacity(),
            used: self.offset,
            available: self.buffer.capacity() - self.offset,
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

/// Zero-copy memory mapping utilities
pub struct MemoryMapper;

impl MemoryMapper {
    /// Create a memory-mapped view of a file with zero-copy access
    /// 
    /// # Arguments
    /// 
    /// * `file_path` - Path to the file to memory map
    /// 
    /// # Returns
    /// 
    /// * `SaicResult<memmap2::Mmap>` - Memory-mapped file
    pub fn map_file(file_path: &std::path::Path) -> SaicResult<memmap2::Mmap> {
        use std::fs::File;
        
        let file = File::open(file_path)
            .map_err(SaicError::Io)?;
        
        // Validate file type before memory mapping
        let file_metadata = file.metadata()
            .map_err(SaicError::Io)?;
        
        // Ensure it's a regular file (not a directory, symlink, device, etc.)
        if !file_metadata.is_file() {
            return Err(SaicError::InvalidPath(
                "File is not a regular file (may be a directory, symlink, or device)".to_string()
            ));
        }
        
        // SAFETY: Memory mapping is safe for regular files
        unsafe { 
            memmap2::Mmap::map(&file)
                .map_err(|e| SaicError::MemoryMappingFailed(
                    format!("Failed to memory map file: {}", e)
                ))
        }
    }
    
    /// Create a memory-mapped view of a slice with zero-copy access
    /// 
    /// # Arguments
    /// 
    /// * `data` - Raw data to memory map
    /// 
    /// # Returns
    /// 
    /// * `memmap2::Mmap` - Memory-mapped data
    pub fn map_slice(data: &[u8]) -> memmap2::Mmap {
        // For now, return an empty mmap
        // TODO: Implement proper slice mapping
        unsafe { memmap2::Mmap::map(&std::fs::File::open("/dev/null").unwrap()).unwrap() }
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
} 