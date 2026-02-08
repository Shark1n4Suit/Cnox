//! Container CLFMM Backend
//!
//! MemoryHAL implementation that uses container-embedded CLFMM for allocate/deallocate/move_memory.
//! The container pulls memory itself via these blobs; no OS-specific allocation in the runtime.
//!
//! Op_ids (per-kind namespace, 1-based): 1 = allocate(size) -> ptr, 2 = deallocate(ptr, size),
//! 3 = move_memory(src, dst, size) for DMA/copy.

use std::sync::Mutex;

use clf::{ClfKind, ClfReaderFromBytes};
use coelanox_core::format::CoelanoxContainer;
use coelanox_core::{
    memory_hal::{
        MemoryCapabilities, MemoryHAL, MemoryRegion, MemoryTarget, MemoryTier, RegionOptions,
    },
    optimization::MaxShapeMetadata,
    CoelanoxError, CoelanoxResult,
};
use tracing::debug;

const OP_ID_ALLOC: u32 = 1;
const OP_ID_DEALLOC: u32 = 2;
const OP_ID_MOVE: u32 = 3;

type AllocFn = unsafe extern "sysv64" fn(usize) -> *mut u8;
type DeallocFn = unsafe extern "sysv64" fn(*mut u8, usize);
type MoveFn = unsafe extern "sysv64" fn(*mut u8, *mut u8, usize) -> ();

struct ClfmmRegion {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for ClfmmRegion {}
unsafe impl Sync for ClfmmRegion {}

impl MemoryRegion for ClfmmRegion {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn size(&self) -> usize {
        self.size
    }
}

/// Memory HAL backend that uses container-embedded CLFMM. Requires bootstrap for loading blobs.
pub struct ContainerClfmmBackend {
    alloc_fn: Mutex<Option<AllocFn>>,
    dealloc_fn: Mutex<Option<DeallocFn>>,
    move_fn: Mutex<Option<MoveFn>>,
    bootstrap_region: Mutex<Option<BootstrapRegion>>,
}

struct BootstrapRegion {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for BootstrapRegion {}

impl Drop for BootstrapRegion {
    fn drop(&mut self) {
        bootstrap_free(self.ptr, self.size);
    }
}

#[cfg(target_os = "linux")]
fn bootstrap_alloc(size: usize) -> CoelanoxResult<(*mut u8, usize)> {
    use libc::{mmap, MAP_PRIVATE, MAP_ANONYMOUS, PROT_READ, PROT_WRITE, PROT_EXEC};
    let page_size = 4096;
    let aligned = ((size + page_size - 1) / page_size) * page_size;
    let result = unsafe {
        mmap(
            std::ptr::null_mut(),
            aligned,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if result == libc::MAP_FAILED {
        return Err(CoelanoxError::Runtime("bootstrap mmap failed for CLFMM".to_string()));
    }
    Ok((result as *mut u8, aligned))
}

#[cfg(target_os = "windows")]
fn bootstrap_alloc(size: usize) -> CoelanoxResult<(*mut u8, usize)> {
    use windows_sys::Win32::System::Memory::{
        VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE,
    };
    let page_size = 4096;
    let aligned = ((size + page_size - 1) / page_size) * page_size;
    let result = unsafe { VirtualAlloc(std::ptr::null_mut(), aligned, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) };
    if result.is_null() {
        return Err(CoelanoxError::Runtime("bootstrap VirtualAlloc failed for CLFMM".to_string()));
    }
    Ok((result as *mut u8, aligned))
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn bootstrap_alloc(_size: usize) -> CoelanoxResult<(*mut u8, usize)> {
    Err(CoelanoxError::Runtime("ContainerClfmmBackend requires Linux or Windows for bootstrap".to_string()))
}

fn bootstrap_free(ptr: *mut u8, size: usize) {
    #[cfg(target_os = "linux")]
    unsafe { libc::munmap(ptr as *mut std::ffi::c_void, size); }
    #[cfg(target_os = "windows")]
    unsafe {
        use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
        let _ = VirtualFree(ptr as *mut std::ffi::c_void, 0, MEM_RELEASE);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        let _ = (ptr, size);
    }
}

impl ContainerClfmmBackend {
    /// Create from container's embedded CLFMM. Returns None if container has no CLFMM.
    pub fn from_container(container: &CoelanoxContainer) -> CoelanoxResult<Option<Self>> {
        let clfmm_data = match &container.clfmm_data {
            Some(d) if !d.is_empty() => d,
            _ => return Ok(None),
        };

        let reader = ClfReaderFromBytes::open(clfmm_data.as_slice(), Some(ClfKind::MemoryMovement))
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMM parse failed: {}", e)))?;

        let alloc_blob = reader
            .get_blob(OP_ID_ALLOC)
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMM get_blob(alloc) failed: {}", e)))?
            .ok_or_else(|| CoelanoxError::Runtime(format!("CLFMM missing op_id {} (allocate)", OP_ID_ALLOC)))?;

        let dealloc_blob = reader
            .get_blob(OP_ID_DEALLOC)
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMM get_blob(dealloc) failed: {}", e)))?
            .ok_or_else(|| CoelanoxError::Runtime(format!("CLFMM missing op_id {} (deallocate)", OP_ID_DEALLOC)))?;

        let move_blob = reader.get_blob(OP_ID_MOVE).ok().flatten();

        let chunk = 4096usize;
        let mut n = 2;
        if move_blob.is_some() {
            n = 3;
        }
        let total = chunk * n;
        let (ptr, size) = bootstrap_alloc(total)?;

        let mut offset = 0usize;
        unsafe {
            std::ptr::copy_nonoverlapping(alloc_blob.as_ptr(), ptr.add(offset), alloc_blob.len().min(chunk));
        }
        let alloc_fn: AllocFn = unsafe { std::mem::transmute(ptr.add(offset)) };
        offset += chunk;

        unsafe {
            std::ptr::copy_nonoverlapping(dealloc_blob.as_ptr(), ptr.add(offset), dealloc_blob.len().min(chunk));
        }
        let dealloc_fn: DeallocFn = unsafe { std::mem::transmute(ptr.add(offset)) };
        offset += chunk;

        let move_fn = if let Some(ref blob) = move_blob {
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), ptr.add(offset), blob.len().min(chunk));
            }
            Some(unsafe { std::mem::transmute(ptr.add(offset)) })
        } else {
            None
        };

        let backend = Self {
            alloc_fn: Mutex::new(Some(alloc_fn)),
            dealloc_fn: Mutex::new(Some(dealloc_fn)),
            move_fn: Mutex::new(move_fn),
            bootstrap_region: Mutex::new(Some(BootstrapRegion { ptr, size })),
        };

        if move_fn.is_some() {
            debug!("ContainerClfmmBackend loaded from container-embedded CLFMM (with move_memory)");
        } else {
            debug!("ContainerClfmmBackend loaded from container-embedded CLFMM (move_memory fallback to memcpy)");
        }
        Ok(Some(backend))
    }
}

impl MemoryHAL for ContainerClfmmBackend {
    fn allocate_region(
        &self,
        size: usize,
        alignment: usize,
        options: &RegionOptions,
    ) -> CoelanoxResult<Box<dyn MemoryRegion>> {
        if !options.executable {
            return Err(CoelanoxError::Runtime(
                "ContainerClfmmBackend only supports executable regions".to_string(),
            ));
        }
        let alloc_fn = self.alloc_fn.lock().unwrap().ok_or_else(|| {
            CoelanoxError::Runtime("ContainerClfmmBackend alloc fn not loaded".to_string())
        })?;
        let aligned_size = ((size + alignment - 1) / alignment) * alignment;
        let ptr = unsafe { alloc_fn(aligned_size) };
        if ptr.is_null() {
            return Err(CoelanoxError::Runtime("CLFMM allocate returned null".to_string()));
        }
        debug!("ContainerClfmmBackend allocated {} bytes", aligned_size);
        Ok(Box::new(ClfmmRegion { ptr, size: aligned_size }))
    }

    fn deallocate_region(&self, region: Box<dyn MemoryRegion>) -> CoelanoxResult<()> {
        let dealloc_fn = self.dealloc_fn.lock().unwrap().ok_or_else(|| {
            CoelanoxError::Runtime("ContainerClfmmBackend dealloc fn not loaded".to_string())
        })?;
        let ptr = region.as_ptr();
        let size = region.size();
        unsafe { dealloc_fn(ptr, size); }
        debug!("ContainerClfmmBackend deallocated {} bytes", size);
        Ok(())
    }

    fn move_memory(
        &self,
        src: &dyn MemoryRegion,
        dst: &dyn MemoryRegion,
        size: usize,
    ) -> CoelanoxResult<()> {
        let s = src.size().min(dst.size()).min(size);
        if let Some(move_fn) = self.move_fn.lock().unwrap().as_ref() {
            unsafe {
                move_fn(src.as_ptr(), dst.as_ptr(), s);
            }
            debug!("ContainerClfmmBackend move_memory {} bytes via CLFMM blob", s);
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), s);
            }
            debug!("ContainerClfmmBackend move_memory {} bytes via memcpy fallback", s);
        }
        Ok(())
    }

    fn create_backing(&self, _metadata: &MaxShapeMetadata) -> CoelanoxResult<Vec<u8>> {
        Err(CoelanoxError::Runtime(
            "ContainerClfmmBackend does not support create_backing".to_string(),
        ))
    }

    fn get_capabilities(&self) -> MemoryCapabilities {
        MemoryCapabilities {
            supported_tiers: vec![MemoryTier::Fast],
            supports_executable: true,
            supports_create_backing: false,
            max_region_size: 0,
            name: "container-clfmm".to_string(),
        }
    }

    fn name(&self) -> &str {
        "container-clfmm"
    }
}
