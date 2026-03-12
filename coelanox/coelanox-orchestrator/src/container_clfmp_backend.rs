//! Container CLFMP Backend
//!
//! ProtectionHAL implementation that uses container-embedded CLFMP for
//! configure_region / enable_protection / disable_protection.
//! The container configures protection itself via these blobs; OS-independent.
//!
//! Op_ids (per-kind namespace, 1-based): 1 = configure(ptr, size, flags), 2 = enable(), 3 = disable().
//! Flags: u8 with bit0=read, bit1=write, bit2=execute.

use std::sync::Mutex;

use coelanox_clf::{ClfKind, ClfReaderFromBytes};
use coelanox_core::format::CoelanoxContainer;
use coelanox_core::{
    memory_hal::MemoryRegion,
    protection_hal::{ProtectionCapabilities, ProtectionHAL, RegionPermissions},
    CoelanoxError, CoelanoxResult,
};
use tracing::debug;

const OP_ID_CONFIGURE: u32 = 1;
const OP_ID_ENABLE: u32 = 2;
const OP_ID_DISABLE: u32 = 3;

type ConfigureFn = unsafe extern "sysv64" fn(*mut u8, usize, u8) -> ();
type EnableFn = unsafe extern "sysv64" fn() -> ();
type DisableFn = unsafe extern "sysv64" fn() -> ();

fn permissions_to_flags(perms: RegionPermissions) -> u8 {
    let mut f = 0u8;
    if perms.read {
        f |= 1;
    }
    if perms.write {
        f |= 2;
    }
    if perms.execute {
        f |= 4;
    }
    f
}

/// Protection HAL backend that uses container-embedded CLFMP.
pub struct ContainerClfmpBackend {
    configure_fn: Mutex<Option<ConfigureFn>>,
    enable_fn: Mutex<Option<EnableFn>>,
    disable_fn: Mutex<Option<DisableFn>>,
    #[allow(dead_code)]
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
        return Err(CoelanoxError::Runtime(
            "bootstrap mmap failed for CLFMP".to_string(),
        ));
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
    let result = unsafe {
        VirtualAlloc(
            std::ptr::null_mut(),
            aligned,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
    };
    if result.is_null() {
        return Err(CoelanoxError::Runtime(
            "bootstrap VirtualAlloc failed for CLFMP".to_string(),
        ));
    }
    Ok((result as *mut u8, aligned))
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn bootstrap_alloc(_size: usize) -> CoelanoxResult<(*mut u8, usize)> {
    Err(CoelanoxError::Runtime(
        "ContainerClfmpBackend requires Linux or Windows for bootstrap".to_string(),
    ))
}

fn bootstrap_free(ptr: *mut u8, size: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::munmap(ptr as *mut std::ffi::c_void, size);
    }
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

impl ContainerClfmpBackend {
    /// Create from container's embedded CLFMP. Returns None if container has no CLFMP.
    pub fn from_container(container: &CoelanoxContainer) -> CoelanoxResult<Option<Self>> {
        let clfmp_data = match &container.clfmp_data {
            Some(d) if !d.is_empty() => d,
            _ => return Ok(None),
        };

        let reader = ClfReaderFromBytes::open(clfmp_data.as_slice(), Some(ClfKind::MemoryProtection))
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMP parse failed: {}", e)))?;

        let configure_blob = reader
            .get_blob(OP_ID_CONFIGURE)
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMP get_blob(configure) failed: {}", e)))?
            .ok_or_else(|| {
                CoelanoxError::Runtime(format!(
                    "CLFMP missing op_id {} (configure_region)",
                    OP_ID_CONFIGURE
                ))
            })?;

        let enable_blob = reader
            .get_blob(OP_ID_ENABLE)
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMP get_blob(enable) failed: {}", e)))?
            .ok_or_else(|| {
                CoelanoxError::Runtime(format!(
                    "CLFMP missing op_id {} (enable_protection)",
                    OP_ID_ENABLE
                ))
            })?;

        let disable_blob = reader
            .get_blob(OP_ID_DISABLE)
            .map_err(|e| CoelanoxError::Runtime(format!("CLFMP get_blob(disable) failed: {}", e)))?
            .ok_or_else(|| {
                CoelanoxError::Runtime(format!(
                    "CLFMP missing op_id {} (disable_protection)",
                    OP_ID_DISABLE
                ))
            })?;

        let chunk = 4096usize;
        let total = chunk * 3;
        let (ptr, size) = bootstrap_alloc(total)?;

        let mut offset = 0usize;
        let cfg_len = configure_blob.len().min(chunk);
        unsafe {
            std::ptr::copy_nonoverlapping(configure_blob.as_ptr(), ptr.add(offset), cfg_len);
        }
        let configure_fn: ConfigureFn = unsafe { std::mem::transmute(ptr.add(offset)) };
        offset += chunk;

        let en_len = enable_blob.len().min(chunk);
        unsafe {
            std::ptr::copy_nonoverlapping(enable_blob.as_ptr(), ptr.add(offset), en_len);
        }
        let enable_fn: EnableFn = unsafe { std::mem::transmute(ptr.add(offset)) };
        offset += chunk;

        let dis_len = disable_blob.len().min(chunk);
        unsafe {
            std::ptr::copy_nonoverlapping(disable_blob.as_ptr(), ptr.add(offset), dis_len);
        }
        let disable_fn: DisableFn = unsafe { std::mem::transmute(ptr.add(offset)) };

        let backend = Self {
            configure_fn: Mutex::new(Some(configure_fn)),
            enable_fn: Mutex::new(Some(enable_fn)),
            disable_fn: Mutex::new(Some(disable_fn)),
            bootstrap_region: Mutex::new(Some(BootstrapRegion { ptr, size })),
        };

        debug!("ContainerClfmpBackend loaded from container-embedded CLFMP");
        Ok(Some(backend))
    }
}

impl ProtectionHAL for ContainerClfmpBackend {
    fn configure_region(
        &self,
        region: &dyn MemoryRegion,
        permissions: RegionPermissions,
    ) -> CoelanoxResult<()> {
        let configure_fn = self.configure_fn.lock().unwrap().ok_or_else(|| {
            CoelanoxError::Runtime("ContainerClfmpBackend configure fn not loaded".to_string())
        })?;
        let ptr = region.as_ptr();
        let size = region.size();
        let flags = permissions_to_flags(permissions);
        unsafe {
            configure_fn(ptr, size, flags);
        }
        debug!("ContainerClfmpBackend configured region {} bytes with flags {:b}", size, flags);
        Ok(())
    }

    fn enable_protection(&self) -> CoelanoxResult<()> {
        let enable_fn = self.enable_fn.lock().unwrap().ok_or_else(|| {
            CoelanoxError::Runtime("ContainerClfmpBackend enable fn not loaded".to_string())
        })?;
        unsafe {
            enable_fn();
        }
        debug!("ContainerClfmpBackend enable_protection called");
        Ok(())
    }

    fn disable_protection(&self) -> CoelanoxResult<()> {
        let disable_fn = self.disable_fn.lock().unwrap().ok_or_else(|| {
            CoelanoxError::Runtime("ContainerClfmpBackend disable fn not loaded".to_string())
        })?;
        unsafe {
            disable_fn();
        }
        debug!("ContainerClfmpBackend disable_protection called");
        Ok(())
    }

    fn get_capabilities(&self) -> ProtectionCapabilities {
        ProtectionCapabilities {
            max_regions: 0,
            supports_execute: true,
            min_alignment: 4096,
            name: "container-clfmp".to_string(),
        }
    }

    fn name(&self) -> &str {
        "container-clfmp"
    }
}
