mod native;

use crate::runtime::{Backend, GpuDialect, MemError};
use native::NativeApi;
pub(crate) use native::{IPC_HANDLE_BYTES, RawIpcMemHandle};
use std::{ffi::c_void, sync::Arc};

/// Process-local device operations. Code generation remains backend-specific.
#[derive(Debug)]
pub(crate) enum GpuApi {
    Native(Arc<NativeApi>),
}

impl GpuApi {
    pub(crate) fn load(backend: Backend) -> Result<Arc<Self>, MemError> {
        let api = match backend {
            Backend::Auto => {
                return Self::load(Backend::Cuda).or_else(|cuda| {
                    Self::load(Backend::Hip).map_err(|hip| MemError::NoBackend {
                        cuda: cuda.to_string(),
                        hip: hip.to_string(),
                    })
                });
            }
            Backend::Hip => Self::Native(NativeApi::load(GpuDialect::Hip)?),
            Backend::Cuda => Self::Native(NativeApi::load(GpuDialect::Cuda)?),
        };
        Ok(Arc::new(api))
    }

    pub(crate) fn dialect(&self) -> GpuDialect {
        match self {
            Self::Native(gpu) => gpu.dialect(),
        }
    }

    pub(crate) fn backend(&self) -> Backend {
        self.dialect().into()
    }

    pub(crate) fn allocate(&self, byte_len: usize) -> Result<*mut c_void, MemError> {
        match self {
            Self::Native(gpu) => gpu.allocate(byte_len),
        }
    }

    pub(crate) fn zero(&self, destination: *mut c_void, byte_len: usize) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.zero(destination, byte_len),
        }
    }

    pub(crate) fn copy_host_to_device(
        &self,
        destination: *mut c_void,
        source: &[u8],
    ) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.copy_host_to_device(destination, source),
        }
    }

    pub(crate) fn copy_device_to_host(
        &self,
        source: *const c_void,
        output: &mut [u8],
    ) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.copy_device_to_host(source, output),
        }
    }

    pub(crate) fn copy_device_to_device(
        &self,
        destination: *mut c_void,
        source: *const c_void,
        byte_len: usize,
    ) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.copy_device_to_device(destination, source, byte_len),
        }
    }

    pub(crate) fn free(&self, data: *mut c_void) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.free(data),
        }
    }

    pub(crate) fn address_range(
        &self,
        data: *const c_void,
    ) -> Result<(*mut c_void, usize), MemError> {
        match self {
            Self::Native(gpu) => gpu.address_range(data),
        }
    }

    pub(crate) fn synchronize(&self) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.synchronize(),
        }
    }

    pub(crate) fn ipc_get_mem_handle(
        &self,
        data: *mut c_void,
    ) -> Result<RawIpcMemHandle, MemError> {
        match self {
            Self::Native(gpu) => gpu.ipc_get_mem_handle(data),
        }
    }
    pub(crate) fn ipc_open_mem_handle(
        &self,
        handle: RawIpcMemHandle,
    ) -> Result<*mut c_void, MemError> {
        match self {
            Self::Native(gpu) => gpu.ipc_open_mem_handle(handle),
        }
    }
    pub(crate) fn ipc_close_mem_handle(&self, data: *mut c_void) -> Result<(), MemError> {
        match self {
            Self::Native(gpu) => gpu.ipc_close_mem_handle(data),
        }
    }
}
