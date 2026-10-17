//! Classify asynchronous device failures that invalidate the execution worker.

use crate::runtime::ExecError;

pub(super) fn gpu_synchronization(error: &anyhow::Error) -> Option<&str> {
    if let Some(ExecError::GpuSynchronization(error)) = error.downcast_ref::<ExecError>() {
        return Some(error.as_str());
    }
    // A failed copy or zero may reflect an asynchronous device fault. An
    // ordinary allocation failure need not invalidate an otherwise healthy GPU.
    if let Some(crate::runtime::MemError::GpuOperation { operation, .. }) =
        error.downcast_ref::<crate::runtime::MemError>()
        && *operation != "allocate device memory"
    {
        return Some(operation);
    }
    None
}
