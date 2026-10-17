//! Invocation-scoped observation of bounded output prefixes.
use super::ValueKind;
use crate::gpu::GpuApi;
use serde::{Deserialize, Serialize};
use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
};

pub const MAX_PUBLICATION_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Publication {
    pub output: u32,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

pub(super) struct Publisher<'a> {
    pub gpu: &'a GpuApi,
    pub outputs: &'a [ValueKind],
    pub lengths: Vec<u64>,
    pub maximum_bytes: u64,
    pub total_bytes: u64,
    pub notify: &'a mut dyn FnMut(Publication) -> bool,
}

impl Publisher<'_> {
    fn publish(&mut self, output: u64, data: *const c_void, length: u64) -> bool {
        let gpu = self.gpu;
        self.publish_with(output, length, &mut |offset, bytes| {
            // The generated primitive passes a live owner and checks length <=
            // allocation length. No device pointer escapes the callback scope.
            let source = unsafe { data.cast::<u8>().add(offset as usize).cast() };
            gpu.copy_device_to_host(source, bytes).is_ok()
        })
    }

    fn publish_with(
        &mut self,
        output: u64,
        length: u64,
        read: &mut dyn FnMut(u64, &mut [u8]) -> bool,
    ) -> bool {
        let Ok(index) = usize::try_from(output) else {
            return false;
        };
        if self.outputs.get(index) != Some(&ValueKind::MemOwn) || output > u64::from(u32::MAX) {
            return false;
        }
        let previous = self.lengths[index];
        let Some(additional) = length.checked_sub(previous) else {
            return false;
        };
        let Some(total) = self.total_bytes.checked_add(additional) else {
            return false;
        };
        if total > self.maximum_bytes || usize::try_from(length).is_err() {
            return false;
        }
        let mut offset = previous;
        while offset < length {
            let count = (length - offset).min(MAX_PUBLICATION_CHUNK_BYTES as u64) as usize;
            let mut bytes = vec![0; count];
            if !read(offset, &mut bytes)
                || !(self.notify)(Publication {
                    output: output as u32,
                    offset,
                    bytes,
                })
            {
                return false;
            }
            offset += count as u64;
        }
        self.lengths[index] = length;
        self.total_bytes = total;
        true
    }
}

pub(super) unsafe extern "C" fn callback(
    context: *mut c_void,
    output: u64,
    data: *const c_void,
    bytes: u64,
) -> u8 {
    // The executor installs this pointer only for the synchronous generated call
    // and clears it on every return/unwind. Rust panics never cross the C ABI.
    let publisher = unsafe { &mut *context.cast::<Publisher<'_>>() };
    u8::from(
        catch_unwind(AssertUnwindSafe(|| publisher.publish(output, data, bytes))).unwrap_or(false),
    )
}
