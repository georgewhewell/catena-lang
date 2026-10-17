//! Calls into prepared programs with worker-owned device memory.
//!
//! Programs supply the computation. This interface knows only Catena value
//! kinds, immutable assets, byte buffers and resource bounds; shapes, formats
//! and the interpretation of results belong to the caller.

use super::{
    AssetSlice, ComputeAssetSlice, ComputeInput, ComputeRequest, Program, ResidentError, Session,
};

pub(super) mod worker;

pub use super::ComputeOutput as Output;
pub use crate::runtime::Publication;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Cancel,
}

/// Maximum aggregate host input bytes or output bytes for one call.
/// Protocol framing and scalar metadata have a separate bounded allowance.
// Leave space for scalar/asset descriptors in the 64 MiB worker frame.
pub const MAX_TRANSFER_BYTES: u64 = 48 * 1024 * 1024;
pub const MAX_INPUTS: usize = 4096;
pub const MAX_OUTPUTS: usize = 64;
/// Maximum cumulative bytes a call's generated code may allocate. Freed
/// buffers are not refunded, so this meters allocation churn, not residency.
pub const MAX_ALLOCATION_BUDGET: u64 = 1 << 44;

#[derive(Debug)]
pub enum Input<'a> {
    Bool(bool),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    /// Uploaded once for this call and borrowed by the program.
    Buffer(&'a [u8]),
    /// Uploaded once and transferred into the program's ownership.
    OwnedBuffer(&'a [u8]),
    /// A new, zero-initialized owned buffer, allocated only in the worker.
    Zeroed(u64),
    /// A read-only slice retained by this session's immutable asset owner.
    Asset(&'a AssetSlice),
}

#[derive(Debug)]
pub struct Request<'a> {
    pub entry_point: &'a str,
    pub inputs: &'a [Input<'a>],
    /// Aggregate returned buffer and scalar bytes, checked before download.
    pub max_output_bytes: u64,
    /// Input staging plus zeroed buffers plus cumulative generated allocations.
    /// Immutable assets are accounted for separately by the asset owner.
    pub allocation_budget: u64,
}

impl Session {
    /// Execute a prepared entry point. No GPU pointers or caller-owned device
    /// allocations cross the process boundary. Failure cannot expose mutable
    /// state to another invocation; only read-only assets survive a call.
    pub fn compute(
        &self,
        program: &Program,
        request: Request<'_>,
    ) -> Result<Vec<Output>, ResidentError> {
        self.compute_streaming(program, request, |_| Control::Continue)
    }

    /// Observe bounded byte deltas before the call completes. A final successful
    /// result verifies every observed prefix. Cancellation invalidates the worker
    /// while preserving its immutable asset owner. The callback must return
    /// promptly and must not re-enter this session; its Rust code is not preemptible.
    pub fn compute_streaming(
        &self,
        program: &Program,
        request: Request<'_>,
        mut publish: impl FnMut(&Publication) -> Control,
    ) -> Result<Vec<Output>, ResidentError> {
        if !program.artifact.belongs_to(self.runtime.id()) {
            return Err(invalid("program belongs to another session"));
        }
        if request.inputs.len() > MAX_INPUTS {
            return Err(invalid("too many compute inputs"));
        }
        let mut uploaded = 0_u64;
        for input in request.inputs {
            match input {
                Input::Buffer(bytes) | Input::OwnedBuffer(bytes) => {
                    uploaded = uploaded
                        .checked_add(bytes.len() as u64)
                        .filter(|&n| n <= MAX_TRANSFER_BYTES)
                        .ok_or_else(|| invalid("compute input exceeds transfer limit"))?;
                }
                Input::Asset(slice) if slice.asset.owner != self.runtime.asset_owner_id() => {
                    return Err(invalid("asset belongs to another owner"));
                }
                _ => {}
            }
        }
        let inputs = request
            .inputs
            .iter()
            .map(|input| match input {
                Input::Bool(v) => ComputeInput::Bool(*v),
                Input::U16(v) => ComputeInput::U16(*v),
                Input::U32(v) => ComputeInput::U32(*v),
                Input::U64(v) => ComputeInput::U64(*v),
                Input::F32(v) => ComputeInput::F32(*v),
                Input::Buffer(v) => ComputeInput::Buffer(v.to_vec()),
                Input::OwnedBuffer(v) => ComputeInput::OwnedBuffer(v.to_vec()),
                Input::Zeroed(v) => ComputeInput::Zeroed(*v),
                Input::Asset(slice) => ComputeInput::Asset(ComputeAssetSlice {
                    asset: slice.asset.resident,
                    offset: slice.offset,
                    byte_len: slice.byte_len,
                }),
            })
            .collect();
        self.runtime.compute(
            &program.artifact,
            ComputeRequest {
                artifact: program.artifact.index(),
                entry_point: request.entry_point.into(),
                inputs,
                max_output_bytes: request.max_output_bytes,
                allocation_budget: request.allocation_budget,
            },
            &mut |part| publish(part) == Control::Continue,
        )
    }
}

fn invalid(message: &str) -> ResidentError {
    ResidentError::InvalidRequest(message.into())
}
