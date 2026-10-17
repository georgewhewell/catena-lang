use anyhow::{Context, Result, ensure};

use super::{MAX_ALLOCATION_BUDGET, MAX_INPUTS, MAX_OUTPUTS, MAX_TRANSFER_BYTES};
use crate::runtime::{Artifact, EntryPoint, Runtime, Value, ValueKind};
use crate::safe_runtime::{
    assets::AssetStore,
    protocol::{ComputeInput, ComputeOutput, WireCompute},
};

/// Validate before allocation, both at the host API and at the worker boundary.
/// Return the private bytes reserved before generated code starts allocating.
pub(in crate::safe_runtime) fn validate(request: &WireCompute, entry: &EntryPoint) -> Result<u64> {
    ensure!(
        !request.entry_point.is_empty() && request.entry_point.len() <= 1024,
        "invalid compute entry point"
    );
    ensure!(
        request.entry_point == entry.name(),
        "compute entry point mismatch"
    );
    ensure!(
        request.inputs.len() <= MAX_INPUTS,
        "too many compute inputs"
    );
    ensure!(
        entry.outputs().len() <= MAX_OUTPUTS,
        "too many compute outputs"
    );
    ensure!(
        !entry.outputs().contains(&ValueKind::MemRef),
        "borrowed compute outputs are unsupported"
    );
    ensure!(
        request.max_output_bytes <= MAX_TRANSFER_BYTES,
        "compute output exceeds transfer limit"
    );
    ensure!(
        request.allocation_budget <= MAX_ALLOCATION_BUDGET,
        "compute allocation budget exceeds its maximum"
    );
    let kinds: Vec<_> = request.inputs.iter().map(input_kind).collect();
    ensure!(entry.inputs() == kinds, "compute input signature mismatch");
    let mut uploaded = 0_u64;
    let mut private = 0_u64;
    for input in &request.inputs {
        let bytes = match input {
            ComputeInput::Buffer(bytes) | ComputeInput::OwnedBuffer(bytes) => {
                uploaded = uploaded
                    .checked_add(bytes.len() as u64)
                    .context("compute input size overflow")?;
                bytes.len() as u64
            }
            ComputeInput::Zeroed(bytes) => *bytes,
            ComputeInput::Asset(slice) => {
                slice
                    .offset
                    .checked_add(slice.byte_len)
                    .context("compute asset range overflow")?;
                0
            }
            _ => 0,
        };
        private = private
            .checked_add(bytes)
            .context("compute private allocation overflow")?;
    }
    ensure!(
        uploaded <= MAX_TRANSFER_BYTES,
        "compute input exceeds transfer limit"
    );
    ensure!(
        private <= request.allocation_budget,
        "compute inputs exceed allocation budget"
    );
    Ok(private)
}

fn input_kind(input: &ComputeInput) -> ValueKind {
    match input {
        ComputeInput::Bool(_) => ValueKind::Bool,
        ComputeInput::U16(_) => ValueKind::U16,
        ComputeInput::U32(_) => ValueKind::U32,
        ComputeInput::U64(_) => ValueKind::U64,
        ComputeInput::F32(_) => ValueKind::F32,
        ComputeInput::Buffer(_) | ComputeInput::Asset(_) => ValueKind::MemRef,
        ComputeInput::OwnedBuffer(_) | ComputeInput::Zeroed(_) => ValueKind::MemOwn,
    }
}

pub(in crate::safe_runtime) fn validate_output(
    output: &[ComputeOutput],
    kinds: &[ValueKind],
    maximum: u64,
) -> Result<()> {
    ensure!(
        output.len() == kinds.len() && output.len() <= MAX_OUTPUTS,
        "compute output arity mismatch"
    );
    let mut total = 0_u64;
    for (output, expected) in output.iter().zip(kinds) {
        let (kind, bytes) = match output {
            ComputeOutput::Bool(_) => (ValueKind::Bool, 1),
            ComputeOutput::U16(_) => (ValueKind::U16, 2),
            ComputeOutput::U32(_) => (ValueKind::U32, 4),
            ComputeOutput::U64(_) => (ValueKind::U64, 8),
            ComputeOutput::F32(_) => (ValueKind::F32, 4),
            ComputeOutput::Buffer(bytes) => (ValueKind::MemOwn, bytes.len() as u64),
        };
        ensure!(kind == *expected, "compute output kind mismatch");
        total = total
            .checked_add(bytes)
            .context("compute output size overflow")?;
        ensure!(total <= maximum, "compute output exceeds byte limit");
    }
    Ok(())
}

pub(in crate::safe_runtime) fn execute(
    runtime: &Runtime,
    artifact: &Artifact,
    assets: &AssetStore,
    request: WireCompute,
    publish: &mut dyn FnMut(crate::runtime::Publication) -> bool,
) -> Result<Vec<ComputeOutput>> {
    let entry = artifact
        .entry_points()
        .iter()
        .find(|e| e.name() == request.entry_point)
        .context("unknown compute entry point")?;
    let private = validate(&request, entry)?;
    // Resolve every asset before allocating inputs or starting generated code.
    let asset_refs = request
        .inputs
        .iter()
        .map(|input| match input {
            ComputeInput::Asset(slice) => assets
                .mem_ref(slice.asset, slice.offset, slice.byte_len)
                .map(Some),
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut borrowed = Vec::with_capacity(request.inputs.len());
    let mut owned = Vec::with_capacity(request.inputs.len());
    for input in &request.inputs {
        let (reference, value) = match input {
            ComputeInput::Buffer(bytes) => (Some(runtime.mem_bytes(bytes)?), None),
            ComputeInput::OwnedBuffer(bytes) => (None, Some(runtime.mem_bytes(bytes)?)),
            ComputeInput::Zeroed(bytes) => (None, Some(runtime.mem_zeroed_bytes(*bytes)?)),
            _ => (None, None),
        };
        borrowed.push(reference);
        owned.push(value);
    }
    let mut values = Vec::with_capacity(request.inputs.len());
    for (index, input) in request.inputs.iter().enumerate() {
        values.push(match input {
            ComputeInput::Bool(v) => Value::from(*v),
            ComputeInput::U16(v) => Value::U16(*v),
            ComputeInput::U32(v) => Value::U32(*v),
            ComputeInput::U64(v) => Value::U64(*v),
            ComputeInput::F32(v) => Value::F32(*v),
            ComputeInput::Buffer(_) => Value::MemRef(
                borrowed[index]
                    .as_ref()
                    .context("missing staged input")?
                    .as_ref(),
            ),
            ComputeInput::OwnedBuffer(_) | ComputeInput::Zeroed(_) => {
                Value::MemOwn(owned[index].take().context("missing owned input")?)
            }
            ComputeInput::Asset(_) => Value::MemRef(asset_refs[index].context("missing asset")?),
        });
    }
    let mut prefixes = vec![Vec::new(); entry.outputs().len()];
    let result = artifact.exec_values_with_publication(
        &request.entry_point,
        values,
        request.allocation_budget - private,
        request.max_output_bytes,
        &mut |part| {
            prefixes[part.output as usize].extend_from_slice(&part.bytes);
            publish(part)
        },
    )?;
    // Inspect all sizes before any potentially large host allocation/download.
    let mut total = 0_u64;
    for value in &result {
        let bytes = match value {
            Value::Bool(v) => {
                ensure!(*v <= 1, "invalid compute boolean");
                1
            }
            Value::U16(_) => 2,
            Value::U32(_) | Value::F32(_) => 4,
            Value::U64(_) => 8,
            Value::MemOwn(value) => value.byte_len(),
            Value::MemRef(_) => anyhow::bail!("borrowed compute output"),
        };
        total = total
            .checked_add(bytes)
            .context("compute output size overflow")?;
        ensure!(
            total <= request.max_output_bytes,
            "compute output exceeds byte limit"
        );
    }
    let output = result
        .into_iter()
        .map(|value| {
            Ok(match value {
                Value::Bool(v) => ComputeOutput::Bool(v != 0),
                Value::U16(v) => ComputeOutput::U16(v),
                Value::U32(v) => ComputeOutput::U32(v),
                Value::U64(v) => ComputeOutput::U64(v),
                Value::F32(v) => ComputeOutput::F32(v),
                Value::MemOwn(value) => ComputeOutput::Buffer(value.try_to_bytes()?),
                Value::MemRef(_) => anyhow::bail!("borrowed compute output"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_output(&output, entry.outputs(), request.max_output_bytes)?;
    for (index, prefix) in prefixes.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        ensure!(
            matches!(&output[index], ComputeOutput::Buffer(bytes) if bytes.starts_with(prefix)),
            "final output differs from its published prefix"
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(inputs: Vec<ComputeInput>) -> WireCompute {
        WireCompute {
            artifact: 0,
            entry_point: "new-program".into(),
            inputs,
            max_output_bytes: 1024,
            allocation_budget: 4096,
        }
    }

    fn signature(inputs: Vec<ValueKind>) -> EntryPoint {
        EntryPoint::new("new-program".into(), inputs, vec![ValueKind::MemOwn])
    }

    #[test]
    fn private_budget_includes_both_kinds_of_upload_and_zero_initialization() {
        let entry = signature(vec![
            ValueKind::MemRef,
            ValueKind::MemOwn,
            ValueKind::MemOwn,
        ]);
        let mut call = request(vec![
            ComputeInput::Buffer(vec![0; 16]),
            ComputeInput::OwnedBuffer(vec![0; 24]),
            ComputeInput::Zeroed(64),
        ]);
        assert_eq!(validate(&call, &entry).unwrap(), 104);
        call.allocation_budget = 103;
        assert!(
            validate(&call, &entry)
                .unwrap_err()
                .to_string()
                .contains("allocation budget")
        );
    }

    #[test]
    fn ownership_is_part_of_the_call_signature() {
        let call = request(vec![ComputeInput::OwnedBuffer(vec![0; 8])]);
        assert!(validate(&call, &signature(vec![ValueKind::MemRef])).is_err());
        assert!(validate(&call, &signature(vec![ValueKind::MemOwn])).is_ok());
    }

    #[test]
    fn invalid_sizes_fail_before_device_allocation() {
        let entry = signature(vec![ValueKind::MemOwn, ValueKind::MemOwn]);
        let call = request(vec![
            ComputeInput::Zeroed(u64::MAX),
            ComputeInput::Zeroed(1),
        ]);
        assert!(
            validate(&call, &entry)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        let mut call = request(Vec::new());
        call.max_output_bytes = MAX_TRANSFER_BYTES + 1;
        assert!(validate(&call, &signature(Vec::new())).is_err());
    }

    #[test]
    fn outputs_are_bounded_in_aggregate_and_checked_by_kind() {
        let output = vec![ComputeOutput::U64(2), ComputeOutput::Buffer(vec![1; 8])];
        let kinds = [ValueKind::U64, ValueKind::MemOwn];
        assert!(validate_output(&output, &kinds, 16).is_ok());
        assert!(validate_output(&output, &kinds, 15).is_err());
        assert!(validate_output(&output, &[ValueKind::F32, ValueKind::MemOwn], 16).is_err());
        assert!(validate_output(&output[..1], &kinds, 16).is_err());
        // Application-specific finite-value restrictions do not belong in transport.
        assert!(validate_output(&[ComputeOutput::F32(f32::NAN)], &[ValueKind::F32], 4).is_ok());
    }

    #[test]
    fn device_faults_in_execution_and_transfers_invalidate_the_worker() {
        use crate::safe_runtime::gpu_error::gpu_synchronization;
        use crate::{
            runtime::GpuDialect,
            runtime::{ExecError, MemError},
        };
        let error =
            anyhow::Error::from(ExecError::GpuSynchronization("fault".into())).context("call");
        assert_eq!(gpu_synchronization(&error), Some("fault"));
        let error = anyhow::Error::from(MemError::GpuOperation {
            dialect: GpuDialect::Cuda,
            operation: "copy device memory to host",
            status: 700,
        })
        .context("download");
        assert!(gpu_synchronization(&error).is_some());
        let error = anyhow::Error::from(MemError::GpuOperation {
            dialect: GpuDialect::Hip,
            operation: "allocate device memory",
            status: 2,
        });
        assert!(gpu_synchronization(&error).is_none());
    }
}
