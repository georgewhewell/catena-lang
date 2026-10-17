//! Parent-side session construction and request dispatch.

use super::{
    Artifact, ComputeInput, ComputeOutput, ComputeRequest, ParentInterface, ResidentError,
    SafeExecError, SafeInitError, SafeRuntime, SafeRuntimeTimeouts, compute, initialize_parent_ipc,
    ipc::{IpcTransport, encode_ipc_buffer, import_ipc_buffers, intern_buffer},
    operation::{
        DeferredRelease, SessionOperationGate, SessionOperationKind, SessionOperationLease,
    },
    protocol::{RemoteExecError, Request, Response, WireExecution, WireIpcBuffer, WireValue},
    worker::{
        DeadlineKind, WorkerError, WorkerProcess, ensure_waitable_children, map_exec_worker_error,
        map_init_worker_error, map_resident_teardown_worker_error, map_resident_worker_error,
    },
};
use crate::{
    codegen::CodegenKind,
    runtime::{Backend, ExecError, RuntimeId, RuntimeModule, Value},
};
use std::{
    env,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

impl SafeRuntime {
    /// Construct an empty process-isolated runtime.
    pub fn new(backend: impl Into<Backend>) -> Result<Self, SafeInitError> {
        Self::with_timeouts(backend, SafeRuntimeTimeouts::default())
    }

    /// Construct a runtime whose complete source-load and GPU-compile step is
    /// bounded by `compile_timeout`.
    ///
    /// A timeout kills the isolated worker and its compiler process group. The
    /// runtime must not be reused after [`SafeInitError::CompileTimedOut`].
    pub fn with_compile_timeout(
        backend: impl Into<Backend>,
        compile_timeout: Duration,
    ) -> Result<Self, SafeInitError> {
        Self::with_timeouts(
            backend,
            SafeRuntimeTimeouts::default().with_compile_timeout(compile_timeout),
        )
    }

    /// Construct a runtime with provider-selected compile and execution limits.
    pub fn with_timeouts(
        backend: impl Into<Backend>,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_interface(backend.into(), timeouts, ParentInterface::RawValues)
    }

    /// Construct the worker-resident protocol used by [`super::Session`] without
    /// loading GPU IPC or native GPU libraries in the parent process.
    pub(crate) fn with_resident_timeouts(
        backend: impl Into<Backend>,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_resident_backend(backend.into(), timeouts)
    }

    pub(crate) fn with_resident_backend(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        let owner = Arc::new(Self::with_asset_owner(backend, timeouts)?);
        Self::with_shared_assets(owner, timeouts)
    }

    pub(crate) fn with_asset_owner(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        Self::with_interface(backend, timeouts, ParentInterface::AssetOwner)
    }

    pub(crate) fn with_asset_owner_executable(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
        executable: &Path,
    ) -> Result<Self, SafeInitError> {
        Self::with_interface_executable(backend, timeouts, ParentInterface::AssetOwner, executable)
    }

    pub(crate) fn with_shared_assets(
        owner: Arc<Self>,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        let mut runtime = Self::with_interface_executable(
            owner.backend,
            timeouts,
            ParentInterface::ResidentProtocol,
            &owner.executable,
        )?;
        runtime.asset_owner = Some(owner);
        Ok(runtime)
    }

    pub(crate) fn asset_owner_id(&self) -> RuntimeId {
        self.asset_owner
            .as_ref()
            .map_or(self.runtime_id, |owner| owner.runtime_id)
    }

    pub(crate) fn asset_owner_is_available(&self) -> bool {
        let owner = self.asset_owner.as_deref().unwrap_or(self);
        let Ok(worker) = owner.worker.lock() else {
            return false;
        };
        if worker.ensure_available().is_err() {
            return false;
        }
        let Some(child) = worker.child.as_ref() else {
            return false;
        };
        // Observe without reaping: the lifecycle gate must retain the numeric
        // process-group identity until terminal cleanup has killed descendants.
        let mut status = std::mem::MaybeUninit::<libc::siginfo_t>::zeroed();
        unsafe {
            libc::waitid(
                libc::P_PID,
                child.id(),
                status.as_mut_ptr(),
                libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
            ) == 0
                && status.assume_init().si_pid() == 0
        }
    }

    pub(super) fn with_interface(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
        interface: ParentInterface,
    ) -> Result<Self, SafeInitError> {
        let executable = env::current_exe().map_err(SafeInitError::CurrentExecutable)?;
        Self::with_interface_executable(backend, timeouts, interface, &executable)
    }

    pub(super) fn with_interface_executable(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
        interface: ParentInterface,
        executable: &Path,
    ) -> Result<Self, SafeInitError> {
        if !executable.is_absolute() {
            return Err(SafeInitError::InvalidExecutable);
        }
        if timeouts.compile_timeout().is_zero() {
            return Err(SafeInitError::InvalidCompileTimeout);
        }
        if timeouts.execution_timeout().is_zero() {
            return Err(SafeInitError::InvalidExecutionTimeout);
        }
        ensure_waitable_children()?;
        let parent_ipc = initialize_parent_ipc(interface, || IpcTransport::load(backend))?;
        let backend = parent_ipc.as_ref().map_or(backend, IpcTransport::backend);
        let mut worker = WorkerProcess::spawn(executable)?;

        match worker
            .with_deadline(
                timeouts.execution_timeout(),
                DeadlineKind::Initialize,
                |worker, deadline| {
                    worker.request(
                        &if interface == ParentInterface::AssetOwner {
                            Request::InitializeAssets { backend }
                        } else {
                            Request::Initialize { backend }
                        },
                        deadline,
                    )
                },
            )
            .map_err(map_init_worker_error)?
        {
            Response::Initialized(Ok(selected))
                if backend != Backend::Auto && backend != selected =>
            {
                Err(SafeInitError::UnexpectedResponse)
            }
            Response::Initialized(Ok(backend)) => Ok(Self {
                backend,
                executable: executable.to_owned(),
                worker: Arc::new(Mutex::new(worker)),
                asset_owner: None,
                operation_gate: Arc::new(SessionOperationGate::default()),
                parent_ipc,
                runtime_id: RuntimeId::new(),
                timeouts,
            }),
            Response::Initialized(Err(error)) => Err(SafeInitError::RemoteInitialization(error)),
            _ => Err(SafeInitError::UnexpectedResponse),
        }
    }

    /// Load a rendered runtime module into the isolated worker.
    pub fn load(&self, module: RuntimeModule) -> Result<Artifact, SafeInitError> {
        self.load_request(Request::LoadModule { module })
    }

    /// Compile and load Catena source inside the isolated worker.
    ///
    /// Compilation runs under the compile deadline, so hosts can accept
    /// untrusted source without running the compiler in their own process.
    pub fn load_sources(
        &self,
        sources: impl IntoIterator<Item = impl AsRef<str>>,
        codegen: CodegenKind,
    ) -> Result<Artifact, SafeInitError> {
        let sources = sources
            .into_iter()
            .map(|source| source.as_ref().to_owned())
            .collect();
        self.load_owned_sources(sources, codegen)
    }

    pub(super) fn load_owned_sources(
        &self,
        sources: Vec<String>,
        codegen: CodegenKind,
    ) -> Result<Artifact, SafeInitError> {
        self.load_request(Request::LoadSources { sources, codegen })
    }

    fn load_request(&self, request: Request) -> Result<Artifact, SafeInitError> {
        let _operation = self.begin_operation().map_err(map_init_worker_error)?;
        let worker = self
            .worker
            .lock()
            .map_err(|_| SafeInitError::Transport("worker lock was poisoned".to_string()))?;
        let mut worker = worker;
        match worker
            .with_deadline(
                self.timeouts.compile_timeout(),
                DeadlineKind::Compile,
                |worker, deadline| worker.request(&request, deadline),
            )
            .map_err(map_init_worker_error)?
        {
            Response::Loaded(Ok((index, entry_points))) => {
                Ok(Artifact::new(self.clone(), index, entry_points.into()))
            }
            Response::Loaded(Err(error)) => Err(SafeInitError::RemoteLoad(error)),
            _ => Err(worker.reject_protocol_response(SafeInitError::UnexpectedResponse)),
        }
    }

    pub(crate) fn backend(&self) -> Backend {
        self.backend
    }

    pub(crate) fn id(&self) -> RuntimeId {
        self.runtime_id
    }

    pub(super) fn begin_operation(&self) -> Result<SessionOperationLease, WorkerError> {
        self.operation_gate
            .acquire(SessionOperationKind::Ordinary)
            .map_err(|()| WorkerError::ReentrantOperation)
    }

    pub(super) fn begin_streaming(&self) -> Result<SessionOperationLease, WorkerError> {
        self.operation_gate
            .acquire(SessionOperationKind::Streaming)
            .map_err(|()| WorkerError::ReentrantOperation)
    }

    /// Execute one complete program with bounded publication callbacks.
    pub(crate) fn compute(
        &self,
        artifact: &Artifact,
        request: ComputeRequest,
        publish: &mut dyn FnMut(&crate::runtime::Publication) -> bool,
    ) -> Result<Vec<ComputeOutput>, ResidentError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(ResidentError::InvalidRequest(
                "program belongs to another session".into(),
            ));
        }
        let entry = artifact
            .entry_points()
            .iter()
            .find(|e| e.name() == request.entry_point)
            .ok_or_else(|| ResidentError::InvalidRequest("unknown compute entry point".into()))?;
        compute::worker::validate(&request, entry)
            .map_err(|e| ResidentError::InvalidRequest(e.to_string()))?;
        let maximum = request.max_output_bytes;
        let slices = request
            .inputs
            .iter()
            .filter_map(|input| match input {
                ComputeInput::Asset(slice) => Some(*slice),
                _ => None,
            })
            .collect::<Vec<_>>();
        let operation = self.begin_streaming().map_err(map_resident_worker_error)?;
        let mut worker = self.resident_worker()?;
        self.import_resident_assets(&mut worker, &slices)?;
        let mut prefixes = vec![Vec::<u8>::new(); entry.outputs().len()];
        let mut published = 0_u64;
        let mut cancelled = false;
        let mut callback_panicked = false;
        let mut invalid = false;
        let response = worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    worker.send(&Request::Compute { request }, deadline)?;
                    operation.open_artifact_release_deferrals();
                    loop {
                        let response = worker.receive(deadline)?;
                        let Response::Published(part) = response else {
                            // The invocation has finished, so destructors may release
                            // their programs without interleaving protocol frames.
                            // Keep the invocation deadline armed through every ack.
                            for release in operation.seal_artifact_release_deferrals() {
                                let released = worker
                                    .request(&release.request(), deadline)
                                    .map_err(|_| WorkerError::Invalidated)?;
                                if !release.acknowledged(&released) {
                                    invalid = true;
                                    return Err(WorkerError::Invalidated);
                                }
                            }
                            return Ok(response);
                        };
                        let index = part.output as usize;
                        if entry.outputs().get(index) != Some(&crate::runtime::ValueKind::MemOwn)
                            || part.bytes.is_empty()
                            || part.bytes.len() > crate::runtime::MAX_PUBLICATION_CHUNK_BYTES
                            || prefixes[index].len() as u64 != part.offset
                            || published
                                .checked_add(part.bytes.len() as u64)
                                .is_none_or(|n| n > maximum)
                        {
                            invalid = true;
                            return Err(WorkerError::Invalidated);
                        }
                        published += part.bytes.len() as u64;
                        prefixes[index].extend_from_slice(&part.bytes);
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            publish(&part)
                        })) {
                            Ok(true) => (),
                            Ok(false) => {
                                cancelled = true;
                                return Err(WorkerError::Invalidated);
                            }
                            Err(_) => {
                                callback_panicked = true;
                                return Err(WorkerError::Invalidated);
                            }
                        }
                    }
                },
            )
            .map_err(map_resident_worker_error);
        if cancelled {
            return Err(ResidentError::Cancelled);
        }
        if callback_panicked {
            return Err(ResidentError::CallbackPanicked);
        }
        if invalid {
            return Err(ResidentError::UnexpectedResponse);
        }
        match response? {
            Response::Computed(Ok(values))
                if compute::worker::validate_output(&values, entry.outputs(), maximum).is_ok()
                    && prefixes.iter().enumerate().all(|(index, prefix)| prefix.is_empty()
                        || matches!(&values[index], ComputeOutput::Buffer(bytes) if bytes.starts_with(prefix))) => Ok(values),
            Response::Computed(Err(error)) => Err(ResidentError::Remote(error)),
            _ => Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse)),
        }
    }

    pub(super) fn release_object(&self, release: DeferredRelease) -> Result<(), ResidentError> {
        let Some(_operation) = self
            .operation_gate
            .acquire_or_defer_release(release)
            .map_err(|()| map_resident_worker_error(WorkerError::ReentrantOperation))?
        else {
            return Ok(());
        };
        let mut worker = worker_for_artifact_teardown(self.worker.lock())?;
        if let Some(termination) = worker.termination() {
            return Err(ResidentError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        let response = match worker.with_deadline(
            self.timeouts.execution_timeout(),
            DeadlineKind::Execution,
            |worker, deadline| worker.request(&release.request(), deadline),
        ) {
            Ok(response) => response,
            Err(error) => {
                // Teardown is stricter than an ordinary request: even a
                // provably local preparation failure would leave the artifact
                // resident, so no other handle may continue on this worker.
                worker.invalidate_after_protocol_failure();
                return Err(map_resident_teardown_worker_error(error));
            }
        };
        match response {
            ref response if release.acknowledged(response) => Ok(()),
            Response::ArtifactReleased(Err(error)) => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::Teardown(error))
            }
            _ => {
                worker.invalidate_after_protocol_failure();
                Err(ResidentError::UnexpectedResponse)
            }
        }
    }

    pub(super) fn resident_worker(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, WorkerProcess>, ResidentError> {
        let worker = self
            .worker
            .lock()
            .map_err(|_| ResidentError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(ResidentError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        Ok(worker)
    }

    /// Run a source-level program with dynamically sized inputs and outputs.
    pub(super) fn exec_values<'a>(
        &self,
        artifact: &Artifact,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        if !artifact.belongs_to(self.runtime_id) {
            return Err(SafeExecError::UnknownArtifact);
        }
        let _operation = self.begin_operation().map_err(map_exec_worker_error)?;
        let ipc = self.parent_ipc()?;
        let (buffers, wire_args) = Self::encode_parent_arguments(ipc, &args)?;
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| SafeExecError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(SafeExecError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }

        let response = worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    worker.request(
                        &Request::Execute {
                            artifact: artifact.index(),
                            name: name.to_string(),
                            buffers,
                            args: wire_args,
                        },
                        deadline,
                    )
                },
            )
            .map_err(map_exec_worker_error)?;
        let execution = execution_from_response(&mut worker, response)?;
        let values = Self::decode_child_outputs(ipc, execution);
        let released = worker.with_deadline(
            self.timeouts.execution_timeout(),
            DeadlineKind::Execution,
            |worker, deadline| worker.request(&Request::ReleaseOutputs, deadline),
        );
        match released {
            Ok(Response::OutputsReleased) => match values {
                Err(error @ (SafeExecError::Transport(_) | SafeExecError::UnexpectedResponse)) => {
                    Err(worker.reject_protocol_response(error))
                }
                result => result,
            },
            Ok(_) => {
                worker.invalidate_after_protocol_failure();
                Err(SafeExecError::UnexpectedResponse)
            }
            Err(error) => {
                worker.invalidate_after_protocol_failure();
                Err(map_exec_worker_error(error))
            }
        }
    }

    /// Exports parent-owned arguments as views for the child to copy into its own allocations.
    pub(super) fn parent_ipc(&self) -> Result<&IpcTransport, SafeExecError> {
        self.parent_ipc.as_ref().ok_or_else(|| {
            SafeExecError::InvalidRequest(
                "raw Value execution is unavailable on a resident-protocol runtime".to_string(),
            )
        })
    }

    pub(super) fn encode_parent_arguments(
        ipc: &IpcTransport,
        args: &[Value<'_>],
    ) -> Result<(Vec<WireIpcBuffer>, Vec<WireValue>), SafeExecError> {
        let mut buffers = Vec::new();
        let mut values = Vec::with_capacity(args.len());
        for (index, value) in args.iter().enumerate() {
            let wire = match value {
                Value::Bool(value) => WireValue::Bool(*value),
                Value::U16(value) => WireValue::U16(*value),
                Value::U32(value) => WireValue::U32(*value),
                Value::U64(value) => WireValue::U64(*value),
                Value::F32(value) => WireValue::F32(*value),
                Value::MemOwn(memory) => {
                    if memory.backend() != ipc.backend() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = ipc.export_view(memory.as_ref())?;
                    let buffer_index = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                    WireValue::MemOwn {
                        buffer: buffer_index,
                        view_offset: exported.view_offset(),
                        byte_len: exported.byte_len(),
                    }
                }
                Value::MemRef(memory) => {
                    if memory.backend() != ipc.backend() {
                        return Err(SafeExecError::Runtime(
                            ExecError::IncompatibleDeviceMemory { index },
                        ));
                    }
                    let exported = ipc.export_view(*memory)?;
                    let buffer_index = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                    WireValue::MemRef {
                        buffer: buffer_index,
                        view_offset: exported.view_offset(),
                        byte_len: exported.byte_len(),
                    }
                }
            };
            values.push(wire);
        }
        if !buffers.is_empty() {
            ipc.synchronize()?;
        }
        Ok((buffers, values))
    }

    /// Copies child-owned outputs into parent-owned allocations before releasing them remotely.
    pub(super) fn decode_child_outputs(
        ipc: &IpcTransport,
        execution: WireExecution,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        let imported = import_ipc_buffers(ipc, execution.buffers).map_err(|error| {
            SafeExecError::Transport(format!("child memory IPC failed: {error}"))
        })?;
        execution
            .values
            .into_iter()
            .map(|value| match value {
                WireValue::Bool(value) => Ok(Value::Bool(value)),
                WireValue::U16(value) => Ok(Value::U16(value)),
                WireValue::U32(value) => Ok(Value::U32(value)),
                WireValue::U64(value) => Ok(Value::U64(value)),
                WireValue::F32(value) => Ok(Value::F32(value)),
                WireValue::MemOwn {
                    buffer,
                    view_offset,
                    byte_len,
                } => imported
                    .get(buffer)
                    .ok_or_else(|| SafeExecError::Transport("invalid IPC memory view".to_string()))?
                    .copy_view_into_owned(view_offset, byte_len)?
                    .map(Value::MemOwn)
                    .ok_or_else(|| SafeExecError::Transport("invalid IPC memory view".to_string())),
                WireValue::MemRef { .. } => Err(SafeExecError::UnexpectedResponse),
            })
            .collect()
    }
}

pub(super) fn worker_for_artifact_teardown(
    worker: std::sync::LockResult<std::sync::MutexGuard<'_, WorkerProcess>>,
) -> Result<std::sync::MutexGuard<'_, WorkerProcess>, ResidentError> {
    match worker {
        Ok(worker) => Ok(worker),
        Err(poisoned) => {
            // A panic while the worker lock was held leaves request progress
            // unknowable. Recover ownership solely to terminate the process;
            // silently abandoning a live artifact here would defeat the
            // teardown acknowledgement protocol.
            let mut worker = poisoned.into_inner();
            worker.invalidate_after_protocol_failure();
            Err(ResidentError::Transport(
                "worker lock was poisoned during artifact teardown".to_string(),
            ))
        }
    }
}

pub(super) fn execution_from_response(
    worker: &mut WorkerProcess,
    response: Response,
) -> Result<WireExecution, SafeExecError> {
    match response {
        Response::Executed(Ok(execution)) => Ok(execution),
        Response::Executed(Err(RemoteExecError::UnknownArtifact)) => {
            Err(SafeExecError::UnknownArtifact)
        }
        Response::Executed(Err(RemoteExecError::Runtime(error))) => {
            Err(SafeExecError::Runtime(error))
        }
        Response::Executed(Err(RemoteExecError::Memory(error))) => {
            Err(SafeExecError::RemoteMemory(error))
        }
        Response::Executed(Err(RemoteExecError::Protocol(error))) => Err(worker
            .reject_protocol_response(SafeExecError::Transport(format!(
                "child execution protocol failed: {error}"
            )))),
        _ => Err(worker.reject_protocol_response(SafeExecError::UnexpectedResponse)),
    }
}

mod assets;
