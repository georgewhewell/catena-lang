//! Worker entrypoint and request dispatch.

use super::{
    CHILD_ASSET_SOCKET_FD, CHILD_MODE_ENV, ChildMainError,
    assets::AssetStore,
    compute,
    fd_transport::{receive_file, send_file},
    gpu_error,
    ipc::{IpcTransport, encode_ipc_buffer, import_ipc_buffers, intern_buffer},
    protocol::{
        ProtocolError, RemoteExecError, Request, Response, WireExecution, WireIpcBuffer, WireValue,
        read_frame, write_frame,
    },
};
use crate::{
    codegen::CodegenKind,
    compile,
    report::CompileReport,
    runtime::{Artifact as RuntimeArtifact, EntryPoint, ExecError, MemOwn, Runtime, Value},
};
use metacat::theory::RawTheorySet;
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{self, Read},
    os::{
        fd::{FromRawFd, OwnedFd, RawFd},
        unix::net::UnixStream,
    },
};

/// Run the SafeRuntime child loop when this executable was spawned as a worker.
///
/// Call this before argument parsing or writing to stdout. The return value is
/// `false` for a normal invocation and `true` after worker mode finishes.
/// Call it before starting threads or child processes: in worker mode fd 3 is
/// the inherited asset socket. The entrypoint duplicates that descriptor for
/// owned use and marks the original close-on-exec without closing it, preventing
/// compiler inheritance without claiming a descriptor from a spoofed caller.
pub fn run_safe_runtime_child_if_requested() -> Result<bool, ChildMainError> {
    if env::var_os(CHILD_MODE_ENV).is_none() {
        return Ok(false);
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    // Worker mode is selected through the environment, which is not itself
    // proof that fd 3 is valid or uniquely owned. Duplicate it first: the
    // syscall rejects an invalid descriptor, and only the returned descriptor
    // is transferred into the owned UnixStream below. The bootstrap descriptor
    // is marked close-on-exec, not closed, so this safe entrypoint neither
    // claims nor double-closes a descriptor owned by a spoofed caller while a
    // genuine worker's compiler descendants cannot retain the socket endpoint.
    let asset_socket = duplicate_unix_stream(CHILD_ASSET_SOCKET_FD).map_err(|error| {
        ChildMainError::Protocol(format!("invalid worker asset socket: {error}"))
    })?;
    #[cfg(target_os = "macos")]
    raise_descriptor_limit();
    run_child_loop(stdin.lock(), stdout.lock(), &asset_socket)?;
    Ok(true)
}

/// A worker holds one descriptor per shared asset allocation, and a host may
/// attach hundreds of assets; macOS starts processes with a soft limit of 256
/// open files. Take the hard limit, bounded by OPEN_MAX (Darwin
/// rejects a larger soft limit). Best effort: an asset that still exceeds the
/// limit fails with its own error.
#[cfg(target_os = "macos")]
fn raise_descriptor_limit() {
    // <sys/syslimits.h> OPEN_MAX; the libc crate does not export it.
    const OPEN_MAX: libc::rlim_t = 10240;
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: plain syscalls on a local struct.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) != 0 {
            return;
        }
        let wanted = limit.rlim_max.min(OPEN_MAX);
        if limit.rlim_cur < wanted {
            limit.rlim_cur = wanted;
            libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
        }
    }
}

pub(super) fn duplicate_unix_stream(descriptor: RawFd) -> io::Result<UnixStream> {
    let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 0) };
    if duplicate == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `F_DUPFD_CLOEXEC` returned a fresh, valid descriptor that this
    // function now exclusively owns.
    let duplicate = unsafe { OwnedFd::from_raw_fd(duplicate) };
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags == -1
        || unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == -1
    {
        return Err(io::Error::last_os_error());
    }
    Ok(UnixStream::from(duplicate))
}

pub(super) fn run_child_loop(
    mut reader: impl Read,
    mut writer: impl io::Write,
    asset_socket: &UnixStream,
) -> Result<(), ChildMainError> {
    let request = read_request(&mut reader)?.ok_or(ChildMainError::ExpectedInitialization)?;
    let (backend, asset_owner) = match request {
        Request::Initialize { backend } => (backend, false),
        Request::InitializeAssets { backend } => (backend, true),
        _ => return Err(ChildMainError::ExpectedInitialization),
    };

    let runtime = match Runtime::with_backend(backend) {
        Ok(runtime) => runtime,
        Err(error) => {
            write_response(&mut writer, &Response::Initialized(Err(error.to_string())))?;
            return Ok(());
        }
    };
    let backend = runtime.backend();
    let ipc = IpcTransport::from_runtime(&runtime);
    let mut assets = AssetStore::new(backend);
    let mut artifacts = HashMap::<usize, RuntimeArtifact>::new();
    let mut next_artifact = 0_usize;
    write_response(&mut writer, &Response::Initialized(Ok(backend)))?;

    let mut pending_outputs = Vec::new();
    while let Some(request) = read_request(&mut reader)? {
        // The owner never loads supplied programs. Its process and allocations
        // survive faults in the separate execution workers.
        if asset_owner
            && !matches!(
                request,
                Request::AttachAsset { .. }
                    | Request::AttachAssetRange { .. }
                    | Request::ExportAsset { .. }
                    | Request::AssetUsage
                    | Request::Shutdown
            )
        {
            return Err(ChildMainError::Protocol(
                "asset owner received an execution request".into(),
            ));
        }
        match request {
            Request::Initialize { .. } | Request::InitializeAssets { .. } => {
                return Err(ChildMainError::AlreadyInitialized);
            }
            Request::LoadSources { sources, codegen } => {
                let result = compile_sources(&sources, codegen, &runtime).map(|artifact| {
                    register_artifact(&mut artifacts, &mut next_artifact, artifact)
                });
                write_response(&mut writer, &Response::Loaded(result))?;
            }
            Request::LoadModule { module } => {
                let result = runtime
                    .load(module)
                    .map(|artifact| register_artifact(&mut artifacts, &mut next_artifact, artifact))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Loaded(result))?;
            }
            Request::AttachAsset { key, byte_len } => {
                // The descriptor channel is part of request framing. A failed
                // receive may have consumed an unknown amount of ancillary
                // data, so it is a fatal protocol error rather than a reusable
                // structured asset rejection.
                let file = receive_asset_file(asset_socket)?;
                let result = assets
                    .attach(key, byte_len, file)
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::AttachAssetRange {
                key,
                file_bytes,
                offset,
                byte_len,
            } => {
                let file = receive_asset_file(asset_socket)?;
                let result = assets
                    .attach_range(key, file_bytes, offset, byte_len, file)
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::AssetUsage => {
                let (count, uploaded, device) = assets.usage();
                write_response(&mut writer, &Response::AssetUsage(count, uploaded, device))?;
            }
            Request::ExportAsset { asset } => {
                if !asset_owner {
                    return Err(ChildMainError::Protocol(
                        "only the asset owner may export allocations".into(),
                    ));
                }
                match assets.export(asset) {
                    Ok((file, byte_len, allocation_len, file_offset)) => {
                        write_response(
                            &mut writer,
                            &Response::Exported(Ok((byte_len, allocation_len, file_offset))),
                        )?;
                        send_file(asset_socket, &file)
                            .map_err(|error| ChildMainError::Protocol(error.to_string()))?;
                    }
                    Err(error) => {
                        write_response(&mut writer, &Response::Exported(Err(error.to_string())))?
                    }
                }
            }
            Request::ImportAsset {
                asset,
                byte_len,
                allocation_len,
                file_offset,
            } => {
                let file = receive_asset_file(asset_socket)?;
                let result = assets
                    .import(asset, byte_len, allocation_len, file_offset, file)
                    .map(|asset| (asset.id, asset.byte_len))
                    .map_err(|error| error.to_string());
                write_response(&mut writer, &Response::Attached(result))?;
            }
            Request::Compute { request } => {
                let result = artifacts
                    .get(&request.artifact)
                    .ok_or_else(|| anyhow::anyhow!("unknown compute artifact"))
                    .and_then(|artifact| {
                        compute::worker::execute(
                            &runtime,
                            artifact,
                            &assets,
                            request,
                            &mut |part| {
                                write_response(&mut writer, &Response::Published(part)).is_ok()
                            },
                        )
                    });
                if let Err(error) = &result
                    && let Some(error) = gpu_error::gpu_synchronization(error)
                {
                    return Err(ChildMainError::GpuSynchronization(error.to_string()));
                }
                write_response(
                    &mut writer,
                    &Response::Computed(result.map_err(|e| e.to_string())),
                )?;
            }
            Request::ReleaseArtifact { artifact } => {
                artifacts.remove(&artifact);
                write_response(&mut writer, &Response::ArtifactReleased(Ok(artifact)))?;
            }
            Request::Shutdown => return Ok(()),
            Request::ReleaseOutputs => {
                pending_outputs.clear();
                write_response(&mut writer, &Response::OutputsReleased)?;
            }
            Request::Execute {
                artifact,
                name,
                buffers,
                args,
            } => {
                let response = if pending_outputs.is_empty() {
                    match artifacts.get(&artifact) {
                        Some(artifact) => execute_in_child(
                            &ipc,
                            artifact,
                            &name,
                            buffers,
                            args,
                            &mut pending_outputs,
                        )?,
                        None => Response::Executed(Err(RemoteExecError::UnknownArtifact)),
                    }
                } else {
                    Response::Executed(Err(RemoteExecError::Protocol(
                        "previous owned outputs have not been released".to_string(),
                    )))
                };
                write_response(&mut writer, &response)?;
            }
        }
    }

    Ok(())
}

/// Compile untrusted source inside the worker, where the compile deadline can
/// terminate it, then load the rendered module for this worker's dialect.
fn compile_sources(
    sources: &[String],
    codegen: CodegenKind,
    runtime: &Runtime,
) -> Result<RuntimeArtifact, String> {
    let sources = sources.iter().map(String::as_str).collect::<Vec<_>>();
    let build = || {
        let raw_theories =
            RawTheorySet::from_texts(sources.iter().copied()).map_err(|e| e.to_string())?;
        let mut report = CompileReport::new(raw_theories);
        compile::compile(&mut report, codegen, runtime.dialect()).map_err(|e| e.to_string())
    };
    // Both cache access and compilation remain in the killable worker and under
    // the source-load deadline. The upstream key includes the compiler build,
    // sources, backend and dialect, including dirty stdlib/compiler changes.
    let module =
        match crate::runtime::ModuleCache::from_sources(&sources, codegen, runtime.dialect()) {
            Some(cache) => cache.get_or_compile(build)?,
            None => build()?,
        };
    runtime.load(module).map_err(|e| e.to_string())
}

fn register_artifact(
    artifacts: &mut HashMap<usize, RuntimeArtifact>,
    next_artifact: &mut usize,
    artifact: RuntimeArtifact,
) -> (usize, Vec<EntryPoint>) {
    let id = *next_artifact;
    *next_artifact = next_artifact
        .checked_add(1)
        .expect("artifact ID space exhausted");
    let entries = artifact.entry_points().to_vec();
    artifacts.insert(id, artifact);
    (id, entries)
}

/// Copies owned arguments into the child, runs the program, and prepares its outputs for export.
pub(super) fn execute_in_child(
    ipc: &IpcTransport,
    artifact: &RuntimeArtifact,
    name: &str,
    buffers: Vec<WireIpcBuffer>,
    wire_args: Vec<WireValue>,
    pending_outputs: &mut Vec<MemOwn>,
) -> Result<Response, ChildMainError> {
    let imported = match import_ipc_buffers(ipc, buffers) {
        Ok(imported) => imported,
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Memory(error)))),
    };

    let args = match wire_args
        .into_iter()
        .map(|value| match value {
            WireValue::Bool(value) => Ok(Value::Bool(value)),
            WireValue::U16(value) => Ok(Value::U16(value)),
            WireValue::U32(value) => Ok(Value::U32(value)),
            WireValue::U64(value) => Ok(Value::U64(value)),
            WireValue::F32(value) => Ok(Value::F32(value)),
            WireValue::MemRef {
                buffer,
                view_offset,
                byte_len,
            } => imported
                .get(buffer)
                .and_then(|allocation| allocation.as_mem_ref(view_offset, byte_len))
                .map(Value::MemRef)
                .ok_or_else(|| "invalid IPC memory view".to_string()),
            WireValue::MemOwn {
                buffer,
                view_offset,
                byte_len,
            } => imported
                .get(buffer)
                .ok_or_else(|| "invalid IPC memory view".to_string())?
                .copy_view_into_owned(view_offset, byte_len)
                .map_err(|error| error.to_string())?
                .map(Value::MemOwn)
                .ok_or_else(|| "invalid IPC memory view".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(args) => args,
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Memory(error)))),
    };

    let values = match artifact.exec_values(name, args) {
        Ok(values) => values,
        Err(ExecError::GpuSynchronization(error)) => {
            return Err(ChildMainError::GpuSynchronization(error));
        }
        Err(error) => return Ok(Response::Executed(Err(RemoteExecError::Runtime(error)))),
    };
    Ok(match encode_child_outputs(ipc, values, pending_outputs) {
        Ok(execution) => Response::Executed(Ok(execution)),
        Err(error) => {
            pending_outputs.clear();
            Response::Executed(Err(error))
        }
    })
}

/// Exports owned outputs and retains them until the parent confirms it has copied them.
pub(super) fn encode_child_outputs(
    ipc: &IpcTransport,
    values: Vec<Value<'static>>,
    pending_outputs: &mut Vec<MemOwn>,
) -> Result<WireExecution, RemoteExecError> {
    let mut buffers = Vec::new();
    let mut wire_values = Vec::with_capacity(values.len());
    for value in values {
        let wire = match value {
            Value::Bool(value) => WireValue::Bool(value),
            Value::U16(value) => WireValue::U16(value),
            Value::U32(value) => WireValue::U32(value),
            Value::U64(value) => WireValue::U64(value),
            Value::F32(value) => WireValue::F32(value),
            Value::MemOwn(memory) => {
                let exported = ipc
                    .export_view(memory.as_ref())
                    .map_err(|error| RemoteExecError::Memory(error.to_string()))?;
                let buffer = intern_buffer(&mut buffers, encode_ipc_buffer(exported));
                let value = WireValue::MemOwn {
                    buffer,
                    view_offset: exported.view_offset(),
                    byte_len: exported.byte_len(),
                };
                pending_outputs.push(memory);
                value
            }
            Value::MemRef(_) => unreachable!("Runtime rejects borrowed memory outputs"),
        };
        wire_values.push(wire);
    }
    Ok(WireExecution {
        buffers,
        values: wire_values,
    })
}

pub(super) fn read_request(reader: &mut impl Read) -> Result<Option<Request>, ChildMainError> {
    read_frame(reader).map_err(child_protocol_error)
}

pub(super) fn receive_asset_file(asset_socket: &UnixStream) -> Result<File, ChildMainError> {
    receive_file(asset_socket).map_err(|error| {
        ChildMainError::Protocol(format!("asset descriptor transport failed: {error}"))
    })
}

pub(super) fn write_response(
    writer: &mut impl io::Write,
    response: &Response,
) -> Result<(), ChildMainError> {
    write_frame(writer, response).map_err(child_protocol_error)
}

pub(super) fn child_protocol_error(error: ProtocolError) -> ChildMainError {
    ChildMainError::Protocol(error.to_string())
}
