//! Process-isolated execution for Catena programs.
//!
//! Generated GPU code runs through native libraries, so a failed Catena assertion,
//! GPU runtime failure, or native crash can terminate the process rather than return
//! a Rust error. [`SafeRuntime`] solves this by executing [`crate::runtime::Runtime`] in a child
//! process and communicating over a framed protocol. If execution terminates the
//! child, the host process survives and receives a structured error with its exit
//! status and stderr.
//!
//! ## Embedding hosts
//!
//! [`Session`] compiles programs in one long-lived execution worker. Programs
//! keep that worker alive after the session handle is dropped.
//! A loaded program exposes its immutable entry-point metadata so a host can
//! reject an incompatible dynamic ABI before execution.
//!
//! [`compute`] executes arbitrary entry points with bounded host inputs
//! and outputs; all device allocations and invocation state remain in the
//! worker. An [`AssetOwner`] can retain immutable allocations across execution
//! sessions. Session construction loads the native GPU runtime only in workers,
//! not in the host process. [`SafeRuntime`]'s value API remains available for
//! callers that exchange device memory with the worker over GPU IPC.
//!
//! ```no_run
//! use catena_lang::{
//!     codegen::CodegenKind,
//!     runtime::Backend,
//!     safe_runtime::{
//!         Session, run_safe_runtime_child_if_requested,
//!         compute::{Input, Output, Request},
//!     },
//!     stdlib,
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Worker mode must be handled before starting threads or using stdout.
//!     if run_safe_runtime_child_if_requested()? { return Ok(()); }
//!     let mut session = Session::new(Backend::Auto)?;
//!     let mut sources: Vec<&str> = stdlib::sources().collect();
//!     sources.push("(def program increment : (u64 val) -> (u64 val) = ({smolcat.value.named (const.u64.0x0000000000000001 smolcat.value.named) stdlib.numeric.numeric.u64} stdlib.numeric.+ :.forget))");
//!     let program = session.prepare(sources, CodegenKind::Default)?;
//!     let output = session.compute(&program, Request {
//!         entry_point: "increment", inputs: &[Input::U64(41)],
//!         max_output_bytes: 8, allocation_budget: 0,
//!     })?;
//!     assert_eq!(output, [Output::U64(42)]);
//!     Ok(())
//! }
//! ```

use self::ipc::IpcTransport;
use self::operation::SessionOperationGate;
use self::worker::WorkerProcess;
use crate::runtime::Backend;
use crate::runtime::EntryPoint;
use crate::runtime::ExecError;
use crate::runtime::MemError;
use crate::runtime::RuntimeId;
use std::io;
use std::os::fd::RawFd;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use thiserror::Error;

mod assets;
pub mod compute;
mod embedding;
pub use embedding::{Asset, AssetOwner, AssetSlice, AssetUsage, Program, Session};
#[cfg(target_os = "macos")]
mod darwin_fds;
mod fd_transport;
mod gpu_error;
mod ipc;
mod protocol;
pub use assets::{MAX_RESIDENT_ASSET_BYTES, MAX_RESIDENT_ASSETS};
pub use protocol::ComputeOutput;
pub(crate) use protocol::{
    ComputeInput, WireAssetSlice as ComputeAssetSlice, WireCompute as ComputeRequest,
};

const CHILD_MODE_ENV: &str = "CATENA_SAFE_RUNTIME_CHILD";
const CHILD_ASSET_SOCKET_FD: RawFd = 3;

/// Default wall-clock limit for one isolated source-load and GPU compilation.
pub const DEFAULT_COMPILE_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Default wall-clock limit for one isolated GPU operation or complete invocation.
pub const DEFAULT_EXECUTION_TIMEOUT: Duration = Duration::from_secs(30 * 60);

const DROP_GRACE_TIMEOUT: Duration = Duration::from_millis(250);
const KILL_REAP_TIMEOUT: Duration = Duration::from_secs(1);
const STDERR_TAIL_BYTES: usize = 64 * 1024;

/// Deadlines applied by a process-isolated runtime.
///
/// The compile deadline covers each complete source or module load request. The execution
/// deadline covers initialization, asset and artifact control requests, generic
/// execution, and one complete compute invocation including publications.
/// Operations first wait to acquire the session lease; that queue wait happens
/// before the corresponding deadline is armed. A compute invocation retains
/// the lease and its watchdog until completion, including synchronous callbacks. Callback code cannot be preempted or
/// bounded by the watchdog and must not re-enter the same session; re-entry is
/// rejected instead of waiting on itself. If a callback never returns after
/// the watchdog kills its worker, the parent cannot reap that worker and
/// process-wide worker admission cannot reopen unless the callback returns and
/// teardown subsequently proves leader exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeRuntimeTimeouts {
    compile: Duration,
    execution: Duration,
}

impl SafeRuntimeTimeouts {
    /// Start from the production defaults.
    pub const fn new() -> Self {
        Self {
            compile: DEFAULT_COMPILE_TIMEOUT,
            execution: DEFAULT_EXECUTION_TIMEOUT,
        }
    }

    /// Set the deadline for one complete source-load and GPU-compile request.
    pub const fn with_compile_timeout(mut self, timeout: Duration) -> Self {
        self.compile = timeout;
        self
    }

    /// Set the deadline for one control/execution operation or complete invocation.
    pub const fn with_execution_timeout(mut self, timeout: Duration) -> Self {
        self.execution = timeout;
        self
    }

    /// Return the deadline for one source-load and compilation request.
    #[must_use]
    pub const fn compile_timeout(self) -> Duration {
        self.compile
    }

    /// Return the deadline for one non-compilation operation or complete invocation.
    #[must_use]
    pub const fn execution_timeout(self) -> Duration {
        self.execution
    }
}

impl Default for SafeRuntimeTimeouts {
    fn default() -> Self {
        Self::new()
    }
}

/// Failures while attaching or slicing a session-resident asset.
#[derive(Debug, Error)]
pub enum AssetError {
    #[error("failed to inspect verified asset descriptor: {0}")]
    FileMetadata(#[source] io::Error),
    #[error("asset length must be between 1 and {maximum} bytes, got {actual}")]
    InvalidLength { actual: u64, maximum: u64 },
    #[error("asset transport failed: {0}")]
    Transport(String),
    #[error("invalid local asset request: {0}")]
    InvalidRequest(String),
    #[error("SafeRuntime asset attachment exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child rejected asset: {0}")]
    Remote(String),
    #[error("SafeRuntime child returned an unexpected asset response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated while attaching an asset with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error("asset belongs to a different GPU asset owner")]
    WrongSession,
    #[error(
        "asset slice at offset {offset} with {byte_len} bytes exceeds its {asset_byte_len}-byte asset"
    )]
    InvalidSlice {
        offset: u64,
        byte_len: u64,
        asset_byte_len: u64,
    },
}

impl AssetError {
    /// Whether the worker protocol can no longer be trusted after this error.
    ///
    /// Caller-side validation, local request preparation, and a structured
    /// rejection from the worker leave the session usable. Transport failures,
    /// termination, and an unexpected response do not: the next operation must
    /// use a new session.
    pub fn invalidates_session(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::TimedOut { .. }
                | Self::UnexpectedResponse
                | Self::ChildTerminated { .. }
                | Self::Unavailable { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttachedAsset {
    pub(crate) id: u64,
    pub(crate) byte_len: u64,
}

/// Failures from worker-resident computation and artifact control.
#[derive(Debug, Error)]
pub enum ResidentError {
    #[error("resident GPU request failed: {0}")]
    Remote(String),
    #[error("invalid local resident GPU request: {0}")]
    InvalidRequest(String),
    #[error("resident GPU teardown failed: {0}")]
    Teardown(String),
    #[error("SafeRuntime child returned an unexpected resident response")]
    UnexpectedResponse,
    #[error("compute publication was cancelled")]
    Cancelled,
    #[error("compute publication callback panicked")]
    CallbackPanicked,
    #[error("resident GPU transport failed: {0}")]
    Transport(String),
    #[error("resident GPU operation exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child terminated during a resident GPU request with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
}

impl ResidentError {
    /// Whether the worker protocol can no longer be trusted after this error.
    pub fn invalidates_session(&self) -> bool {
        !matches!(self, Self::Remote(_) | Self::InvalidRequest(_))
    }
}

/// Initialization failures for [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeInitError {
    #[error("failed to identify the current executable: {0}")]
    CurrentExecutable(#[source] io::Error),
    #[error("SafeRuntime worker executable must be an absolute path")]
    InvalidExecutable,
    #[error("failed to spawn SafeRuntime child {executable}: {source}")]
    Spawn {
        executable: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to create SafeRuntime worker scratch directory under {parent}: {source}")]
    Scratch {
        parent: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to create SafeRuntime asset transport: {0}")]
    AssetTransport(#[source] io::Error),
    #[error("SafeRuntime setup transport failed: {0}")]
    Transport(String),
    #[error("invalid local SafeRuntime setup request: {0}")]
    InvalidRequest(String),
    #[error("failed to inspect the host SIGCHLD disposition: {0}")]
    ChildSignal(#[source] io::Error),
    #[error(
        "SafeRuntime requires waitable child processes; SIGCHLD must not use SIG_IGN or SA_NOCLDWAIT"
    )]
    UnwaitableChildren,
    #[error("failed to start a SafeRuntime lifecycle support thread: {0}")]
    LifecycleThread(#[source] io::Error),
    #[error("cannot start SafeRuntime while {workers} previous worker leader(s) remain unreaped")]
    PreviousWorkerUnreaped { workers: u16 },
    #[error("SafeRuntime process lifecycle tracker exhausted its concurrent-spawn capacity")]
    LifecycleCapacity,
    #[error("SafeRuntime child initialization failed: {0}")]
    RemoteInitialization(String),
    #[error("SafeRuntime child failed to load module: {0}")]
    RemoteLoad(String),
    #[error("SafeRuntime source loading and GPU compilation exceeded {timeout:?}")]
    CompileTimedOut { timeout: Duration },
    #[error("SafeRuntime initialization exceeded {timeout:?}")]
    InitializationTimedOut { timeout: Duration },
    #[error("SafeRuntime compile timeout must be greater than zero")]
    InvalidCompileTimeout,
    #[error("SafeRuntime execution timeout must be greater than zero")]
    InvalidExecutionTimeout,
    #[error("SafeRuntime child returned an unexpected setup response")]
    UnexpectedResponse,
    #[error(
        "SafeRuntime child terminated during initialization or loading with {status}: {stderr}"
    )]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] MemError),
}

impl SafeInitError {
    /// Whether an existing worker protocol can no longer be trusted after this
    /// error.
    ///
    /// Of these variants, only local request preparation and a structured
    /// load rejection can occur without invalidating a live
    /// session. The construction-only variants are conservatively classified
    /// as invalidating so callers never retain a session whose health is
    /// uncertain.
    pub fn invalidates_session(&self) -> bool {
        !matches!(self, Self::RemoteLoad(_) | Self::InvalidRequest(_))
    }
}

/// Execution failures reported by [`SafeRuntime`].
#[derive(Debug, Error)]
pub enum SafeExecError {
    #[error("SafeRuntime artifact is no longer loaded in its child")]
    UnknownArtifact,
    #[error(transparent)]
    Runtime(#[from] ExecError),
    #[error("SafeRuntime transport failed: {0}")]
    Transport(String),
    #[error("invalid local SafeRuntime execution request: {0}")]
    InvalidRequest(String),
    #[error("SafeRuntime child rejected memory IPC: {0}")]
    RemoteMemory(String),
    #[error("SafeRuntime execution exceeded {timeout:?}")]
    TimedOut { timeout: Duration },
    #[error("SafeRuntime child returned an unexpected execution response")]
    UnexpectedResponse,
    #[error("SafeRuntime child terminated with {status}: {stderr}")]
    ChildTerminated { status: ExitStatus, stderr: String },
    #[error("SafeRuntime is unavailable because its child terminated with {status}: {stderr}")]
    Unavailable { status: ExitStatus, stderr: String },
    #[error(transparent)]
    Memory(#[from] MemError),
}

impl SafeExecError {
    /// Whether the worker protocol can no longer be trusted after this error.
    ///
    /// Local validation/allocation errors and structured runtime or child IPC
    /// rejections leave request/response synchronization intact. Transport
    /// failures, deadlines, termination, and semantically impossible responses
    /// require a new session.
    pub fn invalidates_session(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::TimedOut { .. }
                | Self::UnexpectedResponse
                | Self::ChildTerminated { .. }
                | Self::Unavailable { .. }
        )
    }
}

/// Failure in the worker-mode entrypoint itself.
#[derive(Debug, Error)]
pub enum ChildMainError {
    #[error("SafeRuntime child protocol failed: {0}")]
    Protocol(String),
    #[error("SafeRuntime child expected Initialize as its first request")]
    ExpectedInitialization,
    #[error("SafeRuntime child received a second Initialize request")]
    AlreadyInitialized,
    #[error("SafeRuntime child GPU synchronization failed: {0}")]
    GpuSynchronization(String),
}

/// A process-isolated Catena runtime.
///
/// The host executable must call [`run_safe_runtime_child_if_requested`] before
/// parsing arguments or writing to stdout. `SafeRuntime` respawns that same
/// executable and reserves its stdin/stdout for the worker protocol.
///
/// The host must also leave `SIGCHLD` waitable for the session lifetime and
/// must not reap children it did not spawn. Construction rejects `SIG_IGN` and
/// `SA_NOCLDWAIT`; external `waitpid(-1, ...)` use would violate the exclusive
/// ownership required by [`std::process::Child`] and the worker's process-group
/// guard.
///
/// Worker-retirement admission is process-local. A changed PID resets copied
/// admission counters so an exec-style fork child does not inherit its parent's
/// retired leaders. Live `SafeRuntime` values themselves are not fork-safe:
/// after a multi-threaded fork, the child must exec rather than use or drop
/// inherited sessions and synchronization primitives. Independently launched
/// host processes require a deployment-level memory/cgroup boundary.
///
/// The worker intentionally inherits the host environment except for `TMPDIR`,
/// which is replaced by a private, parent-owned scratch directory. GPU runtime
/// and compiler discovery currently depend on platform-specific variables such
/// as `PATH`, `ROCM_PATH`, `HIP_PATH`, `CUDA_PATH`, and loader/wrapper settings;
/// there is no portable safe allowlist at this layer. Deployments that require
/// environment isolation must start the host process with a curated environment
/// rather than assuming `SafeRuntime` clears it.
///
/// This lower-level interface supports raw [`crate::runtime::Value`] memory across the process
/// boundary, so its public constructors initialize GPU IPC in the parent. The
/// typed [`Session`] interface keeps all device memory in the
/// worker and therefore loads the native GPU runtime only there.
#[derive(Debug, Clone)]
pub struct SafeRuntime {
    backend: Backend,
    executable: PathBuf,
    worker: Arc<Mutex<WorkerProcess>>,
    asset_owner: Option<Arc<SafeRuntime>>,
    operation_gate: Arc<SessionOperationGate>,
    parent_ipc: Option<IpcTransport>,
    runtime_id: RuntimeId,
    timeouts: SafeRuntimeTimeouts,
}

/// A compiled program that retains its isolated worker after `SafeRuntime`
/// is dropped. Its final clone releases child-side artifact state.
#[derive(Debug, Clone)]
pub struct Artifact {
    inner: Arc<RemoteArtifact>,
}

#[derive(Debug)]
struct RemoteArtifact {
    runtime: SafeRuntime,
    index: usize,
    entry_points: Arc<[EntryPoint]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParentInterface {
    RawValues,
    ResidentProtocol,
    AssetOwner,
}

fn initialize_parent_ipc<T, E>(
    interface: ParentInterface,
    load: impl FnOnce() -> Result<T, E>,
) -> Result<Option<T>, E> {
    match interface {
        ParentInterface::RawValues => load().map(Some),
        ParentInterface::ResidentProtocol | ParentInterface::AssetOwner => Ok(None),
    }
}

mod artifact;
mod child;
mod deadline;
mod lifecycle;
mod operation;
mod session;
mod worker;

pub use child::run_safe_runtime_child_if_requested;

#[cfg(test)]
mod error_tests;
