//! Worker process transport and bounded shutdown.

#[cfg(test)]
use super::lifecycle::LifecycleTestBarrier;
use super::{
    AssetError, CHILD_ASSET_SOCKET_FD, CHILD_MODE_ENV, DROP_GRACE_TIMEOUT, KILL_REAP_TIMEOUT,
    ResidentError, STDERR_TAIL_BYTES, SafeExecError, SafeInitError,
    deadline::{DeadlineReader, DeadlineWriter, ProcessGroupDeadline, set_nonblocking},
    fd_transport::{FdTransportError, send_file},
    lifecycle::{
        BackgroundReaper, ManagedProcessGroup, ReapResources, WorkerLifecycleGate, reap_resources,
        worker_lifecycle_gate,
    },
    protocol::{
        FrameEncodeError, ProtocolError, Request, Response, encode_frame, read_frame,
        write_encoded_frame,
    },
};
use std::{
    collections::HashSet,
    env,
    fs::{self, File},
    io::{self, BufReader, Read},
    net::Shutdown,
    os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use thiserror::Error;

#[derive(Debug, Clone)]
pub(super) struct Termination {
    pub(super) status: ExitStatus,
    pub(super) stderr: String,
}

#[derive(Debug, Default)]
pub(super) struct StderrTail {
    bytes: Vec<u8>,
    truncated: bool,
}

impl StderrTail {
    pub(super) fn push(&mut self, bytes: &[u8]) {
        let retained = bytes.len().min(STDERR_TAIL_BYTES);
        let required = self.bytes.len().saturating_add(retained);
        let discard = required.saturating_sub(STDERR_TAIL_BYTES);
        if bytes.len() != retained || discard != 0 {
            self.truncated = true;
        }
        if discard != 0 {
            self.bytes.drain(..discard);
        }
        self.bytes
            .extend_from_slice(&bytes[bytes.len() - retained..]);
    }

    pub(super) fn render(&self) -> String {
        let tail = String::from_utf8_lossy(&self.bytes);
        if !self.truncated {
            tail.trim().to_string()
        } else {
            format!(
                "[... stderr truncated; retaining final {} bytes ...]\n{}",
                self.bytes.len(),
                tail.trim()
            )
        }
    }
}

pub(super) fn ensure_waitable_children() -> Result<(), SafeInitError> {
    let mut disposition = std::mem::MaybeUninit::<libc::sigaction>::uninit();
    if unsafe { libc::sigaction(libc::SIGCHLD, std::ptr::null(), disposition.as_mut_ptr()) } == -1 {
        return Err(SafeInitError::ChildSignal(io::Error::last_os_error()));
    }
    let disposition = unsafe { disposition.assume_init() };
    if !sigchld_is_waitable(disposition.sa_sigaction, disposition.sa_flags) {
        Err(SafeInitError::UnwaitableChildren)
    } else {
        Ok(())
    }
}

pub(super) fn sigchld_is_waitable(handler: libc::sighandler_t, flags: libc::c_int) -> bool {
    handler != libc::SIG_IGN && flags & libc::SA_NOCLDWAIT == 0
}

#[derive(Debug)]
pub(super) struct WorkerProcess {
    pub(super) child: Option<Child>,
    pub(super) process_group: Arc<ManagedProcessGroup>,
    pub(super) stdin: Option<ChildStdin>,
    pub(super) stdout: BufReader<ChildStdout>,
    pub(super) asset_socket: UnixStream,
    pub(super) stderr_tail: Arc<Mutex<StderrTail>>,
    pub(super) stderr_done: mpsc::Receiver<()>,
    pub(super) stderr_reader: Option<JoinHandle<()>>,
    pub(super) background_reaper: Option<BackgroundReaper>,
    pub(super) scratch: Option<tempfile::TempDir>,
    pub(super) termination: Option<Termination>,
    pub(super) imported_assets: HashSet<u64>,
    #[cfg(test)]
    pub(super) force_background_handoff: bool,
    #[cfg(test)]
    pub(super) reaper_test_barrier: Option<Box<LifecycleTestBarrier>>,
    #[cfg(test)]
    pub(super) drop_retirement_test_barrier: Option<Box<LifecycleTestBarrier>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeadlineKind {
    Initialize,
    Compile,
    Execution,
}

#[derive(Debug, Error)]
pub(super) enum WorkerError {
    #[error("operation re-entered the same SafeRuntime session from its active callback")]
    ReentrantOperation,
    #[error("local request preparation failed: {0}")]
    LocalRequest(#[source] FrameEncodeError),
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error(transparent)]
    AssetDescriptor(#[from] FdTransportError),
    #[error("failed to wait for SafeRuntime child: {0}")]
    Wait(#[source] io::Error),
    #[error("failed to start the SafeRuntime deadline watchdog: {0}")]
    Watchdog(#[source] io::Error),
    #[error("SafeRuntime child terminated")]
    Terminated(Termination),
    #[error("SafeRuntime worker is invalid after an earlier failed operation")]
    Invalidated,
    #[error("SafeRuntime operation exceeded {timeout:?}")]
    TimedOut {
        kind: DeadlineKind,
        timeout: Duration,
    },
}

impl WorkerError {
    pub(super) fn invalidates_worker(&self) -> bool {
        !matches!(self, Self::ReentrantOperation | Self::LocalRequest(_))
    }
}

impl WorkerProcess {
    pub(super) fn spawn(executable: &Path) -> Result<Self, SafeInitError> {
        let mut command = Command::new(executable);
        command.env(CHILD_MODE_ENV, "1");
        Self::spawn_command_with_gate(command, executable, worker_lifecycle_gate())
    }

    #[cfg(test)]
    pub(super) fn spawn_command(
        command: Command,
        executable: &Path,
    ) -> Result<Self, SafeInitError> {
        Self::spawn_command_with_gate(
            command,
            executable,
            Arc::new(WorkerLifecycleGate::default()),
        )
    }

    pub(super) fn spawn_command_with_gate(
        command: Command,
        executable: &Path,
        lifecycle_gate: Arc<WorkerLifecycleGate>,
    ) -> Result<Self, SafeInitError> {
        let scratch_parent = env::temp_dir();
        Self::spawn_command_in_with_gate(command, executable, &scratch_parent, lifecycle_gate)
    }

    #[cfg(test)]
    pub(super) fn spawn_command_in(
        command: Command,
        executable: &Path,
        scratch_parent: &Path,
    ) -> Result<Self, SafeInitError> {
        Self::spawn_command_in_with_gate(
            command,
            executable,
            scratch_parent,
            Arc::new(WorkerLifecycleGate::default()),
        )
    }

    pub(super) fn spawn_command_in_with_gate(
        mut command: Command,
        executable: &Path,
        scratch_parent: &Path,
        lifecycle_gate: Arc<WorkerLifecycleGate>,
    ) -> Result<Self, SafeInitError> {
        // This permit linearizes worker admission against every detached
        // retirement in the process. It is released once construction either
        // owns a complete WorkerProcess or has registered any spawned child
        // with its reaper.
        let _spawn_permit = lifecycle_gate.begin_spawn()?;
        let scratch = tempfile::Builder::new()
            .prefix("catena-safe-runtime-")
            .tempdir_in(scratch_parent)
            .map_err(|source| SafeInitError::Scratch {
                parent: scratch_parent.to_path_buf(),
                source,
            })?;
        let scratch_path =
            fs::canonicalize(scratch.path()).map_err(|source| SafeInitError::Scratch {
                parent: scratch_parent.to_path_buf(),
                source,
            })?;
        let (asset_socket, child_asset_socket) =
            UnixStream::pair().map_err(SafeInitError::AssetTransport)?;
        let child_asset_fd = child_asset_socket.as_raw_fd();
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Every tempfile made by the worker or its compiler descendants is
            // scoped beneath a parent-owned directory. The parent retains that
            // directory until the private process group has been killed and its
            // leader reaped, so SIGKILL cannot strand Catena build scratch.
            .env("TMPDIR", scratch_path);
        // The worker and every compiler it spawns live in a private process
        // group. A compile deadline can therefore stop hipcc/clang descendants
        // as well as the protocol child that is blocked waiting for them.
        command.process_group(0);
        #[cfg(target_os = "macos")]
        super::darwin_fds::configure(&mut command).map_err(|source| SafeInitError::Spawn {
            executable: executable.to_path_buf(),
            source,
        })?;
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(child_asset_fd, CHILD_ASSET_SOCKET_FD) == -1 {
                    return Err(io::Error::last_os_error());
                }
                if libc::fcntl(CHILD_ASSET_SOCKET_FD, libc::F_SETFD, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        // Create the teardown owner before there is a child it might need to
        // own. Thread exhaustion is therefore a construction error, never a
        // reason for `Drop` to lose the only wait handle.
        let mut background_reaper =
            BackgroundReaper::start().map_err(SafeInitError::LifecycleThread)?;
        let mut child = command.spawn().map_err(|source| SafeInitError::Spawn {
            executable: executable.to_path_buf(),
            source,
        })?;
        let process_group = Arc::new(ManagedProcessGroup::new(child.id(), lifecycle_gate.clone()));
        drop(child_asset_socket);
        let stdin = child
            .stdin
            .take()
            .expect("piped child stdin should be available");
        let stdout = child
            .stdout
            .take()
            .expect("piped child stdout should be available");
        let mut stderr = child
            .stderr
            .take()
            .expect("piped child stderr should be available");
        let stderr_tail = Arc::new(Mutex::new(StderrTail::default()));
        let captured_tail = stderr_tail.clone();
        let (stderr_finished, stderr_done) = mpsc::channel();
        if let Err(error) =
            set_nonblocking(stdin.as_raw_fd()).and_then(|()| set_nonblocking(stdout.as_raw_fd()))
        {
            // A killed process can still be stuck in uninterruptible sleep.
            // Give ownership to a reaper rather than making construction an
            // unbounded wait on this exceptional setup path.
            process_group.kill();
            drop(stderr);
            let resources = ReapResources::new(Some(child), process_group, None, Some(scratch));
            if let Err(resources) = background_reaper.handoff(resources) {
                // The pre-created receiver has no fallible work before recv,
                // so this is only a defensive ownership-preserving fallback.
                reap_resources(resources);
            }
            return Err(SafeInitError::Transport(format!(
                "failed to make SafeRuntime protocol pipes nonblocking: {error}"
            )));
        }
        let stderr_reader = thread::Builder::new()
            .name("catena-safe-runtime-stderr".to_string())
            .spawn(move || {
                let mut bytes = [0_u8; 8 * 1024];
                loop {
                    match stderr.read(&mut bytes) {
                        Ok(0) => break,
                        Ok(read) => {
                            let Ok(mut tail) = captured_tail.lock() else {
                                break;
                            };
                            tail.push(&bytes[..read]);
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
                let _ = stderr_finished.send(());
            });
        let stderr_reader = match stderr_reader {
            Ok(stderr_reader) => stderr_reader,
            Err(error) => {
                process_group.kill();
                let resources = ReapResources::new(Some(child), process_group, None, Some(scratch));
                if let Err(resources) = background_reaper.handoff(resources) {
                    reap_resources(resources);
                }
                return Err(SafeInitError::LifecycleThread(error));
            }
        };

        Ok(Self {
            child: Some(child),
            process_group,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            asset_socket,
            stderr_tail,
            stderr_done,
            stderr_reader: Some(stderr_reader),
            background_reaper: Some(background_reaper),
            scratch: Some(scratch),
            termination: None,
            imported_assets: HashSet::new(),
            #[cfg(test)]
            force_background_handoff: false,
            #[cfg(test)]
            reaper_test_barrier: None,
            #[cfg(test)]
            drop_retirement_test_barrier: None,
        })
    }

    pub(super) fn send(
        &mut self,
        request: &Request,
        deadline: &ProcessGroupDeadline,
    ) -> Result<(), WorkerError> {
        self.ensure_available()?;
        // Serialization and the size check complete before touching the
        // protocol pipe. Their failures are therefore local request errors,
        // unlike every subsequent I/O error whose partial progress is unknown.
        let frame = encode_frame(request).map_err(WorkerError::LocalRequest)?;
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(WorkerError::Invalidated);
        };
        let descriptor = stdin.as_raw_fd();
        write_encoded_frame(
            &mut DeadlineWriter::new(stdin, descriptor, deadline),
            &frame,
        )
        .map_err(WorkerError::Protocol)
    }

    pub(super) fn receive(
        &mut self,
        deadline: &ProcessGroupDeadline,
    ) -> Result<Response, WorkerError> {
        let descriptor = self.stdout.get_ref().as_raw_fd();
        let response = read_frame(&mut DeadlineReader::new(
            &mut self.stdout,
            descriptor,
            deadline,
        ))
        .map_err(WorkerError::Protocol)?;
        match response {
            Some(response) => Ok(response),
            None => Err(WorkerError::Terminated(self.reap(deadline)?)),
        }
    }

    pub(super) fn request(
        &mut self,
        request: &Request,
        deadline: &ProcessGroupDeadline,
    ) -> Result<Response, WorkerError> {
        self.send(request, deadline)?;
        self.receive(deadline)
    }

    pub(super) fn with_deadline<T>(
        &mut self,
        timeout: Duration,
        kind: DeadlineKind,
        operation: impl FnOnce(&mut Self, &ProcessGroupDeadline) -> Result<T, WorkerError>,
    ) -> Result<T, WorkerError> {
        let deadline = self.arm_deadline(timeout)?;
        let result = operation(self, &deadline);
        self.finish_deadline(deadline, timeout, kind)?;
        if result.as_ref().is_err_and(WorkerError::invalidates_worker) {
            // Any failed protocol/descriptor transaction may have made partial
            // progress. Keep the safe API fail-closed: callers need not remember
            // to inspect an error classifier before attempting another request.
            // Request encoding and size validation are the sole exception:
            // both finish before the protocol pipe is touched.
            self.invalidate_after_protocol_failure();
        }
        result
    }

    pub(super) fn arm_deadline(
        &mut self,
        timeout: Duration,
    ) -> Result<ProcessGroupDeadline, WorkerError> {
        self.ensure_available()?;
        match ProcessGroupDeadline::arm(self.process_group.clone(), timeout) {
            Ok(deadline) => Ok(deadline),
            Err(error) => {
                // Running without the promised watchdog is not a recoverable
                // downgrade. Kill and hand off while the worker is still owned.
                self.invalidate_after_protocol_failure();
                Err(WorkerError::Watchdog(error))
            }
        }
    }

    pub(super) fn finish_deadline(
        &mut self,
        deadline: ProcessGroupDeadline,
        timeout: Duration,
        kind: DeadlineKind,
    ) -> Result<(), WorkerError> {
        if deadline.finish() {
            self.invalidate_after_timeout();
            Err(WorkerError::TimedOut { kind, timeout })
        } else {
            Ok(())
        }
    }

    pub(super) fn send_file(&mut self, file: &File) -> Result<(), WorkerError> {
        if let Err(error) = send_file(&self.asset_socket, file) {
            let _ = self.asset_socket.shutdown(Shutdown::Both);
            return Err(error.into());
        }
        Ok(())
    }

    pub(super) fn termination(&self) -> Option<&Termination> {
        self.termination.as_ref()
    }

    pub(super) fn ensure_available(&self) -> Result<(), WorkerError> {
        if let Some(termination) = &self.termination {
            Err(WorkerError::Terminated(termination.clone()))
        } else if self.stdin.is_none() || self.child.is_none() {
            Err(WorkerError::Invalidated)
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(super) fn child_id(&self) -> Result<u32, WorkerError> {
        self.child
            .as_ref()
            .map(Child::id)
            .ok_or(WorkerError::Invalidated)
    }

    pub(super) fn reap(
        &mut self,
        deadline: &ProcessGroupDeadline,
    ) -> Result<Termination, WorkerError> {
        if let Some(termination) = &self.termination {
            return Ok(termination.clone());
        }
        // EOF is never a successful response. Kill the whole private group
        // before reaping its leader so a compiler or other descendant cannot
        // survive the worker that owned it.
        self.process_group.kill();
        self.stdin.take();
        let Some(initial_wait) = deadline
            .remaining()
            .map(|remaining| remaining.min(KILL_REAP_TIMEOUT))
        else {
            self.invalidate_after_timeout();
            return Err(WorkerError::Invalidated);
        };
        let status = match self.wait_for_exit(initial_wait)? {
            Some(status) => status,
            None => {
                self.process_group.kill();
                let Some(kill_wait) = deadline
                    .remaining()
                    .map(|remaining| remaining.min(KILL_REAP_TIMEOUT))
                else {
                    self.invalidate_after_timeout();
                    return Err(WorkerError::Invalidated);
                };
                let Some(status) = self.wait_for_exit(kill_wait)? else {
                    let timed_out = deadline.remaining().is_none();
                    self.handoff_background_reaper();
                    return Err(if timed_out {
                        WorkerError::Invalidated
                    } else {
                        WorkerError::Wait(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "SafeRuntime child could not be reaped",
                        ))
                    });
                };
                status
            }
        };
        let stderr_wait = deadline
            .remaining()
            .unwrap_or_default()
            .min(Duration::from_millis(100));
        Ok(self.record_exit_with_stderr_wait(status, stderr_wait))
    }

    pub(super) fn wait_for_exit(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<ExitStatus>, WorkerError> {
        // This helper is terminal-only: each waitpid probe first sends the
        // group-wide kill under the lifecycle gate. It must never be used to
        // observe a healthy worker.
        let deadline = Instant::now() + timeout;
        loop {
            #[cfg(test)]
            if self.force_background_handoff {
                self.process_group.kill();
                return Ok(None);
            }
            let child = self.child.as_mut().ok_or(WorkerError::Invalidated)?;
            if let Some(status) = self
                .process_group
                .kill_and_try_wait(child)
                .map_err(WorkerError::Wait)?
            {
                return Ok(Some(status));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            thread::sleep((deadline - now).min(Duration::from_millis(5)));
        }
    }

    pub(super) fn invalidate_after_timeout(&mut self) {
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        // The watchdog has already killed this private process group. Repeat
        // the signal defensively, but never extend the operation past its
        // absolute deadline to wait for an uninterruptible child. A dedicated
        // background reaper owns that pathological process until `wait(2)` and
        // stderr capture can finish.
        self.process_group.kill();
        let status = self.try_reap_immediately();
        if let Some(status) = status {
            self.record_exit_now(status);
        } else {
            self.handoff_background_reaper();
        }
    }

    pub(super) fn invalidate_after_protocol_failure(&mut self) {
        if self.termination.is_some() || self.child.is_none() {
            return;
        }
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        self.process_group.kill();
        let status = self.try_reap_immediately();
        if let Some(status) = status {
            self.record_exit_now(status);
        } else {
            self.handoff_background_reaper();
        }
    }

    /// Convert a semantically impossible response into a fail-closed error.
    /// Framing succeeded in this case, but request/response synchronization can
    /// no longer be established, so retaining the worker would be unsafe.
    pub(super) fn reject_protocol_response<E>(&mut self, error: E) -> E {
        self.invalidate_after_protocol_failure();
        error
    }

    pub(super) fn try_reap_immediately(&mut self) -> Option<ExitStatus> {
        #[cfg(test)]
        if self.force_background_handoff {
            self.process_group.kill();
            return None;
        }
        self.child
            .as_mut()
            .and_then(|child| self.process_group.kill_and_try_wait(child).ok().flatten())
    }

    pub(super) fn record_exit(&mut self, status: ExitStatus) -> Termination {
        self.record_exit_with_stderr_wait(status, Duration::from_millis(100))
    }

    pub(super) fn record_exit_with_stderr_wait(
        &mut self,
        status: ExitStatus,
        stderr_wait: Duration,
    ) -> Termination {
        if self.stderr_done.recv_timeout(stderr_wait).is_ok()
            && let Some(stderr_reader) = self.stderr_reader.take()
        {
            let _ = stderr_reader.join();
        }
        self.record_exit_now(status)
    }

    pub(super) fn record_exit_now(&mut self, status: ExitStatus) -> Termination {
        if self.stderr_done.try_recv().is_ok()
            && let Some(stderr_reader) = self.stderr_reader.take()
        {
            let _ = stderr_reader.join();
        }
        let termination = Termination {
            status,
            stderr: self.stderr_snapshot(),
        };
        self.process_group.finish_retirement_after_wait();
        self.child.take();
        self.termination = Some(termination.clone());
        // The process-group leader has been reaped and every member received
        // SIGKILL under the PGID gate. Delegate potentially expensive recursive
        // scratch cleanup rather than extending the caller's deadline.
        self.handoff_background_reaper();
        termination
    }

    pub(super) fn handoff_background_reaper(&mut self) {
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        if self.child.is_some() {
            // Register before relinquishing the child, not in the detached
            // thread. A constructor concurrent with handoff must observe the
            // retirement before this method can return.
            self.process_group.begin_retirement();
        }
        let child = self.child.take();
        let stderr_reader = self.stderr_reader.take();
        if child.is_none() && stderr_reader.is_none() && self.scratch.is_none() {
            return;
        }
        let resources = ReapResources::new(
            child,
            self.process_group.clone(),
            stderr_reader,
            self.scratch.take(),
        );
        #[cfg(test)]
        let resources = {
            let mut resources = resources;
            resources.test_barrier = self.reaper_test_barrier.take();
            resources
        };
        let handed_off = match self.background_reaper.as_mut() {
            Some(reaper) => reaper.handoff(resources),
            None => Err(resources),
        };
        if let Err(resources) = handed_off {
            // The receiver is created before the child and performs no
            // fallible work before recv. Preserve ownership even if that
            // invariant is broken instead of panicking from `Drop`.
            reap_resources(resources);
        }
    }

    pub(super) fn stderr_snapshot(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|tail| tail.render())
            .unwrap_or_else(|_| "[stderr capture unavailable]".to_string())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.termination.is_none() && self.child.is_some() {
            // Close admission before graceful shutdown or SIGKILL. Drop stays
            // bounded: this is one idempotent atomic registration, never a wait
            // for constructors that linearized before it.
            self.process_group.begin_retirement();
            #[cfg(test)]
            if let Some(barrier) = self.drop_retirement_test_barrier.take() {
                let _ = barrier.entered.send(());
                let _ = barrier.release.recv();
            }
            if self.stdin.is_some()
                && let Ok(deadline) =
                    ProcessGroupDeadline::arm(self.process_group.clone(), DROP_GRACE_TIMEOUT)
            {
                let _ = self.send(&Request::Shutdown, &deadline);
                self.stdin.take();
                // Give the worker a bounded chance to run destructors
                // without reaping its leader. The unreaped PID keeps the
                // PGID from being reused; group cleanup remains mandatory.
                let descriptor = self.stdout.get_ref().as_raw_fd();
                let _ = deadline.wait_for_descriptor(descriptor, libc::POLLIN);
                let _ = deadline.finish();
            }
            self.stdin.take();
            let _ = self.asset_socket.shutdown(Shutdown::Both);
            self.process_group.kill();
            let status = self.wait_for_exit(KILL_REAP_TIMEOUT).ok().flatten();
            if let Some(status) = status {
                self.record_exit(status);
            } else {
                self.handoff_background_reaper();
            }
        }
        self.stdin.take();
        let _ = self.asset_socket.shutdown(Shutdown::Both);
        if self.stderr_reader.is_some() {
            self.handoff_background_reaper();
        }
    }
}

pub(super) fn map_init_worker_error(error: WorkerError) -> SafeInitError {
    match error {
        WorkerError::ReentrantOperation => SafeInitError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => SafeInitError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut {
            kind: DeadlineKind::Compile,
            timeout,
        } => SafeInitError::CompileTimedOut { timeout },
        WorkerError::TimedOut {
            kind: DeadlineKind::Initialize,
            timeout,
        } => SafeInitError::InitializationTimedOut { timeout },
        WorkerError::Terminated(termination) => SafeInitError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeInitError::Transport(other.to_string()),
    }
}

pub(super) fn map_exec_worker_error(error: WorkerError) -> SafeExecError {
    match error {
        WorkerError::ReentrantOperation => SafeExecError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => SafeExecError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => SafeExecError::TimedOut { timeout },
        WorkerError::Terminated(termination) => SafeExecError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => SafeExecError::Transport(other.to_string()),
    }
}

pub(super) fn map_asset_worker_error(error: WorkerError) -> AssetError {
    match error {
        WorkerError::ReentrantOperation => AssetError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => AssetError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => AssetError::TimedOut { timeout },
        WorkerError::Terminated(termination) => AssetError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => AssetError::Transport(other.to_string()),
    }
}

pub(super) fn map_resident_worker_error(error: WorkerError) -> ResidentError {
    match error {
        WorkerError::ReentrantOperation => ResidentError::InvalidRequest(
            "operation re-entered the same SafeRuntime session from its active callback".into(),
        ),
        WorkerError::LocalRequest(error) => ResidentError::InvalidRequest(error.to_string()),
        WorkerError::TimedOut { timeout, .. } => ResidentError::TimedOut { timeout },
        WorkerError::Terminated(termination) => ResidentError::ChildTerminated {
            status: termination.status,
            stderr: termination.stderr,
        },
        other => ResidentError::Transport(other.to_string()),
    }
}

pub(super) fn map_resident_teardown_worker_error(error: WorkerError) -> ResidentError {
    match error {
        WorkerError::LocalRequest(error) => ResidentError::Teardown(error.to_string()),
        other => map_resident_worker_error(other),
    }
}

#[cfg(test)]
mod tests {
    use super::{STDERR_TAIL_BYTES, StderrTail};
    #[test]
    fn stderr_capture_retains_only_a_marked_fixed_tail() {
        let mut tail = StderrTail::default();
        tail.push(&vec![b'x'; STDERR_TAIL_BYTES + 17]);
        tail.push(b"final marker");

        assert_eq!(tail.bytes.len(), STDERR_TAIL_BYTES);
        let rendered = tail.render();
        assert!(rendered.starts_with("[... "));
        assert!(rendered.contains("stderr truncated"));
        assert!(rendered.ends_with("final marker"));
    }
}
