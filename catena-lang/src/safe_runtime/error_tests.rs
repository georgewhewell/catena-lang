//! Unit and subprocess regression tests for isolated execution.

use super::child::{duplicate_unix_stream, receive_asset_file};
use super::deadline::{DeadlineReader, DeadlineWriter, ProcessGroupDeadline, set_nonblocking};
use super::lifecycle::{LifecycleTestBarrier, ManagedProcessGroup, WorkerLifecycleGate};
use super::operation::{DeferredRelease, SessionOperationGate, SessionOperationKind};
use super::protocol::{
    self, FrameEncodeError, ProtocolError, RemoteExecError, Request, Response, read_frame,
    write_frame,
};
use super::session::{execution_from_response, worker_for_artifact_teardown};
use super::worker::{DeadlineKind, WorkerError, WorkerProcess, sigchld_is_waitable};
use super::{
    AssetError, ChildMainError, ParentInterface, ResidentError, SafeExecError, SafeInitError,
    SafeRuntime, SafeRuntimeTimeouts, initialize_parent_ipc,
};
use crate::runtime::Backend;
use std::{
    env, fs,
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

const TIMEOUT_FIXTURE_ENV: &str = "CATENA_TEST_PROCESS_GROUP_TIMEOUT_FIXTURE";
const DROP_FIXTURE_ENV: &str = "CATENA_TEST_BOUNDED_DROP_FIXTURE";
const EOF_FIXTURE_ENV: &str = "CATENA_TEST_UNEXPECTED_EOF_FIXTURE";
const EOF_DESCENDANT_ENV: &str = "CATENA_TEST_UNEXPECTED_EOF_DESCENDANT";
const SCRATCH_FIXTURE_ENV: &str = "CATENA_TEST_SCRATCH_FIXTURE";
const SCRATCH_DESCENDANT_ENV: &str = "CATENA_TEST_SCRATCH_DESCENDANT";

fn target_backed_test_directory(prefix: &str) -> tempfile::TempDir {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("catena-lang crate must live in its workspace");
    let parent = env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| workspace.join("target"))
        .join("safe-runtime-lifecycle-tests");
    fs::create_dir_all(&parent).expect("create target-backed lifecycle test directory");
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(parent)
        .expect("create target-backed lifecycle fixture")
}

fn wait_for_path_state(path: &Path, exists: bool, message: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while path.exists() != exists {
        assert!(Instant::now() < deadline, "{message}: {}", path.display());
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn structured_rejections_preserve_the_session() {
    assert!(!SafeInitError::RemoteLoad("invalid source".into()).invalidates_session());
    assert!(!SafeInitError::InvalidRequest("oversized".into()).invalidates_session());
    assert!(!AssetError::Remote("invalid asset".into()).invalidates_session());
    assert!(!AssetError::InvalidRequest("oversized".into()).invalidates_session());
    assert!(!ResidentError::Remote("invalid request".into()).invalidates_session());
    assert!(!ResidentError::InvalidRequest("oversized".into()).invalidates_session());
    assert!(!SafeExecError::UnknownArtifact.invalidates_session());
    assert!(!SafeExecError::RemoteMemory("invalid view".into()).invalidates_session());
    assert!(!SafeExecError::InvalidRequest("oversized".into()).invalidates_session());
}

#[test]
fn protocol_failures_invalidate_the_session() {
    assert!(SafeInitError::UnexpectedResponse.invalidates_session());
    assert!(
        SafeInitError::CompileTimedOut {
            timeout: Duration::from_secs(1)
        }
        .invalidates_session()
    );
    assert!(ResidentError::UnexpectedResponse.invalidates_session());
    assert!(ResidentError::Teardown("missing acknowledgement".into()).invalidates_session());
    assert!(SafeExecError::UnexpectedResponse.invalidates_session());
    assert!(SafeExecError::Transport("partial write".into()).invalidates_session());
}

#[test]
fn invocation_watchdog_never_overlaps_a_queued_session_operation() {
    let gate = Arc::new(SessionOperationGate::default());
    let invocation = gate
        .acquire(SessionOperationKind::Streaming)
        .expect("acquire invocation fixture");
    let process_group = Arc::new(ManagedProcessGroup::new_test(u32::MAX));
    let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_millis(40))
        .expect("arm invocation fixture watchdog");
    let (attempted_sender, attempted) = mpsc::channel();
    let (acquired_sender, acquired) = mpsc::channel();

    thread::scope(|scope| {
        let contender_gate = gate.clone();
        scope.spawn(move || {
            attempted_sender.send(()).expect("announce gate attempt");
            let _operation = contender_gate
                .acquire(SessionOperationKind::Ordinary)
                .expect("queued operation must eventually acquire the gate");
            acquired_sender.send(()).expect("announce gate acquisition");
        });
        attempted
            .recv_timeout(Duration::from_secs(1))
            .expect("contending operation did not start");

        let watchdog_limit = Instant::now() + Duration::from_secs(1);
        while !deadline.watchdog_fired() {
            assert!(
                Instant::now() < watchdog_limit,
                "invocation watchdog did not fire"
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            matches!(acquired.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "an unrelated operation entered while the invocation watchdog was armed"
        );
        // The timeout flag is published before kill(). Join the watchdog before
        // inspecting its side effects, while still holding the invocation gate.
        assert!(deadline.finish());
        assert_eq!(process_group.kill_count(), 1);

        drop(invocation);
        acquired
            .recv_timeout(Duration::from_secs(1))
            .expect("queued operation was not released with the invocation lease");
    });
}

#[test]
fn same_thread_invocation_reentry_is_rejected_and_artifact_drop_is_deferred() {
    let gate = Arc::new(SessionOperationGate::default());
    let invocation = gate
        .acquire(SessionOperationKind::Streaming)
        .expect("acquire invocation fixture");

    assert!(
        gate.acquire(SessionOperationKind::Ordinary).is_err(),
        "same-thread callback re-entry must not deadlock"
    );
    assert!(
        gate.acquire_or_defer_artifact_release(17).is_err(),
        "same-thread artifact release cannot wait during uncommitted startup"
    );
    invocation.open_artifact_release_deferrals();
    assert!(matches!(
        gate.acquire_or_defer_artifact_release(17),
        Ok(None)
    ));
    assert!(matches!(
        gate.acquire_or_defer_artifact_release(23),
        Ok(None)
    ));
    assert_eq!(
        invocation.seal_artifact_release_deferrals(),
        vec![DeferredRelease::Artifact(17), DeferredRelease::Artifact(23)]
    );
    drop(invocation);

    let _next = gate
        .acquire(SessionOperationKind::Ordinary)
        .expect("gate must be reusable after invocation teardown");
}

#[test]
fn artifact_release_from_callback_is_deferred_and_requires_matching_ack() {
    let gate = Arc::new(SessionOperationGate::default());
    let invocation = gate.acquire(SessionOperationKind::Streaming).unwrap();
    invocation.open_artifact_release_deferrals();
    let release = DeferredRelease::Artifact(7);
    assert!(matches!(gate.acquire_or_defer_release(release), Ok(None)));
    assert_eq!(invocation.seal_artifact_release_deferrals(), vec![release]);
    assert!(matches!(
        release.request(),
        Request::ReleaseArtifact { artifact: 7 }
    ));
    assert!(release.acknowledged(&Response::ArtifactReleased(Ok(7))));
    assert!(!release.acknowledged(&Response::ArtifactReleased(Ok(8))));
    assert!(!release.acknowledged(&Response::OutputsReleased));
    drop(invocation);
    let _next = gate.acquire(SessionOperationKind::Ordinary).unwrap();
}

#[test]
fn preopen_artifact_release_defers_after_start_and_waits_after_seal() {
    let gate = Arc::new(SessionOperationGate::default());
    let invocation = gate
        .acquire(SessionOperationKind::Streaming)
        .expect("acquire invocation fixture");

    thread::scope(|scope| {
        let callback_drop_gate = gate.clone();
        let (completed_sender, completed) = mpsc::sync_channel(1);
        scope.spawn(move || {
            let release = callback_drop_gate
                .acquire_or_defer_artifact_release(17)
                .expect("cross-thread artifact release admission");
            completed_sender
                .send((17, release.is_none()))
                .expect("report pre-open artifact drop outcome");
        });
        gate.wait_for_artifact_release_waiter();
        assert!(
            matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "artifact release completed before invocation startup committed"
        );

        invocation.open_artifact_release_deferrals();
        assert_eq!(
            completed
                .recv_timeout(Duration::from_secs(1))
                .expect("pre-open artifact release did not join deferral after start"),
            (17, true)
        );
        assert_eq!(
            invocation.seal_artifact_release_deferrals(),
            vec![DeferredRelease::Artifact(17)]
        );

        let post_seal_gate = gate.clone();
        let (completed_sender, completed) = mpsc::sync_channel(1);
        scope.spawn(move || {
            let release = post_seal_gate
                .acquire_or_defer_artifact_release(23)
                .expect("post-seal artifact release admission");
            completed_sender
                .send((23, release.is_none()))
                .expect("report post-seal artifact drop outcome");
        });
        gate.wait_for_artifact_release_waiter();
        assert!(
            matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "post-seal artifact release entered before invocation teardown"
        );

        drop(invocation);
        assert_eq!(
            completed
                .recv_timeout(Duration::from_secs(1))
                .expect("post-seal artifact release did not resume after invocation teardown"),
            (23, false)
        );
    });
}

#[test]
fn preopen_artifact_release_acquires_after_failed_invocation_start() {
    let gate = Arc::new(SessionOperationGate::default());
    let invocation = gate
        .acquire(SessionOperationKind::Streaming)
        .expect("acquire invocation-start fixture");

    thread::scope(|scope| {
        let artifact_drop_gate = gate.clone();
        let (completed_sender, completed) = mpsc::sync_channel(1);
        scope.spawn(move || {
            let release = artifact_drop_gate
                .acquire_or_defer_artifact_release(29)
                .expect("failed-start artifact release admission");
            completed_sender
                .send((29, release.is_none()))
                .expect("report failed-start artifact drop outcome");
        });
        gate.wait_for_artifact_release_waiter();
        assert!(
            matches!(completed.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "artifact release completed before invocation startup resolved"
        );

        // Dropping a still-closed lease models every Compute startup error.
        // The waiting destructor must retain its artifact ID and acquire the
        // ordinary release lease rather than disappear into a stale queue.
        drop(invocation);
        assert_eq!(
            completed
                .recv_timeout(Duration::from_secs(1))
                .expect("artifact release did not resume after failed startup"),
            (29, false)
        );
    });

    let _next = gate
        .acquire(SessionOperationKind::Ordinary)
        .expect("failed-start artifact release did not relinquish the gate");
}

#[test]
fn descriptor_transport_failure_is_a_fatal_child_protocol_error() {
    let (mut sender, receiver) = UnixStream::pair().expect("descriptor fixture pair");
    sender
        .write_all(&[0])
        .expect("write malformed descriptor message");

    assert!(matches!(
        receive_asset_file(&receiver),
        Err(ChildMainError::Protocol(message))
            if message.contains("asset descriptor transport failed")
    ));
}

#[test]
fn worker_asset_socket_duplicates_instead_of_claiming_the_supplied_descriptor() {
    let (original, mut peer) = UnixStream::pair().expect("asset socket fixture pair");
    let original_fd = original.as_raw_fd();
    let flags = unsafe { libc::fcntl(original_fd, libc::F_GETFD) };
    assert_ne!(flags, -1);
    assert_ne!(
        unsafe { libc::fcntl(original_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) },
        -1
    );
    let mut duplicate = duplicate_unix_stream(original_fd).expect("duplicate asset socket");
    let original_flags = unsafe { libc::fcntl(original_fd, libc::F_GETFD) };
    assert_ne!(original_flags & libc::FD_CLOEXEC, 0);
    let duplicate_flags = unsafe { libc::fcntl(duplicate.as_raw_fd(), libc::F_GETFD) };
    assert_ne!(duplicate_flags & libc::FD_CLOEXEC, 0);
    drop(original);

    duplicate.write_all(b"x").expect("write through duplicate");
    let mut byte = [0_u8; 1];
    peer.read_exact(&mut byte).expect("read duplicate payload");
    assert_eq!(byte, *b"x");
    assert_eq!(
        duplicate_unix_stream(-1)
            .expect_err("invalid descriptor must be rejected")
            .raw_os_error(),
        Some(libc::EBADF)
    );
}

#[test]
fn sigchld_must_leave_children_waitable() {
    assert!(sigchld_is_waitable(libc::SIG_DFL, 0));
    assert!(!sigchld_is_waitable(libc::SIG_IGN, 0));
    assert!(!sigchld_is_waitable(libc::SIG_DFL, libc::SA_NOCLDWAIT));
}

#[test]
fn zero_compile_timeout_is_rejected_before_spawning() {
    assert!(matches!(
        SafeRuntime::with_compile_timeout(Backend::Hip, Duration::ZERO),
        Err(SafeInitError::InvalidCompileTimeout)
    ));
}

#[test]
fn zero_execution_timeout_is_rejected_before_spawning() {
    let timeouts = SafeRuntimeTimeouts::default().with_execution_timeout(Duration::ZERO);
    assert!(matches!(
        SafeRuntime::with_timeouts(Backend::Hip, timeouts),
        Err(SafeInitError::InvalidExecutionTimeout)
    ));
}

#[test]
fn resident_protocol_construction_does_not_initialize_parent_ipc() {
    let calls = std::cell::Cell::new(0);
    let ipc = initialize_parent_ipc(ParentInterface::ResidentProtocol, || {
        calls.set(calls.get() + 1);
        Ok::<_, std::convert::Infallible>(())
    })
    .expect("infallible parent IPC fixture");

    assert!(ipc.is_none());
    assert_eq!(calls.get(), 0);
}

#[test]
fn raw_value_construction_initializes_parent_ipc() {
    let calls = std::cell::Cell::new(0);
    let ipc = initialize_parent_ipc(ParentInterface::RawValues, || {
        calls.set(calls.get() + 1);
        Ok::<_, std::convert::Infallible>("parent IPC")
    })
    .expect("infallible parent IPC fixture");

    assert_eq!(ipc, Some("parent IPC"));
    assert_eq!(calls.get(), 1);
}

#[test]
fn compile_deadline_kills_the_worker_process_group() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::compile_deadline_process_fixture",
        ])
        .env(TIMEOUT_FIXTURE_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0);
    let mut child = command.spawn().expect("spawn timeout fixture");
    let process_group = Arc::new(ManagedProcessGroup::new_test(child.id()));
    let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_millis(50))
        .expect("spawn deadline watchdog");
    let wait_deadline = Instant::now() + Duration::from_secs(2);
    while !deadline.watchdog_fired() {
        assert!(
            Instant::now() < wait_deadline,
            "deadline watchdog did not fire"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let status = loop {
        if let Some(status) = process_group
            .kill_and_try_wait(&mut child)
            .expect("poll timeout fixture")
        {
            break status;
        }
        assert!(
            Instant::now() < wait_deadline,
            "deadline did not stop its process fixture"
        );
        thread::sleep(Duration::from_millis(5));
    };
    assert!(deadline.finish(), "fixture exited before its deadline");
    assert!(!status.success(), "deadline did not kill the fixture");
}

#[test]
fn protocol_pipe_read_and_write_share_an_absolute_deadline() {
    let (mut reader, mut held_writer) = UnixStream::pair().expect("reader fixture pair");
    held_writer
        .write_all(&[1, 0])
        .expect("write partial frame header");
    set_nonblocking(reader.as_raw_fd()).expect("nonblocking reader fixture");
    let read_deadline = ProcessGroupDeadline::arm(
        Arc::new(ManagedProcessGroup::new_test(u32::MAX)),
        Duration::from_millis(40),
    )
    .expect("spawn read deadline watchdog");
    let descriptor = reader.as_raw_fd();
    let started = Instant::now();
    let error = read_frame::<Request>(&mut DeadlineReader::new(
        &mut reader,
        descriptor,
        &read_deadline,
    ))
    .expect_err("partial frame should reach its deadline");
    assert!(matches!(
        error,
        ProtocolError::Io(error) if error.kind() == io::ErrorKind::TimedOut
    ));
    assert!(read_deadline.finish());
    assert!(started.elapsed() < Duration::from_secs(1));

    let (mut writer, _held_reader) = UnixStream::pair().expect("writer fixture pair");
    set_nonblocking(writer.as_raw_fd()).expect("nonblocking writer fixture");
    let write_deadline = ProcessGroupDeadline::arm(
        Arc::new(ManagedProcessGroup::new_test(u32::MAX)),
        Duration::from_millis(40),
    )
    .expect("spawn write deadline watchdog");
    let descriptor = writer.as_raw_fd();
    let started = Instant::now();
    let error = write_frame(
        &mut DeadlineWriter::new(&mut writer, descriptor, &write_deadline),
        &Request::LoadSources {
            sources: vec!["x".repeat(8 * 1024 * 1024)],
            codegen: crate::codegen::CodegenKind::Default,
        },
    )
    .expect_err("undrained partial frame should reach its deadline");
    assert!(matches!(
        error,
        ProtocolError::Io(error) if error.kind() == io::ErrorKind::TimedOut
    ));
    assert!(write_deadline.finish());
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn worker_drop_is_bounded_when_the_child_ignores_shutdown() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let worker =
        WorkerProcess::spawn_command(command, &executable).expect("spawn bounded-drop fixture");
    thread::sleep(Duration::from_millis(20));
    let process = worker.child_id().expect("worker process ID");
    assert_eq!(
        unsafe { libc::kill(i32::try_from(process).unwrap(), 0) },
        0,
        "drop fixture exited before WorkerProcess was dropped"
    );

    let started = Instant::now();
    drop(worker);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "WorkerProcess::drop exceeded its grace and reap bounds"
    );
}

#[test]
fn forced_worker_termination_repeatedly_cleans_parent_owned_scratch() {
    let scratch_parent = target_backed_test_directory("scratch-parent-");
    let executable = env::current_exe().expect("test executable path");

    for teardown in ["watchdog", "reaper", "drop", "watchdog", "reaper", "drop"] {
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::scratch_process_fixture",
                "--nocapture",
            ])
            .env(SCRATCH_FIXTURE_ENV, "1");
        let mut worker =
            WorkerProcess::spawn_command_in(command, &executable, scratch_parent.path())
                .expect("spawn scratch fixture");
        let scratch = worker
            .scratch
            .as_ref()
            .expect("worker owns scratch")
            .path()
            .to_path_buf();
        wait_for_path_state(
            &scratch.join("ready"),
            true,
            "scratch fixture did not become ready",
        );

        match teardown {
            "watchdog" => {
                let result: Result<(), WorkerError> = worker.with_deadline(
                    Duration::from_millis(25),
                    DeadlineKind::Compile,
                    |_worker, _deadline| {
                        thread::sleep(Duration::from_millis(75));
                        Ok(())
                    },
                );
                assert!(matches!(result, Err(WorkerError::TimedOut { .. })));
                drop(worker);
            }
            "reaper" => {
                worker.handoff_background_reaper();
                drop(worker);
            }
            "drop" => drop(worker),
            _ => unreachable!(),
        }

        wait_for_path_state(
            &scratch,
            false,
            "forced worker teardown leaked its scratch directory",
        );
    }
}

#[test]
fn pathological_worker_resources_are_handed_to_a_background_reaper() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let mut worker =
        WorkerProcess::spawn_command(command, &executable).expect("spawn reap fixture");
    let process = worker.child_id().expect("worker process ID");
    assert!(
        worker
            .background_reaper
            .as_ref()
            .is_some_and(|reaper| reaper.is_running()),
        "reaper must exist before teardown begins"
    );
    worker.handoff_background_reaper();
    assert!(worker.child.is_none());
    assert!(worker.stderr_reader.is_none());

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let result = unsafe { libc::kill(i32::try_from(process).unwrap(), 0) };
        if result == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "background reaper did not wait for the killed worker"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn every_detached_teardown_gates_replacement_until_reap() {
    for teardown in ["handoff", "watchdog", "drop"] {
        let executable = env::current_exe().expect("test executable path");
        let mut command = Command::new(&executable);
        command
            .args([
                "--ignored",
                "--exact",
                "safe_runtime::error_tests::bounded_drop_process_fixture",
            ])
            .env(DROP_FIXTURE_ENV, "1");
        let gate = Arc::new(WorkerLifecycleGate::default());
        let mut worker = WorkerProcess::spawn_command_with_gate(command, &executable, gate.clone())
            .expect("spawn retirement-gate fixture");
        let (entered_sender, entered) = mpsc::sync_channel(1);
        let (release, release_receiver) = mpsc::channel();
        worker.force_background_handoff = true;
        worker.reaper_test_barrier = Some(Box::new(LifecycleTestBarrier {
            entered: entered_sender,
            release: release_receiver,
        }));

        match teardown {
            "handoff" => worker.handoff_background_reaper(),
            "watchdog" => {
                let result: Result<(), WorkerError> = worker.with_deadline(
                    Duration::from_millis(20),
                    DeadlineKind::Execution,
                    |_worker, _deadline| {
                        thread::sleep(Duration::from_millis(60));
                        Ok(())
                    },
                );
                assert!(matches!(result, Err(WorkerError::TimedOut { .. })));
            }
            "drop" => drop(worker),
            _ => unreachable!(),
        }

        entered
            .recv_timeout(Duration::from_secs(2))
            .unwrap_or_else(|_| panic!("{teardown} did not hand its leader to the reaper"));
        assert!(matches!(
            gate.begin_spawn(),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));
        release
            .send(())
            .expect("release retirement-gate fixture reaper");

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            match gate.begin_spawn() {
                Ok(permit) => {
                    drop(permit);
                    break;
                }
                Err(SafeInitError::PreviousWorkerUnreaped { .. }) => {
                    assert!(
                        Instant::now() < deadline,
                        "{teardown} did not reopen worker admission after wait"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("unexpected lifecycle error after {teardown}: {error}"),
            }
        }
    }
}

#[test]
fn drop_registers_retirement_before_graceful_shutdown() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let gate = Arc::new(WorkerLifecycleGate::default());
    let mut worker = WorkerProcess::spawn_command_with_gate(command, &executable, gate.clone())
        .expect("spawn pre-handoff retirement fixture");
    let (entered_sender, entered) = mpsc::sync_channel(1);
    let (release, release_receiver) = mpsc::channel();
    worker.drop_retirement_test_barrier = Some(Box::new(LifecycleTestBarrier {
        entered: entered_sender,
        release: release_receiver,
    }));

    thread::scope(|scope| {
        let dropper = scope.spawn(move || drop(worker));
        entered
            .recv_timeout(Duration::from_secs(2))
            .expect("Drop did not register retirement before graceful shutdown");
        assert!(matches!(
            gate.begin_spawn(),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));
        release
            .send(())
            .expect("release pre-handoff retirement fixture");
        dropper.join().expect("Drop fixture thread panicked");
    });

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match gate.begin_spawn() {
            Ok(permit) => {
                drop(permit);
                break;
            }
            Err(SafeInitError::PreviousWorkerUnreaped { .. }) => {
                assert!(
                    Instant::now() < deadline,
                    "Drop did not reopen admission after foreground or detached wait"
                );
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("unexpected lifecycle error after Drop: {error}"),
        }
    }
}

#[test]
fn unexpected_eof_kills_the_worker_process_group_before_reaping() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::unexpected_eof_process_fixture",
            "--nocapture",
        ])
        .env(EOF_FIXTURE_ENV, "1");
    let mut worker =
        WorkerProcess::spawn_command(command, &executable).expect("spawn unexpected-EOF fixture");
    let deadline = worker
        .arm_deadline(Duration::from_secs(5))
        .expect("arm EOF fixture deadline");

    // Drain test-harness chatter until the worker leader closes its
    // protocol pipe. Its sleeping descendant deliberately retains stderr.
    let descriptor = worker.stdout.get_ref().as_raw_fd();
    let mut output = Vec::new();
    let mut bytes = [0_u8; 1024];
    loop {
        let read = DeadlineReader::new(&mut worker.stdout, descriptor, &deadline)
            .read(&mut bytes)
            .expect("drain EOF fixture stdout");
        if read == 0 {
            break;
        }
        output.extend_from_slice(&bytes[..read]);
    }
    assert!(
        String::from_utf8_lossy(&output).contains("CATENA_DESCENDANT_STARTED"),
        "EOF fixture did not start its inherited process-group descendant"
    );
    loop {
        let ready = worker
            .stderr_tail
            .lock()
            .expect("lock EOF fixture stderr")
            .render()
            .contains("CATENA_DESCENDANT_READY");
        if ready {
            break;
        }
        deadline
            .ensure_io_time_remaining()
            .expect("descendant did not become ready before EOF cleanup deadline");
        thread::sleep(Duration::from_millis(5));
    }

    let termination = worker.reap(&deadline).expect("reap EOF fixture");
    assert!(!termination.status.success());
    assert!(worker.process_group.kill_count() >= 1);
    assert!(!deadline.finish(), "EOF cleanup exceeded its deadline");

    // The descendant inherited the worker's stderr descriptor and sleeps
    // for a minute. EOF here therefore proves group cleanup stopped it.
    if let Some(stderr_reader) = worker.stderr_reader.take() {
        worker
            .stderr_done
            .recv_timeout(Duration::from_secs(2))
            .expect("worker descendant retained stderr after group cleanup");
        stderr_reader
            .join()
            .expect("join EOF fixture stderr reader");
    }
}

#[test]
fn stale_watchdog_cannot_signal_after_the_group_leader_is_reaped() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(executable);
    command
        .arg("--list")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0);
    let mut child = command.spawn().expect("spawn short-lived group leader");
    let process_group = Arc::new(ManagedProcessGroup::new_test(child.id()));
    let deadline = ProcessGroupDeadline::arm(process_group.clone(), Duration::from_secs(2))
        .expect("spawn stale deadline watchdog");
    let wait_deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if process_group
            .kill_and_try_wait(&mut child)
            .expect("poll short-lived group leader")
            .is_some()
        {
            break;
        }
        assert!(
            Instant::now() < wait_deadline,
            "short-lived group leader did not exit before its watchdog"
        );
        thread::sleep(Duration::from_millis(5));
    }

    let watchdog_deadline = Instant::now() + Duration::from_secs(3);
    while !deadline.watchdog_fired() {
        assert!(
            Instant::now() < watchdog_deadline,
            "stale watchdog did not fire"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let kills_at_reap = process_group.kill_count();
    assert!(deadline.finish());
    assert_eq!(
        process_group.kill_count(),
        kills_at_reap,
        "retired process-group identity was signalled by a stale watchdog"
    );
}

#[test]
fn protocol_failure_automatically_poisons_the_worker() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let mut worker =
        WorkerProcess::spawn_command(command, &executable).expect("spawn poison fixture");
    let result: Result<(), WorkerError> = worker.with_deadline(
        Duration::from_secs(1),
        DeadlineKind::Execution,
        |_worker, _deadline| {
            Err(WorkerError::Protocol(ProtocolError::Io(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "fixture transport failure",
            ))))
        },
    );

    assert!(matches!(result, Err(WorkerError::Protocol(_))));
    assert!(worker.stdin.is_none());
    assert!(worker.child.is_none());
    assert!(matches!(
        worker.ensure_available(),
        Err(WorkerError::Invalidated)
    ));
}

#[test]
fn local_request_preparation_failure_preserves_the_worker() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let mut worker =
        WorkerProcess::spawn_command(command, &executable).expect("spawn local-request fixture");
    let process = worker.child_id().expect("worker process ID");
    let result: Result<(), WorkerError> = worker.with_deadline(
        Duration::from_secs(1),
        DeadlineKind::Execution,
        |_worker, _deadline| {
            Err(WorkerError::LocalRequest(FrameEncodeError::FrameTooLarge {
                actual: protocol::MAX_FRAME_LEN + 1,
                maximum: protocol::MAX_FRAME_LEN,
            }))
        },
    );

    assert!(matches!(result, Err(WorkerError::LocalRequest(_))));
    assert_eq!(worker.child_id().expect("worker remains owned"), process);
    assert!(worker.ensure_available().is_ok());
}

#[test]
fn structured_memory_rejection_preserves_the_worker() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let mut worker = WorkerProcess::spawn_command(command, &executable)
        .expect("spawn structured-memory-rejection fixture");
    let process = worker.child_id().expect("worker process ID");

    let error = execution_from_response(
        &mut worker,
        Response::Executed(Err(RemoteExecError::Memory("invalid IPC view".into()))),
    )
    .expect_err("structured memory rejection must remain an error");

    assert!(matches!(error, SafeExecError::RemoteMemory(_)));
    assert_eq!(worker.child_id().expect("worker remains owned"), process);
    assert!(worker.ensure_available().is_ok());

    let error = execution_from_response(
        &mut worker,
        Response::Executed(Err(RemoteExecError::Protocol("pending outputs".into()))),
    )
    .expect_err("protocol-state rejection must remain an error");
    assert!(matches!(error, SafeExecError::Transport(_)));
    assert!(matches!(
        worker.ensure_available(),
        Err(WorkerError::Invalidated)
    ));
}

#[test]
fn poisoned_artifact_teardown_lock_terminates_the_worker() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let worker = Mutex::new(
        WorkerProcess::spawn_command(command, &executable)
            .expect("spawn poisoned-artifact-teardown fixture"),
    );
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held_worker = worker.lock().expect("lock artifact-teardown fixture");
        panic!("poison artifact-teardown fixture");
    }));
    assert!(panicked.is_err());

    let error = worker_for_artifact_teardown(worker.lock())
        .expect_err("poisoned teardown lock must not yield a live worker");
    assert!(matches!(error, ResidentError::Transport(_)));
    let worker = worker
        .lock()
        .expect_err("fixture mutex remains poisoned")
        .into_inner();
    assert!(matches!(
        worker.ensure_available(),
        Err(WorkerError::Invalidated)
    ));
}

#[test]
fn unexpected_response_automatically_poisons_the_worker() {
    let executable = env::current_exe().expect("test executable path");
    let mut command = Command::new(&executable);
    command
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
        ])
        .env(DROP_FIXTURE_ENV, "1");
    let mut worker = WorkerProcess::spawn_command(command, &executable)
        .expect("spawn unexpected-response fixture");

    let error = worker.reject_protocol_response(ResidentError::UnexpectedResponse);

    assert!(error.invalidates_session());
    assert!(worker.stdin.is_none());
    assert!(worker.child.is_none());
    assert!(matches!(
        worker.ensure_available(),
        Err(WorkerError::Invalidated)
    ));
}

#[test]
#[ignore = "spawned only by compile_deadline_kills_the_worker_process_group"]
fn compile_deadline_process_fixture() {
    assert_eq!(
        env::var_os(TIMEOUT_FIXTURE_ENV).as_deref(),
        Some(std::ffi::OsStr::new("1"))
    );
    thread::sleep(Duration::from_secs(60));
}

#[test]
#[ignore = "spawned only by worker_drop_is_bounded_when_the_child_ignores_shutdown"]
fn bounded_drop_process_fixture() {
    assert_eq!(
        env::var_os(DROP_FIXTURE_ENV).as_deref(),
        Some(std::ffi::OsStr::new("1"))
    );
    if env::var_os(EOF_DESCENDANT_ENV).is_some() {
        eprintln!("CATENA_DESCENDANT_READY");
        io::stderr().flush().expect("flush EOF descendant marker");
    }
    thread::sleep(Duration::from_secs(60));
}

#[test]
#[ignore = "spawned only by unexpected_eof_kills_the_worker_process_group_before_reaping"]
fn unexpected_eof_process_fixture() {
    assert_eq!(
        env::var_os(EOF_FIXTURE_ENV).as_deref(),
        Some(std::ffi::OsStr::new("1"))
    );
    let executable = env::current_exe().expect("test executable path");
    Command::new(executable)
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::bounded_drop_process_fixture",
            "--nocapture",
        ])
        .env(DROP_FIXTURE_ENV, "1")
        .env(EOF_DESCENDANT_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn inherited process-group descendant");
    println!("CATENA_DESCENDANT_STARTED");
    io::stdout().flush().expect("flush EOF fixture marker");
    std::process::exit(23);
}

#[test]
#[ignore = "spawned only by forced_worker_termination_repeatedly_cleans_parent_owned_scratch"]
fn scratch_process_fixture() {
    assert_eq!(
        env::var_os(SCRATCH_FIXTURE_ENV).as_deref(),
        Some(std::ffi::OsStr::new("1"))
    );
    let scratch = env::temp_dir();
    if env::var_os(SCRATCH_DESCENDANT_ENV).is_some() {
        let descendant = scratch.join("compiler-descendant");
        fs::create_dir_all(&descendant).expect("create compiler descendant scratch");
        fs::write(descendant.join("output.o"), b"partial compiler output")
            .expect("write compiler descendant scratch");
        thread::sleep(Duration::from_secs(60));
        return;
    }

    let worker = scratch.join("worker-build");
    fs::create_dir_all(&worker).expect("create worker scratch");
    fs::write(worker.join("module.cpp"), b"partial generated source")
        .expect("write worker scratch");
    let executable = env::current_exe().expect("test executable path");
    let mut compiler = Command::new(executable)
        .args([
            "--ignored",
            "--exact",
            "safe_runtime::error_tests::scratch_process_fixture",
            "--nocapture",
        ])
        .env(SCRATCH_FIXTURE_ENV, "1")
        .env(SCRATCH_DESCENDANT_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn compiler descendant fixture");
    wait_for_path_state(
        &scratch.join("compiler-descendant/output.o"),
        true,
        "compiler descendant did not create scratch",
    );
    fs::write(scratch.join("ready"), b"ready").expect("write scratch ready marker");
    thread::sleep(Duration::from_secs(60));
    let _ = compiler.kill();
    let _ = compiler.wait();
}
