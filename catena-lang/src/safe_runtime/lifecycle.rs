//! Process-local worker admission, process-group identity and reaping.

use super::SafeInitError;
use std::{
    io,
    process::{Child, ExitStatus},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

// One atomic word makes admission and retirement linearizable without putting
// a poisonable or fork-inherited mutex on SafeRuntime's process-wide spawn
// path. The high half identifies the process; the low half contains two u16
// counters (workers being spawned, then leaders awaiting a proven wait). The
// maximum retirement value is a sticky saturation sentinel, not a count.
//
// `spawning` is an admission claim, not something retirement waits for. Every
// constructor must mutate this word so it is totally ordered with a terminal
// transition. A constructor whose claim wins first may finish; retirement wins
// against every later constructor. Counting claims permits concurrent healthy
// session construction without serializing `Command::spawn`, while terminal
// paths stay bounded and never wait for those earlier operations.
const LIFECYCLE_PROCESS_SHIFT: u32 = 32;
const LIFECYCLE_RETIRING_SHIFT: u32 = 16;
const LIFECYCLE_COUNTER_MASK: u64 = u16::MAX as u64;
const LIFECYCLE_RETIREMENT_SATURATED: u16 = u16::MAX;

#[derive(Debug, Default)]
pub(super) struct WorkerLifecycleGate {
    state: AtomicU64,
}

#[derive(Debug)]
pub(super) struct WorkerSpawnPermit {
    gate: Arc<WorkerLifecycleGate>,
    process: u32,
}

#[derive(Debug)]
pub(super) struct WorkerRetirementPermit {
    gate: Arc<WorkerLifecycleGate>,
    process: u32,
    completable: bool,
    completed: AtomicBool,
}

impl WorkerLifecycleGate {
    pub(super) fn process_id() -> u32 {
        let process = unsafe { libc::getpid() };
        u32::try_from(process).expect("getpid must return a positive value representable as u32")
    }

    pub(super) fn encode(process: u32, spawning: u16, retiring: u16) -> u64 {
        (u64::from(process) << LIFECYCLE_PROCESS_SHIFT)
            | (u64::from(retiring) << LIFECYCLE_RETIRING_SHIFT)
            | u64::from(spawning)
    }

    pub(super) fn decode(state: u64) -> (u32, u16, u16) {
        let process = (state >> LIFECYCLE_PROCESS_SHIFT) as u32;
        let retiring = ((state >> LIFECYCLE_RETIRING_SHIFT) & LIFECYCLE_COUNTER_MASK) as u16;
        let spawning = (state & LIFECYCLE_COUNTER_MASK) as u16;
        (process, spawning, retiring)
    }

    pub(super) fn begin_spawn(self: &Arc<Self>) -> Result<WorkerSpawnPermit, SafeInitError> {
        self.begin_spawn_for(Self::process_id())
    }

    pub(super) fn begin_spawn_for(
        self: &Arc<Self>,
        process: u32,
    ) -> Result<WorkerSpawnPermit, SafeInitError> {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            let next = if owner != process {
                // A post-fork child has no ownership over the parent's worker
                // leaders. Reset the copied counters before its first spawn.
                Self::encode(process, 1, 0)
            } else {
                if retiring != 0 {
                    return Err(SafeInitError::PreviousWorkerUnreaped { workers: retiring });
                }
                let Some(spawning) = spawning.checked_add(1) else {
                    return Err(SafeInitError::LifecycleCapacity);
                };
                Self::encode(process, spawning, retiring)
            };
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(WorkerSpawnPermit {
                    gate: self.clone(),
                    process,
                });
            }
        }
    }

    pub(super) fn begin_retirement(self: &Arc<Self>) -> WorkerRetirementPermit {
        self.begin_retirement_for(Self::process_id())
    }

    pub(super) fn begin_retirement_for(self: &Arc<Self>, process: u32) -> WorkerRetirementPermit {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            let (next, completable) = if owner != process {
                (Self::encode(process, 0, 1), true)
            } else {
                // Reserve u16::MAX as a sticky saturation sentinel. Once an
                // additional leader cannot be counted, no tracked completion
                // may reopen admission while that untracked leader can remain.
                if retiring >= LIFECYCLE_RETIREMENT_SATURATED - 1 {
                    (
                        Self::encode(process, spawning, LIFECYCLE_RETIREMENT_SATURATED),
                        false,
                    )
                } else {
                    (Self::encode(process, spawning, retiring + 1), true)
                }
            };
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return WorkerRetirementPermit {
                    gate: self.clone(),
                    process,
                    completable,
                    completed: AtomicBool::new(false),
                };
            }
        }
    }

    pub(super) fn finish_spawn(&self, process: u32) {
        self.update_current_process(process, |spawning, retiring| {
            (spawning.saturating_sub(1), retiring)
        });
    }

    pub(super) fn finish_retirement(&self, process: u32) {
        self.update_current_process(process, |spawning, retiring| {
            let retiring = if retiring == LIFECYCLE_RETIREMENT_SATURATED {
                LIFECYCLE_RETIREMENT_SATURATED
            } else {
                retiring.saturating_sub(1)
            };
            (spawning, retiring)
        });
    }

    pub(super) fn update_current_process(
        &self,
        process: u32,
        update: impl Fn(u16, u16) -> (u16, u16),
    ) {
        loop {
            let current = self.state.load(Ordering::Acquire);
            let (owner, spawning, retiring) = Self::decode(current);
            if owner != process {
                // The permit was copied across fork. It belongs to the parent
                // epoch and must not alter the child's freshly reset counters.
                return;
            }
            let (spawning, retiring) = update(spawning, retiring);
            let next = Self::encode(process, spawning, retiring);
            if self
                .state
                .compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

impl Drop for WorkerSpawnPermit {
    fn drop(&mut self) {
        self.gate.finish_spawn(self.process);
    }
}

impl WorkerRetirementPermit {
    /// Complete only after `wait(2)` has positively established leader exit.
    /// Merely dropping this value is deliberately fail-closed.
    pub(super) fn finish_after_wait(&self) {
        if self.completable && !self.completed.swap(true, Ordering::AcqRel) {
            self.gate.finish_retirement(self.process);
        }
    }
}

pub(super) fn worker_lifecycle_gate() -> Arc<WorkerLifecycleGate> {
    static GATE: std::sync::OnceLock<Arc<WorkerLifecycleGate>> = std::sync::OnceLock::new();
    GATE.get_or_init(|| Arc::new(WorkerLifecycleGate::default()))
        .clone()
}

/// Owns the usable lifetime of one private process-group ID.
///
/// Linux may reuse a numeric PID/PGID as soon as the group leader is reaped.
/// Every watchdog signal and every kill-plus-`try_wait` therefore takes the
/// same gate; the successful reap retires the ID before releasing that gate. A
/// stale invocation watchdog can observe retirement, but can never signal a
/// later process that happened to receive the same numeric ID. Reaping is
/// deliberately exposed only together with a preceding group-wide kill, so a
/// descendant cannot outlive the leader whose PID pins this identity.
#[derive(Debug)]
pub(super) struct ManagedProcessGroup {
    id: u32,
    state: Mutex<ManagedProcessGroupState>,
    lifecycle_gate: Arc<WorkerLifecycleGate>,
    retirement: std::sync::OnceLock<WorkerRetirementPermit>,
}

#[derive(Debug)]
pub(super) struct ManagedProcessGroupState {
    active: bool,
    kill_count: usize,
}

impl ManagedProcessGroup {
    pub(super) fn new(id: u32, lifecycle_gate: Arc<WorkerLifecycleGate>) -> Self {
        Self {
            id,
            state: Mutex::new(ManagedProcessGroupState {
                active: true,
                kill_count: 0,
            }),
            lifecycle_gate,
            retirement: std::sync::OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn new_test(id: u32) -> Self {
        Self::new(id, Arc::new(WorkerLifecycleGate::default()))
    }

    pub(super) fn state(&self) -> std::sync::MutexGuard<'_, ManagedProcessGroupState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn kill(&self) {
        let mut state = self.state();
        if !state.active {
            return;
        }
        self.begin_retirement_while_active();
        state.kill_count = state.kill_count.saturating_add(1);
        kill_process_group(self.id);
    }

    /// Close process-wide worker admission before this live leader enters any
    /// terminal path. The OnceLock makes concurrent watchdog, protocol, and
    /// Drop transitions idempotent without a poisonable lifecycle mutex.
    pub(super) fn begin_retirement(&self) {
        let state = self.state();
        if state.active {
            self.begin_retirement_while_active();
        }
    }

    pub(super) fn begin_retirement_while_active(&self) {
        self.retirement
            .get_or_init(|| self.lifecycle_gate.begin_retirement());
    }

    pub(super) fn finish_retirement_after_wait(&self) {
        if let Some(retirement) = self.retirement.get() {
            retirement.finish_after_wait();
        }
    }

    /// Kill every current member while the numeric PGID is still protected,
    /// then reap the leader under that same gate. Keeping these inseparable is
    /// what makes both descendant cleanup and PGID retirement hard to bypass.
    pub(super) fn kill_and_try_wait(&self, child: &mut Child) -> io::Result<Option<ExitStatus>> {
        let mut state = self.state();
        if state.active {
            self.begin_retirement_while_active();
            state.kill_count = state.kill_count.saturating_add(1);
            kill_process_group(self.id);
        }
        let status = child.try_wait()?;
        if status.is_some() {
            state.active = false;
            self.finish_retirement_after_wait();
        }
        Ok(status)
    }

    /// Used only on an already-killed exceptional path before a blocking
    /// background wait. No future signal may use this numeric identity.
    pub(super) fn retire(&self) {
        self.state().active = false;
    }

    #[cfg(test)]
    pub(super) fn kill_count(&self) -> usize {
        self.state().kill_count
    }
}

#[derive(Debug)]
pub(super) struct ReapResources {
    child: Option<Child>,
    process_group: Arc<ManagedProcessGroup>,
    stderr_reader: Option<JoinHandle<()>>,
    scratch: Option<tempfile::TempDir>,
    #[cfg(test)]
    pub(super) test_barrier: Option<Box<LifecycleTestBarrier>>,
}

#[cfg(test)]
#[derive(Debug)]
pub(super) struct LifecycleTestBarrier {
    pub(super) entered: mpsc::SyncSender<()>,
    pub(super) release: mpsc::Receiver<()>,
}

impl ReapResources {
    pub(super) fn new(
        child: Option<Child>,
        process_group: Arc<ManagedProcessGroup>,
        stderr_reader: Option<JoinHandle<()>>,
        scratch: Option<tempfile::TempDir>,
    ) -> Self {
        Self {
            child,
            process_group,
            stderr_reader,
            scratch,
            #[cfg(test)]
            test_barrier: None,
        }
    }
}

/// A reaper is created fallibly before its worker child. Teardown can then
/// transfer ownership without attempting to create a thread from `Drop`, when
/// failure would otherwise strand a killed child without a waiter.
#[derive(Debug)]
pub(super) struct BackgroundReaper {
    sender: Option<mpsc::SyncSender<ReapResources>>,
    handle: Option<JoinHandle<()>>,
}

impl BackgroundReaper {
    #[cfg(test)]
    pub(super) fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    pub(super) fn start() -> io::Result<Self> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle = thread::Builder::new()
            .name("catena-safe-runtime-reaper".to_string())
            .spawn(move || {
                if let Ok(resources) = receiver.recv() {
                    reap_resources(resources);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    pub(super) fn handoff(&mut self, resources: ReapResources) -> Result<(), ReapResources> {
        let Some(sender) = self.sender.take() else {
            return Err(resources);
        };
        match sender.send(resources) {
            Ok(()) => {
                // Detach only after the live receiver has accepted ownership.
                self.handle.take();
                Ok(())
            }
            Err(mpsc::SendError(resources)) => Err(resources),
        }
    }
}

impl Drop for BackgroundReaper {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(super) fn reap_resources(mut resources: ReapResources) {
    #[cfg(test)]
    if resources.child.is_some()
        && let Some(barrier) = resources.test_barrier.take()
    {
        let _ = barrier.entered.send(());
        let _ = barrier.release.recv();
    }
    if let Some(mut child) = resources.child.take() {
        resources.process_group.kill();
        loop {
            match resources.process_group.kill_and_try_wait(&mut child) {
                Ok(Some(_)) => break,
                Ok(None) => thread::sleep(Duration::from_millis(5)),
                Err(_) => {
                    // The group has already received SIGKILL. Retire its
                    // numeric identity before falling back to a blocking wait,
                    // so stale watchdogs cannot target a reused PGID.
                    resources.process_group.retire();
                    if child.wait().is_ok() {
                        resources.process_group.finish_retirement_after_wait();
                    }
                    break;
                }
            }
        }
    }
    if let Some(stderr_reader) = resources.stderr_reader {
        let _ = stderr_reader.join();
    }
    // TempDir cleanup can traverse compiler output. Keep it on the lifecycle
    // thread so a forced operation and WorkerProcess::drop remain bounded.
    drop(resources.scratch.take());
}

pub(super) fn kill_process_group(process_group: u32) {
    let Ok(process_group) = i32::try_from(process_group) else {
        return;
    };
    // WorkerProcess::spawn makes the child PID its process-group ID. Killing
    // the negative ID stops that child and all compiler descendants. ESRCH is
    // benign: the process may have exited at the deadline boundary.
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::{LIFECYCLE_RETIREMENT_SATURATED, WorkerLifecycleGate};
    use crate::safe_runtime::SafeInitError;
    use std::sync::{Arc, atomic::Ordering};
    #[test]
    fn retirement_gate_counts_callers_and_resets_in_a_forked_process() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let parent = 41;
        let already_admitted = gate
            .begin_spawn_for(parent)
            .expect("initial spawn must be admitted");
        let first = gate.begin_retirement_for(parent);
        let second = gate.begin_retirement_for(parent);
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 2 })
        ));
        drop(already_admitted);
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 2 })
        ));
        first.finish_after_wait();
        assert!(matches!(
            gate.begin_spawn_for(parent),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));

        // A forked child copied the atomic word, but none of the parent's
        // workers are its children. Its PID starts a fresh lifecycle epoch.
        let child = 42;
        let child_spawn = gate
            .begin_spawn_for(child)
            .expect("forked process must reset copied retirement state");
        drop(child_spawn);
        second.finish_after_wait();
        let child_spawn = gate
            .begin_spawn_for(child)
            .expect("parent permit must not alter child epoch");
        drop(child_spawn);
    }

    #[test]
    fn dropped_retirement_permit_keeps_worker_admission_closed() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let process = 41;
        let retirement = gate.begin_retirement_for(process);
        drop(retirement);
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped { workers: 1 })
        ));
    }

    #[test]
    fn retirement_saturation_is_sticky_after_tracked_completions() {
        let gate = Arc::new(WorkerLifecycleGate::default());
        let process = 41;
        gate.state.store(
            WorkerLifecycleGate::encode(process, 0, LIFECYCLE_RETIREMENT_SATURATED - 2),
            Ordering::Release,
        );

        let last_tracked = gate.begin_retirement_for(process);
        assert!(last_tracked.completable);
        assert_eq!(
            WorkerLifecycleGate::decode(gate.state.load(Ordering::Acquire)).2,
            LIFECYCLE_RETIREMENT_SATURATED - 1
        );

        let saturated = gate.begin_retirement_for(process);
        assert!(!saturated.completable);
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped {
                workers: LIFECYCLE_RETIREMENT_SATURATED
            })
        ));

        last_tracked.finish_after_wait();
        saturated.finish_after_wait();
        assert_eq!(
            WorkerLifecycleGate::decode(gate.state.load(Ordering::Acquire)).2,
            LIFECYCLE_RETIREMENT_SATURATED
        );
        assert!(matches!(
            gate.begin_spawn_for(process),
            Err(SafeInitError::PreviousWorkerUnreaped {
                workers: LIFECYCLE_RETIREMENT_SATURATED
            })
        ));
    }
}
