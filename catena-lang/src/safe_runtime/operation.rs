//! Exclusive session leases and deferred object destruction.

use super::protocol::{Request, Response};
#[cfg(test)]
use std::time::{Duration, Instant};
use std::{
    sync::{Arc, Condvar, Mutex},
    thread,
};

/// Serializes every operation that can touch one worker session. An invocation
/// keeps its lease across publication callbacks and artifact release acknowledgements, so its
/// still-armed watchdog can never kill the worker while an unrelated operation
/// is using it.
#[derive(Debug, Default)]
pub(super) struct SessionOperationGate {
    state: Mutex<SessionOperationState>,
    available: Condvar,
}

#[derive(Debug, Default)]
pub(super) struct SessionOperationState {
    active: Option<ActiveSessionOperation>,
    artifact_release_deferrals: ArtifactReleaseDeferrals,
    #[cfg(test)]
    artifact_release_waiters: usize,
}

#[derive(Debug, Default)]
pub(super) enum ArtifactReleaseDeferrals {
    #[default]
    Closed,
    Open(Vec<DeferredRelease>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeferredRelease {
    Artifact(usize),
}

impl DeferredRelease {
    pub(super) fn request(self) -> Request {
        match self {
            Self::Artifact(artifact) => Request::ReleaseArtifact { artifact },
        }
    }

    pub(super) fn acknowledged(self, response: &Response) -> bool {
        match (self, response) {
            (Self::Artifact(expected), Response::ArtifactReleased(Ok(id))) => expected == *id,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveSessionOperation {
    owner: thread::ThreadId,
    kind: SessionOperationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SessionOperationKind {
    Ordinary,
    Streaming,
}

#[derive(Debug)]
pub(super) struct SessionOperationLease {
    gate: Arc<SessionOperationGate>,
    operation: ActiveSessionOperation,
}

impl SessionOperationGate {
    pub(super) fn acquire(
        self: &Arc<Self>,
        kind: SessionOperationKind,
    ) -> Result<SessionOperationLease, ()> {
        let operation = ActiveSessionOperation {
            owner: thread::current().id(),
            kind,
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match state.active {
                None => {
                    debug_assert!(matches!(
                        &state.artifact_release_deferrals,
                        ArtifactReleaseDeferrals::Closed
                    ));
                    state.active = Some(operation);
                    return Ok(SessionOperationLease {
                        gate: self.clone(),
                        operation,
                    });
                }
                Some(active) if active.owner == operation.owner => return Err(()),
                Some(_) => {
                    state = self
                        .available
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        }
    }

    /// Admit one artifact destructor without racing invocation startup. A release
    /// waits while startup is uncommitted, then either joins the invocation's
    /// open deferral batch or acquires the ordinary lease after startup rolls
    /// back. Same-thread non-invocation re-entry remains an error.
    #[cfg(test)]
    pub(super) fn acquire_or_defer_artifact_release(
        self: &Arc<Self>,
        artifact: usize,
    ) -> Result<Option<SessionOperationLease>, ()> {
        self.acquire_or_defer_release(DeferredRelease::Artifact(artifact))
    }

    pub(super) fn acquire_or_defer_release(
        self: &Arc<Self>,
        release: DeferredRelease,
    ) -> Result<Option<SessionOperationLease>, ()> {
        let operation = ActiveSessionOperation {
            owner: thread::current().id(),
            kind: SessionOperationKind::Ordinary,
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match state.active {
                None => {
                    debug_assert!(matches!(
                        &state.artifact_release_deferrals,
                        ArtifactReleaseDeferrals::Closed
                    ));
                    state.active = Some(operation);
                    return Ok(Some(SessionOperationLease {
                        gate: self.clone(),
                        operation,
                    }));
                }
                Some(active) if active.kind == SessionOperationKind::Streaming => {
                    if let ArtifactReleaseDeferrals::Open(releases) =
                        &mut state.artifact_release_deferrals
                    {
                        releases.push(release);
                        return Ok(None);
                    }
                    if active.owner == operation.owner {
                        return Err(());
                    }
                }
                Some(active) if active.owner == operation.owner => return Err(()),
                Some(_) => {}
            }
            #[cfg(test)]
            {
                state.artifact_release_waiters += 1;
                self.available.notify_all();
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            #[cfg(test)]
            {
                debug_assert!(state.artifact_release_waiters != 0);
                state.artifact_release_waiters -= 1;
            }
        }
    }

    #[cfg(test)]
    pub(super) fn wait_for_artifact_release_waiter(&self) {
        let deadline = Instant::now() + Duration::from_secs(1);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.artifact_release_waiters == 0 {
            let now = Instant::now();
            assert!(now < deadline, "artifact release did not wait on the gate");
            let (next, result) = self
                .available
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next;
            assert!(
                !result.timed_out() || state.artifact_release_waiters != 0,
                "artifact release did not wait on the gate"
            );
        }
    }

    /// Commit a successfully started invocation to accepting artifact-release
    /// deferrals, then wake every destructor that waited during worker I/O.
    pub(super) fn open_artifact_release_deferrals(&self, operation: ActiveSessionOperation) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(operation));
        debug_assert_eq!(operation.kind, SessionOperationKind::Streaming);
        if matches!(
            &state.artifact_release_deferrals,
            ArtifactReleaseDeferrals::Closed
        ) {
            state.artifact_release_deferrals = ArtifactReleaseDeferrals::Open(Vec::new());
            self.available.notify_all();
        } else {
            debug_assert!(
                false,
                "invocation artifact-release deferral was already open"
            );
        }
    }

    /// Atomically close deferral and take every release accepted before this
    /// point. A concurrent destructor is therefore either in this returned
    /// batch or observes `Closed` and waits for the ordinary operation lease.
    pub(super) fn seal_artifact_release_deferrals(
        &self,
        operation: ActiveSessionOperation,
    ) -> Vec<DeferredRelease> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(operation));
        debug_assert_eq!(operation.kind, SessionOperationKind::Streaming);
        match std::mem::take(&mut state.artifact_release_deferrals) {
            ArtifactReleaseDeferrals::Open(releases) => releases,
            ArtifactReleaseDeferrals::Closed => {
                debug_assert!(
                    false,
                    "invocation artifact-release deferral was already sealed"
                );
                Vec::new()
            }
        }
    }
}

impl SessionOperationLease {
    pub(super) fn open_artifact_release_deferrals(&self) {
        self.gate.open_artifact_release_deferrals(self.operation);
    }

    pub(super) fn seal_artifact_release_deferrals(&self) -> Vec<DeferredRelease> {
        self.gate.seal_artifact_release_deferrals(self.operation)
    }
}

impl Drop for SessionOperationLease {
    fn drop(&mut self) {
        let mut state = self
            .gate
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.active, Some(self.operation));
        // A missing protocol teardown already invalidates the worker. Never
        // let stale destructor work escape into the next operation's lease.
        state.artifact_release_deferrals = ArtifactReleaseDeferrals::Closed;
        state.active = None;
        self.gate.available.notify_all();
    }
}
