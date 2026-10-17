//! Compiled-program handles and child-side release.

use super::{Artifact, RemoteArtifact, SafeExecError, SafeRuntime, operation::DeferredRelease};
use crate::runtime::{EntryPoint, RuntimeId, Value};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

impl Artifact {
    pub(super) fn new(runtime: SafeRuntime, index: usize, entry_points: Arc<[EntryPoint]>) -> Self {
        Self {
            inner: Arc::new(RemoteArtifact {
                runtime,
                index,
                entry_points,
            }),
        }
    }

    pub(crate) fn belongs_to(&self, runtime: RuntimeId) -> bool {
        self.inner.runtime.runtime_id == runtime
    }

    pub(crate) fn index(&self) -> usize {
        self.inner.index
    }

    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.inner.entry_points
    }

    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], SafeExecError> {
        let values = self.exec_values(name, args.into())?;
        let actual = values.len();
        values.try_into().map_err(|_| {
            SafeExecError::InvalidRequest(format!(
                "caller expected {N} outputs, but the entry point returned {actual}"
            ))
        })
    }

    pub fn exec_values<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, SafeExecError> {
        self.inner.runtime.exec_values(self, name, args)
    }
}

impl PartialEq for Artifact {
    fn eq(&self, other: &Self) -> bool {
        self.inner.runtime.runtime_id == other.inner.runtime.runtime_id
            && self.index() == other.index()
    }
}
impl Eq for Artifact {}
impl Hash for Artifact {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.runtime.runtime_id.hash(state);
        self.index().hash(state);
    }
}
impl Drop for RemoteArtifact {
    fn drop(&mut self) {
        // Release failures invalidate the worker, as with artifact teardown.
        let _ = self
            .runtime
            .release_object(DeferredRelease::Artifact(self.index));
    }
}
