//! Worker sessions and immutable assets for embedding hosts.

use std::{fs::File, sync::Arc, time::Duration};

use super::{Artifact, AssetError, SafeInitError, SafeRuntime, SafeRuntimeTimeouts};
use crate::{
    codegen::CodegenKind,
    runtime::{Backend, EntryPoint, RuntimeId, RuntimeModule},
};

/// A compiled program retaining the execution worker that created it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Program {
    pub(super) artifact: Artifact,
}

impl Program {
    pub fn entry_points(&self) -> &[EntryPoint] {
        self.artifact.entry_points()
    }
}

/// Opaque identity for one read-only asset resident in an asset owner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asset {
    pub(super) owner: RuntimeId,
    pub(super) resident: u64,
    pub(super) byte_len: u64,
}

impl Asset {
    /// Length of the uploaded range of the verified file.
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A bounds-checked view into an owner-resident [`Asset`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssetSlice {
    pub(super) asset: Asset,
    pub(super) offset: u64,
    pub(super) byte_len: u64,
}

impl AssetSlice {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Usage of an owner's immutable asset allocations at the time of the query.
/// GPU context memory and private execution state are excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetUsage {
    /// Number of distinct owned allocations; shared imports are not counted.
    pub allocation_count: u64,
    /// Successfully uploaded source payload bytes, excluding allocation padding.
    /// Reattaching a cached content key does not add uploaded bytes.
    pub uploaded_bytes: u64,
    /// Owned device bytes, including padding to GPU allocation granularity.
    pub device_bytes: u64,
}

/// A process-isolated owner of immutable GPU assets shared by execution sessions.
///
/// Keeping this owner alive lets a replacement [`Session`] reuse uploaded assets
/// after an execution worker is recycled or fails. Programs and mutable invocation
/// state still belong to their individual execution session.
#[derive(Debug, Clone)]
pub struct AssetOwner {
    pub(super) runtime: Arc<SafeRuntime>,
}

impl AssetOwner {
    /// Start a host-selected worker executable. It must implement
    /// `run_safe_runtime_child_if_requested`; execution sessions inherit the same path.
    /// This supports embedding in hosts that cannot dispatch worker mode from
    /// their own main function, without process-global environment settings.
    pub fn with_executable(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
        executable: &std::path::Path,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_asset_owner_executable(backend, timeouts, executable).map(|runtime| {
            Self {
                runtime: Arc::new(runtime),
            }
        })
    }

    pub fn with_backend(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_asset_owner(backend, timeouts).map(|runtime| Self {
            runtime: Arc::new(runtime),
        })
    }

    pub fn new(backend: impl Into<Backend>) -> Result<Self, SafeInitError> {
        Self::with_backend(backend.into(), SafeRuntimeTimeouts::default())
    }

    pub fn backend(&self) -> Backend {
        self.runtime.backend()
    }

    /// Whether the owner's worker is still available for asset operations.
    /// This is a health snapshot; a later operation can still fail.
    pub fn is_available(&self) -> bool {
        self.runtime.asset_owner_is_available()
    }

    /// Query owned asset allocation usage without counting shared imports,
    /// GPU context memory, or private execution state.
    pub fn usage(&self) -> Result<AssetUsage, AssetError> {
        let (allocation_count, uploaded_bytes, device_bytes) = self.runtime.asset_usage()?;
        Ok(AssetUsage {
            allocation_count,
            uploaded_bytes,
            device_bytes,
        })
    }

    /// Upload a verified immutable file once for all sessions using this owner.
    /// The caller is responsible for verifying the content identity in `key`.
    pub fn attach(&self, key: [u8; 32], file: File) -> Result<Asset, AssetError> {
        let attached = self.runtime.attach_asset(key, file)?;
        Ok(Asset {
            owner: self.runtime.asset_owner_id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }

    /// Upload only a range of a verified immutable file. The content key names
    /// the complete file; the owner deduplicates by key, offset and length.
    /// The caller verifies the file identity before supplying its descriptor.
    /// Explicit worker hosts must support the same Catena worker protocol.
    pub fn attach_range(
        &self,
        key: [u8; 32],
        file: File,
        offset: u64,
        byte_len: u64,
    ) -> Result<Asset, AssetError> {
        let attached = self
            .runtime
            .attach_asset_range(key, file, offset, byte_len)?;
        Ok(Asset {
            owner: self.runtime.asset_owner_id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }
}

/// One process-isolated GPU worker session.
#[derive(Debug)]
pub struct Session {
    pub(super) runtime: Arc<SafeRuntime>,
}

impl Session {
    fn from_runtime(runtime: SafeRuntime) -> Self {
        Self {
            runtime: Arc::new(runtime),
        }
    }

    /// Start an execution worker sharing immutable assets with this owner.
    pub fn with_assets(
        owner: &AssetOwner,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_shared_assets(Arc::clone(&owner.runtime), timeouts)
            .map(Self::from_runtime)
    }

    /// Select a backend inside the isolated worker. Auto tries CUDA then HIP;
    /// native runtime libraries are never probed in the parent process.
    pub fn with_backend(
        backend: Backend,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_backend(backend, timeouts).map(Self::from_runtime)
    }

    /// The backend selected for this session's device, programs, and assets.
    pub fn backend(&self) -> Backend {
        self.runtime.backend()
    }

    /// Start a worker for the selected provider-local GPU backend.
    pub fn new(backend: impl Into<Backend>) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(backend, SafeRuntimeTimeouts::default())
            .map(Self::from_runtime)
    }

    /// Start a worker with a provider-selected deadline for complete Catena
    /// source loading and GPU compilation.
    pub fn with_compile_timeout(
        backend: impl Into<Backend>,
        compile_timeout: Duration,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(
            backend,
            SafeRuntimeTimeouts::default().with_compile_timeout(compile_timeout),
        )
        .map(Self::from_runtime)
    }

    /// Start a worker with provider-selected compile and execution deadlines.
    ///
    /// The execution deadline covers each control operation and the worker-side
    /// protocol for a complete compute invocation, including all publications.
    /// Operations are serialized within a session: every caller may first wait for
    /// the in-flight operation holding the session, and its deadline starts only
    /// after it acquires the worker. A synchronous publication callback runs on the
    /// calling thread; the watchdog remains armed but cannot preempt that code.
    pub fn with_timeouts(
        backend: impl Into<Backend>,
        timeouts: SafeRuntimeTimeouts,
    ) -> Result<Self, SafeInitError> {
        SafeRuntime::with_resident_timeouts(backend, timeouts).map(Self::from_runtime)
    }

    /// Compile `sources` inside the worker with the selected code generator.
    ///
    /// The sources are the complete program, including any standard library
    /// bundle it depends on: selecting a code generator does not select a library.
    pub fn prepare(
        &mut self,
        sources: impl IntoIterator<Item = impl AsRef<str>>,
        codegen: CodegenKind,
    ) -> Result<Program, SafeInitError> {
        self.runtime
            .load_sources(sources, codegen)
            .map(|artifact| Program { artifact })
    }

    /// Load a module compiled by [`crate::compile::compile`] for this session's
    /// dialect. Only the native build and load run in the worker.
    pub fn load(&mut self, module: RuntimeModule) -> Result<Program, SafeInitError> {
        self.runtime
            .load(module)
            .map(|artifact| Program { artifact })
    }

    /// Attach an already-open, verified file as a persistent read-only asset.
    ///
    /// `key` is opaque to Catena: the caller is responsible for verifying that
    /// the immutable file contents have this identity. Catena passes only the
    /// descriptor to its asset owner; it never receives a path or copies file
    /// bytes through the host. Sessions using the same owner can reuse the asset.
    pub fn attach(&self, key: [u8; 32], file: File) -> Result<Asset, AssetError> {
        let attached = self.runtime.attach_asset(key, file)?;
        Ok(Asset {
            owner: self.runtime.asset_owner_id(),
            resident: attached.id,
            byte_len: attached.byte_len,
        })
    }

    /// Construct a checked view into an asset belonging to this session's owner.
    pub fn slice(
        &self,
        asset: &Asset,
        offset: u64,
        byte_len: u64,
    ) -> Result<AssetSlice, AssetError> {
        if asset.owner != self.runtime.asset_owner_id() {
            return Err(AssetError::WrongSession);
        }
        let in_bounds = offset
            .checked_add(byte_len)
            .is_some_and(|end| end <= asset.byte_len);
        if !in_bounds {
            return Err(AssetError::InvalidSlice {
                offset,
                byte_len,
                asset_byte_len: asset.byte_len,
            });
        }
        Ok(AssetSlice {
            asset: asset.clone(),
            offset,
            byte_len,
        })
    }
}
