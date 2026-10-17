use thiserror::Error;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use libloading::Library;
use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
use serde::{Deserialize, Serialize};

use super::Backend;
use super::artifact::{ArtifactError, SharedObject};
use super::executor::{AbiValue, Executor, ExecutorError};
use super::mem::{MemError, MemOwn};
use super::{
    GeneratedFunction, GpuDialect, RuntimeModule,
    signature::{
        EntryPoint, FunctionSignature, SignatureTable, entry_points, generated_signatures,
    },
    value::{Value, ValueKind},
};
use crate::gpu::GpuApi;

/// A process-local GPU context for compiling artifacts and allocating memory.
#[derive(Debug)]
pub struct Runtime {
    /// GPU operations used to validate and release memory crossing the ABI.
    gpu: Arc<GpuApi>,
}

/// A compiled Catena program executable in this process.
///
/// It owns its execution state and remains usable after its [`Runtime`] is
/// dropped. Native code remains mapped until process exit.
#[derive(Debug)]
pub struct Artifact {
    /// GPU operations used to validate and release memory crossing the ABI.
    gpu: Arc<GpuApi>,
    entry_points: Vec<EntryPoint>,
    _shared_object: SharedObject,
    executor: Executor,
    /// Function signatures (runtime Rust ↔ C typechecking)
    signatures: SignatureTable,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("failed to write generated GPU source to {path}: {source}")]
    WriteGeneratedSource {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create generated GPU build directory {path}: {source}")]
    CreateBuildDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("failed to load compiled shared object {path}: {source}")]
    LoadLibrary {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
    #[error("Function '{name}' has unsupported cap.ref output at index {index}")]
    UnsupportedRefOutput { name: String, index: usize },
    #[error("runtime module targets {module:?}, but this runtime uses {runtime:?}")]
    IncompatibleModuleDialect {
        runtime: GpuDialect,
        module: GpuDialect,
    },
    #[error(transparent)]
    Mem(#[from] MemError),
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExecError {
    #[error("Unknown source function '{0}'")]
    UnknownSourceFunction(String),
    #[error("Argument {index} expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        index: usize,
        expected: ValueKind,
        actual: ValueKind,
    },
    #[error("Function '{name}' expected {expected} inputs, got {actual}")]
    InputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Function '{name}' expected {expected} outputs, got {actual}")]
    OutputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Argument {index} contains device memory from a different GPU backend")]
    IncompatibleDeviceMemory { index: usize },
    #[error("Argument {index} violates its dependent input type")]
    InvalidArgumentRefinement { index: usize },
    #[error("artifact does not provide generated-device-allocation controls")]
    AllocationBudgetUnavailable,
    #[error("argument check returned {result}, which names no argument")]
    InvalidArgumentCheck { result: i64 },
    #[error(
        "an artifact cannot be called again while one of its calls is in progress on this thread"
    )]
    ReentrantCall,
    #[error("GPU execution failed while synchronizing: {0}")]
    GpuSynchronization(String),
}

impl Runtime {
    /// Select a provider-local native runtime. Auto probes CUDA before HIP
    /// and reports both failures.
    pub fn with_backend(backend: Backend) -> Result<Self, InitError> {
        Ok(Self {
            gpu: GpuApi::load(backend)?,
        })
    }

    pub fn backend(&self) -> Backend {
        self.gpu.backend()
    }
    pub(crate) fn gpu_api(&self) -> Arc<GpuApi> {
        self.gpu.clone()
    }

    /// Construct an empty runtime for the selected GPU backend.
    pub fn new(backend: impl Into<Backend>) -> Result<Runtime, InitError> {
        Self::with_backend(backend.into())
    }

    /// The GPU dialect used by this runtime.
    pub fn dialect(&self) -> GpuDialect {
        self.gpu.dialect()
    }

    /// Compile a rendered runtime module and load its public entry points.
    pub fn load(&self, module: RuntimeModule) -> Result<Artifact, InitError> {
        let dialect = self.gpu.dialect();
        if module.dialect != dialect {
            return Err(InitError::IncompatibleModuleDialect {
                runtime: dialect,
                module: module.dialect,
            });
        }
        self.load_generated(&module.source, generated_signatures(module.functions))
    }

    /// Compatibility wrapper for callers that have separate source and ABI metadata.
    #[deprecated(note = "construct a RuntimeModule and call Runtime::load")]
    pub fn load_generated_source(
        &self,
        source: &str,
        functions: impl IntoIterator<Item = GeneratedFunction>,
    ) -> Result<Artifact, InitError> {
        self.load(RuntimeModule::new(self.gpu.dialect(), source, functions))
    }

    fn load_generated(
        &self,
        source: &str,
        signature_table: SignatureTable,
    ) -> Result<Artifact, InitError> {
        if let Some((name, index)) = ref_output(&signature_table) {
            return Err(InitError::UnsupportedRefOutput { name, index });
        }

        let dialect = self.gpu.dialect();

        let report_dir = tempfile::Builder::new()
            .prefix("catena-report-")
            .tempdir()
            .map_err(|source| InitError::CreateBuildDir {
                path: std::env::temp_dir(),
                source,
            })?;
        let cpp_path = report_dir.path().join("module.cpp");
        fs::write(&cpp_path, source).map_err(|source| InitError::WriteGeneratedSource {
            path: cpp_path.clone(),
            source,
        })?;
        let shared_object = super::artifact::compile(&cpp_path, dialect)?;

        let library = load_generated_library(shared_object.path())?;
        let executor = Executor::new(library, &signature_table).map_err(|error| match error {
            ExecutorError::LoadSymbol { symbol, source } => {
                InitError::LoadSymbol { symbol, source }
            }
        })?;
        Ok(Artifact {
            gpu: self.gpu.clone(),
            entry_points: entry_points(&signature_table),
            _shared_object: shared_object,
            executor,
            signatures: signature_table,
        })
    }

    /// Copy bytes into owned device memory without interpreting their contents.
    /// The synchronous upload completes before this method returns.
    pub fn mem_bytes(&self, bytes: &[u8]) -> Result<MemOwn, MemError> {
        let data = self.gpu.allocate(bytes.len())?;
        // SAFETY: this is the unique allocation returned by this GPU API.
        // Establish its owner before copying so a failed upload also frees it.
        let memory =
            unsafe { MemOwn::from_raw_parts_with_gpu(data, bytes.len() as u64, self.gpu.clone()) };
        self.gpu.copy_host_to_device(data, bytes)?;
        Ok(memory)
    }

    /// Copy `u64` values into device memory for this context.
    pub fn mem_u64(&self, values: &[u64]) -> Result<MemOwn, MemError> {
        MemOwn::from_u64_slice(values, self.gpu.backend())
    }

    /// Copy `u16` values into device memory for this context.
    pub fn mem_u16(&self, values: &[u16]) -> Result<MemOwn, MemError> {
        MemOwn::from_u16_slice(values, self.gpu.backend())
    }

    /// Copy `f32` values into device memory for this context.
    pub fn mem_f32(&self, values: &[f32]) -> Result<MemOwn, MemError> {
        MemOwn::from_f32_slice(values, self.gpu.backend())
    }

    /// Allocate an application-owned F32 device buffer initialized to zero.
    pub fn mem_f32_zeroed(&self, element_count: usize) -> Result<MemOwn, MemError> {
        let element_size = std::mem::size_of::<f32>();
        let byte_len =
            element_count
                .checked_mul(element_size)
                .ok_or(MemError::AllocationSizeOverflow {
                    element_count,
                    element_size,
                })?;
        self.mem_zeroed_bytes(byte_len as u64)
    }

    /// Allocate an application-owned BF16 device buffer initialized to zero.
    pub fn mem_bf16_zeroed(&self, element_count: usize) -> Result<MemOwn, MemError> {
        let element_size = std::mem::size_of::<u16>();
        let bytes =
            element_count
                .checked_mul(element_size)
                .ok_or(MemError::AllocationSizeOverflow {
                    element_count,
                    element_size,
                })?;
        self.mem_zeroed_bytes(bytes as u64)
    }

    pub(crate) fn mem_zeroed_bytes(&self, byte_len: u64) -> Result<MemOwn, MemError> {
        let byte_len =
            usize::try_from(byte_len).map_err(|_| MemError::LengthTooLarge { byte_len })?;
        let data = self.gpu.allocate(byte_len)?;
        // SAFETY: `data` is the unique allocation returned immediately above
        // by this same GPU API, and ownership is transferred into `MemOwn`.
        let memory =
            unsafe { MemOwn::from_raw_parts_with_gpu(data, byte_len as u64, self.gpu.clone()) };
        self.gpu.zero(data, byte_len)?;
        Ok(memory)
    }
}

impl Artifact {
    /// Source-level entry signatures for this artifact, sorted by name.
    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.entry_points
    }

    /// Run a source-level `program` definition.
    pub fn exec<'a, const M: usize, const N: usize>(
        &self,
        name: &str,
        args: [Value<'a>; M],
    ) -> Result<[Value<'static>; N], ExecError> {
        let signature = self
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: name.to_string(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }

        self.exec_symbol(name, signature, args.into())
            .map(|values| values.try_into().expect("output arity already validated"))
    }

    /// Run a source-level `program` with dynamically sized
    /// input and output collections.
    ///
    /// This is the public execution boundary for adapters such as SafeRuntime,
    /// whose arities are known from a runtime protocol rather than const
    /// generics.
    pub fn exec_values<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        let signature = self
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        self.exec_symbol(name, signature, args)
    }

    /// Run a source-level program while conservatively bounding every device
    /// allocation made by generated code during this invocation. Allocations
    /// are charged cumulatively; generated frees do not refund the limit.
    /// Caller-created input memory is not charged by this lower-level boundary.
    /// A native budget or allocation-size violation traps the executing process,
    /// like a failed Catena assertion; containment requires a worker process.
    pub fn exec_values_with_generated_allocation_budget<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
        byte_limit: u64,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        let signature = self
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        self.exec_symbol_with_generated_allocation_budget(name, signature, args, Some(byte_limit))
    }

    /// Run a source-level program while observing published output prefixes.
    ///
    /// GPU pointers stay inside the synchronous generated call; `notify` receives
    /// bounded host copies of each newly published range and returns `false` to
    /// cancel. Cancellation and invalid publications trap the executing process,
    /// like a failed Catena assertion; containment requires a worker process.
    /// `notify` must not call this artifact again; such calls return
    /// [`ExecError::ReentrantCall`].
    /// Allocations made by generated code are bounded as in
    /// [`Self::exec_values_with_generated_allocation_budget`].
    pub fn exec_values_with_publication<'a>(
        &self,
        name: &str,
        args: Vec<Value<'a>>,
        allocation_limit: u64,
        maximum_output_bytes: u64,
        notify: &mut dyn FnMut(super::Publication) -> bool,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        let signature = self
            .signatures
            .get(name)
            .ok_or_else(|| ExecError::UnknownSourceFunction(name.to_string()))?;
        let mut publisher = super::publication::Publisher {
            gpu: &self.gpu,
            outputs: &signature.outputs,
            lengths: vec![0; signature.outputs.len()],
            maximum_bytes: maximum_output_bytes,
            total_bytes: 0,
            notify,
        };
        let executor = &self.executor;
        let _active = executor.enter().ok_or(ExecError::ReentrantCall)?;
        if !executor.supports_allocation_budget() {
            return Err(ExecError::AllocationBudgetUnavailable);
        }
        if !executor.supports_publication() {
            // A module without the publication service cannot publish.
            return self.exec_entered(name, signature, args, Some(allocation_limit));
        }
        // Publisher and callback remain live for this synchronous call. The
        // executor resets the generated-library callback before returning.
        unsafe {
            executor.with_publication(
                (&mut publisher as *mut super::publication::Publisher<'_>).cast(),
                super::publication::callback,
                || self.exec_entered(name, signature, args, Some(allocation_limit)),
            )
        }
    }

    fn exec_symbol<'a>(
        &self,
        name: &str,
        signature: &FunctionSignature,
        args: Vec<Value<'a>>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        self.exec_symbol_with_generated_allocation_budget(name, signature, args, None)
    }

    fn exec_symbol_with_generated_allocation_budget<'a>(
        &self,
        name: &str,
        signature: &FunctionSignature,
        args: Vec<Value<'a>>,
        allocation_budget: Option<u64>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        let _active = self.executor.enter().ok_or(ExecError::ReentrantCall)?;
        self.exec_entered(name, signature, args, allocation_budget)
    }

    /// Execute while this thread holds the artifact's call guard.
    fn exec_entered<'a>(
        &self,
        name: &str,
        signature: &FunctionSignature,
        args: Vec<Value<'a>>,
        allocation_budget: Option<u64>,
    ) -> Result<Vec<Value<'static>>, ExecError> {
        // Reject unsupported limits while the caller's owned inputs are still
        // Rust values, before transferring them across the generated-code ABI.
        let supports_budget = self.executor.supports_allocation_budget();
        if allocation_budget.is_some() && !supports_budget {
            return Err(ExecError::AllocationBudgetUnavailable);
        }
        // Check input arity lines up with what's in the function signature.
        if signature.inputs.len() != args.len() {
            return Err(ExecError::InputArityMismatch {
                name: name.to_string(),
                expected: signature.inputs.len(),
                actual: args.len(),
            });
        }
        for (index, (value, expected)) in args
            .iter()
            .zip(signature.inputs.iter().copied())
            .enumerate()
        {
            if value.kind() != expected {
                return Err(ExecError::TypeMismatch {
                    index,
                    expected,
                    actual: value.kind(),
                });
            }
            if matches!(value, Value::Bool(value) if *value > 1) {
                return Err(ExecError::InvalidArgumentRefinement { index });
            }
            let memory_backend = match value {
                Value::MemOwn(memory) => Some(memory.backend()),
                Value::MemRef(memory) => Some(memory.backend()),
                _ => None,
            };
            if let Some(backend) = memory_backend
                && self.gpu.backend() != backend
            {
                return Err(ExecError::IncompatibleDeviceMemory { index });
            }
        }

        // The module's own argument check only reads the inputs, so it runs
        // while the caller still owns them.
        let borrowed = args.iter().map(abi_view).collect::<Vec<_>>();
        match self
            .executor
            .rejected_argument(&signature.symbol, &borrowed)
        {
            Ok(None) => {}
            Ok(Some(index)) => return Err(ExecError::InvalidArgumentRefinement { index }),
            Err(result) => return Err(ExecError::InvalidArgumentCheck { result }),
        }

        let executor = &self.executor;

        let mut raw_outputs = signature
            .outputs
            .iter()
            .copied()
            .map(AbiValue::zeroed)
            .collect::<Vec<_>>();
        let raw_inputs = args
            .into_iter()
            .map(|value| match value {
                Value::Bool(value) => AbiValue::Bool(value),
                Value::U16(value) => AbiValue::U16(value),
                Value::U32(value) => AbiValue::U32(value),
                Value::U64(value) => AbiValue::U64(value),
                Value::F32(value) => AbiValue::F32(value),
                Value::MemOwn(memory) => AbiValue::Mem(memory.into_abi()),
                Value::MemRef(memory) => AbiValue::Mem(memory.abi),
            })
            .collect::<Vec<_>>();

        match allocation_budget {
            Some(byte_limit) => executor.call_with_generated_allocation_budget(
                &signature.symbol,
                &raw_inputs,
                &mut raw_outputs,
                byte_limit,
            ),
            None => executor.call(&signature.symbol, &raw_inputs, &mut raw_outputs),
        }

        // Re-establish Rust ownership before the synchronization boundary. If
        // synchronization reports a device fault, dropping `outputs` still
        // attempts to release every allocation returned by generated code.
        let outputs = raw_outputs
            .into_iter()
            .map(|output| self.resolve_output(output))
            .collect::<Result<Vec<_>, _>>()?;
        self.gpu
            .synchronize()
            .map_err(|error| ExecError::GpuSynchronization(error.to_string()))?;
        Ok(outputs)
    }

    fn resolve_output(&self, output: AbiValue) -> Result<Value<'static>, ExecError> {
        match output {
            AbiValue::Bool(value) => Ok(Value::Bool(value)),
            AbiValue::U16(value) => Ok(Value::U16(value)),
            AbiValue::U32(value) => Ok(Value::U32(value)),
            AbiValue::U64(value) => Ok(Value::U64(value)),
            AbiValue::F32(value) => Ok(Value::F32(value)),
            AbiValue::Mem(abi) => {
                // SAFETY: cap.ref outputs are rejected at initialization, so
                // every memory output transfers a GPU allocation owned by the
                // generated program to its caller.
                let memory =
                    unsafe { MemOwn::from_raw_parts_with_gpu(abi.data, abi.len, self.gpu.clone()) };
                Ok(Value::from(memory))
            }
        }
    }
}

/// A non-owning view of an argument for checks that only read it.
fn abi_view(value: &Value<'_>) -> AbiValue {
    match value {
        Value::Bool(value) => AbiValue::Bool(*value),
        Value::U16(value) => AbiValue::U16(*value),
        Value::U32(value) => AbiValue::U32(*value),
        Value::U64(value) => AbiValue::U64(*value),
        Value::F32(value) => AbiValue::F32(*value),
        Value::MemOwn(memory) => AbiValue::Mem(memory.abi),
        Value::MemRef(memory) => AbiValue::Mem(memory.abi),
    }
}

fn ref_output(signatures: &SignatureTable) -> Option<(String, usize)> {
    signatures.iter().find_map(|(name, signature)| {
        signature
            .outputs
            .iter()
            .position(|kind| *kind == ValueKind::MemRef)
            .map(|index| (name.clone(), index))
    })
}

fn load_generated_library(path: &Path) -> Result<Library, InitError> {
    // Generated GPU shared objects must remain resident for the process lifetime.
    // If one is unloaded and a generated GPU object is loaded again later, ROCm/LLVM
    // initialization can re-register process-global LLVM command-line options and
    // abort with "Option 'ubsan-guard-checks' registered more than once".
    // RTLD_NODELETE lets the Rust handle be dropped while preventing that unload.
    #[cfg(not(target_os = "macos"))]
    let flags = RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NODELETE;
    #[cfg(target_os = "macos")]
    let flags = RTLD_LAZY | RTLD_LOCAL;
    let library = unsafe { UnixLibrary::open(Some(path), flags) }.map_err(|source| {
        InitError::LoadLibrary {
            path: path.to_path_buf(),
            source,
        }
    })?;
    Ok(library.into())
}
