//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::{cell::RefCell, collections::HashMap, ffi::c_void};

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

use super::{module_abi, signature::SignatureTable, value::ValueKind};

type AllocationBudgetBegin = unsafe extern "C" fn(u64);
type AllocationBudgetEnd = unsafe extern "C" fn();

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(super) struct CatenaMem {
    pub(super) data: *mut c_void,
    pub(super) len: u64,
}

/// Temporary storage for values crossing the generated C ABI.
///
/// Unlike [`super::value::Value`], this type does not express Rust ownership:
/// an owned input has already relinquished its allocation before becoming an
/// `AbiValue`, while a memory output is not wrapped in an owning Rust value
/// until generated code has filled its [`CatenaMem`] slot. It must therefore
/// remain private to the synchronous executor boundary.
#[derive(Debug)]
pub(super) enum AbiValue {
    Bool(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    Mem(CatenaMem),
}

impl AbiValue {
    pub(super) fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Bool => Self::Bool(0),
            ValueKind::U16 => Self::U16(0),
            ValueKind::U32 => Self::U32(0),
            ValueKind::U64 => Self::U64(0),
            ValueKind::F32 => Self::F32(0.0),
            ValueKind::MemOwn | ValueKind::MemRef => Self::Mem(CatenaMem {
                data: std::ptr::null_mut(),
                len: 0,
            }),
        }
    }
}

#[derive(Debug)]
struct PreparedFunction {
    code: CodePtr,
    cif: Cif,
}

#[derive(Debug, Clone, Copy)]
struct AllocationBudgetControl {
    begin: AllocationBudgetBegin,
    end: AllocationBudgetEnd,
}

impl AllocationBudgetControl {
    fn under_limit<T>(&self, byte_limit: u64, invoke: impl FnOnce() -> T) -> T {
        unsafe { (self.begin)(byte_limit) };
        let _reset = AllocationBudgetReset { control: self };
        invoke()
    }
}

struct AllocationBudgetReset<'a> {
    control: &'a AllocationBudgetControl,
}

impl Drop for AllocationBudgetReset<'_> {
    fn drop(&mut self) {
        unsafe { (self.control.end)() };
    }
}

thread_local! {
    /// Executors with a generated call in progress on this thread.
    static ACTIVE_CALLS: RefCell<Vec<*const Executor>> = const { RefCell::new(Vec::new()) };
}

/// Releases an executor's per-thread call guard.
pub(super) struct ActiveCall(*const Executor);

impl Drop for ActiveCall {
    fn drop(&mut self) {
        ACTIVE_CALLS.with(|active| active.borrow_mut().retain(|executor| *executor != self.0));
    }
}

/// Loaded generated code and its prepared dynamic call interfaces.
#[derive(Debug)]
pub(super) struct Executor {
    // Keep the library loaded for as long as any cached code pointer can be used.
    _library: Library,
    functions: HashMap<String, PreparedFunction>,
    allocation_budget: Option<AllocationBudgetControl>,
}

#[derive(Debug, Error)]
pub(super) enum ExecutorError {
    #[error("failed to resolve generated symbol `{symbol}`: {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
}

impl Executor {
    /// Resolve generated entry points and prepare their libffi call interfaces once.
    ///
    /// The module's service symbols are optional: a module without them still
    /// runs, but cannot meter its allocations.
    pub(super) fn new(
        library: Library,
        signatures: &SignatureTable,
    ) -> Result<Self, ExecutorError> {
        let allocation_budget = match (
            optional_symbol(&library, module_abi::ALLOCATION_BUDGET_BEGIN),
            optional_symbol(&library, module_abi::ALLOCATION_BUDGET_END),
        ) {
            (Some(begin), Some(end)) => Some(AllocationBudgetControl { begin, end }),
            _ => None,
        };
        let mut functions = HashMap::with_capacity(signatures.len());
        for signature in signatures.values() {
            if functions.contains_key(&signature.symbol) {
                continue;
            }

            let symbol_name = format!("{}\0", signature.symbol);
            let function = unsafe { library.get::<*mut c_void>(symbol_name.as_bytes()) }.map_err(
                |source| ExecutorError::LoadSymbol {
                    symbol: signature.symbol.clone(),
                    source,
                },
            )?;

            let argument_types = signature
                .inputs
                .iter()
                .copied()
                .map(ffi_type)
                .chain(signature.outputs.iter().map(|_| Type::pointer()))
                .collect::<Vec<_>>();
            functions.insert(
                signature.symbol.clone(),
                PreparedFunction {
                    code: CodePtr(*function),
                    cif: Cif::new(argument_types, Type::void()),
                },
            );
        }

        Ok(Self {
            _library: library,
            functions,
            allocation_budget,
        })
    }

    /// Invoke a prepared symbol. Runtime validation guarantees the value shapes and kinds.
    pub(super) fn call(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
        match self.allocation_budget {
            Some(control) => {
                control.under_limit(u64::MAX, || self.call_raw(symbol, inputs, outputs))
            }
            None => self.call_raw(symbol, inputs, outputs),
        }
    }

    /// Invoke a prepared symbol with a cumulative generated-device-allocation limit.
    pub(super) fn call_with_generated_allocation_budget(
        &self,
        symbol: &str,
        inputs: &[AbiValue],
        outputs: &mut [AbiValue],
        byte_limit: u64,
    ) {
        self.allocation_budget
            .expect("budget support is checked before input ownership is transferred")
            .under_limit(byte_limit, || self.call_raw(symbol, inputs, outputs));
    }

    pub(super) fn supports_allocation_budget(&self) -> bool {
        self.allocation_budget.is_some()
    }

    /// Mark a generated call in progress on this thread, or return `None` if one
    /// already is. A module keeps its invocation state per thread, so a nested
    /// call (for example from a publication callback) would reset the outer
    /// call's allocation budget and publication callback.
    pub(super) fn enter(&self) -> Option<ActiveCall> {
        let executor = self as *const Self;
        ACTIVE_CALLS.with(|active| {
            let mut active = active.borrow_mut();
            if active.contains(&executor) {
                return None;
            }
            active.push(executor);
            Some(ActiveCall(executor))
        })
    }

    fn call_raw(&self, symbol: &str, inputs: &[AbiValue], outputs: &mut [AbiValue]) {
        let function = self
            .functions
            .get(symbol)
            .expect("source signature should have a prepared generated function");

        // A generated output parameter is itself a pointer value. libffi therefore needs
        // an address containing that pointer while it constructs the native call frame.
        let output_pointers = outputs.iter_mut().map(output_pointer).collect::<Vec<_>>();
        let arguments = inputs
            .iter()
            .map(input_arg)
            .chain(output_pointers.iter().map(Arg::new))
            .collect::<Vec<_>>();

        unsafe { function.cif.call::<()>(function.code, &arguments) };
    }
}

fn optional_symbol<T: Copy>(library: &Library, symbol: &str) -> Option<T> {
    let symbol_name = format!("{symbol}\0");
    unsafe { library.get::<T>(symbol_name.as_bytes()) }
        .ok()
        .map(|loaded| *loaded)
}

fn ffi_type(kind: ValueKind) -> Type {
    match kind {
        ValueKind::Bool => Type::u8(),
        ValueKind::U16 => Type::u16(),
        ValueKind::U32 => Type::u32(),
        ValueKind::U64 => Type::u64(),
        ValueKind::F32 => Type::f32(),
        ValueKind::MemOwn | ValueKind::MemRef => Type::structure([Type::pointer(), Type::u64()]),
    }
}

fn input_arg(value: &AbiValue) -> Arg<'_> {
    match value {
        AbiValue::Bool(value) => Arg::new(value),
        AbiValue::U16(value) => Arg::new(value),
        AbiValue::U32(value) => Arg::new(value),
        AbiValue::U64(value) => Arg::new(value),
        AbiValue::F32(value) => Arg::new(value),
        AbiValue::Mem(value) => Arg::new(value),
    }
}

fn output_pointer(value: &mut AbiValue) -> *mut c_void {
    match value {
        AbiValue::Bool(value) => (value as *mut u8).cast(),
        AbiValue::U16(value) => (value as *mut u16).cast(),
        AbiValue::U32(value) => (value as *mut u32).cast(),
        AbiValue::U64(value) => (value as *mut u64).cast(),
        AbiValue::F32(value) => (value as *mut f32).cast(),
        AbiValue::Mem(value) => (value as *mut CatenaMem).cast(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static ACTIVE_LIMIT: AtomicU64 = AtomicU64::new(u64::MAX);
    static END_COUNT: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" fn begin(byte_limit: u64) {
        ACTIVE_LIMIT.store(byte_limit, Ordering::SeqCst);
    }

    unsafe extern "C" fn end() {
        ACTIVE_LIMIT.store(u64::MAX, Ordering::SeqCst);
        END_COUNT.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn an_executor_cannot_be_entered_twice_on_one_thread() {
        let load = || {
            Executor::new(
                libloading::os::unix::Library::this().into(),
                &SignatureTable::new(),
            )
            .unwrap()
        };
        let executor = load();
        let other = load();
        let outer = executor.enter().expect("first entry");
        assert!(executor.enter().is_none());
        // Another module keeps separate invocation state.
        let nested_other = other.enter().expect("independent executor");
        drop(nested_other);
        drop(outer);
        assert!(executor.enter().is_some());
    }

    #[test]
    fn modules_without_service_symbols_load_without_services() {
        let library = libloading::os::unix::Library::this().into();
        let executor = Executor::new(library, &SignatureTable::new()).unwrap();
        assert!(!executor.supports_allocation_budget());
    }

    #[test]
    fn allocation_budget_is_reset_between_invocations_and_on_unwind() {
        let control = AllocationBudgetControl { begin, end };
        END_COUNT.store(0, Ordering::SeqCst);

        control.under_limit(17, || {
            assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), 17);
        });
        assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), u64::MAX);

        let panic = std::panic::catch_unwind(|| {
            control.under_limit(23, || panic!("fixture panic"));
        });
        assert!(panic.is_err());
        assert_eq!(ACTIVE_LIMIT.load(Ordering::SeqCst), u64::MAX);
        assert_eq!(END_COUNT.load(Ordering::SeqCst), 2);
    }
}
