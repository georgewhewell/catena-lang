//! Symbols a generated module exports for the runtime besides its entry points.
//!
//! Every code generator that produces a [`crate::runtime::RuntimeModule`] implements this
//! contract, so execution does not depend on which generator built a module. The
//! runtime resolves these symbols by name when it loads the module. All state
//! they control is per host thread and scoped to one entry-point invocation; the
//! runtime does not re-enter a module while one of its calls is in progress.
//!
//! The services are optional: a module without them still loads, runs without a
//! generated-allocation limit and cannot publish outputs.
//!
//! A generator owns the safety of its entry points. It must not export an entry
//! point whose inputs carry a relation that its code relies on but neither checks
//! itself nor exposes through an argument check.
//!
//! Entry points and checks use the C ABI with inputs in `GeneratedFunction` order:
//! `Bool` as `uint8_t` (0 or 1), `U16`/`U32`/`U64` as the unsigned integer of that
//! width, `F32` as `float`, and `MemOwn`/`MemRef` as `struct { void *data;
//! uint64_t len; }` by value. Entry points then take one pointer per output.

/// `extern "C" void (uint64_t byte_limit)`: start charging device allocations
/// made by generated code against `byte_limit`. Charges are cumulative: freeing
/// a buffer does not refund it. Exceeding the limit traps the process.
pub const ALLOCATION_BUDGET_BEGIN: &str = "catena_generated_allocation_budget_begin";

/// `extern "C" void ()`: stop charging and lift the limit.
pub const ALLOCATION_BUDGET_END: &str = "catena_generated_allocation_budget_end";

/// `extern "C" void (void *context, uint8_t (*callback)(void *context, uint64_t output,
/// const void *device_data, uint64_t bytes))`: install the callback that observes
/// published output prefixes, or clear it with a null callback. A zero return
/// from the callback cancels the invocation by trapping the process.
pub const SET_PUBLICATION_CALLBACK: &str = "catena_set_publication_callback";

/// Name of an entry point's optional argument check.
///
/// `extern "C" int64_t (inputs...)` takes the entry point's inputs and returns the
/// index of the first argument that violates its refinement, or `-1`. The runtime
/// calls it, when present, before transferring any input to the entry point.
pub fn argument_check_symbol(entry_symbol: &str) -> String {
    format!("catena_check_arguments__{entry_symbol}")
}
