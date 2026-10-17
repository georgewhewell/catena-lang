# Embedding generic compute

`catena_lang::safe_runtime::{Session, AssetOwner, compute}` is the boundary for
a host that executes Catena programs without interpreting what they compute.
The host supplies source, verified immutable assets and typed inputs. Catena
prepares the program, reports its callable signatures and executes bounded calls
in a worker.

The interface has no application schema. Byte buffers carry the host's
numerical representation unchanged; encoding inputs, interpreting outputs,
authenticating content and accounting for work belong to the host.

## Ownership and lifecycle

1. Call `run_safe_runtime_child_if_requested()` at executable startup, before starting
   threads or writing to stdout. An explicitly selected worker executable must
   use the same Catena version and implement this entry point.
2. Create an `AssetOwner` with a host-local `Backend` and `SafeRuntimeTimeouts`.
   The parent does not initialize HIP or CUDA. Native GPU access stays in workers.
3. Attach read-only file descriptors using `attach` or `attach_range`. The host
   verifies the content key and supplies immutable files; Catena treats that key
   as opaque and checks descriptors, ranges and capacity. It does not accept
   client-selected filesystem paths. Execution workers import read-only mappings.
4. Start a `Session::with_assets` and `prepare` the program's complete sources
   with a `CodegenKind`. Selecting a code generator does not select a standard
   library, so the sources include whichever bundle the program uses.
   Compilation runs in the worker under the compile deadline, so untrusted
   source never runs the compiler in the host process. Preparation reuses the
   upstream `ModuleCache` inside that same worker and deadline; its keys include
   source texts, compiler build and dialect. `Program::entry_points`
   exposes sorted names, argument kinds and result kinds for admission. A
   `Program` keeps its worker alive; it cannot be used in another session.
5. Execute `compute` or `compute_streaming`. A failed execution worker can be
   discarded and replaced while the separate asset owner retains its mappings.

Programs use the default code generator and standard library. The
implementation supports HIP and CUDA.

Hosts that compile ahead of time can instead pass the `RuntimeModule` returned
by `compile::compile` to `Session::load`; the worker then only builds and loads
it. A loaded module is trusted as supplied, including its entry-point metadata.

## One call

`compute::Request` names an entry point, an ordered list of `Input` values,
`max_output_bytes` and an `allocation_budget`. Inputs are:

| Input | Meaning |
|---|---|
| `Bool`, `U16`, `U32`, `U64`, `F32` | Scalar values; floating-point transport preserves bits |
| `Buffer` | Uploaded bytes borrowed read-only during the call |
| `OwnedBuffer` | Uploaded bytes whose ownership passes into the program |
| `Zeroed` | A worker-allocated, zero-initialized owned byte buffer |
| `Asset` | A checked read-only view into the session owner's immutable asset |

The transport also supports `U16` for supplied native modules; the current
language backend exports `Bool`, `U32`, `U64` and `F32` scalar inputs.

Results are ordered scalars and owned byte buffers. Borrowed output pointers are
not exposed. No device pointer crosses the generic host API. The host supplies
buffer names, shapes and formats and validates application-specific constraints.

Both host and worker validate input kinds and resource bounds. The code
generator decides which definitions are foreign entry points and how their
arguments are validated; it does not export definitions whose
erased proof requirements a foreign caller cannot supply. When a module exports
an argument check for an entry point (see `runtime::module_abi`), the runtime
runs it before transferring any input and reports `InvalidArgumentRefinement`.
These checks do not turn the worker into a sandbox for arbitrary native code or
provide complete device-driver isolation.

The allocation budget counts uploaded private buffers, zeroed buffers and
cumulative generated allocations. Freeing an allocation does not refund it.
Generated allocations are metered through the runtime service.
Immutable assets have separate capacity accounting. Output bounds are checked
before downloading returned buffers. Limits describe bytes and time, not an
application-specific unit of work.

## Streaming and failure

`mem.publish`, provided by the default code generator, observes a prefix of an
owned final output while retaining program ownership. A module that does not
export the publication service simply runs without publications. The host receives `Publication { output, offset, bytes }` deltas of at
most 64 KiB. Prefix lengths must grow monotonically and the total must fit the
request's output bound. Completion verifies published bytes against the final
output. Treat publications as provisional until the call succeeds; cancellation,
panic, a changed prefix or a device fault can prevent completion.

Return `Control::Cancel` from the callback to stop a call. Compile and execution
deadlines retire unresponsive workers. An invocation holds an exclusive session
lease through publication callbacks; a callback must not re-enter that same
session. Callbacks are synchronous host code and cannot be forcibly preempted by
the worker watchdog. Queue wait occurs before the operation deadline starts.
Use `invalidates_session()` on errors to decide whether to replace a session.

## The host side

The host owns content identities, program packaging, application admission,
stage placement, request authentication, durable retry/replay and result commitments.
Catena supplies local program preparation and execution. A returned byte
artifact can become a later stage's input on any chosen worker; scheduling and
authenticated transfer are above this boundary.

Calling Catena again executes again; this API provides no distributed
exactly-once guarantee, so retries belong to the host's own work lifecycle.
Successful publication is likewise not a delivery receipt. Backend choice,
executable paths and process policy belong to the host and are independent of
portable program identity.

The `safe_runtime` module documentation contains a complete minimal host example.
`tests/safe_runtime_compute.rs` and `tests/safe_runtime_assets.rs` exercise
scalar and buffer transport, prepared stage composition, streaming,
cancellation, limits, shared immutable assets, worker replacement and the
existing raw runtime API.
