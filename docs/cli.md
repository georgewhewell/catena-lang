# Compiler CLI

Run the compiler from the repository root:

```sh
cargo run -p catena-lang -- program.hex -o report
```

After `cargo install catena-lang`, use `catena-lang program.hex -o report`.
The CLI compiles input files and writes compiler reports and generated code.
It uses the experimental GPU code generator and its embedded standard library.
It does not execute programs. On compilation failure it writes available partial
reports and exits with an error. Input-loading errors occur before reporting.

Stage progress and elapsed times are printed to stderr. `timings.json` records
completed or failed stages, including partial runs. Report generation is timed
separately. The separate `catena-cli` package provides a `status` command.

## Options

| Argument or option | Behavior |
| --- | --- |
| `<PATHS>...` | Input files or directories; directories load `.hex` files recursively. |
| `-o`, `--output-dir <PATH>` | Required directory for compilation reports. |
| `--no-svg` | Skip SVG graph rendering. |
| `--dialect <NAME>` | Select `hip` (default) or `cuda`. |
| `--codegen <NAME>` | Accepts `experimental`, the default and only code generator. |
| `--stdlib <NAME>` | Select a named embedded bundle; repeat for multiple selections. |
| `--stdlib-dir <PATH>` | Load `.hex` files recursively from a directory; repeat for multiple directories. |
| `--no-stdlib` | Load only input files; conflicts with both stdlib selection options. |
| `-h`, `--help` | Show usage and options. |
| `-V`, `--version` | Show the version. |

The compiler emits one runtime module for the selected dialect. Its source is
written to `gpu/hip.cpp` or `gpu/cuda.cpp`. Unsupported operations and runtime
types are reported as errors.

## Standard library selection

Without selection flags, the CLI loads the embedded `stdlib` bundle from
[`catena-lang/stdlib/experimental/`](../catena-lang/stdlib/experimental/).
Explicit selections replace that implicit choice:

```sh
# Explicitly select the embedded library
cargo run -p catena-lang -- program.hex -o report --stdlib stdlib

# Load a directory of library sources
cargo run -p catena-lang -- program.hex -o report --stdlib-dir ./my-stdlib

# Combine the embedded library and additional sources
cargo run -p catena-lang -- program.hex -o report \
  --stdlib stdlib --stdlib-dir ./my-stdlib

# Supply every definition through input files
cargo run -p catena-lang -- definitions.hex program.hex -o report --no-stdlib
```

Named selections are loaded first in flag order, then directory sources in flag
order. Input files follow library sources. Duplicate definitions are errors.
Directory loading ignores manifests and reads `.hex` files recursively.

## Bundle manifests for library clients

Rust clients can use `stdlib::sources()` to access the embedded library, or
`stdlib::BundleRegistry` to register bundles with `add_directory`.
The experimental library is embedded in the order declared by its
[`stdlib.json`](../catena-lang/stdlib/experimental/stdlib.json) manifest.

A local manifest requires a `name`. An optional `extends` list names dependencies,
which must also be registered or embedded. Dependencies are resolved first and
loaded once. Unknown names, dependency cycles, and duplicate bundle names are
errors. For example:

```json
{"name": "my-stdlib", "extends": ["stdlib"], "files": ["operations.hex"]}
```

An optional `files` array specifies an ordered list of relative source paths,
including subdirectories. Paths cannot contain `..`. An empty list loads no
source files. Without `files`, the registry loads `.hex` files directly in the
bundle directory, sorted by filename. Unknown manifest fields are rejected.
