//! Run with `cargo run -p catena-lang --example safe_runtime`.
use catena_lang::{
    codegen::CodegenKind,
    runtime::Backend,
    safe_runtime::{
        Session,
        compute::{Control, Input, Output, Request},
        run_safe_runtime_child_if_requested,
    },
    stdlib,
};

const SOURCE: &str = r#"
(def program increment : (u64 val) -> (u64 val)
= ({smolcat.value.named (const.u64.0x0000000000000001 smolcat.value.named) stdlib.numeric.numeric.u64}
   stdlib.numeric.+ :.forget))
(def program publish : {((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val)
= mem.publish)
"#;

fn main() -> anyhow::Result<()> {
    // Each child re-enters this executable. Dispatch before starting host work.
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }
    let backend = std::env::var("CATENA_GPU_DIALECT")
        .unwrap_or_else(|_| "auto".into())
        .parse::<Backend>()
        .map_err(anyhow::Error::msg)?;
    let mut session = Session::new(backend)?;
    let program = session.prepare(stdlib::sources().chain([SOURCE]), CodegenKind::Default)?;
    let result = session.compute(
        &program,
        Request {
            entry_point: "increment",
            inputs: &[Input::U64(41)],
            max_output_bytes: 8,
            allocation_budget: 0,
        },
    )?;
    anyhow::ensure!(result == [Output::U64(42)]);
    println!("increment: 42");

    let bytes = [17_u64, 19, 23]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let mut published = Vec::new();
    let output = session.compute_streaming(
        &program,
        Request {
            entry_point: "publish",
            inputs: &[
                Input::OwnedBuffer(&bytes),
                Input::U64(0),
                Input::U64(bytes.len() as u64),
            ],
            max_output_bytes: bytes.len() as u64,
            allocation_budget: bytes.len() as u64,
        },
        |part| {
            published.extend_from_slice(&part.bytes);
            Control::Continue
        },
    )?;
    anyhow::ensure!(published == bytes && output == [Output::Buffer(bytes)]);
    println!("published and returned 24 bytes from the isolated worker");
    Ok(())
}
