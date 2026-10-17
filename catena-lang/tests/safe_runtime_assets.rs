use std::{fs::File, io::Write};

use catena_lang::{
    runtime::Backend,
    safe_runtime::{
        Asset, AssetError, AssetOwner, Program, SafeInitError, SafeRuntimeTimeouts, Session,
        compute::{Control, Input, Output, Request},
        run_safe_runtime_child_if_requested,
    },
};

/// Prepare a program written against the default standard library.
fn prepare_default(session: &mut Session, source: &str) -> Result<Program, SafeInitError> {
    let mut sources: Vec<&str> = catena_lang::stdlib::sources().collect();
    sources.push(source);
    session.prepare(sources, catena_lang::codegen::CodegenKind::Default)
}

fn main() -> anyhow::Result<()> {
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }

    let functional_only = std::env::args()
        .skip(1)
        .any(|arg| arg == "--functional-only");
    if functional_only {
        eprintln!("functional-only selected: deliberate worker fault coverage is skipped");
    }
    let backend: Backend = std::env::var("CATENA_GPU_DIALECT")
        .unwrap_or_else(|_| "auto".into())
        .parse()
        .map_err(anyhow::Error::msg)?;
    let session = Session::with_backend(backend, SafeRuntimeTimeouts::default())?;
    eprintln!("resident asset test backend: {:?}", session.backend());
    let key = [0x42; 32];
    let asset = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(asset.byte_len() == 4096);
    anyhow::ensure!(asset == asset.clone());

    let slice = session.slice(&asset, 128, 1024)?;
    anyhow::ensure!(slice.offset() == 128);
    anyhow::ensure!(slice.byte_len() == 1024);
    anyhow::ensure!(session.slice(&asset, 4096, 0).is_ok());
    anyhow::ensure!(matches!(
        session.slice(&asset, 4090, 8),
        Err(AssetError::InvalidSlice { .. })
    ));
    anyhow::ensure!(matches!(
        session.slice(&asset, u64::MAX, 2),
        Err(AssetError::InvalidSlice { .. })
    ));

    // A stable opaque handle proves the resident mapping was reused rather
    // than registered again for the same content identity.
    let reused = session.attach(key, unlinked_read_only_file(&[7; 4096])?)?;
    anyhow::ensure!(reused == asset);

    let conflicting = session.attach(key, unlinked_read_only_file(&[7; 8192])?);
    anyhow::ensure!(matches!(conflicting, Err(AssetError::Remote(_))));

    let empty = session.attach([0x43; 32], unlinked_read_only_file(&[])?);
    anyhow::ensure!(matches!(empty, Err(AssetError::InvalidLength { .. })));
    let writable = tempfile::tempfile()?;
    writable.set_len(4096)?;
    anyhow::ensure!(matches!(
        session.attach([0x44; 32], writable),
        Err(AssetError::Remote(_))
    ));

    let other_session = Session::new(session.backend())?;
    anyhow::ensure!(matches!(
        other_session.slice(&asset, 0, 1),
        Err(AssetError::WrongSession)
    ));
    let dialect = session.backend();
    drop(other_session);
    drop(session);
    verify_remote_artifact_lifetime(dialect)?;
    verify_shared_owner(dialect, functional_only)?;
    Ok(())
}

fn unlinked_read_only_file(bytes: &[u8]) -> anyhow::Result<File> {
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    let path = temporary.path().to_owned();
    let file = File::open(&path)?;
    temporary.close()?;
    anyhow::ensure!(!path.exists(), "temporary asset path still exists");
    Ok(file)
}

fn verify_shared_owner(dialect: Backend, functional_only: bool) -> anyhow::Result<()> {
    let owner = AssetOwner::new(dialect)?;
    let mut source = tempfile::NamedTempFile::new()?;
    source.write_all(&1_u64.to_ne_bytes())?;
    source.as_file().set_len(4096)?;
    source.as_file().sync_all()?;
    let source_path = source.path().to_owned();
    let asset = owner.attach([0x61; 32], File::open(&source_path)?)?;
    source.close()?;
    anyhow::ensure!(!source_path.exists(), "source survived first owner upload");

    let initial_usage = owner.usage()?;
    anyhow::ensure!(initial_usage.allocation_count == 1);
    anyhow::ensure!(initial_usage.uploaded_bytes == 4096);
    anyhow::ensure!(initial_usage.device_bytes >= 4096);
    let mut identical = [0_u8; 4096];
    identical[..8].copy_from_slice(&1_u64.to_ne_bytes());
    let reattached = owner.attach([0x61; 32], unlinked_read_only_file(&identical)?)?;
    anyhow::ensure!(reattached == asset);
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "reattach uploaded the asset twice"
    );

    let mut first = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    let mut second = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    anyhow::ensure!(first.slice(&asset, 0, 8)? == second.slice(&asset, 0, 8)?);
    let first_program = prepare_default(&mut first, SHARED_SOURCE)?;
    let second_program = prepare_default(&mut second, SHARED_SOURCE)?;
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "session imports duplicated owner allocations"
    );

    // A publishes a 1 MiB prefix, forcing worker-pipe backpressure while B
    // independently computes [5, 5] in the first callback. A must
    // return its own buffer containing [2, 2]. Program destruction
    // on this thread and another thread must defer until the invocation ends.
    let mut unused_program = Some(prepare_default(
        &mut first,
        "(def program unused-during-compute : (bool val) -> (bool val) = [value])",
    )?);
    let mut cross_thread_program = Some(prepare_default(
        &mut first,
        "(def program unused-cross-thread : (bool val) -> (bool val) = [value])",
    )?);
    let mut callbacks = 0;
    let interleaved = shared_compute(&first, &first_program, &asset, 1, |_| {
        callbacks += 1;
        if callbacks == 1 {
            drop(unused_program.take());
            let program = cross_thread_program.take().unwrap();
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let dropper = std::thread::spawn(move || {
                drop(program);
                sender.send(()).unwrap();
            });
            receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            dropper.join().unwrap();
            assert_eq!(
                shared_compute(&second, &second_program, &asset, 4, |_| {}).unwrap(),
                [5, 5]
            );
        }
    })?;
    anyhow::ensure!(callbacks == 17);
    anyhow::ensure!(unused_program.is_none() && cross_thread_program.is_none());
    anyhow::ensure!(interleaved == [2, 2]);
    anyhow::ensure!(
        owner.usage()? == initial_usage,
        "compute changed asset usage"
    );
    anyhow::ensure!(shared_compute(&second, &second_program, &asset, 1, |_| {})? == [2, 2]);

    if !functional_only {
        let fault = first
            .compute(
                &first_program,
                Request {
                    entry_point: "compute-fault",
                    inputs: &[Input::U64(1)],
                    max_output_bytes: 8,
                    allocation_budget: 4096,
                },
            )
            .unwrap_err();
        anyhow::ensure!(fault.invalidates_session(), "unexpected fault: {fault:?}");
        anyhow::ensure!(matches!(
            first.compute(
                &first_program,
                Request {
                    entry_point: "compute-owned",
                    inputs: &[Input::Zeroed(8)],
                    max_output_bytes: 8,
                    allocation_budget: 8,
                }
            ),
            Err(catena_lang::safe_runtime::ResidentError::Unavailable { .. })
        ));
        anyhow::ensure!(owner.is_available(), "execution fault killed asset owner");
        anyhow::ensure!(shared_compute(&second, &second_program, &asset, 1, |_| {})? == [2, 2]);
        anyhow::ensure!(
            owner.usage()? == initial_usage,
            "fault changed owner allocations"
        );
    }
    drop(first_program);
    drop(first);

    // Replacement uses the same opaque asset with no second file upload.
    let mut replacement = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    anyhow::ensure!(replacement.slice(&asset, 0, 8)? == second.slice(&asset, 0, 8)?);
    let replacement_program = prepare_default(&mut replacement, SHARED_SOURCE)?;
    anyhow::ensure!(
        shared_compute(&replacement, &replacement_program, &asset, 1, |_| {})? == [2, 2]
    );
    anyhow::ensure!(shared_compute(&second, &second_program, &asset, 4, |_| {})? == [5, 5]);
    let final_usage = owner.usage()?;
    anyhow::ensure!(
        final_usage == initial_usage,
        "replacement duplicated owner allocations"
    );
    eprintln!(
        "shared owner after reuse/replacement (fault_tested={}): allocations={}, uploaded_bytes={}, device_bytes={}",
        !functional_only,
        final_usage.allocation_count,
        final_usage.uploaded_bytes,
        final_usage.device_bytes,
    );
    Ok(())
}

const SHARED_SOURCE: &str = concat!(
    include_str!("fixtures/compute.hex"),
    "\n",
    r#"(def program shared-state-test : {((cap.ref mem) val) (u64 val)} -> ((cap.own mem) val)
    = ([weight seed.]
      ({[.weight] const.u64.0x0000000000020001 [.seed]} compute-add-fill [output.])
      ({[.output] const.u64.0x0000000000000000 const.u64.0x0000000000100008} mem.publish)))"#,
);

fn shared_compute(
    session: &Session,
    program: &Program,
    asset: &Asset,
    seed: u64,
    mut published: impl FnMut(&catena_lang::runtime::Publication),
) -> anyhow::Result<Vec<u64>> {
    let weight = session.slice(asset, 0, 8)?;
    let mut prefix = Vec::new();
    let outputs = session.compute_streaming(
        program,
        Request {
            entry_point: "shared-state-test",
            inputs: &[Input::Asset(&weight), Input::U64(seed)],
            max_output_bytes: (1 << 20) + 8,
            allocation_budget: 2 << 20,
        },
        |part| {
            assert_eq!(part.output, 0);
            assert_eq!(part.offset, prefix.len() as u64);
            prefix.extend_from_slice(&part.bytes);
            published(part);
            Control::Continue
        },
    )?;
    let [Output::Buffer(bytes)] = outputs.as_slice() else {
        anyhow::bail!("unexpected result")
    };
    anyhow::ensure!(prefix == *bytes && bytes.len() == (1 << 20) + 8);
    let expected = (seed + 1).to_le_bytes();
    anyhow::ensure!(bytes.chunks_exact(8).all(|word| word == expected));
    Ok(vec![
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[1 << 20..].try_into().unwrap()),
    ])
}

fn verify_remote_artifact_lifetime(dialect: Backend) -> anyhow::Result<()> {
    use catena_lang::{codegen::CodegenKind, runtime::Value, safe_runtime::SafeRuntime, stdlib};

    let runtime = SafeRuntime::new(dialect)?;
    let artifact = runtime.load_sources(
        stdlib::sources().chain([
            "(def program lifetime-add-one : (u64 val) -> (u64 val) = ({smolcat.value.named (const.u64.0x0000000000000001 smolcat.value.named) stdlib.numeric.numeric.u64} stdlib.numeric.+ :.forget))",
        ]),
        CodegenKind::Default,
    )?;
    let sibling = runtime.load_sources(
        stdlib::sources()
            .chain(["(def program lifetime-identity : (u64 val) -> (u64 val) = [value])"]),
        CodegenKind::Default,
    )?;
    drop(runtime);
    drop(sibling);

    anyhow::ensure!(
        artifact
            .entry_points()
            .iter()
            .any(|entry| entry.name() == "lifetime-add-one")
    );
    let [result] = artifact.exec("lifetime-add-one", [41_u64.into()])?;
    anyhow::ensure!(matches!(result, Value::U64(42)));
    let outputs = artifact.exec_values("lifetime-add-one", vec![6_u64.into()])?;
    anyhow::ensure!(matches!(outputs.as_slice(), [Value::U64(7)]));
    Ok(())
}
