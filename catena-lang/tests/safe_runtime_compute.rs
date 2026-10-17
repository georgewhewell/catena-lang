use std::{fs::File, io::Write, os::unix::fs::FileExt, time::Duration};

use anyhow::{Result, ensure};
use catena_lang::{
    runtime::Backend,
    safe_runtime::{
        AssetError, AssetOwner, Program, ResidentError, SafeInitError, SafeRuntimeTimeouts,
        Session,
        compute::{Control, Input, Output, Request},
        run_safe_runtime_child_if_requested,
    },
};

const SOURCE: &str = concat!(
    include_str!("fixtures/compute.hex"),
    "\n",
    include_str!("fixtures/publication.hex")
);

/// Prepare a program written against the default standard library.
fn prepare_default(session: &mut Session, source: &str) -> Result<Program, SafeInitError> {
    let mut sources: Vec<&str> = catena_lang::stdlib::sources().collect();
    sources.push(source);
    session.prepare(sources, catena_lang::codegen::CodegenKind::Default)
}

fn main() -> Result<()> {
    if run_safe_runtime_child_if_requested()? {
        return Ok(());
    }
    let backend: Backend = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("CATENA_GPU_DIALECT").ok())
        .unwrap_or_else(|| "auto".into())
        .parse()
        .map_err(anyhow::Error::msg)?;
    let owner = match std::env::args().nth(2) {
        Some(path) => AssetOwner::with_executable(
            backend,
            SafeRuntimeTimeouts::default(),
            std::path::Path::new(&path),
        )?,
        None => AssetOwner::with_backend(backend, SafeRuntimeTimeouts::default())?,
    };
    publication_checks(&owner)?;
    native_allocation_overflow_is_isolated(&owner)?;
    let mut temporary = tempfile::NamedTempFile::new()?;
    temporary.write_all(&41_u64.to_le_bytes())?;
    temporary.as_file_mut().set_len(4096)?;
    temporary.as_file_mut().sync_all()?;
    let asset = owner.attach([0xc1; 32], File::open(temporary.path())?)?;
    temporary.close()?;
    let before_ranges = owner.usage()?;
    let mut ranges = tempfile::NamedTempFile::new()?;
    ranges.write_all(&[0xff; 3])?;
    ranges.write_all(&41_u64.to_le_bytes())?;
    ranges.write_all(&99_u64.to_le_bytes())?;
    let wide_offset = (1 << 33) + 16;
    let file_bytes = wide_offset + 8;
    ranges
        .as_file()
        .write_all_at(&123_u64.to_le_bytes(), wide_offset)?;
    ranges.as_file_mut().set_len(file_bytes)?;
    ranges.as_file_mut().sync_all()?;
    let range_key = [0xc2; 32];
    let first = owner.attach_range(range_key, File::open(ranges.path())?, 3, 8)?;
    let second_range = owner.attach_range(range_key, File::open(ranges.path())?, 11, 8)?;
    let wide_range = owner.attach_range(range_key, File::open(ranges.path())?, wide_offset, 8)?;
    ensure!(first != second_range);
    ensure!(first == owner.attach_range(range_key, File::open(ranges.path())?, 3, 8)?);
    ensure!(matches!(
        owner.attach_range(range_key, File::open(ranges.path())?, u64::MAX, 8),
        Err(AssetError::InvalidSlice { .. })
    ));
    ensure!(matches!(
        owner.attach_range(range_key, File::open(ranges.path())?, file_bytes - 1, 8),
        Err(AssetError::InvalidSlice { .. })
    ));
    ranges.close()?;
    let usage = owner.usage()?;
    ensure!(usage.allocation_count == before_ranges.allocation_count + 3);
    ensure!(
        usage.uploaded_bytes == before_ranges.uploaded_bytes + 24,
        "range attachment uploaded unreferenced file contents"
    );
    let mut session = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    let program = prepare_default(&mut session, SOURCE)?;
    ensure!(call(&session, &program, "compute-empty", &[], 0, 0)? == [Output::Buffer(vec![])]);
    for (asset, expected) in [
        (&first, 42_u64),
        (&second_range, 100_u64),
        (&wide_range, 124_u64),
    ] {
        let range = session.slice(asset, 0, 8)?;
        ensure!(
            call(
                &session,
                &program,
                "compute-add",
                &[Input::Asset(&range), Input::U64(1)],
                8,
                4096
            )? == [Output::Buffer(expected.to_le_bytes().to_vec())]
        );
    }
    let weight = session.slice(&asset, 0, 8)?;
    let inputs = [Input::Asset(&weight), Input::U64(1)];
    let bytes = 10_u64.to_le_bytes();
    // Exercise the transport above 32 MiB while leaving descriptor space in
    // the bounded frame. The transport does not interpret the bytes.
    let large = vec![0xa5; 40 * 1024 * 1024];
    ensure!(
        call(
            &session,
            &program,
            "compute-owned",
            &[Input::OwnedBuffer(&large)],
            large.len() as u64,
            large.len() as u64
        )? == [Output::Buffer(large)]
    );
    ensure!(
        call(
            &session,
            &program,
            "compute-add",
            &[Input::Buffer(&bytes), Input::U64(3)],
            8,
            4096
        )? == [Output::Buffer(13_u64.to_le_bytes().to_vec())]
    );
    // Odd-sized bytes must retain their representation; the transport has no
    // element codec.
    let bits = [0, 255, 1, 2, 3];
    ensure!(
        call(
            &session,
            &program,
            "compute-return",
            &[
                Input::Bool(true),
                Input::U32(u32::MAX),
                Input::U64(u64::MAX),
                Input::F32(-1.25),
                Input::OwnedBuffer(&bits),
            ],
            24,
            4096
        )? == [
            Output::Bool(true),
            Output::U32(u32::MAX),
            Output::U64(u64::MAX),
            Output::F32(-1.25),
            Output::Buffer(bits.to_vec())
        ]
    );
    ensure!(
        call(
            &session,
            &program,
            "compute-owned",
            &[Input::Zeroed(17)],
            17,
            17
        )? == [Output::Buffer(vec![0; 17])]
    );
    ensure!(matches!(
        call(
            &session,
            &program,
            "compute-owned",
            &[Input::Zeroed(17)],
            17,
            16
        ),
        Err(ResidentError::InvalidRequest(_))
    ));
    ensure!(matches!(
        call(
            &session,
            &program,
            "compute-owned",
            &[Input::Buffer(&bits)],
            17,
            4096
        ),
        Err(ResidentError::InvalidRequest(_))
    ));
    ensure!(matches!(
        call(&session, &program, "compute-add", &inputs, 7, 4096),
        Err(ResidentError::Remote(_))
    ));
    ensure!(
        call(&session, &program, "compute-add", &inputs, 8, 4096)?.len() == 1,
        "output limit rejection damaged the session"
    );

    // Preserve generated-allocation overflow and exhausted-envelope isolation
    // using checked element-to-byte conversion in the source fixture.
    for (length, budget, message) in [
        (u64::MAX, 4096, "byte count overflowed"),
        (1, 1, "allocation budget"),
    ] {
        let mut bounded = publication_session(&owner, SafeRuntimeTimeouts::default())?;
        let bounded_program = prepare_default(&mut bounded, SOURCE)?;
        let error = call(
            &bounded,
            &bounded_program,
            "compute-fill",
            &[Input::U64(length), Input::U64(7)],
            8,
            budget,
        )
        .unwrap_err();
        ensure!(
            error.invalidates_session(),
            "{message} kept worker alive: {error:?}"
        );
        ensure!(
            owner.is_available(),
            "generated allocation failure killed asset owner"
        );
    }

    let mut second = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    let second_program = prepare_default(&mut second, SOURCE)?;
    ensure!(matches!(
        call(&second, &program, "compute-add", &inputs, 8, 4096),
        Err(ResidentError::InvalidRequest(_))
    ));
    ensure!(call(&second, &second_program, "compute-add", &inputs, 8, 4096)?.len() == 1);
    // A returned byte artifact can feed another prepared stage in a different
    // worker. Transport and stage placement are owned by the embedding system.
    let stage_one = call(&session, &program, "compute-add", &inputs, 8, 4096)?;
    let [Output::Buffer(intermediate)] = stage_one.as_slice() else {
        anyhow::bail!("stage one did not return a byte artifact");
    };
    ensure!(
        call(
            &second,
            &second_program,
            "compute-add",
            &[Input::Buffer(intermediate), Input::U64(7)],
            8,
            4096,
        )? == [Output::Buffer(49_u64.to_le_bytes().to_vec())]
    );
    let other_owner = AssetOwner::with_backend(backend, SafeRuntimeTimeouts::default())?;
    let mut other = Session::with_assets(&other_owner, SafeRuntimeTimeouts::default())?;
    let other_program = prepare_default(&mut other, SOURCE)?;
    ensure!(matches!(
        call(&other, &other_program, "compute-add", &inputs, 8, 4096),
        Err(ResidentError::InvalidRequest(_))
    ));
    drop(other_program);
    drop(other);
    drop(other_owner);

    let error = call(
        &session,
        &program,
        "compute-fault",
        &[Input::U64(1)],
        8,
        4096,
    )
    .unwrap_err();
    ensure!(
        error.invalidates_session(),
        "assertion did not retire the worker: {error:?}"
    );
    ensure!(
        owner.is_available(),
        "execution failure killed the asset owner"
    );
    ensure!(call(&second, &second_program, "compute-add", &inputs, 8, 4096)?.len() == 1);
    drop(program);
    drop(session);
    let mut replacement = Session::with_assets(&owner, SafeRuntimeTimeouts::default())?;
    let replacement_program = prepare_default(&mut replacement, SOURCE)?;
    ensure!(
        call(
            &replacement,
            &replacement_program,
            "compute-add",
            &inputs,
            8,
            4096
        )? == [Output::Buffer(42_u64.to_le_bytes().to_vec())]
    );
    ensure!(
        owner.usage()? == usage,
        "execution/recovery duplicated immutable assets"
    );
    eprintln!(
        "generic compute passed on {:?}: scalars, buffers, assets, ownership, fault isolation and recovery",
        replacement.backend()
    );
    Ok(())
}

fn call(
    session: &Session,
    program: &Program,
    entry_point: &str,
    inputs: &[Input<'_>],
    max_output_bytes: u64,
    allocation_budget: u64,
) -> Result<Vec<Output>, ResidentError> {
    session.compute(
        program,
        Request {
            entry_point,
            inputs,
            max_output_bytes,
            allocation_budget,
        },
    )
}

fn publication_checks(owner: &AssetOwner) -> Result<()> {
    let mut session = publication_session(owner, SafeRuntimeTimeouts::default())?;
    let program = prepare_default(&mut session, SOURCE)?;
    let a: Vec<_> = (0..16).collect();
    let b = vec![41; 8];
    let mut parts = Vec::new();
    let request = Request {
        entry_point: "publish-pair",
        inputs: &[Input::OwnedBuffer(&a), Input::OwnedBuffer(&b)],
        max_output_bytes: 24,
        allocation_budget: 24,
    };
    let result = session.compute_streaming(&program, request, |part| {
        parts.push(part.clone());
        Control::Continue
    })?;
    ensure!(result == [Output::Buffer(a.clone()), Output::Buffer(b.clone())]);
    for (index, expected) in [(0, &a), (1, &b)] {
        let mut prefix = Vec::new();
        for part in parts.iter().filter(|p| p.output == index) {
            ensure!(part.offset == prefix.len() as u64);
            prefix.extend_from_slice(&part.bytes);
        }
        ensure!(&prefix == expected);
    }
    ensure!(parts.len() == 4);
    let large = vec![7; 200_000];
    let mut lengths = Vec::new();
    ensure!(
        session.compute_streaming(
            &program,
            Request {
                entry_point: "publish-prefix",
                inputs: &[
                    Input::OwnedBuffer(&large),
                    Input::U64(0),
                    Input::U64(large.len() as u64)
                ],
                max_output_bytes: large.len() as u64,
                allocation_budget: large.len() as u64,
            },
            |part| {
                lengths.push(part.bytes.len());
                Control::Continue
            }
        )? == [Output::Buffer(large)]
    );
    ensure!(lengths == [65536, 65536, 65536, 3392]);
    let zero = session.compute_streaming(
        &program,
        Request {
            entry_point: "publish-prefix",
            inputs: &[Input::Zeroed(8), Input::U64(0), Input::U64(0)],
            max_output_bytes: 8,
            allocation_budget: 8,
        },
        |_| panic!("empty publication should not emit a chunk"),
    )?;
    ensure!(zero == [Output::Buffer(vec![0; 8])]);
    let error = call(
        &session,
        &program,
        "publish-changed",
        &[Input::Zeroed(8)],
        8,
        16,
    )
    .unwrap_err();
    ensure!(
        matches!(error, ResidentError::Remote(ref message) if message.contains("published prefix")),
        "unexpected changed-prefix error: {error:?}"
    );
    ensure!(
        call(
            &session,
            &program,
            "compute-owned",
            &[Input::Zeroed(8)],
            8,
            8
        )?
        .len()
            == 1
    );
    drop(program);
    drop(session);
    for case in 0..6 {
        let mut session = publication_session(owner, SafeRuntimeTimeouts::default())?;
        let program = prepare_default(&mut session, SOURCE)?;
        let mut parts = 0;
        let entry = if case == 2 {
            "publish-regression"
        } else {
            "publish-prefix"
        };
        let inputs = if case == 2 {
            vec![Input::Zeroed(8)]
        } else {
            vec![
                Input::Zeroed(8),
                Input::U64(if case == 1 { 1 } else { 0 }),
                Input::U64(if case == 0 { 9 } else { 8 }),
            ]
        };
        let result = session.compute_streaming(
            &program,
            Request {
                entry_point: entry,
                inputs: &inputs,
                max_output_bytes: if case == 5 { 4 } else { 8 },
                allocation_budget: 8,
            },
            |_| {
                parts += 1;
                match case {
                    3 => Control::Cancel,
                    4 => panic!("publication callback fixture"),
                    _ => Control::Continue,
                }
            },
        );
        let error = result.unwrap_err();
        if case == 3 {
            ensure!(matches!(error, ResidentError::Cancelled) && parts == 1);
        }
        if case == 4 {
            ensure!(matches!(error, ResidentError::CallbackPanicked) && parts == 1);
        }
        ensure!(
            error.invalidates_session(),
            "publication failure kept worker: {error:?}"
        );
        ensure!(
            owner.is_available(),
            "publication failure killed immutable assets"
        );
    }
    let mut timed = publication_session(
        owner,
        SafeRuntimeTimeouts::default().with_execution_timeout(Duration::from_millis(100)),
    )?;
    let program = prepare_default(&mut timed, SOURCE)?;
    let error = timed
        .compute_streaming(
            &program,
            Request {
                entry_point: "publish-prefix",
                inputs: &[Input::Zeroed(8), Input::U64(0), Input::U64(8)],
                max_output_bytes: 8,
                allocation_budget: 8,
            },
            |_| {
                std::thread::sleep(Duration::from_millis(250));
                Control::Continue
            },
        )
        .unwrap_err();
    ensure!(
        matches!(error, ResidentError::TimedOut { .. }),
        "publication deadline: {error:?}"
    );
    ensure!(owner.is_available());
    drop(program);
    drop(timed);
    // Prove bounded recovery before returning to the rest of the GPU suite.
    drop(publication_session(owner, SafeRuntimeTimeouts::default())?);
    Ok(())
}

// Forced teardown deliberately prevents replacement until the old process is
// reaped. Retry that specific admission error; never mask another init failure.
fn publication_session(owner: &AssetOwner, timeouts: SafeRuntimeTimeouts) -> Result<Session> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match Session::with_assets(owner, timeouts) {
            Err(catena_lang::safe_runtime::SafeInitError::PreviousWorkerUnreaped { .. })
                if std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(5))
            }
            result => return Ok(result?),
        }
    }
}

// The language allocator now takes a byte count. Independently exercise the
// module service's multiplication guard, using a trusted native-module probe.
fn native_allocation_overflow_is_isolated(owner: &AssetOwner) -> Result<()> {
    use catena_lang::{
        codegen::CodegenKind,
        compile::compile,
        report::CompileReport,
        runtime::{GeneratedFunction, ValueKind},
    };
    let mut session = publication_session(owner, SafeRuntimeTimeouts::default())?;
    let dialect = session.backend().dialect().unwrap();
    let mut report = CompileReport::new(metacat::theory::RawTheorySet::from_text("")?);
    let mut module = compile(&mut report, CodegenKind::Default, dialect)?;
    module.source.push_str(
        r#"
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
extern "C" __host__ void allocation_overflow_probe(catena_mem_own_t* output) {
    *output = {catena_runtime_allocate(UINT64_MAX, 8), 0};
}
#endif
"#,
    );
    module.functions.push(GeneratedFunction {
        source_name: "allocation-overflow-probe".into(),
        symbol: "allocation_overflow_probe".into(),
        inputs: vec![],
        outputs: vec![ValueKind::MemOwn],
    });
    let program = session.load(module)?;
    let error = call(
        &session,
        &program,
        "allocation-overflow-probe",
        &[],
        8,
        4096,
    )
    .unwrap_err();
    ensure!(error.invalidates_session());
    ensure!(
        error
            .to_string()
            .contains("allocation byte count overflowed"),
        "{error}"
    );
    ensure!(owner.is_available());
    Ok(())
}
