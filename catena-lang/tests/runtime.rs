//! Runtime execution of hex fixtures against the standalone default stdlib.

#[path = "runtime/matmul.rs"]
mod matmul;
#[path = "runtime/support.rs"]
mod support;

use catena_lang::runtime::{ExecError, Value};

#[test]
fn foreign_calls_validate_dependent_sizes_and_indices_before_execution() -> anyhow::Result<()> {
    let (_device, runtime) =
        support::runtime_with_source(include_str!("cases/abi-refinements.hex"))?;
    for (n, m) in [(0_u64, 7_u64), (3, 5), (1, u64::MAX), (1 << 32, 1)] {
        let [Value::U64(size)] =
            runtime.exec("checked-matrix-size", [(n * m).into(), n.into(), m.into()])?
        else {
            anyhow::bail!("expected size")
        };
        assert_eq!(size, n * m);
        let [Value::U64(size)] = runtime.exec("checked-repeated-size", [n.into(), n.into()])?
        else {
            anyhow::bail!("expected repeated size")
        };
        assert_eq!(size, n);
    }
    for args in [[14, 3, 5], [1, 0, 7], [0, 1 << 32, 1 << 32]] {
        assert!(matches!(
            runtime.exec_values(
                "checked-matrix-size",
                args.into_iter().map(Value::U64).collect()
            ),
            Err(ExecError::InvalidArgumentRefinement { index: 0 })
        ));
    }
    assert!(matches!(
        runtime.exec_values("checked-repeated-size", vec![2_u64.into(), 3_u64.into()]),
        Err(ExecError::InvalidArgumentRefinement { index: 1 })
    ));
    for n in [1_u64, 3, 1 << 32, u64::MAX] {
        let [Value::U64(index)] = runtime.exec("checked-index", [n.into(), (n - 1).into()])? else {
            anyhow::bail!("expected index")
        };
        assert_eq!(index, n - 1);
    }
    for (n, index) in [(0_u64, 0_u64), (3, 3), (3, u64::MAX)] {
        assert!(matches!(
            runtime.exec_values("checked-index", vec![n.into(), index.into()]),
            Err(ExecError::InvalidArgumentRefinement { index: 1 })
        ));
    }
    for name in ["erased-index-bound", "erased-proof"] {
        assert!(matches!(
            runtime.exec_values(name, vec![0_u64.into()]),
            Err(ExecError::UnknownSourceFunction(_))
        ));
    }
    let [] = runtime.exec("checked-named-bool", [true.into()])?;
    for value in [0, 1] {
        let [Value::Bool(actual)] = runtime.exec("canonical-bool", [Value::Bool(value)])? else {
            anyhow::bail!("expected boolean")
        };
        assert_eq!(actual, value);
    }
    for value in [2, 255] {
        assert!(matches!(
            runtime.exec_values("canonical-bool", vec![Value::Bool(value)]),
            Err(ExecError::InvalidArgumentRefinement { index: 0 })
        ));
    }
    let [Value::U64(zero)] = runtime.exec("checked-zero", [0_u64.into()])? else {
        anyhow::bail!("expected zero")
    };
    assert_eq!(zero, 0);
    assert!(matches!(
        runtime.exec_values("checked-zero", vec![1_u64.into()]),
        Err(ExecError::InvalidArgumentRefinement { index: 0 })
    ));
    // A zero outer dimension remains valid even when an inner product exceeds u64.
    let [Value::U64(volume)] = runtime.exec(
        "checked-volume",
        [0_u64.into(), 0_u64.into(), u64::MAX.into(), u64::MAX.into()],
    )?
    else {
        anyhow::bail!("expected zero volume")
    };
    assert_eq!(volume, 0);
    // Admission failures leave the artifact reusable.
    let [Value::U64(index)] = runtime.exec("checked-index", [8_u64.into(), 7_u64.into()])? else {
        anyhow::bail!("expected recovered index")
    };
    assert_eq!(index, 7);
    Ok(())
}

#[test]
fn publication_streams_prefixes_and_rejects_reentry() -> anyhow::Result<()> {
    let (runtime, artifact) = support::runtime_with_source(
        "(def program publish-prefix : {((cap.own mem) val) (u64 val) (u64 val)} -> ((cap.own mem) val) = mem.publish)",
    )?;
    let values = [1_u64, 2, 3, 4];
    let buffer = runtime.mem_u64(&values)?;
    let mut published = Vec::new();
    let mut nested = None;
    let outputs = artifact.exec_values_with_publication(
        "publish-prefix",
        vec![buffer.into(), 0_u64.into(), 16_u64.into()],
        u64::MAX,
        64,
        &mut |publication| {
            assert_eq!(publication.output, 0);
            published.extend_from_slice(&publication.bytes);
            nested.get_or_insert_with(|| artifact.exec_values("publish-prefix", vec![]));
            true
        },
    )?;
    let [Value::MemOwn(output)] = outputs.as_slice() else {
        anyhow::bail!("publish-prefix returned {outputs:?}");
    };
    assert_eq!(output.try_to_u64_vec()?, values);
    let expected = values[..2]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(published, expected);
    assert!(matches!(nested, Some(Err(ExecError::ReentrantCall))));
    Ok(())
}
