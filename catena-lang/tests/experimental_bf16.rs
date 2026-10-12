//! BF16 has typed memory/kernel values but keeps the public memory ABI.
use catena_lang::{
    codegen::CodegenKind,
    compile::compile,
    report::CompileReport,
    runtime::{GpuDialect, RuntimeModule},
    stdlib,
};
use metacat::theory::RawTheorySet;
fn build(source: &str, dialect: GpuDialect) -> anyhow::Result<RuntimeModule> {
    let mut sources = stdlib::experimental_core_sources().collect::<Vec<_>>();
    sources.push(source);
    let raw = RawTheorySet::from_texts(sources)?;
    Ok(compile(
        &mut CompileReport::new(raw),
        CodegenKind::Experimental,
        dialect,
    )?)
}
#[test]
fn scalar_operations_and_bitcasts_keep_bf16_types() -> anyhow::Result<()> {
    let source = r#"(def program main : (f32 val) -> (f32 val) =
 ([a.] ([.a] f32.to-bf16 [b.]) ([.b b] bf16.add [c.])
 ([.c b] bf16.sub [d.]) ([.d b] bf16.mul [e.])
 ([.e b] bf16.div [f.]) ([.f b b] bf16.fma [g.])
 ([.g] bf16.neg bf16.bitcast-u16 u16.bitcast-bf16 [h.])
 ([.h b] bf16x2.pack bf16x2.high bf16.to-f32)))"#;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let module = build(source, dialect)?;
        assert!(module.source.contains("catena_bf16_mul("));
        assert!(module.source.contains("__hmul_rn(a,b)"));
        assert!(module.source.contains("uint16_t"));
        assert!(!module.source.contains("__builtin_amdgcn_fdot2"));
    }
    Ok(())
}
#[test]
fn typed_memory_and_device_dot_require_native_hip() -> anyhow::Result<()> {
    let source = include_str!("experimental_runtime/fixtures/bf16.hex");
    let module = build(source, GpuDialect::Hip)?;
    assert!(module.source.contains("__builtin_amdgcn_fdot2_f32_bf16"));
    assert!(module.source.contains("catena_global<catena_bf16x2_t>"));
    assert!(module.source.contains("catena_bf16_load_pair(&"));
    assert!(module.source.contains("catena_global<catena_bf16_t>"));
    assert!(module.source.contains("catena_mem_count<catena_bf16x2_t>"));
    assert!(module.source.contains("#error \"native packed BF16 dot"));
    let error = build(source, GpuDialect::Cuda).unwrap_err().to_string();
    assert!(error.contains("requires HIP gfx11"), "{error}");
    Ok(())
}
#[test]
fn native_dot_and_exp_cannot_execute_on_host() {
    for body in [
        "([.acc] f32.to-bf16 [b.]) ([.b b] bf16x2.pack [p.]) ([.p p acc] bf16x2.dot-f32)",
        "([.acc] f32.to-bf16 bf16.exp bf16.to-f32)",
    ] {
        let source = format!("(def program main : (f32 val) -> (f32 val) = ([acc.] {body}))");
        let error = build(&source, GpuDialect::Hip).unwrap_err().to_string();
        assert!(error.contains("requires device execution"), "{error}");
    }
}
#[test]
fn f32_modules_do_not_include_bf16_support() -> anyhow::Result<()> {
    let module = build(
        "(def program main : (f32 val) -> (f32 val) = ([a.] [.a]))",
        GpuDialect::Hip,
    )?;
    assert!(!module.source.contains("hip_bf16.h"));
    assert!(!module.source.contains("catena_bf16"));
    Ok(())
}
