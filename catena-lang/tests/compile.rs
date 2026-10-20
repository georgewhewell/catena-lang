use catena_lang::{
    codegen,
    compile::compile,
    report::CompileReport,
    runtime::{GpuDialect, ValueKind},
};
use metacat::theory::RawTheorySet;

#[test]
fn codegen_produces_a_runtime_module() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(
        catena_lang::stdlib::sources()
            .chain(["(def program identity : (u64 val) -> (u64 val) = [value])"]),
    )?;
    for dialect in [GpuDialect::Hip, GpuDialect::Cuda] {
        let mut report = CompileReport::new(raw.clone());
        let module = compile(&mut report, codegen::CodegenKind::Experimental, dialect)?;
        assert_eq!(module.dialect, dialect);
        assert!(!module.source.is_empty());
        let identity = module
            .functions
            .iter()
            .find(|function| function.source_name == "identity")
            .expect("identity should be exported");
        assert_eq!(identity.inputs, [ValueKind::U64]);
        assert_eq!(identity.outputs, [ValueKind::U64]);
    }
    Ok(())
}

#[test]
fn experimental_codegen_uses_shared_passes_and_dispatch() -> anyhow::Result<()> {
    let mut report = CompileReport::new(RawTheorySet::from_text("")?);
    let module = compile(
        &mut report,
        codegen::CodegenKind::Experimental,
        GpuDialect::Hip,
    )?;
    assert_eq!(module.dialect, GpuDialect::Hip);
    assert!(report.closure_conversion.is_some());
    assert!(report.unpacked_products.is_some());
    assert!(module.functions.is_empty());
    Ok(())
}
