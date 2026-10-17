use catena_lang::{
    codegen,
    compile::compile,
    report::CompileReport,
    runtime::{GpuDialect, ValueKind, module_abi},
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
        let module = compile(&mut report, codegen::CodegenKind::Default, dialect)?;
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
fn default_codegen_uses_shared_passes_and_dispatch() -> anyhow::Result<()> {
    let mut report = CompileReport::new(RawTheorySet::from_text("")?);
    let module = compile(&mut report, codegen::CodegenKind::Default, GpuDialect::Hip)?;
    assert_eq!(module.dialect, GpuDialect::Hip);
    assert!(report.closure_conversion.is_some());
    assert!(report.unpacked_products.is_some());
    assert!(module.functions.is_empty());
    Ok(())
}

#[test]
fn runtime_modules_advertise_only_checkable_foreign_entry_points() -> anyhow::Result<()> {
    let raw = RawTheorySet::from_texts(
        catena_lang::stdlib::sources().chain([include_str!("cases/abi-refinements.hex")]),
    )?;
    let mut report = CompileReport::new(raw);
    let module = compile(&mut report, codegen::CodegenKind::Default, GpuDialect::Hip)?;
    let function = |name: &str| {
        module
            .functions
            .iter()
            .find(|function| function.source_name == name)
    };
    let check_symbol = |name: &str| {
        let entry = function(name).unwrap_or_else(|| panic!("{name} should be exported"));
        module_abi::argument_check_symbol(&entry.symbol)
    };

    for name in ["checked-named-bool"] {
        let entry = function(name).unwrap_or_else(|| panic!("{name} should be exported"));
        assert_eq!(entry.inputs, [ValueKind::Bool]);
        assert!(entry.outputs.is_empty());
    }
    // Refined entry points carry their own check; unrefined ones need none.
    for name in ["checked-index", "checked-matrix-size", "checked-zero"] {
        assert!(
            module
                .source
                .contains(&format!("int64_t {}(", check_symbol(name)))
        );
    }
    for name in ["canonical-bool", "checked-named-bool"] {
        assert!(!module.source.contains(&check_symbol(name)));
    }
    for name in [
        "erased-index-bound",
        "erased-proof",
        "constant-named-u32",
        "repeated-named-bool",
    ] {
        assert!(function(name).is_none(), "{name} must not be exported");
    }
    Ok(())
}

#[test]
fn every_code_generator_exports_the_runtime_services() -> anyhow::Result<()> {
    let codegen = codegen::CodegenKind::Default;
    let sources = catena_lang::stdlib::sources().collect::<Vec<_>>();
    {
        let dialects = &[GpuDialect::Hip, GpuDialect::Cuda];
        for &dialect in dialects {
            let mut report = CompileReport::new(RawTheorySet::from_texts(sources.iter().copied())?);
            let module = compile(&mut report, codegen, dialect)?;
            for symbol in [
                module_abi::ALLOCATION_BUDGET_BEGIN,
                module_abi::ALLOCATION_BUDGET_END,
            ] {
                assert!(
                    module
                        .source
                        .contains(&format!("extern \"C\" __host__ void {symbol}(")),
                    "{codegen:?} {dialect:?} module must export {symbol}"
                );
            }
            // Every allocation path goes through the metered service.
            assert!(module.source.contains("catena_runtime_allocate("));
            assert!(
                !module
                    .source
                    .contains(&format!("catena_gpu_check({}(", dialect_alloc(dialect)))
            );
            assert!(!module.source.contains(&format!(
                "catena_host_gpu_check({}(",
                dialect_alloc(dialect)
            )));
        }
    }
    Ok(())
}

fn dialect_alloc(dialect: GpuDialect) -> &'static str {
    match dialect {
        GpuDialect::Hip => "hipMalloc",
        GpuDialect::Cuda => "cudaMalloc",
    }
}
