//! Embedded standard library sources match the on-disk manifest.
use catena_lang::stdlib::{self, BundleRegistry};
#[test]
fn embedded_sources_match_manifest_and_core_excludes_only_matmul() -> anyhow::Result<()> {
    let mut registry = BundleRegistry::new(&[])?;
    let name = registry
        .add_directory(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("stdlib/default"))?;
    let files = registry.resolve(&[&name])?;
    assert_eq!(
        stdlib::sources().collect::<Vec<_>>(),
        files.iter().map(|f| f.source.as_ref()).collect::<Vec<_>>()
    );
    let expected = files
        .iter()
        .filter(|file| !file.filename.ends_with("gpu/matmul.hex"))
        .map(|file| file.source.as_ref())
        .collect::<Vec<_>>();
    assert_eq!(stdlib::core_sources().collect::<Vec<_>>(), expected);
    assert_eq!(stdlib::sources().len(), stdlib::core_sources().count() + 1);
    Ok(())
}
