//! Validate dependent u64 arguments at the foreign call boundary, before their
//! refinements are erased. Internal calls still use the checked Hex types.
//!
//! The checks are rendered into the module as the entry point's argument check
//! (see [`crate::runtime::module_abi::argument_check_symbol`]), so the runtime can
//! reject a foreign call without interpreting refinements itself.
use std::collections::BTreeMap;

use metacat::tree::Tree;

use super::{
    ir::{Function, Variable},
    lower_types::{children as node, runtime},
    values::Obj,
};
use crate::runtime::{GpuDialect, module_abi};

type Type = Obj;

/// A mathematical size built from foreign arguments. Wrapping arithmetic cannot
/// establish a buffer/index relationship, so overflow makes the input invalid
/// unless a factor is zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Extent {
    Argument(usize),
    Constant(u64),
    Multiply(Box<Extent>, Box<Extent>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArgumentCheck {
    Equal { argument: usize, value: Extent },
    Index { argument: usize, extent: Extent },
}

/// Render `function`'s argument check, which returns the index of the first
/// argument violating its refinement or `-1`. Entry points without refinements
/// need no check.
pub(super) fn render_argument_check(
    out: &mut String,
    function: &Function,
    checks: &[ArgumentCheck],
    dialect: GpuDialect,
) {
    if checks.is_empty() {
        return;
    }
    let params = function
        .inputs
        .iter()
        .map(|source| format!("{} {}", source.ty.c_name(), source.name))
        .collect::<Vec<_>>();
    out.push_str(&format!(
        "#ifndef {}\nextern \"C\" __host__ int64_t {}({}) {{\n",
        dialect.device_compile_guard(),
        module_abi::argument_check_symbol(&function.symbol),
        params.join(", ")
    ));
    for check in checks {
        let (argument, condition) = match check {
            ArgumentCheck::Equal { argument, value } => (
                *argument,
                format!(
                    "catena_extent_equals({}, {})",
                    function.inputs[*argument].name,
                    render_extent(value, &function.inputs)
                ),
            ),
            ArgumentCheck::Index { argument, extent } => (
                *argument,
                format!(
                    "catena_extent_bounds({}, {})",
                    function.inputs[*argument].name,
                    render_extent(extent, &function.inputs)
                ),
            ),
        };
        out.push_str(&format!("    if (!{condition}) return {argument};\n"));
    }
    out.push_str("    return -1;\n}\n#endif\n");
}

fn render_extent(extent: &Extent, sources: &[Variable]) -> String {
    match extent {
        Extent::Argument(index) => format!("catena_extent{{{}, 1}}", sources[*index].name),
        Extent::Constant(value) => format!("catena_extent{{{value}ull, 1}}"),
        Extent::Multiply(a, b) => format!(
            "catena_extent_mul({}, {})",
            render_extent(a, sources),
            render_extent(b, sources)
        ),
    }
}

fn named_u64(ty: &Type) -> Option<&Type> {
    let [name, kind] = node(ty, ":")? else {
        return None;
    };
    (node(kind, "u64") == Some(&[])).then_some(name)
}

fn extent(ty: &Type, bindings: &BTreeMap<usize, Extent>) -> Option<Extent> {
    match ty {
        Tree::Leaf(index, ()) => bindings.get(index).cloned(),
        Tree::Node(op, 0, children) => match (op.as_str(), children.as_slice()) {
            ("0", []) => Some(Extent::Constant(0)),
            ("1", []) => Some(Extent::Constant(1)),
            ("type.stdlib.numeric.*", [a, b]) => Some(Extent::Multiply(
                Box::new(extent(a, bindings)?),
                Box::new(extent(b, bindings)?),
            )),
            (name, []) => name
                .strip_prefix("type.const.u64.0x")
                .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                .map(Extent::Constant),
            _ => None,
        },
        _ => None,
    }
}

/// None means this interface has a refinement the foreign ABI cannot validate.
/// Such helpers remain callable from checked Hex, but are not runtime entries.
pub(super) fn argument_checks(inputs: &[Obj]) -> Option<Vec<ArgumentCheck>> {
    // Use exactly the same product traversal order as Lowerer::input.
    fn flatten<'a>(ty: &'a Obj, fields: &mut Vec<&'a Obj>) -> Option<()> {
        if let Some(children) = node(ty, "*") {
            for child in children {
                flatten(child, fields)?;
            }
        } else {
            // Erasure cannot supply a proof or a type witness for a foreign caller.
            runtime(ty).ok()??;
            fields.push(ty);
        }
        Some(())
    }
    let mut runtime = Vec::new();
    for ty in inputs {
        flatten(ty, &mut runtime)?;
    }
    let mut bindings = BTreeMap::new();
    for (argument, ty) in runtime.iter().enumerate() {
        if let Some(Tree::Leaf(name, ())) = named_u64(ty) {
            bindings.entry(*name).or_insert(Extent::Argument(argument));
        }
    }
    let mut checks = Vec::new();
    for (argument, &ty) in runtime.iter().enumerate() {
        if let Some(name) = named_u64(ty) {
            let value = extent(name, &bindings)?;
            if value != Extent::Argument(argument) {
                checks.push(ArgumentCheck::Equal { argument, value });
            }
        } else if let Some([inner]) = node(ty, "val") {
            if let Some([bound]) = node(inner, "stdlib.gpu.memory.type.Ix") {
                checks.push(ArgumentCheck::Index {
                    argument,
                    extent: extent(bound, &bindings)?,
                });
            } else if !plain_value(inner) {
                return None;
            }
        } else if let Some([name, kind]) = node(ty, ":") {
            // A named non-u64 scalar needs no check when its name is a variable
            // that no other input mentions: no relation constrains its value.
            // Constant names and repeated names are relations the ABI cannot
            // check, so such definitions are not entry points.
            let Tree::Leaf(variable, ()) = name else {
                return None;
            };
            let mentions: usize = runtime.iter().map(|ty| leaf_count(ty, *variable)).sum();
            if mentions != 1 || !plain_value(kind) {
                return None;
            }
        } else if node(ty, "mem").is_none() {
            return None;
        }
    }
    Some(checks)
}

fn leaf_count(ty: &Type, variable: usize) -> usize {
    match ty {
        Tree::Leaf(index, ()) => usize::from(*index == variable),
        Tree::Node(_, _, children) => children
            .iter()
            .map(|child| leaf_count(child, variable))
            .sum(),
        Tree::Empty => 0,
    }
}

fn plain_value(ty: &Type) -> bool {
    node(ty, "mem").is_some()
        || matches!(ty, Tree::Node(op, 0, children)
            if children.is_empty() && matches!(op.as_str(), "bool" | "u32" | "u64" | "f32"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::default::lower_types::node as ty;

    #[test]
    fn products_preserve_argument_order_and_cannot_hide_proofs_or_relations() {
        let named = |name, kind| ty(":", vec![name, ty(kind, vec![])]);
        let n = Tree::Leaf(0, ());
        let index = ty(
            "val",
            vec![ty("stdlib.gpu.memory.type.Ix", vec![n.clone()])],
        );
        let input = ty(
            "*",
            vec![
                index,
                ty("*", vec![named(n.clone(), "u64"), named(n, "u64")]),
            ],
        );
        assert_eq!(
            argument_checks(&[input]),
            Some(vec![
                ArgumentCheck::Index {
                    argument: 0,
                    extent: Extent::Argument(1)
                },
                ArgumentCheck::Equal {
                    argument: 2,
                    value: Extent::Argument(1)
                },
            ])
        );
        let scalar = ty("val", vec![ty("u32", vec![])]);
        for field in [
            ty("|-", vec![Tree::Leaf(1, ())]),
            named(ty("type.const.u32.0x00000000", vec![]), "u32"),
            ty(
                "*",
                vec![
                    named(Tree::Leaf(2, ()), "bool"),
                    named(Tree::Leaf(2, ()), "bool"),
                ],
            ),
        ] {
            assert!(argument_checks(&[ty("*", vec![scalar.clone(), field])]).is_none());
        }
    }
}
