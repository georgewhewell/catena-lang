//! Lower an explicit kernel function/environment pair into a GPU launch.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::*,
    lower_types::{CType, children, runtime},
    values::*,
};

/// Full callback/capture types determine specialization; runtime expressions
/// remain arguments. Canonicalize scoped witness IDs jointly across the key.
#[derive(PartialEq, Eq)]
pub(in crate::codegen::default) struct KernelKey {
    function: hexpr::Operation,
    types: Vec<Obj>,
    functions: Vec<Option<hexpr::Operation>>,
}
fn key(callback: &Callback<'_>) -> Option<KernelKey> {
    let Repr::Function(function) = &callback.function.repr else {
        return None;
    };
    let mut types = vec![callback.function.ty.clone()];
    let mut functions = Vec::new();
    for capture in &callback.captures {
        types.push(capture.ty.clone());
        functions.push(match &capture.repr {
            Repr::Function(op) => Some(op.clone()),
            Repr::Runtime(_) | Repr::Erased => None,
            Repr::Product(_) => return None,
        });
    }
    fn canonical(t: &Obj, ids: &mut std::collections::BTreeMap<usize, usize>) -> Obj {
        use metacat::tree::Tree;
        match t {
            Tree::Leaf(id, ()) => {
                let next = ids.len();
                Tree::Leaf(*ids.entry(*id).or_insert(next), ())
            }
            Tree::Node(op, port, children) => Tree::Node(
                op.clone(),
                *port,
                children.iter().map(|t| canonical(t, ids)).collect(),
            ),
            Tree::Empty => Tree::Empty,
        }
    }
    let mut ids = std::collections::BTreeMap::new();
    Some(KernelKey {
        function: function.clone(),
        types: types.iter().map(|t| canonical(t, &mut ids)).collect(),
        functions,
    })
}

fn kernel_argument(
    l: &mut Lowerer<'_>,
    op: &str,
    ty: &Obj,
    grid: &str,
) -> Result<Value, CodegenError> {
    if let Some(fields) = children(ty, "*") {
        return Ok(super::super::product(
            ty.clone(),
            fields
                .iter()
                .map(|ty| kernel_argument(l, op, ty, grid))
                .collect::<Result<_, _>>()?,
        ));
    }
    match runtime(ty)? {
        Some(CType::Block) => l.emit(
            ty,
            format!("catena_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}"),
        ),
        Some(CType::Thread) => {
            let block = format!("catena_block{{{grid},{{blockIdx.x,blockIdx.y,blockIdx.z}}}}");
            l.emit(
                ty,
                format!("catena_thread{{{block},{{threadIdx.x,threadIdx.y,threadIdx.z}}}}"),
            )
        }
        Some(_) => Err(invalid(
            op,
            "kernel input must contain only block/thread data and proofs",
        )),
        None => l.erased(ty),
    }
}

pub(in crate::codegen::default) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    if l.place != Place::Host {
        return Err(invalid(op, "nested device launch is unsupported"));
    }
    // Closure conversion expands the kernel into environment + function operands.
    let (grid, shared, environment, kernel) = match (op, args) {
        ("unsafe.launch", [grid, environment, kernel]) => (grid, None, environment, kernel),
        ("unsafe.launch_shared", [grid, shared, environment, kernel]) => {
            (grid, Some(shared), environment, kernel)
        }
        _ => return Err(invalid(op, "invalid launch operands")),
    };
    if runtime(&grid.ty)? != Some(CType::Grid) {
        return Err(invalid(op, "expected a grid"));
    }
    let shared_bytes = if let Some(shared) = shared {
        if !matches!(runtime(&shared.ty)?, Some(CType::Layout(_))) {
            return Err(invalid(op, "expected shared memory"));
        }
        format!("{}.bytes", expr(shared)?)
    } else {
        "0".into()
    };
    let callback = Callback::new(kernel, environment)?;
    let specialization = key(&callback);
    if let Some(symbol) = specialization.as_ref().and_then(|key| {
        l.kernel_cache
            .iter()
            .find(|(previous, _)| previous == key)
            .map(|(_, symbol)| symbol.clone())
    }) {
        let mut arguments = vec![expr(grid)?.to_owned()];
        for capture in &callback.captures {
            if let Repr::Runtime(expression) = &capture.repr {
                arguments.push(expression.clone());
            }
        }
        l.body.push(Instruction::Launch {
            kernel: symbol,
            grid: expr(grid)?.to_owned(),
            shared_bytes,
            arguments,
        });
        return outputs.iter().map(|ty| l.erased(ty)).collect();
    }

    let grid_expression = expr(grid)?.to_owned();
    let grid_parameter = l.fresh("grid");
    let mut inputs = vec![Variable {
        name: grid_parameter.clone(),
        ty: CType::Grid,
    }];
    let mut arguments = vec![grid_expression.clone()];
    let mut captured = callback.captures;
    for value in &mut captured {
        if let Repr::Runtime(expression) = &mut value.repr {
            let name = l.fresh("capture");
            arguments.push(expression.clone());
            *expression = name.clone();
            inputs.push(Variable {
                name,
                ty: runtime(&value.ty)?.ok_or_else(|| invalid(op, "invalid capture type"))?,
            });
        }
    }
    let outer = std::mem::take(&mut l.body);
    l.place = Place::Device;
    for ty in &callback.parameters {
        captured.push(kernel_argument(l, op, ty, &grid_parameter)?);
    }
    let results = l.call_function(callback.function, captured, &callback.outputs)?;
    for result in results {
        l.erased(&result.ty)?;
    }
    let body = std::mem::replace(&mut l.body, outer);
    l.place = Place::Host;
    let symbol = l.fresh("catena_kernel");
    l.modules.kernels.insert(
        symbol.clone(),
        Function {
            symbol: symbol.clone(),
            inputs,
            outputs: vec![],
            body,
        },
    );
    if let Some(key) = specialization {
        l.kernel_cache.push((key, symbol.clone()));
    }
    l.body.push(Instruction::Launch {
        kernel: symbol,
        grid: grid_expression,
        shared_bytes,
        arguments,
    });
    outputs.iter().map(|ty| l.erased(ty)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use metacat::tree::Tree;

    fn specialization(
        a: usize,
        b: usize,
        capture: usize,
        expression: &str,
        nested: Option<&str>,
    ) -> KernelKey {
        let function = Value {
            ty: Tree::Node(
                "function-type".parse().unwrap(),
                0,
                vec![Tree::Leaf(a, ()), Tree::Leaf(b, ())],
            ),
            repr: Repr::Function("kernel".parse().unwrap()),
        };
        let callback = Callback {
            function: &function,
            captures: vec![Value {
                ty: Tree::Node(
                    "capture-type".parse().unwrap(),
                    0,
                    vec![Tree::Leaf(capture, ())],
                ),
                repr: nested.map_or_else(
                    || Repr::Runtime(expression.into()),
                    |f| Repr::Function(f.parse().unwrap()),
                ),
            }],
            parameters: vec![],
            outputs: vec![],
        };
        key(&callback).unwrap()
    }

    #[test]
    fn kernel_specializations_preserve_witness_relationships_and_function_captures() {
        let first = specialization(1, 2, 1, "first_buffer", None);
        assert!(first == specialization(41, 82, 41, "other_buffer", None));
        assert!(first != specialization(41, 82, 82, "other_buffer", None));
        assert!(first != specialization(41, 41, 41, "other_buffer", None));
        assert!(
            specialization(1, 2, 1, "", Some("first"))
                != specialization(1, 2, 1, "", Some("second"))
        );
        assert!(first != specialization(1, 2, 1, "", Some("first")));
    }
}
