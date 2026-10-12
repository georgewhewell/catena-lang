//! BF16 operations retain BF16 operands/results. Packed dot is native BF16/F32.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::Place,
    lower_types::{CType, runtime},
    values::*,
};
use super::{results, runtime_args, single_output};
pub(super) fn lower(
    l: &mut Lowerer<'_>,
    op: &str,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let ty = single_output(outputs)?;
    let out = runtime(&ty)?.ok_or_else(|| invalid(op, "expected runtime result"))?;
    let args = runtime_args(args);
    let types = args
        .iter()
        .map(|a| runtime(&a.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let xs = args.iter().map(expr).collect::<Result<Vec<_>, _>>()?;
    let b = Some(CType::Bf16);
    let p = Some(CType::Bf16x2);
    let (expected, inputs, expression) = match op {
        "bf16.one" => (CType::Bf16, vec![], "catena_bf16_from_f32(1.0f)".into()),
        "bf16.add" | "bf16.sub" | "bf16.mul" | "bf16.div" => (
            CType::Bf16,
            vec![b.clone(), b.clone()],
            format!(
                "catena_bf16_{}({})",
                op.trim_start_matches("bf16."),
                xs.join(",")
            ),
        ),
        "bf16.neg" | "bf16.exp" => (
            CType::Bf16,
            vec![b.clone()],
            format!(
                "catena_bf16_{}({})",
                op.trim_start_matches("bf16."),
                xs.join(",")
            ),
        ),
        "bf16.fma" => (
            CType::Bf16,
            vec![b.clone(); 3],
            format!("catena_bf16_fma({})", xs.join(",")),
        ),
        "bf16.select" => (
            CType::Bf16,
            vec![Some(CType::Bool), b.clone(), b.clone()],
            format!(
                "({} ? {} : {})",
                xs.first().unwrap_or(&""),
                xs.get(1).unwrap_or(&""),
                xs.get(2).unwrap_or(&"")
            ),
        ),
        "bf16.eq" | "bf16.ne" | "bf16.lt" | "bf16.lte" | "bf16.gt" | "bf16.gte" => {
            let symbol = match op {
                "bf16.eq" => "==",
                "bf16.ne" => "!=",
                "bf16.lt" => "<",
                "bf16.lte" => "<=",
                "bf16.gt" => ">",
                _ => ">=",
            };
            (
                CType::Bool,
                vec![b.clone(); 2],
                format!(
                    "({} {symbol} {})",
                    xs.first().unwrap_or(&""),
                    xs.get(1).unwrap_or(&"")
                ),
            )
        }
        "bf16.to-f32" => (
            CType::F32,
            vec![b.clone()],
            format!("catena_bf16_to_f32({})", xs.join(",")),
        ),
        "f32.to-bf16" => (
            CType::Bf16,
            vec![Some(CType::F32)],
            format!("catena_bf16_from_f32({})", xs.join(",")),
        ),
        "bf16.bitcast-u16" => (
            CType::U16,
            vec![b.clone()],
            format!("catena_bf16_bitcast_u16({})", xs.join(",")),
        ),
        "u16.bitcast-bf16" => (
            CType::Bf16,
            vec![Some(CType::U16)],
            format!("catena_u16_bitcast_bf16({})", xs.join(",")),
        ),
        "bf16x2.pack" => (
            CType::Bf16x2,
            vec![b.clone(); 2],
            format!("catena_bf16_pack({})", xs.join(",")),
        ),
        "bf16x2.low" | "bf16x2.high" => (
            CType::Bf16,
            vec![p.clone()],
            format!(
                "catena_bf16_{}({})",
                op.trim_start_matches("bf16x2."),
                xs.join(",")
            ),
        ),
        "bf16x2.dot-f32" => {
            if l.place != Place::Device {
                return Err(invalid(op, "packed dot requires device execution"));
            }
            l.modules.bf16_dot = true;
            (
                CType::F32,
                vec![p.clone(), p, Some(CType::F32)],
                format!("catena_bf16_dot({})", xs.join(",")),
            )
        }
        _ => return Err(CodegenError::Unsupported(op.into())),
    };
    if types != inputs || out != expected {
        return Err(invalid(op, "BF16 operand or result type mismatch"));
    }
    if op == "bf16.exp" && l.place != Place::Device {
        return Err(invalid(op, "BF16 exp requires device execution"));
    }
    l.modules.bf16 = true;
    let v = l.emit(&ty, expression)?;
    results(l, outputs, vec![v])
}
