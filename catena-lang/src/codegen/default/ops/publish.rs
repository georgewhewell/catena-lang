//! Publish a prefix while preserving the program's owned buffer.
use super::super::{
    CodegenError, Lowerer, expr, invalid,
    ir::{Instruction, Place, Variable},
    lower_types::{CType, runtime},
    values::{Obj, Value},
};
use super::{results, runtime_args, single_output};

pub(super) fn lower(
    l: &mut Lowerer<'_>,
    args: &[Value],
    outputs: &[Obj],
) -> Result<Vec<Value>, CodegenError> {
    let values = runtime_args(args);
    let [buffer, output, bytes] = values.as_slice() else {
        return Err(invalid(
            "mem.publish",
            "expected buffer, output and prefix length",
        ));
    };
    if l.place != Place::Host
        || runtime(&buffer.ty)? != Some(CType::MemOwn)
        || runtime(&output.ty)? != Some(CType::U64)
        || runtime(&bytes.ty)? != Some(CType::U64)
        || runtime(&single_output(outputs)?)? != Some(CType::MemOwn)
    {
        return Err(invalid(
            "mem.publish",
            "requires owned memory and two u64 arguments on the host",
        ));
    }
    let operand = |value: &Value, ty| -> Result<Variable, CodegenError> {
        Ok(Variable {
            name: expr(value)?.into(),
            ty,
        })
    };
    l.body.push(Instruction::Publish {
        buffer: operand(buffer, CType::MemOwn)?,
        output: operand(output, CType::U64)?,
        bytes: operand(bytes, CType::U64)?,
    });
    results(l, outputs, vec![buffer.clone()])
}
