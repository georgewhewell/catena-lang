use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::runtime::{GeneratedFunction, ValueKind};

/// Read-only source-level function signature exposed by a loaded artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    name: String,
    inputs: Vec<ValueKind>,
    outputs: Vec<ValueKind>,
}

impl EntryPoint {
    pub(crate) fn new(name: String, inputs: Vec<ValueKind>, outputs: Vec<ValueKind>) -> Self {
        Self {
            name,
            inputs,
            outputs,
        }
    }

    /// Source-level function name used for execution.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Ordered input kinds, including memory ownership.
    pub fn inputs(&self) -> &[ValueKind] {
        &self.inputs
    }

    /// Ordered output kinds, including memory ownership.
    pub fn outputs(&self) -> &[ValueKind] {
        &self.outputs
    }
}

#[derive(Debug, Clone)]
pub(super) struct FunctionSignature {
    pub(super) symbol: String,
    pub(super) inputs: Vec<ValueKind>,
    pub(super) outputs: Vec<ValueKind>,
}

/// Source-level program names and their generated C ABI signatures.
pub(super) type SignatureTable = HashMap<String, FunctionSignature>;

pub(super) fn entry_points(signatures: &SignatureTable) -> Vec<EntryPoint> {
    let mut entries = signatures
        .iter()
        .map(|(name, signature)| {
            EntryPoint::new(
                name.clone(),
                signature.inputs.clone(),
                signature.outputs.clone(),
            )
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    entries
}

pub(super) fn generated_signatures(
    functions: impl IntoIterator<Item = GeneratedFunction>,
) -> SignatureTable {
    functions
        .into_iter()
        .map(|function| {
            (
                function.source_name,
                FunctionSignature {
                    symbol: function.symbol,
                    inputs: function.inputs,
                    outputs: function.outputs,
                },
            )
        })
        .collect()
}
