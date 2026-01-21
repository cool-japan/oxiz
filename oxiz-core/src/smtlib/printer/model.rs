//! Model printing functionality

use crate::ast::{model::Model, model::ModelValue};
use std::fmt::Write;

use super::basic::Printer;

impl<'a> Printer<'a> {
    /// Print a model in SMT-LIB2 format
    pub fn print_model(&self, model: &Model) -> String {
        let mut buf = String::new();
        self.write_model(&mut buf, model);
        buf
    }

    /// Write a model in SMT-LIB2 format
    pub fn write_model(&self, w: &mut impl Write, model: &Model) {
        let _ = writeln!(w, "(model");

        // Print variable assignments as define-fun declarations
        for (term_id, value) in model.assignments() {
            if let Some(term) = self.manager.get(*term_id)
                && let crate::ast::TermKind::Var(name_spur) = term.kind
            {
                let var_name = self.manager.resolve_str(name_spur);
                let _ = write!(w, "  (define-fun {} () ", var_name);
                self.write_sort(w, term.sort);
                let _ = write!(w, " ");
                self.write_model_value(w, value);
                let _ = writeln!(w, ")");
            }
        }

        // Print function interpretations
        for (name_spur, func_interp) in model.functions() {
            let func_name = self.manager.resolve_str(*name_spur);
            let _ = write!(w, "  (define-fun {} ", func_name);

            // For now, we'll print a simplified version
            // A full implementation would need parameter sorts from the function signature
            if func_interp.table().is_empty() {
                // Just print the default value if there is one
                if let Some(default) = func_interp.default_value() {
                    let _ = write!(w, "() ");
                    // We'd need sort information here
                    let _ = write!(w, "Int "); // Placeholder
                    self.write_model_value(w, default);
                    let _ = writeln!(w, ")");
                }
            } else {
                // For functions with explicit table entries, we'd need to construct
                // an ITE chain or similar. This is a placeholder.
                let _ = writeln!(w, "...)");
            }
        }

        let _ = writeln!(w, ")");
    }

    /// Write a model value
    fn write_model_value(&self, w: &mut impl Write, value: &ModelValue) {
        match value {
            ModelValue::Bool(b) => {
                let _ = write!(w, "{}", b);
            }
            ModelValue::Int(n) => {
                let _ = write!(w, "{}", n);
            }
            ModelValue::Real(r) => {
                let _ = write!(w, "{}", r);
            }
            ModelValue::BitVec { value, width } => {
                let _ = write!(
                    w,
                    "#x{:0>width$x}",
                    value,
                    width = (*width as usize).div_ceil(4)
                );
            }
            ModelValue::Uninterpreted { sort, id } => {
                let _ = write!(w, "uninterp_{}_{}", sort.0, id);
            }
        }
    }

}
