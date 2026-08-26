//! `(declare-datatype ..)` / `(declare-datatypes ..)` follow-up registration
//! for [`Context`].
//!
//! Lives in a child module so the (already large) `context` module stays under
//! the 2000-line policy limit; being a child of `context`, it retains full
//! access to `Context`'s private fields.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::sort::SortId;

use super::Context;

impl Context {
    /// Expose a declared datatype's constructors and selectors as callable
    /// functions in this context's own function registry.
    ///
    /// The parser has already fully registered each datatype's sort and its
    /// constructor/selector definitions directly on `self.terms.sorts` —
    /// including selector sorts resolved through the full sort grammar — so
    /// in-script constructor application (e.g. `(cons 1 nil)`) already works
    /// without help from here. What is missing is the registry entry Z3
    /// implicitly creates for each constructor and selector, so that
    /// introspection (`get_fun_signature`, `declared_function_names`) sees
    /// them.
    ///
    /// `names` is the comma-joined list of every datatype the command declared
    /// (see the parser's `DeclareDatatype` doc comment, covering both the
    /// multi- and the mutually recursive `declare-datatypes` forms). Each one's
    /// authoritative definition is looked up directly on the sort manager
    /// rather than re-derived from the weaker, string-typed `constructors`
    /// field of the command.
    pub(super) fn register_datatype_functions(&mut self, names: &str) {
        for dt_name in names.split(',') {
            let dt_name = dt_name.trim();
            if dt_name.is_empty() {
                continue;
            }
            let dt_sort = self.terms.sorts.mk_datatype_sort(dt_name);
            let Some(ctors) = self
                .terms
                .sorts
                .get_datatype(dt_name)
                .map(|def| def.constructors.clone())
            else {
                continue;
            };
            for ctor in &ctors {
                let ctor_name = self.terms.resolve_str(ctor.name).to_string();
                let selector_sorts: Vec<SortId> =
                    ctor.selectors.iter().map(|&(_, sort)| sort).collect();
                self.declare_fun(&ctor_name, selector_sorts, dt_sort);
                for &(sel_spur, sel_sort) in &ctor.selectors {
                    let sel_name = self.terms.resolve_str(sel_spur).to_string();
                    self.declare_fun(&sel_name, vec![dt_sort], sel_sort);
                }
            }
        }
    }
}
