//! Rendering an array constant's model value and a declared function's
//! interpretation for `(get-model)` (`#P2b-34`).
//!
//! Before this module an array constant printed the *sort default* and nothing
//! else — `arr = ((as const (Array (_ BitVec 8) (_ BitVec 8))) #x00)` — even
//! when the same model answered `(get-value ((select arr j)))` with `#x05`.
//! The two statements contradict each other: read the printed array at `j` and
//! it is `#x00`.  A declared function fared worse still: `(get-model)` omitted
//! it altogether, so a model over `(f x)` could be neither printed nor checked.
//!
//! Both are rendered here from what the model actually holds:
//!
//! * an array becomes the `store` chain of its published reads laid over the
//!   constant default, which is the canonical SMT-LIB spelling of exactly the
//!   function those reads describe;
//! * a function becomes the nested `ite` over its congruence-closed
//!   application entries with the interpretation's else-value at the bottom,
//!   the spelling Z3 prints.
//!
//! Neither rendering invents anything: every entry comes from
//! `Solver::publish_array_reads` / `Context::get_func_interp_raw`, and an array
//! with no published read still prints the constant default it always did.

use super::*;
use oxiz_core::ast::TermManager;

impl Context {
    /// The `store`-chain rendering of `array`'s value, or `None` when the model
    /// published no read of it (the caller then keeps the constant default).
    ///
    /// # Why the chain is deduplicated by index *value*
    ///
    /// A `store` chain is read outermost-first, so two writes at the same index
    /// would let the outer one shadow the inner — and the shadowed read is one
    /// the model published, i.e. the printed array would contradict its own
    /// `(get-value ((select arr i)))` answer.  That is the very defect this
    /// rendering exists to remove, so entries are keyed by the *evaluated*
    /// index (two syntactically different indices that the model maps to the
    /// same value are one entry) and the first entry in term order wins.
    ///
    /// Two model-equal indices carrying *different* values would be a genuine
    /// inconsistency in the model rather than a rendering choice; it is the
    /// model gate's job to refuse such a candidate, and this rendering does not
    /// try to repair one.  What is actually checked is narrower:
    /// `p2b34_published_array_reads_agree_with_the_printed_array` replays the
    /// printed model of three read-under-operator goals as assertions and
    /// requires the result to stay `sat`, which fails outright if the chain is
    /// dropped — that replay only became meaningful once `#P2b-36` gave the
    /// chain's `((as const …) d)` base a value the solver reasons about.
    ///
    /// Reads are matched syntactically against `array`: an entry for
    /// `(select (store arr i v) j)` belongs to the *store term*, not to `arr`,
    /// and is rendered as part of no constant's value (nothing declares it).
    pub(super) fn array_model_value(
        &self,
        array: TermId,
        sort: SortId,
        model: &crate::solver::Model,
    ) -> Option<String> {
        let mut reads: Vec<(TermId, TermId, TermId)> = Vec::new();
        for (&term, &value) in model.assignments() {
            let Some(data) = self.terms.get(term) else {
                continue;
            };
            if let TermKind::Select(base, index) = data.kind
                && base == array
            {
                reads.push((term, index, value));
            }
        }
        if reads.is_empty() {
            return None;
        }
        // Deterministic: the model map is a hash table, the printed chain is
        // not allowed to depend on its iteration order.
        reads.sort_unstable_by_key(|&(term, _, _)| term.raw());

        let mut seen: crate::prelude::HashSet<String> = crate::prelude::HashSet::new();
        let mut chain = self.default_value(sort);
        let mut entries: Vec<(String, String)> = Vec::new();
        for (_, index, value) in reads {
            // An index the model leaves unassigned cannot be placed in the
            // chain: the read is real, but *where* it sits is not decided.
            let index_value = model.get(index).unwrap_or(index);
            if !is_ground_value(index_value, &self.terms) {
                continue;
            }
            let index_str = self.format_value(index_value);
            if !seen.insert(index_str.clone()) {
                continue;
            }
            entries.push((index_str, self.format_value(value)));
        }
        if entries.is_empty() {
            return None;
        }
        // Innermost write first, so the chain reads left to right in the same
        // order the entries were collected.
        for (index_str, value_str) in entries {
            chain = format!("(store {chain} {index_str} {value_str})");
        }
        Some(chain)
    }

    /// `(define-fun f ((x!0 S0) …) Ret …)` for every declared uninterpreted
    /// function, in declaration order.
    ///
    /// The body is the nested `ite` over the interpretation's entries with the
    /// else-value at the bottom — a total function, as SMT-LIB requires a
    /// model's interpretation to be.  A function with no application in the
    /// assertions is the constant else-value, which is what
    /// [`Context::get_func_interp_raw`] reports for it.
    ///
    /// An entry whose arity disagrees with the declaration is skipped rather
    /// than rendered with the wrong number of guards; that cannot happen for a
    /// well-formed declaration, and printing a malformed `define-fun` would be
    /// worse than printing the else-value alone.
    pub(super) fn func_interp_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        let uninterpreted: Vec<(String, Vec<SortId>, SortId)> = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| (d.name.clone(), d.arg_sorts.clone(), d.ret_sort))
            .collect();
        for (name, arg_sorts, ret_sort) in uninterpreted {
            let name = name.as_str();
            let Some((entries, else_value, arity)) = self.get_func_interp_raw(name) else {
                continue;
            };
            let params: Vec<String> = arg_sorts
                .iter()
                .enumerate()
                .map(|(i, &s)| format!("(x!{i} {})", self.format_sort_name(s)))
                .collect();
            let mut body = else_value;
            // Innermost `ite` last, so the first entry is tested first.
            for (args, value) in entries.into_iter().rev() {
                if args.len() != arity {
                    continue;
                }
                // An entry that agrees with what the body already says adds a
                // guard and no information: `(ite (= x!0 #x01) #x07 #x07)` is
                // the else-value spelled twice.
                if value == body {
                    continue;
                }
                let guard = match args.len() {
                    1 => match args.first() {
                        Some(arg) => format!("(= x!0 {arg})"),
                        None => continue,
                    },
                    _ => {
                        let conjuncts: Vec<String> = args
                            .iter()
                            .enumerate()
                            .map(|(i, arg)| format!("(= x!{i} {arg})"))
                            .collect();
                        format!("(and {})", conjuncts.join(" "))
                    }
                };
                body = format!("(ite {guard} {value} {body})");
            }
            lines.push(format!(
                "  (define-fun {name} ({}) {} {body})",
                params.join(" "),
                self.format_sort_name(ret_sort)
            ));
        }
        lines
    }
}

/// Whether `term` is a ground value literal — the shapes a model entry can
/// carry and the shapes `format_value` renders verbatim.
///
/// Used to decide whether an index can be placed in a `store` chain: an index
/// the model left symbolic names no position in the array.
fn is_ground_value(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        matches!(
            t.kind,
            TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::True
                | TermKind::False
                | TermKind::StringLit(_)
        )
    })
}
