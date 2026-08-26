//! Model and function-interpretation output formatting for [`Context`].
//!
//! These methods turn the solver's model, term values, and sorts into
//! SMT-LIB2 display strings for `(get-model)` / `(get-value ...)` and the Z3
//! function-interpretation extensions.  They live in a child module so the
//! (already large) `context` module stays under the 2000-line policy limit;
//! being a child of `context`, they retain full access to `Context`'s private
//! fields.

#[allow(unused_imports)]
use crate::prelude::*;
use crate::solver::SolverResult;
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use oxiz_core::ast::{TermId, TermKind};
use oxiz_core::sort::{SortId, SortKind};
use oxiz_theories::nl_witness::{AlgebraicValue, NlWitnessValue};

use super::{Context, RawFuncInterp};

/// Render a nonlinear-real exact value as SMT-LIB2 text.
///
/// Rational values print exactly as the ordinary `RealConst` arm of
/// [`Context::format_value`] prints them, so the two channels agree
/// character-for-character on a value both could carry. Algebraic values print
/// as `root-obj`; see [`render_root_obj`].
fn render_nl_witness_value(value: &NlWitnessValue) -> String {
    match value {
        NlWitnessValue::Rational(rational) => {
            if rational.denom().is_one() {
                format!("{}.0", rational.numer())
            } else {
                format!("(/ {} {})", rational.numer(), rational.denom())
            }
        }
        NlWitnessValue::Algebraic(algebraic) => render_root_obj(algebraic),
    }
}

/// Render an exact algebraic number as SMT-LIB2 `root-obj` notation.
///
/// # The spelling is Z3's, verified against it
///
/// Every rule below was read off `z3 4.15.4` rather than inferred, by running
/// the corresponding goal and capturing `(get-value (x))`:
///
/// | goal | Z3 answer |
/// |---|---|
/// | `x² = 2` | `(root-obj (+ (^ x 2) (- 2)) 1)` |
/// | `x² = 2 ∧ x > 0` | `(root-obj (+ (^ x 2) (- 2)) 2)` |
/// | `2x² = 3` | `(root-obj (+ (* 2 (^ x 2)) (- 3)) 1)` |
/// | `x² + x = 1 ∧ x > 0` | `(root-obj (+ (^ x 2) x (- 1)) 2)` |
/// | `x² − 2x = 1 ∧ x > 0` | `(root-obj (+ (^ x 2) (* (- 2) x) (- 1)) 2)` |
/// | `x⁵ − 3x = 1` | `(root-obj (+ (^ x 5) (* (- 3) x) (- 1)) 1)` |
///
/// So: monomials in **descending** degree, zero coefficients omitted; degree
/// `k ≥ 2` spells `(^ x k)`, degree 1 spells bare `x`, degree 0 spells the
/// bare integer; a coefficient of exactly `1` is elided rather than written as
/// `(* 1 ..)`; a negative integer anywhere spells `(- n)`, since SMT-LIB has no
/// negative numerals.
///
/// The bound variable is **always** printed `x`, whatever the SMT constant is
/// called — confirmed by asking Z3 for a model of `yy² = 2`, which answers
/// `(root-obj (+ (^ x 2) (- 2)) 1)` for `yy`. `root-obj`'s first argument is a
/// closed polynomial term with its own binder, not an expression over the
/// model's constants.
fn render_root_obj(value: &AlgebraicValue) -> String {
    let mut monomials: Vec<String> = Vec::new();
    for (degree, coefficient) in value.coefficients.iter().enumerate().rev() {
        if coefficient.is_zero() {
            continue;
        }
        let power = match degree {
            0 => None,
            1 => Some("x".to_string()),
            k => Some(format!("(^ x {k})")),
        };
        monomials.push(match (power, coefficient.is_one()) {
            // Degree 0: the bare integer.
            (None, _) => render_integer(coefficient),
            // A unit coefficient is elided: `(^ x 2)`, never `(* 1 (^ x 2))`.
            (Some(power), true) => power,
            (Some(power), false) => format!("(* {} {power})", render_integer(coefficient)),
        });
    }
    // A `root-obj` polynomial always has at least two monomials in practice —
    // a single-monomial polynomial `c·xᵏ` has only the rational root 0, which
    // never reaches this type. Handle it anyway rather than emitting a
    // one-argument `(+ ..)`, which some readers reject.
    let polynomial = match monomials.len() {
        1 => monomials.join(""),
        _ => format!("(+ {})", monomials.join(" ")),
    };
    format!("(root-obj {polynomial} {})", value.root_index)
}

/// An integer as an SMT-LIB2 term: negatives need the `-` *operator*, since
/// the grammar has no negative numeral.
fn render_integer(value: &BigInt) -> String {
    if value.is_negative() {
        format!("(- {})", -value)
    } else {
        value.to_string()
    }
}

/// Constructor-expansion budget for [`Context::default_value`]'s datatype
/// arm.  Mirrors the solver-side ground-default budget; see
/// [`crate::solver::model_builder::ground_default_term`].
///
/// This is the *only* bound in the default-value walk, and it is deliberate:
///
/// - Array nesting needs no bound.  `Array` sorts intern bottom-up, so the
///   sort graph itself is acyclic and an array-only descent always
///   terminates; the walk is an explicit heap stack
///   ([`DefaultValueFrame`]), so depth cannot take the native stack down
///   either.  A cap here (there used to be one at 512) could only replace a
///   perfectly computable value with a wrong `?` placeholder.
/// - Datatype constructor expansion **must** be bounded, because the walk
///   steps through the datatype *definition* table and that edge can close
///   a genuine cycle: `(declare-datatype T ((c (f (Array Int T)))))` has no
///   finite ground value at all — every expansion of `c` needs a value of
///   `(Array Int T)`, which needs a value of `T` again.  (An earlier fix
///   threaded a shared budget through both halves of the cycle after the
///   original code reset its counter to 0 at every datatype re-entry and
///   recursed forever on `(get-model)`.)  When the budget is exhausted the
///   walk yields the honest placeholder `?` — visibly not a value — rather
///   than fabricating a wrong finite value for an ill-founded sort.
const DT_DEFAULT_EXPANSION_LIMIT: u32 = 16;

/// One suspended node of [`Context::default_value`]'s iterative walk.
///
/// The walk evaluates one `(sort, expansions)` task at a time; a compound
/// sort pushes a frame here, evaluates its first child, and each finished
/// child value is folded back into the innermost frame.
enum DefaultValueFrame {
    /// `((as const {sort_name}) {range_default})`, waiting for the range
    /// default.
    ArrayConst {
        /// The rendered name of the array sort itself.
        sort_name: String,
    },
    /// `({name} {field defaults...})`, collecting field defaults in order.
    Constructor {
        /// The constructor's name.
        name: String,
        /// Field default values collected so far.
        done: Vec<String>,
        /// All field sorts of the constructor.
        fields: Vec<SortId>,
        /// Index into `fields` of the next field to evaluate.
        next_field: usize,
        /// The constructor-expansion count charged to every field.
        expansions: u32,
    },
}

/// One suspended compound node of [`Context::format_sort_name`]'s iterative
/// walk: an `(Array dom rng)` or a `(Name arg...)` parametric application.
enum SortNameFrame {
    /// The domain is being rendered; the range is next.
    RangeAfter {
        /// The array sort this frame will finish.
        sort: SortId,
        /// The not-yet-rendered range sort.
        range: SortId,
    },
    /// The range is being rendered; the domain's text is finished.
    Finish {
        /// The array sort this frame will finish.
        sort: SortId,
        /// The rendered domain text.
        domain: String,
    },
    /// A parametric application `(name arg...)` collecting its arguments in
    /// order.
    Parametric {
        /// The parametric sort this frame will finish.
        sort: SortId,
        /// The already-resolved head name.
        name: String,
        /// Argument texts rendered so far.
        done: Vec<String>,
        /// All argument sorts of the application.
        args: Vec<SortId>,
        /// Index into `args` of the next argument to render.
        next: usize,
    },
}

impl Context {
    /// Get the model (if SAT)
    /// Returns a list of (name, sort, value) tuples
    pub fn get_model(&self) -> Option<Vec<(String, String, String)>> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }

        let mut model = Vec::new();
        // An empty assertion stack is satisfied by *every* assignment, so the
        // solver answers `sat` without ever building a model.  Completing from
        // the sort defaults is exact there, not a guess — with nothing asserted
        // no constraint exists that a completion could violate — whereas
        // reporting "no model available" for a `sat` verdict is simply wrong.
        // Any other missing model is a genuine extraction failure and stays
        // `None`, so a real assignment is never fabricated.
        //
        // A nonlinear-real model whose values are algebraic is the second
        // exception. There is no `Model` for it — `√2` has no term in the
        // rational term language — and yet the assignment is fully known: it
        // lives in `Solver::nl_algebraic_values`, which is populated only
        // all-or-nothing and only for a real cell decomposition's `Sat` (see
        // that field). So the gate opens on that map being non-empty, and on
        // nothing weaker: widening it to "assertions non-empty" would fabricate
        // a sort-default model for every other modelless `sat` in the tree.
        let empty_model;
        let solver_model = match self.solver.model() {
            Some(solver_model) => solver_model,
            None if self.assertions.is_empty() || !self.solver.nl_algebraic_values().is_empty() => {
                empty_model = crate::solver::Model::new();
                &empty_model
            }
            None => return None,
        };

        // Witness bookkeeping for unconstrained uninterpreted-sort constants:
        // `per_sort_next` is the next fresh witness index for a sort, and
        // `class_witness` maps an EUF congruence class (or a lone, never-equated
        // term) to its already-assigned index — so constants proven equal share
        // one witness while distinct constants get distinct ones.
        let mut per_sort_next: crate::prelude::HashMap<SortId, usize> =
            crate::prelude::HashMap::new();
        let mut class_witness: crate::prelude::HashMap<(SortId, u64), usize> =
            crate::prelude::HashMap::new();

        for decl in &self.declared_consts {
            // The exact-value side-channel is consulted first. It is only ever
            // populated when the ordinary `Model` is absent, so this cannot
            // shadow a real model entry today; consulting it first is what
            // keeps that true if one ever coexisted, because it is the more
            // precise of the two (a `Model` value for the same constant could
            // only be a rounded stand-in). Without this arm the sqrt2 goal
            // falls through to `default_value(Real)` and reports `0.0` — a
            // number satisfying none of its assertions.
            let value = if let Some(exact) = self.solver.nl_algebraic_value(decl.term) {
                render_nl_witness_value(exact)
            } else if let Some(val) = solver_model.get(decl.term) {
                self.format_value(val)
            } else if self.is_uninterpreted_sort(decl.sort) {
                // No direct model entry for an uninterpreted-sort constant:
                // synthesize a Z3-style `@uc_S_n` abstract witness.  Group by
                // EUF congruence class so equal constants share a witness;
                // never-equated constants key by their own term id (a disjoint
                // namespace via the high bit) so they stay distinct.  Always a
                // valid value, unlike the previous invalid `?`.
                let class_key: u64 = match self.solver.euf_class_representative(decl.term) {
                    Some(rep) => (1u64 << 32) | u64::from(rep),
                    None => u64::from(decl.term.0),
                };
                let idx = if let Some(&i) = class_witness.get(&(decl.sort, class_key)) {
                    i
                } else {
                    let next = per_sort_next.entry(decl.sort).or_insert(0);
                    let i = *next;
                    *next += 1;
                    class_witness.insert((decl.sort, class_key), i);
                    i
                };
                format!("@uc_{}_{}", self.format_sort_name(decl.sort), idx)
            } else {
                // Default value based on sort
                self.default_value(decl.sort)
            };
            let sort_name = self.format_sort_name(decl.sort);
            model.push((decl.name.clone(), sort_name, value));
        }

        Some(model)
    }

    /// Build a raw function interpretation for a declared uninterpreted function.
    ///
    /// Derives entries from the EUF congruence closure rather than from raw
    /// `Apply` terms alone.  For every application `f(a1, …, an)` interned in the
    /// E-graph, the arguments and the result are canonicalized through their
    /// equivalence-class representatives, so:
    ///
    /// - Two applications whose arguments are pairwise congruent (e.g. `f(a)` and
    ///   `f(b)` when `a = b` is implied by the assertions) collapse to a **single**
    ///   entry keyed by the shared argument class.
    /// - The reported argument and result strings are **model values** taken from
    ///   the class (resolving through the representative), not raw term ids.
    /// - When an application has no direct model value, the value of any congruent
    ///   member of its class is used.
    ///
    /// `else_value` is chosen as the most frequently occurring entry value (ties
    /// broken by first occurrence), mirroring how Z3 selects a default; if there
    /// are no entries it falls back to the return sort's default value.
    ///
    /// Returns `None` when:
    /// - the last check was not `Sat`, or
    /// - no model is available, or
    /// - `func_name` is not a declared function.
    ///
    /// The return type is `(entries, else_value_string, arity)` to avoid
    /// pulling `oxiz_core::model` types into this file.
    pub fn get_func_interp_raw(&self, func_name: &str) -> Option<RawFuncInterp> {
        if self.last_result != Some(SolverResult::Sat) {
            return None;
        }
        let solver_model = self.solver.model()?;

        // Find the declared function so we know its arity and default sort.
        let decl = self.declared_funs.iter().find(|d| d.name == func_name)?;
        let arity = decl.arg_sorts.len();
        let default_else = self.default_value(decl.ret_sort);

        // Resolve `func_name` to the EUF function-symbol id.  For an `Apply`
        // term the EUF id is the underlying value of the function-name `Spur`,
        // so we recover it from any matching application term (read-only — no
        // mutable interner access required).
        let mut func_id: Option<u32> = None;
        for idx in 0..(self.terms.len() as u32) {
            let tid = TermId(idx);
            let Some(term) = self.terms.get(tid) else {
                continue;
            };
            if let TermKind::Apply {
                func: func_spur, ..
            } = &term.kind
                && self.terms.resolve_str(*func_spur) == func_name
            {
                func_id = Some(func_spur.into_inner().get());
                break;
            }
        }

        // No application of this function exists in the E-graph: the function is
        // declared but never applied, so its interpretation is purely the default.
        let Some(func_id) = func_id else {
            return Some((Vec::new(), default_else, arity));
        };

        // Pull congruence-closed application entries from the EUF solver.  Each
        // entry already has its argument and result classes canonicalized, so
        // congruence (e.g. f(a) == f(b) when a == b) is applied for us.
        let euf_entries = self.solver.euf_function_entries(func_id);

        // Deduplicate on the canonical argument-class representative tuple so
        // congruent applications produce exactly one entry.  Because congruence
        // forces congruent applications into the same result class, the values
        // agree in a consistent model.
        let mut seen_arg_keys: crate::prelude::HashSet<smallvec::SmallVec<[u32; 4]>> =
            crate::prelude::HashSet::new();
        let mut entries: Vec<(Vec<String>, String)> = Vec::new();
        for entry in &euf_entries {
            // Resolve the result value first: skip applications whose class has
            // no concrete model value (an unconstrained application contributes
            // nothing observable beyond the else-branch).
            let Some(val_str) = self.class_value_string(&entry.result_class_terms, solver_model)
            else {
                continue;
            };

            if !seen_arg_keys.insert(entry.arg_reps.clone()) {
                continue; // already emitted this congruence class of arguments
            }

            // Resolve each argument to its canonical model value.  Falls back to
            // the default value for the corresponding argument sort when the
            // class carries no concrete value (rare: an unconstrained argument).
            let arg_strs: Vec<String> = entry
                .arg_class_terms
                .iter()
                .enumerate()
                .map(|(i, members)| {
                    self.class_value_string(members, solver_model)
                        .unwrap_or_else(|| {
                            decl.arg_sorts
                                .get(i)
                                .map_or_else(|| "?".to_string(), |&s| self.default_value(s))
                        })
                })
                .collect();
            entries.push((arg_strs, val_str));
        }

        // Pick `else_value`: the most common entry value (ties → first seen),
        // matching Z3's habit of reusing an existing value as the default.
        let else_value = Self::most_common_value(&entries).unwrap_or(default_else);

        Some((entries, else_value, arity))
    }

    /// Resolve an equivalence class (its member `TermId`s) to a formatted model
    /// value string, by finding the first member that carries either a direct
    /// model assignment or is itself a literal constant.
    ///
    /// Returns `None` when no member of the class has an observable value.
    fn class_value_string(
        &self,
        members: &[TermId],
        solver_model: &crate::solver::Model,
    ) -> Option<String> {
        for &member in members {
            // Direct model assignment (covers variables and applications whose
            // value was extracted from an equality constraint).
            if let Some(val_term) = solver_model.get(member) {
                return Some(self.format_value(val_term));
            }
            // The member may itself be a literal constant (e.g. the term `5` in
            // `f(a) = 5`), which has no separate model entry but is its own value.
            if let Some(term) = self.terms.get(member)
                && matches!(
                    term.kind,
                    TermKind::True
                        | TermKind::False
                        | TermKind::IntConst(_)
                        | TermKind::RealConst(_)
                        | TermKind::BitVecConst { .. }
                )
            {
                return Some(self.format_value(member));
            }
        }
        None
    }

    /// Choose the most frequently occurring value among the interpretation
    /// entries, breaking ties in favour of the earliest occurrence.  Returns
    /// `None` for an empty entry list.
    fn most_common_value(entries: &[(Vec<String>, String)]) -> Option<String> {
        let mut counts: crate::prelude::HashMap<&str, (usize, usize)> =
            crate::prelude::HashMap::new();
        for (order, (_, value)) in entries.iter().enumerate() {
            let slot = counts.entry(value.as_str()).or_insert((0, order));
            slot.0 += 1;
        }
        counts
            .into_iter()
            .max_by(|(_, (count_a, order_a)), (_, (count_b, order_b))| {
                // Higher count wins; on a tie the smaller insertion order wins,
                // so we reverse the order comparison.
                count_a.cmp(count_b).then_with(|| order_b.cmp(order_a))
            })
            .map(|(value, _)| value.to_string())
    }

    /// Format a sort ID to its SMT-LIB2 name.
    ///
    /// Handles every `SortKind` that [`Context::parse_sort_name`] can
    /// produce (its inverse), including compound `(Array ..)`/`(_
    /// BitVec ..)`/`(_ FloatingPoint ..)` forms and previously
    /// declared uninterpreted/datatype sorts by name, so
    /// `get-model`/`get-value` output reflects a declared constant's
    /// real sort instead of falling back to a generic placeholder.
    /// Sort parameters render as their name and parametric applications as
    /// `(Name arg...)`, matching [`oxiz_core::smtlib::SmtLibPrinter`]; both
    /// used to print as the literal string `Unknown`, which collapsed every
    /// such sort — however many and however different — onto one name.
    ///
    /// The walk over nested `Array`/parametric sorts is an explicit heap stack
    /// ([`SortNameFrame`]) — array-sort nesting is input-controlled (the
    /// builder API composes sorts in O(1) stack, and chained `define-sort`
    /// grows a sort by one level per command), so native recursion here was
    /// an unbounded stack risk with no error channel to cap it honestly.
    /// Sub-sorts reachable along more than one path are rendered once and
    /// memoized; the memo is restricted to genuinely shared sorts so that a
    /// deep unshared chain does not pay quadratic memory for cached copies
    /// of every suffix.  (The *output* still spells shared sorts out in
    /// full each time — SMT-LIB sort syntax has no sharing — so output
    /// size, not the walk, remains the inherent cost on heavily shared
    /// sorts.)
    fn format_sort_name(&self, sort: SortId) -> String {
        // Pre-pass over the sort DAG (visited-set bounded): find the sorts
        // reachable along more than one path, the only ones worth caching.
        let mut seen: crate::prelude::HashSet<SortId> = crate::prelude::HashSet::new();
        let mut shared: crate::prelude::HashSet<SortId> = crate::prelude::HashSet::new();
        let mut scan = vec![sort];
        while let Some(s) = scan.pop() {
            if !seen.insert(s) {
                shared.insert(s);
                continue;
            }
            if let Some(node) = self.terms.sorts.get(s) {
                // Only the compound kinds have sub-sorts to visit; the leaf
                // kinds are listed rather than swept up by a wildcard so a
                // future sort kind with children cannot silently skip the
                // pre-pass (and lose its memoization).
                match &node.kind {
                    SortKind::Array { domain, range } => {
                        scan.push(*domain);
                        scan.push(*range);
                    }
                    SortKind::Parametric { args, .. } => scan.extend(args.iter().copied()),
                    SortKind::Bool
                    | SortKind::Int
                    | SortKind::Real
                    | SortKind::String
                    | SortKind::BitVec(_)
                    | SortKind::FloatingPoint { .. }
                    | SortKind::RoundingMode
                    | SortKind::Uninterpreted(_)
                    | SortKind::Parameter(_)
                    | SortKind::Datatype(_) => {}
                }
            }
        }

        let mut memo: crate::prelude::HashMap<SortId, String> = crate::prelude::HashMap::new();
        let mut pending: Vec<SortNameFrame> = Vec::new();
        let mut current = sort;
        'render: loop {
            // Render `current`, descending through `Array` domains until a
            // leaf (or memoized) sort is reached.
            let mut text: String = loop {
                if let Some(hit) = memo.get(&current) {
                    break hit.clone();
                }
                let Some(s) = self.terms.sorts.get(current) else {
                    break "Unknown".to_string();
                };
                let leaf = match &s.kind {
                    SortKind::Array { domain, range } => {
                        pending.push(SortNameFrame::RangeAfter {
                            sort: current,
                            range: *range,
                        });
                        current = *domain;
                        continue;
                    }
                    SortKind::Bool => "Bool".to_string(),
                    SortKind::Int => "Int".to_string(),
                    SortKind::Real => "Real".to_string(),
                    SortKind::String => "String".to_string(),
                    SortKind::BitVec(w) => format!("(_ BitVec {w})"),
                    SortKind::FloatingPoint { eb, sb } => format!("(_ FloatingPoint {eb} {sb})"),
                    SortKind::RoundingMode => "RoundingMode".to_string(),
                    // An uninterpreted sort's name is interned by the *term*
                    // manager (`Parser::parse_sort` for `declare-sort` names
                    // and `TermManager::reglan_sort` both call
                    // `TermManager::intern_str`), so it resolves through
                    // `self.terms`.
                    SortKind::Uninterpreted(spur) => self.terms.resolve_str(*spur).to_string(),
                    // A datatype/parameter/parametric name, by contrast, is
                    // interned by the *sort* manager's own `Rodeo`
                    // (`mk_datatype_sort` / `declare_datatype`,
                    // `mk_sort_parameter`, `declare_parametric_sort` /
                    // `instantiate_parametric_sort` / `define_parametric_sort`
                    // all go through `SortManager::interner`).  The two
                    // interners are separate, so a key from one resolved
                    // through the other yields an unrelated string or indexes
                    // out of range; each name below therefore goes through
                    // `self.terms.sorts.resolve_spur`.  Mirrors
                    // `SmtLibPrinter::write_sort`, where the same hazard is
                    // documented.
                    SortKind::Datatype(spur) => {
                        self.terms.sorts.datatype_name(current).map_or_else(
                            || self.terms.sorts.resolve_spur(*spur).to_string(),
                            ToString::to_string,
                        )
                    }
                    SortKind::Parameter(spur) => self.terms.sorts.resolve_spur(*spur).to_string(),
                    // Rendered exactly as the printer spells it: `(Name
                    // arg...)`, arguments rendered through this same walk.
                    // Printing every parametric sort as the single literal
                    // "Unknown" used to collapse distinct sorts -- `(List
                    // Int)` and a bare parameter `T` alike -- onto one
                    // meaningless name in `(get-model)` output.
                    SortKind::Parametric { name, args } => {
                        let head = self.terms.sorts.resolve_spur(*name).to_string();
                        let args: Vec<SortId> = args.iter().copied().collect();
                        match args.first().copied() {
                            // A nullary application has no arguments to walk:
                            // `(Name)`, as `write_sort` prints it.
                            None => format!("({head})"),
                            Some(first) => {
                                pending.push(SortNameFrame::Parametric {
                                    sort: current,
                                    name: head,
                                    done: Vec::with_capacity(args.len()),
                                    args,
                                    next: 1,
                                });
                                current = first;
                                continue;
                            }
                        }
                    }
                };
                if shared.contains(&current) {
                    memo.insert(current, leaf.clone());
                }
                break leaf;
            };
            // Fold the rendered text upward: a finished domain schedules its
            // partner range; a finished range completes its `(Array ..)`.
            loop {
                match pending.pop() {
                    None => return text,
                    Some(SortNameFrame::RangeAfter { sort, range }) => {
                        pending.push(SortNameFrame::Finish { sort, domain: text });
                        current = range;
                        continue 'render;
                    }
                    Some(SortNameFrame::Finish { sort, domain }) => {
                        text = format!("(Array {domain} {text})");
                        if shared.contains(&sort) {
                            memo.insert(sort, text.clone());
                        }
                    }
                    Some(SortNameFrame::Parametric {
                        sort,
                        name,
                        mut done,
                        args,
                        next,
                    }) => {
                        done.push(text);
                        match args.get(next).copied() {
                            Some(arg) => {
                                pending.push(SortNameFrame::Parametric {
                                    sort,
                                    name,
                                    done,
                                    args,
                                    next: next + 1,
                                });
                                current = arg;
                                continue 'render;
                            }
                            None => {
                                text = format!("({name} {})", done.join(" "));
                                if shared.contains(&sort) {
                                    memo.insert(sort, text.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Whether `sort` is an uninterpreted (user-declared) sort.
    fn is_uninterpreted_sort(&self, sort: SortId) -> bool {
        self.terms
            .sorts
            .get(sort)
            .is_some_and(|s| matches!(s.kind, SortKind::Uninterpreted(_)))
    }

    /// Format a model value
    fn format_value(&self, term: TermId) -> String {
        match self.terms.get(term).map(|t| &t.kind) {
            Some(TermKind::True) => "true".to_string(),
            Some(TermKind::False) => "false".to_string(),
            Some(TermKind::IntConst(n)) => n.to_string(),
            Some(TermKind::RealConst(r)) => {
                if *r.denom() == 1 {
                    format!("{}.0", r.numer())
                } else {
                    format!("(/ {} {})", r.numer(), r.denom())
                }
            }
            Some(TermKind::BitVecConst { value, width }) => {
                format!(
                    "#b{:0>width$}",
                    format!("{:b}", value),
                    width = *width as usize
                )
            }
            // Floating-point constants, array store/const-array chains, string
            // literals and datatype constructor applications are structured
            // values that the shared SMT-LIB printer renders faithfully
            // (`(fp ..)`, `(_ +zero eb sb)`, `(store ..)`, `"..."`,
            // `(cons 7 nil)`), so delegate rather than emitting an invalid `?`
            // placeholder.
            //
            // Delegating matters beyond brevity for the datatype case:
            // `(get-value ..)` renders through the very same printer (see
            // [`Context::format_get_value`]), so routing `(get-model)` through
            // it is what keeps the two commands character-for-character in
            // agreement about a reconstructed datatype value.
            Some(
                TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. }
                | TermKind::Store(..)
                | TermKind::StringLit(_)
                | TermKind::DtConstructor { .. },
            ) => {
                let printer = oxiz_core::smtlib::Printer::new(&self.terms);
                printer.print_term(term)
            }
            // A rounding mode is a nullary `Var` interned at the reserved
            // `RoundingMode` sort under its canonical long name, so the name
            // *is* the value.  Without this arm every solved rounding mode
            // reported as the invalid `?` placeholder, since `Var` otherwise
            // falls through to the catch-all below.
            Some(TermKind::Var(spur)) if self.is_rounding_mode_term(term) => {
                self.terms.resolve_str(*spur).to_string()
            }
            _ => "?".to_string(),
        }
    }

    /// Whether `term` is interned at the reserved `RoundingMode` sort.
    fn is_rounding_mode_term(&self, term: TermId) -> bool {
        self.terms
            .get(term)
            .and_then(|t| self.terms.sorts.get(t.sort))
            .is_some_and(oxiz_core::sort::Sort::is_rounding_mode)
    }

    /// Get a default value for a sort.
    ///
    /// Used to complete `get-model` output for a declared constant that the
    /// model left unconstrained.  Every case yields a *valid* SMT-LIB value of
    /// the sort (never the invalid `?` placeholder), except uninhabited or
    /// ill-founded datatypes, which have no ground value at all.
    ///
    /// The walk is an explicit heap stack ([`DefaultValueFrame`]), so array
    /// nesting of any depth is rendered in full — the pre-conversion code
    /// both recursed natively and truncated every value below 512 sort
    /// levels to a wrong `?`.  The datatype arm expands constructors:
    ///
    /// - `nil`-style nullary constructors render as their bare name;
    /// - compound constructors render as an application of the field
    ///   defaults (`(mk-pair 0 0)`), the constructor chosen by the shared
    ///   [`default_constructor_index`](crate::solver::model_builder::default_constructor_index)
    ///   policy — the first one with no datatype-sorted field, so a
    ///   well-founded construction bottoms out immediately, matching the
    ///   solver's own reconstruction of an underdetermined datatype term;
    /// - expansion is bounded by [`DT_DEFAULT_EXPANSION_LIMIT`], the one
    ///   *deliberate* bound in this walk: an ill-founded declaration such
    ///   as `(declare-datatype T ((c (f T))))`, or one that closes the
    ///   cycle through an array field as in
    ///   `(declare-datatype D ((c (f (Array Int D)))))`, has no finite
    ///   ground value at all, and yields the honest `?` placeholder rather
    ///   than a non-terminating expansion (see the limit's doc comment).
    fn default_value(&self, sort: SortId) -> String {
        let mut pending: Vec<DefaultValueFrame> = Vec::new();
        // The task being evaluated: a sort, and the number of datatype
        // constructor expansions already charged on the path to it.
        let mut task: (SortId, u32) = (sort, 0);
        'task: loop {
            // Evaluate `task` down to a leaf value, pushing a frame for
            // each compound sort passed through on the way.
            let mut value: String = loop {
                let (sort, expansions) = task;
                let Some(s) = self.terms.sorts.get(sort) else {
                    break "?".to_string();
                };
                match &s.kind {
                    SortKind::Bool => break "false".to_string(),
                    SortKind::Int => break "0".to_string(),
                    SortKind::Real => break "0.0".to_string(),
                    // The empty string is the canonical ground `String`
                    // value; the old `?` fallback was not valid SMT-LIB
                    // output at all.
                    SortKind::String => break "\"\"".to_string(),
                    SortKind::BitVec(w) => {
                        break format!("#b{:0>width$}", "0", width = *w as usize);
                    }
                    // Positive zero is a canonical, valid ground FP value.
                    SortKind::FloatingPoint { eb, sb } => break format!("(_ +zero {eb} {sb})"),
                    // The five-element `RoundingMode` sort defaults to IEEE
                    // 754's own default mode, spelled the long way exactly as
                    // Z3 prints it.  Emphatically *not* the `@uc_..._0`
                    // witness of the uninterpreted arm below: the sort has
                    // named inhabitants, and a witness would be neither valid
                    // SMT-LIB input nor equal to any mode.
                    SortKind::RoundingMode => {
                        break oxiz_core::ast::TermManager::rounding_mode_name(
                            oxiz_core::ast::RoundingMode::RNE,
                        )
                        .to_string();
                    }
                    // A constant array whose every entry is the range's
                    // default value.  Descending into the range charges no
                    // constructor expansion — the sort graph alone is
                    // acyclic, so this edge cannot close a cycle.
                    SortKind::Array { range, .. } => {
                        let range = *range;
                        let sort_name = self.format_sort_name(sort);
                        pending.push(DefaultValueFrame::ArrayConst { sort_name });
                        task = (range, expansions);
                    }
                    // A datatype default is a ground constructor
                    // application built from the field sorts' own
                    // defaults.  Every field charges one constructor
                    // expansion, whatever its sort: the pre-fix code
                    // charged nothing for non-datatype fields and
                    // re-entered the walk at depth zero, which is exactly
                    // the reset that let an array-mediated cycle run
                    // forever.
                    SortKind::Datatype(_) => {
                        if expansions >= DT_DEFAULT_EXPANSION_LIMIT {
                            break "?".to_string();
                        }
                        let Some(dt_name) = self.terms.sorts.datatype_name(sort) else {
                            break "?".to_string();
                        };
                        let dt_name = dt_name.to_string();
                        let Some(def) = self.terms.sorts.get_datatype(&dt_name) else {
                            break "?".to_string();
                        };
                        let Some(index) = crate::solver::model_builder::default_constructor_index(
                            def,
                            &self.terms.sorts,
                        ) else {
                            break "?".to_string();
                        };
                        let Some(constructor) = def.constructors.get(index) else {
                            break "?".to_string();
                        };
                        let name = self.terms.resolve_str(constructor.name).to_string();
                        let fields: Vec<SortId> = constructor
                            .selectors
                            .iter()
                            .map(|&(_, field_sort)| field_sort)
                            .collect();
                        // A nullary constructor is already its own ground
                        // value.  (`fields.first()` returning `None` is the
                        // same statement as `selectors.is_empty()`; matching
                        // on it keeps the non-empty arm unwrap-free.)
                        let charged = expansions.saturating_add(1);
                        match fields.first().copied() {
                            None => break name,
                            Some(first) => {
                                let arity = fields.len();
                                pending.push(DefaultValueFrame::Constructor {
                                    name,
                                    done: Vec::with_capacity(arity),
                                    fields,
                                    next_field: 1,
                                    expansions: charged,
                                });
                                task = (first, charged);
                            }
                        }
                    }
                    // Uninterpreted-sort defaults are abstract witnesses.
                    // `get_model` assigns a distinct per-constant index; as
                    // a standalone fallback emit the zero-th witness for
                    // the sort.
                    SortKind::Uninterpreted(_) => {
                        break format!("@uc_{}_0", self.format_sort_name(sort));
                    }
                    // Sort parameters and unapplied parametric sorts have
                    // no ground values.
                    SortKind::Parameter(_) | SortKind::Parametric { .. } => {
                        break "?".to_string();
                    }
                }
            };
            // Fold the finished value into the innermost pending frame; a
            // constructor frame with fields left schedules the next one.
            loop {
                match pending.pop() {
                    None => return value,
                    Some(DefaultValueFrame::ArrayConst { sort_name }) => {
                        value = format!("((as const {sort_name}) {value})");
                    }
                    Some(DefaultValueFrame::Constructor {
                        name,
                        mut done,
                        fields,
                        next_field,
                        expansions,
                    }) => {
                        done.push(value);
                        match fields.get(next_field).copied() {
                            Some(field_sort) => {
                                pending.push(DefaultValueFrame::Constructor {
                                    name,
                                    done,
                                    fields,
                                    next_field: next_field + 1,
                                    expansions,
                                });
                                task = (field_sort, expansions);
                                continue 'task;
                            }
                            None => value = format!("({} {})", name, done.join(" ")),
                        }
                    }
                }
            }
        }
    }

    /// A ground *term* carrying the same default value that
    /// [`Context::default_value`] renders as a string, or `None` for sorts with
    /// no constructible ground witness (uninterpreted and array sorts, sort
    /// parameters, and ill-founded datatypes).
    ///
    /// Used to complete the model before a `(get-value ...)` evaluation, so a
    /// query over an unconstrained constant — including inside a compound term
    /// such as `(+ x 1)` — reduces to a real value instead of echoing itself.
    ///
    /// Delegates to the solver's
    /// [`ground_default_term`](crate::solver::model_builder::ground_default_term),
    /// which is also what fills in an unconstrained *field* of a reconstructed
    /// datatype value — one definition of "the default of this sort" rather
    /// than one per caller.
    fn default_value_term(&mut self, sort: SortId) -> Option<TermId> {
        crate::solver::model_builder::ground_default_term(&mut self.terms, sort)
    }

    /// The value string [`Context::get_model`] reports for `term`, when `term`
    /// is a declared constant that the model left unassigned.
    ///
    /// `(get-value ...)` and `(get-model)` must never disagree about the same
    /// constant, and `get_model`'s uninterpreted-sort witnesses (`@uc_S_n`) are
    /// numbered across the whole declaration list — so the answer is read back
    /// out of `get_model` itself rather than recomputed.
    fn unassigned_const_value(&self, term: TermId, model: &crate::solver::Model) -> Option<String> {
        let index = self.declared_consts.iter().position(|d| d.term == term)?;
        // A constant the model *did* assign keeps the ordinary evaluation path.
        if model.get(term).is_some() {
            return None;
        }
        let (_, _, value) = self.get_model()?.into_iter().nth(index)?;
        Some(value)
    }

    /// Answer a `(get-value (t1 .. tn))` request.
    ///
    /// SMT-LIB 2.6 §4.1.1: the command is available only in `sat` mode, so a
    /// missing/superseded check result is reported as an error rather than
    /// answered from stale state.  Each term is evaluated in the current model,
    /// which is first *completed* with the sort defaults `get_model` reports for
    /// unconstrained declared constants — otherwise `Model::eval` returns an
    /// unassigned constant unchanged and `(get-value (x))` answered `((x x))`,
    /// echoing the term instead of producing a value.
    pub(super) fn format_get_value(&mut self, terms: &[TermId]) -> String {
        const NO_MODEL: &str = "(error \"No model available\")";
        if self.last_result != Some(SolverResult::Sat) {
            return NO_MODEL.to_string();
        }
        // Owned so the evaluation below can borrow `self.terms` mutably; see
        // `get_model` for why an empty assertion stack — or a populated
        // algebraic side-channel — yields an empty model rather than an error.
        let model = match self.solver.model() {
            Some(model) => model.clone(),
            None if self.assertions.is_empty() || !self.solver.nl_algebraic_values().is_empty() => {
                crate::solver::Model::new()
            }
            None => return NO_MODEL.to_string(),
        };

        // Completion substitution: every declared constant with no model entry
        // maps to its sort default.
        //
        // A constant the algebraic side-channel *does* pin is excluded. Its
        // sort default is `0.0`, and substituting that would answer a compound
        // query like `(get-value ((* x x)))` with `0.0` for a goal whose
        // witness is `√2` — a fabricated value, and one contradicting the `2.0`
        // that the very same model implies. Left out of the map the term
        // survives evaluation unreduced and echoes back, which is the same
        // honest non-answer this path already gives for anything else it
        // cannot fold. (A bare `(get-value (x))` never reaches the completion
        // at all: `unassigned_const_value` answers it from `get_model` below,
        // which is where the `root-obj` rendering lives.)
        let unassigned: Vec<(TermId, SortId)> = self
            .declared_consts
            .iter()
            .filter(|d| model.get(d.term).is_none())
            .filter(|d| self.solver.nl_algebraic_value(d.term).is_none())
            .map(|d| (d.term, d.sort))
            .collect();
        let mut completion: crate::prelude::FxHashMap<TermId, TermId> =
            crate::prelude::FxHashMap::default();
        for (term, sort) in unassigned {
            if let Some(value) = self.default_value_term(sort) {
                completion.insert(term, value);
            }
        }

        let mut values = Vec::with_capacity(terms.len());
        for &term in terms {
            let value_str = if let Some(value) = self.unassigned_const_value(term, &model) {
                // A bare unconstrained constant: report exactly what
                // `(get-model)` reports for it, witnesses included.
                value
            } else {
                let completed = if completion.is_empty() {
                    term
                } else {
                    self.terms.substitute(term, &completion)
                };
                // `Model::eval` substitutes and folds the Boolean structure but
                // leaves arithmetic/bit-vector applications of the substituted
                // constants unreduced (`(+ x 1)` → `(+ 0 1)`), so run the
                // rewriter over the result to reach an actual value.
                let value = model.eval(completed, &mut self.terms);
                let value = self.terms.simplify(value);
                oxiz_core::smtlib::Printer::new(&self.terms).print_term(value)
            };
            let term_str = oxiz_core::smtlib::Printer::new(&self.terms).print_term(term);
            values.push(format!("({} {})", term_str, value_str));
        }
        format!("({})", values.join("\n "))
    }

    /// Format the model as SMT-LIB2
    ///
    /// Recursive definitions in scope are echoed back as `define-fun-rec`
    /// entries. A model that omitted them would be unusable: the constants it
    /// does report are only meaningful together with the definitions that
    /// constrain them, and a reader replaying the model would see applications
    /// of a symbol with no interpretation at all.
    pub fn format_model(&self) -> String {
        let recfun_lines = self.recfun_model_lines();
        match self.get_model() {
            None => "(error \"No model available\")".to_string(),
            Some(model) if model.is_empty() && recfun_lines.is_empty() => "(model)".to_string(),
            Some(model) => {
                let mut lines = vec!["(model".to_string()];
                for (name, sort, value) in model {
                    lines.push(format!("  (define-fun {} () {} {})", name, sort, value));
                }
                lines.extend(recfun_lines);
                lines.push(")".to_string());
                lines.join("\n")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== root-obj rendering, pinned against z3 4.15.4 ==================
    //
    // Every expected string below was captured by running the named goal
    // through `z3 4.15.4` and reading its `(get-value (x))` answer, so these
    // are conformance pins rather than pins on what this code happens to do.

    fn algebraic(coefficients: &[i64], root_index: u32) -> NlWitnessValue {
        NlWitnessValue::Algebraic(AlgebraicValue {
            coefficients: coefficients.iter().copied().map(BigInt::from).collect(),
            root_index,
            // The bracket is diagnostics-only and never appears in the output;
            // a placeholder is honest here precisely because nothing reads it.
            lower: num_rational::BigRational::from_integer(0.into()),
            upper: num_rational::BigRational::from_integer(0.into()),
        })
    }

    /// `x² = 2` → `(root-obj (+ (^ x 2) (- 2)) 1)`; with `x > 0`, index 2.
    #[test]
    fn root_obj_pins_the_sqrt2_spelling() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-2, 0, 1], 1)),
            "(root-obj (+ (^ x 2) (- 2)) 1)"
        );
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-2, 0, 1], 2)),
            "(root-obj (+ (^ x 2) (- 2)) 2)"
        );
    }

    /// `2x² = 3` → a non-unit leading coefficient is a `*` application.
    #[test]
    fn root_obj_pins_a_non_unit_leading_coefficient() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-3, 0, 2], 1)),
            "(root-obj (+ (* 2 (^ x 2)) (- 3)) 1)"
        );
    }

    /// `x² + x = 1 ∧ x > 0` → a unit degree-1 term is the bare variable.
    #[test]
    fn root_obj_pins_a_bare_linear_term() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-1, 1, 1], 2)),
            "(root-obj (+ (^ x 2) x (- 1)) 2)"
        );
    }

    /// `x² − 2x = 1 ∧ x > 0` → a negative coefficient inside a `*`.
    #[test]
    fn root_obj_pins_a_negative_linear_coefficient() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-1, -2, 1], 2)),
            "(root-obj (+ (^ x 2) (* (- 2) x) (- 1)) 2)"
        );
    }

    /// `x⁵ − 3x = 1` → interior zero coefficients are omitted, and the
    /// surviving monomials stay in descending degree order.
    #[test]
    fn root_obj_pins_omitted_zero_coefficients() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-1, -3, 0, 0, 0, 1], 1)),
            "(root-obj (+ (^ x 5) (* (- 3) x) (- 1)) 1)"
        );
    }

    /// `x³ = 2` and `x⁴ = 5`: higher powers keep the `(^ x k)` spelling.
    #[test]
    fn root_obj_pins_higher_degrees() {
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-2, 0, 0, 1], 1)),
            "(root-obj (+ (^ x 3) (- 2)) 1)"
        );
        assert_eq!(
            render_nl_witness_value(&algebraic(&[-5, 0, 0, 0, 1], 1)),
            "(root-obj (+ (^ x 4) (- 5)) 1)"
        );
    }

    /// The rational arm of the side-channel must render a value *identically*
    /// to [`Context::format_value`]'s `RealConst` arm.
    ///
    /// A mixed model is reported partly through each — `x` from the
    /// side-channel, `y` from `Model` — so a divergence would print two
    /// spellings of `Real` inside one `(get-model)`. This compares the two
    /// renderers against **each other** rather than against a copied
    /// expectation, so a later change to either one is caught even if both
    /// were changed to agree on something new.
    #[test]
    fn the_rational_arm_agrees_with_format_value_itself() {
        let mut ctx = Context::new();
        for (numerator, denominator) in [
            (3_i64, 1_i64),
            (-3, 1),
            (0, 1),
            (1, 2),
            (-1, 2),
            (7, 3),
            (-22, 7),
        ] {
            let term = ctx
                .terms
                .mk_real(num_rational::Rational64::new(numerator, denominator));
            let through_model = ctx.format_value(term);
            let through_side_channel = render_nl_witness_value(&NlWitnessValue::Rational(
                num_rational::BigRational::new(numerator.into(), denominator.into()),
            ));
            assert_eq!(
                through_side_channel, through_model,
                "the two renderings of {numerator}/{denominator} must not diverge"
            );
        }
    }

    // ===== default_value semantic pins ==================================
    //
    // Every expected string below was captured from the pre-conversion
    // recursive implementation, so these tests prove the explicit-stack
    // conversion is behavior-preserving.

    /// Atomic sorts keep their canonical ground defaults.
    #[test]
    fn test_default_value_atomic_pins() {
        let mut ctx = Context::new();
        let bool_sort = ctx.terms.sorts.bool_sort;
        let int_sort = ctx.terms.sorts.int_sort;
        let real_sort = ctx.terms.sorts.real_sort;
        let string_sort = ctx.terms.sorts.string_sort();
        let bv8 = ctx.terms.sorts.bitvec(8);
        let f32_sort = ctx.terms.sorts.float_sort(8, 24);
        assert_eq!(ctx.default_value(bool_sort), "false");
        assert_eq!(ctx.default_value(int_sort), "0");
        assert_eq!(ctx.default_value(real_sort), "0.0");
        assert_eq!(ctx.default_value(string_sort), "\"\"");
        assert_eq!(ctx.default_value(bv8), "#b00000000");
        assert_eq!(ctx.default_value(f32_sort), "(_ +zero 8 24)");

        let spur = ctx.terms.intern_str("Widget");
        let widget = ctx
            .terms
            .sorts
            .intern(oxiz_core::sort::SortKind::Uninterpreted(spur));
        assert_eq!(ctx.default_value(widget), "@uc_Widget_0");
    }

    /// Array defaults are const-arrays over the range default, spelled
    /// exactly as before the conversion.
    #[test]
    fn test_default_value_array_pins() {
        let mut ctx = Context::new();
        let int_sort = ctx.terms.sorts.int_sort;
        let a1 = ctx.terms.sorts.array(int_sort, int_sort);
        let a2 = ctx.terms.sorts.array(int_sort, a1);
        assert_eq!(ctx.default_value(a1), "((as const (Array Int Int)) 0)");
        assert_eq!(
            ctx.default_value(a2),
            "((as const (Array Int (Array Int Int))) ((as const (Array Int Int)) 0))"
        );
    }

    /// Well-founded datatype defaults: a compound constructor renders as an
    /// application of the field defaults; a nullary constructor renders as
    /// its bare name.
    #[test]
    fn test_default_value_datatype_pins() {
        let mut ctx = Context::new();
        ctx.execute_script("(declare-datatype Pair ((mk-pair (first Int) (second Int))))")
            .expect("datatype declaration script");
        let pair = ctx.terms.sorts.mk_datatype_sort("Pair");
        assert_eq!(ctx.default_value(pair), "(mk-pair 0 0)");

        let mut ctx2 = Context::new();
        ctx2.execute_script("(declare-datatype Color ((red) (green)))")
            .expect("datatype declaration script");
        let color = ctx2.terms.sorts.mk_datatype_sort("Color");
        assert_eq!(ctx2.default_value(color), "red");
    }

    /// The `(get-model)` infinite-recursion repro: a datatype whose only
    /// constructor closes a cycle through an array field,
    /// `(declare-datatype T ((c (f (Array Int T)))))`.
    ///
    /// The declaration IS accepted (verified against the working tree), and
    /// the sort has no finite ground value, so the default expands the
    /// constructor exactly [`DT_DEFAULT_EXPANSION_LIMIT`] times and bottoms
    /// out in the honest `?` placeholder.  The original code reset its
    /// depth counter to 0 at every re-entry and recursed forever; the
    /// budgeted fix produced exactly this value, and the explicit-stack
    /// conversion must keep producing it.
    #[test]
    fn test_default_value_ill_founded_array_cycle_pins() {
        let mut ctx = Context::new();
        ctx.execute_script("(declare-datatype T ((c (f (Array Int T)))))")
            .expect("self-referential array-field datatype is accepted");
        assert!(ctx.terms.sorts.is_datatype_declared("T"));
        let t_sort = ctx.terms.sorts.mk_datatype_sort("T");

        let mut expected = "?".to_string();
        for _ in 0..DT_DEFAULT_EXPANSION_LIMIT {
            expected = format!("(c ((as const (Array Int T)) {expected}))");
        }
        assert_eq!(ctx.default_value(t_sort), expected);
    }

    /// A *directly* self-referential ill-founded datatype
    /// (`(declare-datatype W ((c (f W))))`) also terminates with the honest
    /// placeholder at the same budget.
    #[test]
    fn test_default_value_ill_founded_direct_cycle_terminates() {
        let mut ctx = Context::new();
        ctx.execute_script("(declare-datatype W ((c (f W))))")
            .expect("self-referential datatype is accepted");
        let w_sort = ctx.terms.sorts.mk_datatype_sort("W");

        let mut expected = "?".to_string();
        for _ in 0..DT_DEFAULT_EXPANSION_LIMIT {
            expected = format!("(c {expected})");
        }
        assert_eq!(ctx.default_value(w_sort), expected);
    }

    /// The full script-level repro from the audit terminates and reports a
    /// model for the declared constant.
    #[test]
    fn test_get_model_ill_founded_datatype_script_terminates() {
        let mut ctx = Context::new();
        let out = ctx
            .execute_script(
                "(declare-datatype T ((c (f (Array Int T)))))\n\
                 (declare-const a T)\n\
                 (check-sat)\n\
                 (get-model)",
            )
            .expect("repro script executes");
        assert_eq!(out.first().map(String::as_str), Some("sat"));
        let model = out.get(1).map(String::as_str).unwrap_or_default();
        assert!(
            model.contains("(define-fun a () T "),
            "model must report the declared constant: {model}"
        );
    }

    /// Beyond the removed 512-level cap the walk now renders the true
    /// value: the pre-conversion code answered a wrong `?` for every array
    /// sort nested deeper than 512.
    ///
    /// Depth is 2000 rather than the usual 50k-100k deep-nesting depth
    /// because the *output* of `default_value` is inherently quadratic in
    /// array depth — every level embeds its own full sort name — so 100k
    /// levels would be a multi-gigabyte string; output construction, not
    /// the walk, is the bottleneck.  The walk machinery itself is proven at
    /// 12 500 levels on a 128 KiB stack by the `format_sort_name` tests
    /// below, which share the same explicit-stack shape with linear output
    /// and pin the same ~10 bytes-per-frame threshold a 1 MiB / 100k pair
    /// would.  This test's own stack stays at 1 MiB: its depth is 2000, well
    /// under the scaling threshold, and nothing here is quadratic in stack.
    #[test]
    fn test_default_value_deep_array_chain_beyond_old_cap() {
        const DEPTH: usize = 2000;
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — depth is
        // 2000 (sub-10,000), well under the scaling threshold and nothing
        // here is quadratic in stack. See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut ctx = Context::new();
                let int_sort = ctx.terms.sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..DEPTH {
                    sort = ctx.terms.sorts.array(int_sort, sort);
                }
                let value = ctx.default_value(sort);
                assert!(
                    !value.contains('?'),
                    "a finite array sort must never render the `?` placeholder"
                );
                assert_eq!(value.matches("((as const ").count(), DEPTH);
                assert!(value.starts_with("((as const (Array Int "));
                // The innermost value is the Int default `0` wrapped by one
                // closing paren per level.
                let expected_suffix = format!(" 0{}", ")".repeat(DEPTH));
                assert!(
                    value.ends_with(&expected_suffix),
                    "value must bottom out in the Int default"
                );
            })
            .expect("spawn deep default-value thread");
        handle.join().expect("deep default-value must not overflow");
    }

    // ===== format_sort_name =============================================

    /// Leaf and compound sort names, pinned against the pre-conversion
    /// recursive renderer.
    #[test]
    fn test_format_sort_name_pins() {
        let mut ctx = Context::new();
        let int_sort = ctx.terms.sorts.int_sort;
        let bool_sort = ctx.terms.sorts.bool_sort;
        let bv8 = ctx.terms.sorts.bitvec(8);
        let f32_sort = ctx.terms.sorts.float_sort(8, 24);
        let arr = ctx.terms.sorts.array(int_sort, bv8);
        let nested = ctx.terms.sorts.array(arr, bool_sort);
        assert_eq!(ctx.format_sort_name(int_sort), "Int");
        assert_eq!(ctx.format_sort_name(bv8), "(_ BitVec 8)");
        assert_eq!(ctx.format_sort_name(f32_sort), "(_ FloatingPoint 8 24)");
        assert_eq!(ctx.format_sort_name(arr), "(Array Int (_ BitVec 8))");
        assert_eq!(
            ctx.format_sort_name(nested),
            "(Array (Array Int (_ BitVec 8)) Bool)"
        );

        ctx.execute_script("(declare-datatype Pair ((mk-pair (first Int) (second Int))))")
            .expect("datatype declaration script");
        let pair = ctx.terms.sorts.mk_datatype_sort("Pair");
        assert_eq!(ctx.format_sort_name(pair), "Pair");
    }

    /// Deep-nesting regression: a 12 500-level range-nested array sort (built
    /// through the builder API in O(1) stack) renders on a 128 KiB thread
    /// stack.  The pre-conversion renderer recursed natively per level, so
    /// returning at all is the proof; the exact length pins correctness
    /// (`len("(Array Int ") + len(")") = 12` added bytes per level).
    #[test]
    fn test_format_sort_name_deep_range_chain_small_stack() {
        // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
        // ~10 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 12_500;
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut ctx = Context::new();
                let int_sort = ctx.terms.sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..DEPTH {
                    sort = ctx.terms.sorts.array(int_sort, sort);
                }
                let name = ctx.format_sort_name(sort);
                assert_eq!(name.len(), 12 * DEPTH + 3);
                assert!(name.starts_with("(Array Int (Array Int "));
                // The innermost sort is `Int`, closed by one paren per level.
                let suffix = format!("Int{}", ")".repeat(DEPTH));
                assert!(name.ends_with(&suffix));
            })
            .expect("spawn deep-format thread");
        handle.join().expect("deep-format thread must not overflow");
    }

    /// Same regression with the nesting on the domain side, which exercises
    /// the domain-first descent frames.
    #[test]
    fn test_format_sort_name_deep_domain_chain_small_stack() {
        // Stack and depth scale together (1 MiB/100k -> 128 KiB/12.5k): the
        // ~10 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 12_500;
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut ctx = Context::new();
                let int_sort = ctx.terms.sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..DEPTH {
                    sort = ctx.terms.sorts.array(sort, int_sort);
                }
                let name = ctx.format_sort_name(sort);
                assert_eq!(name.len(), 12 * DEPTH + 3);
                assert!(name.starts_with("(Array (Array "));
                assert!(name.ends_with(" Int) Int)"));
            })
            .expect("spawn deep-format thread");
        handle.join().expect("deep-format thread must not overflow");
    }

    /// Shared-DAG regression for the memo: a doubling array sort
    /// (`s[i+1] = (Array s[i] s[i])`) is rendered once per *distinct* sort,
    /// not once per path.
    ///
    /// Depth is 20 rather than the usual 50-60 doubling levels because the
    /// *output* is inherently exponential in doubling depth — SMT-LIB sort
    /// syntax has no sharing, so the rendered text of level `n` has length
    /// `12 * 2^n - 9` and 60 levels would be an exabyte-scale string no
    /// algorithm could materialize.  At depth 20 the test pins the exact
    /// (12.6 MB) length and must complete quickly.
    #[test]
    fn test_format_sort_name_shared_doubling_dag() {
        const LEVELS: u32 = 20;
        let mut ctx = Context::new();
        let int_sort = ctx.terms.sorts.int_sort;
        let mut sort = int_sort;
        for _ in 0..LEVELS {
            sort = ctx.terms.sorts.array(sort, sort);
        }
        let name = ctx.format_sort_name(sort);
        // len(n) = 2 * len(n-1) + len("(Array ") + len(" ") + len(")")
        //        = 2 * len(n-1) + 9, len(0) = 3  =>  len(n) = 12 * 2^n - 9.
        let expected_len = 12usize * (1usize << LEVELS) - 9;
        assert_eq!(name.len(), expected_len);
        assert!(name.starts_with("(Array (Array "));
    }

    // ===== get_model through the public surface =========================

    /// An unconstrained array constant round-trips its sort and default
    /// value through `(get-model)` exactly as before the conversion.
    #[test]
    fn test_get_model_array_const_roundtrip() {
        let mut ctx = Context::new();
        let out = ctx
            .execute_script("(declare-const a (Array Int Int))\n(check-sat)\n(get-model)")
            .expect("valid script executes");
        let model = out.last().map(String::as_str).unwrap_or_default();
        assert!(
            model.contains("(define-fun a () (Array Int Int) "),
            "model must render the array sort name: {model}"
        );
    }

    // ===== parameter / parametric sort names ============================

    /// A sort *parameter* renders as its declared name. It used to render as
    /// the literal string "Unknown", so every parameter of every definition
    /// printed identically.
    #[test]
    fn test_format_sort_name_parameter_uses_its_real_name() {
        let mut ctx = Context::new();
        let t_param = ctx.terms.sorts.mk_sort_parameter("T");
        let u_param = ctx.terms.sorts.mk_sort_parameter("U");

        assert_eq!(ctx.format_sort_name(t_param), "T");
        assert_eq!(ctx.format_sort_name(u_param), "U");
        assert_ne!(
            ctx.format_sort_name(t_param),
            ctx.format_sort_name(u_param),
            "two distinct parameters must not collapse onto one printed name"
        );
    }

    /// A parametric application renders as `(Name arg...)`, exactly as
    /// `SmtLibPrinter::write_sort` spells it, with the head name resolved
    /// through the *sort* manager's interner.
    #[test]
    fn test_format_sort_name_parametric_instance_uses_its_real_name() {
        let mut ctx = Context::new();
        let int_sort = ctx.terms.sorts.int_sort;
        let bool_sort = ctx.terms.sorts.bool_sort;
        ctx.terms.sorts.declare_parametric_sort("List", 1);
        ctx.terms.sorts.declare_parametric_sort("Pair", 2);

        let list_int = ctx
            .terms
            .sorts
            .instantiate_parametric_sort("List", &[int_sort])
            .expect("List has arity 1");
        let pair = ctx
            .terms
            .sorts
            .instantiate_parametric_sort("Pair", &[list_int, bool_sort])
            .expect("Pair has arity 2");

        assert_eq!(ctx.format_sort_name(list_int), "(List Int)");
        assert_eq!(ctx.format_sort_name(pair), "(Pair (List Int) Bool)");
    }

    /// The parametric arguments are walked on the same explicit heap stack as
    /// array domains, so a deeply nested application does not touch the
    /// native stack: `(List (List ... Int ...))`.
    #[test]
    fn test_format_sort_name_deeply_nested_parametric() {
        // Stack and depth scale together (1 MiB/20k -> 128 KiB/2.5k): the
        // ~52 B-per-frame threshold is the pin, so never raise one alone.
        const DEPTH: usize = 2_500;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut ctx = Context::new();
                let int_sort = ctx.terms.sorts.int_sort;
                ctx.terms.sorts.declare_parametric_sort("List", 1);
                let mut sort = int_sort;
                for _ in 0..DEPTH {
                    sort = ctx
                        .terms
                        .sorts
                        .instantiate_parametric_sort("List", &[sort])
                        .expect("List has arity 1");
                }
                let name = ctx.format_sort_name(sort);
                // "(List " * DEPTH + "Int" + ")" * DEPTH
                assert_eq!(name.len(), 7 * DEPTH + 3);
                assert!(name.starts_with("(List (List "));
                // The innermost `Int` is followed by one closing paren per
                // level: `(List (List ... (List Int)...))`.
                assert!(name.ends_with(&format!(" Int{}", ")".repeat(DEPTH))));
            })
            .expect("spawn deep-parametric thread");
        handle
            .join()
            .expect("deep-parametric thread must not overflow");
    }

    /// End to end through `get_model`: constants declared at a parametric
    /// instance and at a sort parameter report their real sort names.  Both
    /// used to report the single name "Unknown", so a model could not even
    /// tell the two constants' sorts apart.
    ///
    /// Declared through the `Context` API rather than a script because the
    /// SMT-LIB front end does not yet accept an applied user sort
    /// constructor in sort position (`parse_sort` handles only `Array`
    /// there); the model formatter is what is under test here either way.
    #[test]
    fn test_get_model_parameter_and_parametric_sorts_show_real_names() {
        let mut ctx = Context::new();
        let int_sort = ctx.terms.sorts.int_sort;
        ctx.terms.sorts.declare_parametric_sort("List", 1);
        let list_int = ctx
            .terms
            .sorts
            .instantiate_parametric_sort("List", &[int_sort])
            .expect("List has arity 1");
        let t_param = ctx.terms.sorts.mk_sort_parameter("T");

        ctx.declare_const("xs", list_int);
        ctx.declare_const("t", t_param);
        assert_eq!(ctx.check_sat(), SolverResult::Sat);

        let model = ctx.get_model().expect("sat check produces a model");
        let sorts: crate::prelude::HashMap<&str, &str> = model
            .iter()
            .map(|(name, sort, _)| (name.as_str(), sort.as_str()))
            .collect();
        assert_eq!(sorts.get("xs").copied(), Some("(List Int)"));
        assert_eq!(sorts.get("t").copied(), Some("T"));
    }
}
