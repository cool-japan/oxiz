//! Faithful re-interning of a CHC system into a fresh, independent
//! [`TermManager`].
//!
//! Spacer borrows `&mut TermManager` for the whole of a solve, and term
//! interning is not thread-safe, so genuinely parallel solving requires each
//! worker thread to own a *private* term arena. A [`ChcSystem`] is not `Clone`
//! (it holds atomic id counters) and its rule terms live in one specific
//! arena, so to hand a worker an independent copy we must rebuild every rule
//! term in a brand-new arena — which is exactly what this module does.
//!
//! The translation is **fail-closed**: it only handles the linear
//! arithmetic / boolean fragment that the Spacer engine solves soundly
//! (`is_supported_fragment` in `pdr.rs`). Any term or sort outside that
//! fragment (bit-vectors, arrays, strings, floating-point, uninterpreted
//! functions, quantifiers) makes [`translate_system`] return `None`, so the
//! caller can fall back to the sound single-arena sequential engine rather
//! than risk an unfaithful copy.
//!
//! Predicates and rules are rebuilt in their original insertion order, so the
//! translated [`ChcSystem`] reproduces the exact same [`crate::chc::PredId`]/
//! [`crate::chc::RuleId`] numbering — the copy is structurally identical to
//! the original and yields the same verdict.
//!
//! Reference: Z3's `ast_translation` (`src/ast/ast_translation.cpp`).

use crate::chc::{ChcSystem, PredicateApp, RuleBody, RuleHead};
use oxiz_core::ast::TermKind;
use oxiz_core::{SortId, SortKind, TermId, TermManager};
use rustc_hash::FxHashMap;

/// Rebuild `(src_terms, src_system)` as an independent `(TermManager,
/// ChcSystem)` pair whose terms live in a fresh arena.
///
/// Returns `None` if any predicate sort, rule variable sort, or rule term
/// falls outside the translatable linear-arithmetic/boolean fragment, or if
/// the rebuilt system fails to reproduce the original id numbering (a
/// defensive invariant check). On `None` the caller must not assume any
/// parallel copy is available.
#[must_use]
pub fn translate_system(
    src_terms: &TermManager,
    src_system: &ChcSystem,
) -> Option<(TermManager, ChcSystem)> {
    let mut dest_terms = TermManager::new();
    let mut dest_system = ChcSystem::new();
    let mut tr = Translator {
        src: src_terms,
        memo: FxHashMap::default(),
    };

    // Rebuild predicate declarations in insertion order so ids line up.
    for pred in src_system.predicates() {
        let mut params: Vec<SortId> = Vec::with_capacity(pred.params.len());
        for &sort in &pred.params {
            params.push(tr.sort(&mut dest_terms, sort)?);
        }
        let id = dest_system.declare_predicate(pred.name.clone(), params);
        if id != pred.id {
            return None;
        }
    }

    // Rebuild rules in insertion order so ids line up.
    for rule in src_system.rules() {
        let mut vars: Vec<(String, SortId)> = Vec::with_capacity(rule.vars.len());
        for (name, sort) in &rule.vars {
            vars.push((name.clone(), tr.sort(&mut dest_terms, *sort)?));
        }

        let mut body_preds: Vec<PredicateApp> = Vec::with_capacity(rule.body.predicates.len());
        for app in &rule.body.predicates {
            let mut args: Vec<TermId> = Vec::with_capacity(app.args.len());
            for &arg in &app.args {
                args.push(tr.term(&mut dest_terms, arg)?);
            }
            body_preds.push(PredicateApp::new(app.pred, args));
        }
        let constraint = tr.term(&mut dest_terms, rule.body.constraint)?;
        let body = RuleBody::new(body_preds, constraint);

        let head = match &rule.head {
            RuleHead::Query => RuleHead::Query,
            RuleHead::Predicate(app) => {
                let mut args: Vec<TermId> = Vec::with_capacity(app.args.len());
                for &arg in &app.args {
                    args.push(tr.term(&mut dest_terms, arg)?);
                }
                RuleHead::Predicate(PredicateApp::new(app.pred, args))
            }
        };

        let rid = dest_system.add_rule(vars, body, head, rule.name.clone());
        if rid != rule.id {
            return None;
        }
    }

    Some((dest_terms, dest_system))
}

/// Stateful helper that memoizes translated terms so shared subterms are
/// rebuilt once (preserving structural sharing and avoiding exponential
/// blow-up on DAG-shaped terms).
struct Translator<'s> {
    src: &'s TermManager,
    memo: FxHashMap<TermId, TermId>,
}

impl Translator<'_> {
    /// Translate a sort id from the source arena into the destination arena.
    ///
    /// Only the base sorts the Spacer engine reasons about are supported;
    /// everything else fails closed with `None`.
    fn sort(&self, dest: &mut TermManager, sort: SortId) -> Option<SortId> {
        match self.src.sorts.get(sort)?.kind {
            SortKind::Bool => Some(dest.sorts.bool_sort),
            SortKind::Int => Some(dest.sorts.int_sort),
            SortKind::Real => Some(dest.sorts.real_sort),
            _ => None,
        }
    }

    /// Translate a term id from the source arena into the destination arena,
    /// returning `None` for any kind outside the supported fragment.
    ///
    /// Walked with an explicit heap stack. The previous form recursed once
    /// per nesting level (mutually with `terms`); the memo made it linear in
    /// DAG size but did nothing about depth, and the return type is
    /// `Option<TermId>` whose `None` already means "outside the supported
    /// fragment" — so a depth cap could not be distinguished from that and
    /// would silently drop a translatable rule instead of translating it.
    fn term(&mut self, dest: &mut TermManager, root: TermId) -> Option<TermId> {
        let mut stack: Vec<(TermId, bool)> = vec![(root, false)];

        while let Some((current, expanded)) = stack.pop() {
            if self.memo.contains_key(&current) {
                continue;
            }

            let kind = self.src.get(current)?.kind.clone();
            // `None` here means the kind is outside the fragment; the whole
            // translation fails closed, exactly as before.
            let children = Self::translatable_children(&kind)?;

            if expanded {
                let mut mapped = Vec::with_capacity(children.len());
                for child in &children {
                    mapped.push(self.memo.get(child).copied()?);
                }
                let out = self.rebuild(dest, current, &kind, &mapped)?;
                self.memo.insert(current, out);
            } else {
                stack.push((current, true));
                for child in children {
                    if !self.memo.contains_key(&child) {
                        stack.push((child, false));
                    }
                }
            }
        }

        self.memo.get(&root).copied()
    }

    /// The subterms of a translatable kind, or `None` when the kind is
    /// outside the fragment Spacer reasons about soundly (bit-vectors,
    /// arrays, strings, floating-point, uninterpreted functions,
    /// quantifiers).
    fn translatable_children(kind: &TermKind) -> Option<Vec<TermId>> {
        match kind {
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::Var(_) => Some(Vec::new()),
            TermKind::Not(a) | TermKind::Neg(a) => Some(vec![*a]),
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Distinct(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => Some(args.to_vec()),
            TermKind::Xor(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Eq(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b) => Some(vec![*a, *b]),
            TermKind::Ite(c, t, e) => Some(vec![*c, *t, *e]),
            _ => None,
        }
    }

    /// Rebuild one already-translated node in `dest`.
    ///
    /// `mapped` holds the destination ids of the children
    /// [`Self::translatable_children`] reported, in the same order.
    fn rebuild(
        &self,
        dest: &mut TermManager,
        source: TermId,
        kind: &TermKind,
        mapped: &[TermId],
    ) -> Option<TermId> {
        // Helpers keep the arity assumptions in one place: `translatable_children`
        // decided how many children each kind has, so a mismatch here would be
        // an internal inconsistency, reported as `None` rather than a panic.
        let unary = || mapped.first().copied();
        let binary = || Some((mapped.first().copied()?, mapped.get(1).copied()?));

        Some(match kind {
            TermKind::True => dest.mk_true(),
            TermKind::False => dest.mk_false(),
            TermKind::IntConst(v) => dest.mk_int(v.clone()),
            TermKind::RealConst(r) => dest.mk_real(*r),
            TermKind::Var(spur) => {
                let name = self.src.resolve_str(*spur).to_string();
                let sort = self.sort(dest, self.src.get(source)?.sort)?;
                dest.mk_var(&name, sort)
            }
            TermKind::Not(_) => dest.mk_not(unary()?),
            TermKind::Neg(_) => dest.mk_neg(unary()?),
            TermKind::And(_) => dest.mk_and(mapped.to_vec()),
            TermKind::Or(_) => dest.mk_or(mapped.to_vec()),
            TermKind::Distinct(_) => dest.mk_distinct(mapped.to_vec()),
            TermKind::Add(_) => dest.mk_add(mapped.to_vec()),
            TermKind::Mul(_) => dest.mk_mul(mapped.to_vec()),
            TermKind::Xor(_, _) => {
                let (a, b) = binary()?;
                dest.mk_xor(a, b)
            }
            TermKind::Implies(_, _) => {
                let (a, b) = binary()?;
                dest.mk_implies(a, b)
            }
            TermKind::Eq(_, _) => {
                let (a, b) = binary()?;
                dest.mk_eq(a, b)
            }
            TermKind::Sub(_, _) => {
                let (a, b) = binary()?;
                dest.mk_sub(a, b)
            }
            TermKind::Div(_, _) => {
                let (a, b) = binary()?;
                dest.mk_div(a, b)
            }
            TermKind::Mod(_, _) => {
                let (a, b) = binary()?;
                dest.mk_mod(a, b)
            }
            TermKind::Lt(_, _) => {
                let (a, b) = binary()?;
                dest.mk_lt(a, b)
            }
            TermKind::Le(_, _) => {
                let (a, b) = binary()?;
                dest.mk_le(a, b)
            }
            TermKind::Gt(_, _) => {
                let (a, b) = binary()?;
                dest.mk_gt(a, b)
            }
            TermKind::Ge(_, _) => {
                let (a, b) = binary()?;
                dest.mk_ge(a, b)
            }
            TermKind::Ite(_, _, _) => {
                let cond = mapped.first().copied()?;
                let then_branch = mapped.get(1).copied()?;
                let else_branch = mapped.get(2).copied()?;
                dest.mk_ite(cond, then_branch, else_branch)
            }
            // `translatable_children` already refused every other kind.
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chc::PredicateApp;

    /// Stack size and nesting depth for the deep-recursion test below.
    ///
    /// The two are scaled together on purpose: what the test actually pins is
    /// the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive translation needs far more
    /// than that per frame and still overflows, so the regression keeps every
    /// bit of its detection power. The pair used to be 1 MiB / 50_000 -- the
    /// same 21 bytes. `mk_and` flattens its arguments (so `acc = mk_and([acc,
    /// atom])` never actually nests, and is quadratic to boot), so the deep
    /// term below is built with `TermManager::intern_term` directly, which
    /// neither flattens nor re-interns already-built prefixes. Never raise
    /// `DEEP_DEPTH` without raising `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

    #[test]
    fn test_translate_preserves_structure_and_ids() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort, terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let y = terms.mk_var("y", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let eq_x = terms.mk_eq(x, zero);
        let eq_y = terms.mk_eq(y, zero);
        let init = terms.mk_and([eq_x, eq_y]);
        system.add_init_rule(
            [
                ("x".to_string(), terms.sorts.int_sort),
                ("y".to_string(), terms.sorts.int_sort),
            ],
            init,
            inv,
            [x, y],
        );
        let neg = terms.mk_lt(x, zero);
        system.add_query(
            [
                ("x".to_string(), terms.sorts.int_sort),
                ("y".to_string(), terms.sorts.int_sort),
            ],
            [PredicateApp::new(inv, [x, y])],
            neg,
        );

        let (dest_terms, dest_system) =
            translate_system(&terms, &system).expect("int fragment must translate");

        // Structural identity: same predicate/rule counts and ids.
        assert_eq!(dest_system.num_predicates(), system.num_predicates());
        assert_eq!(dest_system.num_rules(), system.num_rules());
        assert_eq!(dest_system.queries().count(), 1);
        assert_eq!(dest_system.entries().count(), 1);

        // Independent arena: the translated Inv predicate exists with the same
        // arity, and its terms are valid in the fresh manager.
        let dest_inv = dest_system
            .get_predicate_by_name("Inv")
            .expect("Inv must survive translation");
        assert_eq!(dest_inv.arity(), 2);
        // The fresh manager is genuinely separate.
        assert!(dest_terms.get(dest_terms.mk_true()).is_some());
    }

    #[test]
    fn test_translate_rejects_unsupported_sort() {
        // A bit-vector variable is outside the supported fragment, so
        // translation must fail closed rather than silently drop it.
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();
        let bv_sort = terms.sorts.bitvec(8);
        let p = system.declare_predicate("P", [bv_sort]);
        let bv = terms.mk_var("b", bv_sort);
        let zero = terms.mk_bitvec(0, 8);
        let c = terms.mk_eq(bv, zero);
        system.add_query(
            [("b".to_string(), bv_sort)],
            [PredicateApp::new(p, [bv])],
            c,
        );

        assert!(
            translate_system(&terms, &system).is_none(),
            "bit-vector sorts must make translation fail closed"
        );
    }

    /// Deeply nested terms must translate without overflowing the stack.
    #[test]
    fn translate_survives_deep_nesting() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut src = TermManager::new();
                let mut system = ChcSystem::new();
                let bool_sort = src.sorts.bool_sort;
                let int_sort = src.sorts.int_sort;
                let inv = system.declare_predicate("Inv", [int_sort]);

                let x = src.mk_var("x", int_sort);
                let zero = src.mk_int(0);
                let mut constraint = src.mk_eq(x, zero);
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the tree really is
                // `DEEP_DEPTH` deep.
                for i in 0..DEEP_DEPTH {
                    let k = src.mk_int(i);
                    let atom = src.mk_ge(x, k);
                    constraint =
                        src.intern_term(TermKind::And(vec![constraint, atom].into()), bool_sort);
                }
                system.add_init_rule([("x".to_string(), int_sort)], constraint, inv, [x]);

                let translated = translate_system(&src, &system);
                assert!(
                    translated.is_some(),
                    "a deeply nested but fully supported system must translate"
                );
            })
            .expect("thread spawn should succeed");
        handle.join().expect("deep translation must return");
    }

    /// Structural sharing keeps translation linear in DAG size: a 60-level
    /// doubling DAG would be 2^60 tree nodes without the memo.
    #[test]
    fn translate_is_linear_on_shared_dag() {
        let mut src = TermManager::new();
        let mut system = ChcSystem::new();
        let int_sort = src.sorts.int_sort;
        let inv = system.declare_predicate("Inv", [int_sort]);

        let x = src.mk_var("x", int_sort);
        let one = src.mk_int(1);
        let mut shared = src.mk_add([x, one]);
        for _ in 0..60 {
            shared = src.mk_add([shared, shared]);
        }
        let zero = src.mk_int(0);
        let constraint = src.mk_ge(shared, zero);
        system.add_init_rule([("x".to_string(), int_sort)], constraint, inv, [x]);

        assert!(translate_system(&src, &system).is_some());
    }
}
