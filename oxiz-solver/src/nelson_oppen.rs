//! Nelson-Oppen Theory Combination.
#![allow(dead_code, clippy::result_unit_err)] // Under development
//!
//! Implements the Nelson-Oppen framework for combining decision procedures
//! of disjoint theories through equality sharing.

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager, collect_subterms, get_children};

/// Nelson-Oppen theory combination engine.
pub struct NelsonOppenCombiner {
    /// Shared terms between theories
    shared_terms: FxHashSet<TermId>,
    /// Equality classes for shared terms
    equality_classes: UnionFind,
    /// Pending equalities to propagate
    pending_equalities: VecDeque<(TermId, TermId)>,
    /// Already-propagated equalities (normalized so lhs <= rhs).
    /// Prevents the fixed-point loop from re-discovering known equalities.
    propagated_equalities: FxHashSet<(TermId, TermId)>,
    /// Theory assignments for shared terms
    theory_assignments: FxHashMap<TermId, TheoryId>,
    /// Statistics
    stats: NelsonOppenStats,
    /// Counter for generating fresh variable names during purification
    fresh_var_counter: u64,
    /// Alias introduced by [`NelsonOppenCombiner::purify_term`] for each
    /// interface subterm, so repeated calls (one per assertion, typically)
    /// reuse one shared variable per subterm instead of minting a new one and
    /// a new definitional equality every time.
    purification_aliases: FxHashMap<TermId, TermId>,
}

/// Theory identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheoryId(pub usize);

/// Which theory owns a term's top-level symbol.
///
/// This is the signature partition Nelson-Oppen purification is defined over:
/// a term is *pure* in theory `T` when every symbol in it belongs to `T` or is
/// shared by all theories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermTheory {
    /// Owned by no single theory, so never a purification boundary.
    ///
    /// Two groups land here:
    ///
    /// * **Nullary symbols** -- variables and literals. They have no arguments
    ///   to leak into a foreign signature, and they are exactly the interface
    ///   terms Nelson-Oppen shares between theories, so a variable or a
    ///   numeral is already pure wherever it appears.
    /// * **The Boolean skeleton and equality** -- `not`, `and`, `or`, `xor`,
    ///   `=>`, `ite`, `=`, `distinct`. In this framework the propositional
    ///   structure is handled by the SAT layer and `=` is shared by every
    ///   theory, so a bit-vector term sitting directly under `and` or `=` is
    ///   not a mixed *term*: it is a pure literal in the bit-vector theory.
    ///   Purification is only needed where a term of one theory appears as an
    ///   argument of an *operator* of a different theory.
    Shared,
    /// Integer / real arithmetic.
    Arithmetic,
    /// Fixed-width bit-vectors.
    BitVector,
    /// Extensional arrays.
    Array,
    /// Unicode strings and regular languages.
    String,
    /// IEEE-754 floating point.
    FloatingPoint,
    /// Algebraic datatypes.
    Datatype,
    /// Uninterpreted functions.
    Uf,
    /// Not a theory term at all: a binding form (`forall`, `exists`, `let`,
    /// `match`).
    ///
    /// This combiner implements the quantifier-free Nelson-Oppen procedure, so
    /// it has no purification rule for binders; [`NelsonOppenCombiner::purify_term`]
    /// reports an error rather than returning a term it has not purified.
    Binder,
}

impl TermTheory {
    /// A stable [`TheoryId`] per theory, for `theory_assignments`.
    #[must_use]
    pub fn theory_id(self) -> TheoryId {
        TheoryId(match self {
            Self::Shared => 0,
            Self::Arithmetic => 1,
            Self::BitVector => 2,
            Self::Array => 3,
            Self::String => 4,
            Self::FloatingPoint => 5,
            Self::Datatype => 6,
            Self::Uf => 7,
            Self::Binder => 8,
        })
    }
}

/// Nelson-Oppen statistics
#[derive(Debug, Clone, Default)]
pub struct NelsonOppenStats {
    /// Number of shared terms
    pub shared_terms_count: usize,
    /// Number of equalities propagated
    pub equalities_propagated: usize,
    /// Number of theory conflicts detected
    pub theory_conflicts: usize,
    /// Number of purification steps: one per fresh shared variable
    /// [`NelsonOppenCombiner::purify_term`] introduced for a mixed-theory
    /// subterm.
    ///
    /// This used to be incremented once per recursive `purify_term` call --
    /// including the calls that purified nothing, which was all of them -- so
    /// it reported activity that had not happened.
    pub purifications: usize,
}

impl NelsonOppenCombiner {
    /// Create a new Nelson-Oppen combiner.
    pub fn new() -> Self {
        Self {
            shared_terms: FxHashSet::default(),
            equality_classes: UnionFind::new(),
            pending_equalities: VecDeque::new(),
            propagated_equalities: FxHashSet::default(),
            theory_assignments: FxHashMap::default(),
            stats: NelsonOppenStats::default(),
            fresh_var_counter: 0,
            purification_aliases: FxHashMap::default(),
        }
    }

    /// Register a shared term between theories.
    pub fn register_shared_term(&mut self, term_id: TermId, theory1: TheoryId, _theory2: TheoryId) {
        self.shared_terms.insert(term_id);
        self.theory_assignments.insert(term_id, theory1);
        self.equality_classes.make_set(term_id);
        self.stats.shared_terms_count += 1;
    }

    /// Normalize an equality pair so that the smaller TermId comes first.
    /// This ensures (a,b) and (b,a) are treated as the same equality.
    fn normalize_pair(lhs: TermId, rhs: TermId) -> (TermId, TermId) {
        if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
    }

    /// Assert an equality between shared terms.
    ///
    /// Returns Ok(()) if consistent, Err(()) if conflict detected.
    pub fn assert_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<(), ()> {
        if !self.shared_terms.contains(&lhs) || !self.shared_terms.contains(&rhs) {
            return Err(()); // Only shared terms can be equated
        }

        // Normalize and check if this equality was already propagated
        let key = Self::normalize_pair(lhs, rhs);
        if self.propagated_equalities.contains(&key) {
            return Ok(());
        }

        // Check if already in same equivalence class
        if self.equality_classes.find(lhs) == self.equality_classes.find(rhs) {
            self.propagated_equalities.insert(key);
            return Ok(());
        }

        // Merge equivalence classes
        self.equality_classes.union(lhs, rhs);
        self.pending_equalities.push_back((lhs, rhs));
        self.propagated_equalities.insert(key);
        self.stats.equalities_propagated += 1;

        Ok(())
    }

    /// Generate a fresh variable name for purification.
    fn fresh_var_name(&mut self) -> String {
        let name =
            oxiz_core::smtlib::reserved_name("nopurify", &self.fresh_var_counter.to_string());
        self.fresh_var_counter += 1;
        name
    }

    /// Which theory owns `kind`'s top-level symbol.
    ///
    /// The `match` is exhaustive with no `_` arm on purpose: a new `TermKind`
    /// variant must be classified explicitly, because an unclassified operator
    /// silently becomes a non-boundary and purification stops being
    /// purification. See [`TermTheory`] for the two judgement calls (nullary
    /// symbols and the Boolean/equality skeleton are `Shared`).
    #[must_use]
    pub fn theory_of(kind: &TermKind) -> TermTheory {
        match kind {
            // Nullary symbols and the Boolean / equality skeleton.
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::Var(_)
            | TermKind::StringLit(_)
            | TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. }
            | TermKind::Not(_)
            | TermKind::And(_)
            | TermKind::Or(_)
            | TermKind::Xor(_, _)
            | TermKind::Implies(_, _)
            | TermKind::Ite(_, _, _)
            | TermKind::Eq(_, _)
            | TermKind::Distinct(_) => TermTheory::Shared,

            TermKind::Neg(_)
            | TermKind::Add(_)
            | TermKind::Sub(_, _)
            | TermKind::Mul(_)
            | TermKind::Div(_, _)
            | TermKind::Mod(_, _)
            | TermKind::Lt(_, _)
            | TermKind::Le(_, _)
            | TermKind::Gt(_, _)
            | TermKind::Ge(_, _) => TermTheory::Arithmetic,

            TermKind::BvConcat(_, _)
            | TermKind::BvExtract { .. }
            | TermKind::BvNot(_)
            | TermKind::BvAnd(_, _)
            | TermKind::BvOr(_, _)
            | TermKind::BvXor(_, _)
            | TermKind::BvAdd(_, _)
            | TermKind::BvSub(_, _)
            | TermKind::BvMul(_, _)
            | TermKind::BvUdiv(_, _)
            | TermKind::BvSdiv(_, _)
            | TermKind::BvUrem(_, _)
            | TermKind::BvSrem(_, _)
            | TermKind::BvShl(_, _)
            | TermKind::BvLshr(_, _)
            | TermKind::BvAshr(_, _)
            | TermKind::BvUlt(_, _)
            | TermKind::BvUle(_, _)
            | TermKind::BvSlt(_, _)
            | TermKind::BvSle(_, _) => TermTheory::BitVector,

            TermKind::Select(_, _) | TermKind::Store(_, _, _) => TermTheory::Array,

            TermKind::StrConcat(_, _)
            | TermKind::StrLen(_)
            | TermKind::StrSubstr(_, _, _)
            | TermKind::StrAt(_, _)
            | TermKind::StrContains(_, _)
            | TermKind::StrPrefixOf(_, _)
            | TermKind::StrSuffixOf(_, _)
            | TermKind::StrIndexOf(_, _, _)
            | TermKind::StrReplace(_, _, _)
            | TermKind::StrReplaceAll(_, _, _)
            | TermKind::StrReplaceRe(_, _, _)
            | TermKind::StrReplaceReAll(_, _, _)
            | TermKind::StrToInt(_)
            | TermKind::IntToStr(_)
            | TermKind::StrInRe(_, _)
            | TermKind::StrLt(_, _)
            | TermKind::StrLe(_, _)
            | TermKind::StrToCode(_)
            | TermKind::StrFromCode(_) => TermTheory::String,

            TermKind::FpAbs(_)
            | TermKind::FpNeg(_)
            | TermKind::FpSqrt(_, _)
            | TermKind::FpRoundToIntegral(_, _)
            | TermKind::FpAdd(_, _, _)
            | TermKind::FpSub(_, _, _)
            | TermKind::FpMul(_, _, _)
            | TermKind::FpDiv(_, _, _)
            | TermKind::FpRem(_, _)
            | TermKind::FpMin(_, _)
            | TermKind::FpMax(_, _)
            | TermKind::FpLeq(_, _)
            | TermKind::FpLt(_, _)
            | TermKind::FpGeq(_, _)
            | TermKind::FpGt(_, _)
            | TermKind::FpEq(_, _)
            | TermKind::FpFma(_, _, _, _)
            | TermKind::FpIsNormal(_)
            | TermKind::FpIsSubnormal(_)
            | TermKind::FpIsZero(_)
            | TermKind::FpIsInfinite(_)
            | TermKind::FpIsNaN(_)
            | TermKind::FpIsNegative(_)
            | TermKind::FpIsPositive(_)
            | TermKind::FpToFp { .. }
            | TermKind::FpToSBV { .. }
            | TermKind::FpToUBV { .. }
            | TermKind::FpToReal(_)
            | TermKind::RealToFp { .. }
            | TermKind::SBVToFp { .. }
            | TermKind::UBVToFp { .. } => TermTheory::FloatingPoint,

            TermKind::DtConstructor { .. }
            | TermKind::DtTester { .. }
            | TermKind::DtSelector { .. } => TermTheory::Datatype,

            TermKind::Apply { .. } => TermTheory::Uf,

            TermKind::Forall { .. }
            | TermKind::Exists { .. }
            | TermKind::Let { .. }
            | TermKind::Match { .. } => TermTheory::Binder,
        }
    }

    /// Register `term` as shared without disturbing an existing registration.
    ///
    /// [`Self::register_shared_term`] calls `make_set`, which resets the
    /// term's union-find parent to itself and would therefore *forget* any
    /// equalities already asserted about it.
    fn ensure_shared(&mut self, term: TermId, theory: TermTheory) {
        if !self.shared_terms.contains(&term) {
            self.register_shared_term(term, theory.theory_id(), theory.theory_id());
        }
    }

    /// Purify a term by introducing fresh variables for foreign subterms.
    ///
    /// Purification is what makes Nelson-Oppen theory combination correct: each
    /// theory solver must see only its own signature plus shared variables. A
    /// subterm whose top symbol belongs to a theory other than its parent's is
    /// replaced everywhere by a fresh variable `v` of the same sort, and the
    /// definitional equality `v = <purified subterm>` is registered and
    /// asserted so it reaches the theories through
    /// [`Self::get_pending_equalities`]. The returned term is pure in the
    /// theory of its own top symbol.
    ///
    /// For example `(+ (f (bvadd x y)) 1)` becomes `(+ v1 1)` with the
    /// definitions `v1 = (f v0)` and `v0 = (bvadd x y)`: three terms, each pure
    /// in one theory, connected only by shared variables.
    ///
    /// # What this replaced
    ///
    /// The previous implementation matched `TermKind::Apply` and ended in
    /// `_ => Ok(term_id)`, so every other kind -- all of arithmetic, all
    /// bit-vector, array, string, floating-point and datatype operators, and
    /// every binder -- was returned **unpurified** while
    /// `stats.purifications` was incremented as if work had been done. It also
    /// never purified anything at all, even for `Apply`: its
    /// `needs_purification` test compared `get_theory(purified_arg)` with
    /// `get_theory(original_arg)`, i.e. the `theory_assignments` entry of two
    /// term ids that are *equal* whenever the recursive call changed nothing,
    /// and the base case could never change anything. Its one definitional
    /// equality was recorded with `let _ = self.assert_equality(..)`, which
    /// always failed and was always discarded, because the right-hand side was
    /// never registered as a shared term.
    ///
    /// # Errors
    ///
    /// * The term, or one of its subterms, is not in `tm`.
    /// * The term contains a binding form. This combiner implements the
    ///   quantifier-free procedure and has no purification rule for binders;
    ///   returning the term unchanged would be a silent claim to have purified
    ///   it.
    ///
    /// # Notes
    ///
    /// Aliasing is *per subterm*, not per occurrence: because terms are
    /// hash-consed, a subterm that needs an alias in one context is aliased in
    /// every context. That is still a valid purification -- the extra
    /// occurrences of `v` are variables, which are pure everywhere, and the
    /// definitional equality is unchanged -- it merely introduces more shared
    /// variables than the minimum.
    ///
    /// The walk is iterative (via [`collect_subterms`] and
    /// [`TermManager::substitute`], both of which use explicit heap stacks), so
    /// a deeply nested term cannot overflow the stack.
    pub fn purify_term(&mut self, term_id: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        if tm.get(term_id).is_none() {
            return Err(format!("term {term_id:?} not found"));
        }

        // Post-order, so children precede their parents.
        let subterms = collect_subterms(term_id, tm);

        // Reject binders up front rather than half-purifying around them.
        for &sub in &subterms {
            let kind = &tm
                .get(sub)
                .ok_or_else(|| format!("subterm {sub:?} not found"))?
                .kind;
            if Self::theory_of(kind) == TermTheory::Binder {
                return Err(format!(
                    "cannot purify {term_id:?}: subterm {sub:?} is a binding form, \
                     which the quantifier-free Nelson-Oppen procedure does not handle"
                ));
            }
        }

        // A subterm is an interface term when it appears as an argument of an
        // operator owned by a different theory.
        let mut boundary: FxHashSet<TermId> = FxHashSet::default();
        for &parent in &subterms {
            let parent_kind = tm
                .get(parent)
                .ok_or_else(|| format!("subterm {parent:?} not found"))?
                .kind
                .clone();
            let parent_theory = Self::theory_of(&parent_kind);
            if parent_theory == TermTheory::Shared {
                continue;
            }
            for child in get_children(&parent_kind) {
                let child_theory = Self::theory_of(
                    &tm.get(child)
                        .ok_or_else(|| format!("subterm {child:?} not found"))?
                        .kind,
                );
                if child_theory != TermTheory::Shared && child_theory != parent_theory {
                    boundary.insert(child);
                }
            }
        }

        if boundary.is_empty() {
            return Ok(term_id);
        }

        // Innermost first, so a nested interface term is already aliased by the
        // time its enclosing definition is built.
        let ordered: Vec<TermId> = subterms
            .iter()
            .copied()
            .filter(|s| boundary.contains(s))
            .collect();

        // One fresh variable per interface term. `existing` keeps the generated
        // name from colliding with a variable the caller already put in the
        // term, which would alias two unrelated terms together.
        let existing: FxHashSet<TermId> = subterms.iter().copied().collect();
        let mut alias: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut fresh_this_call: Vec<TermId> = Vec::new();
        for &sub in &ordered {
            if let Some(&known) = self.purification_aliases.get(&sub) {
                // Already purified by an earlier call: reuse the alias, and do
                // not re-assert its definition below.
                alias.insert(sub, known);
                continue;
            }
            let sort = tm
                .get(sub)
                .ok_or_else(|| format!("subterm {sub:?} not found"))?
                .sort;
            let fresh = loop {
                let name = self.fresh_var_name();
                let var = tm.mk_var(&name, sort);
                if !existing.contains(&var)
                    && !alias.values().any(|&v| v == var)
                    && !self.purification_aliases.values().any(|&v| v == var)
                {
                    break var;
                }
            };
            alias.insert(sub, fresh);
            self.purification_aliases.insert(sub, fresh);
            fresh_this_call.push(sub);
        }

        // Definitional equalities `v = purified(subterm)`.
        for &sub in &fresh_this_call {
            let Some(fresh) = alias.remove(&sub) else {
                return Err(format!("missing alias for interface term {sub:?}"));
            };
            let purified = if alias.is_empty() {
                sub
            } else {
                tm.substitute(sub, &alias)
            };
            alias.insert(sub, fresh);

            let theory = Self::theory_of(
                &tm.get(purified)
                    .ok_or_else(|| format!("purified term {purified:?} not found"))?
                    .kind,
            );
            self.ensure_shared(fresh, TermTheory::Shared);
            self.ensure_shared(purified, theory);
            self.assert_equality(fresh, purified).map_err(|()| {
                format!("failed to record purification equality {fresh:?} = {purified:?}")
            })?;
            self.stats.purifications += 1;
        }

        Ok(tm.substitute(term_id, &alias))
    }

    /// Get pending equalities to propagate to theories.
    pub fn get_pending_equalities(&mut self) -> Vec<(TermId, TermId)> {
        let mut result = Vec::new();
        while let Some(eq) = self.pending_equalities.pop_front() {
            result.push(eq);
        }
        result
    }

    /// Check if two terms are in the same equivalence class.
    pub fn are_equal(&self, lhs: TermId, rhs: TermId) -> bool {
        self.equality_classes.find(lhs) == self.equality_classes.find(rhs)
    }

    /// Get all terms in the equivalence class of a term.
    pub fn get_equivalence_class(&self, term_id: TermId) -> Vec<TermId> {
        let rep = self.equality_classes.find(term_id);
        self.shared_terms
            .iter()
            .filter(|&&t| self.equality_classes.find(t) == rep)
            .copied()
            .collect()
    }

    /// Get theory assignment for a term.
    fn get_theory(&self, term_id: TermId) -> Option<TheoryId> {
        self.theory_assignments.get(&term_id).copied()
    }

    /// Convexity closure: generate implied equalities.
    ///
    /// For convex theories, if we have equalities in each class,
    /// we must propagate all pairwise equalities.
    /// Only returns equalities that have NOT already been propagated.
    pub fn convexity_closure(&mut self) -> Vec<(TermId, TermId)> {
        let mut implied_equalities = Vec::new();

        // Group terms by equivalence class
        let mut classes: FxHashMap<TermId, Vec<TermId>> = FxHashMap::default();
        for &term in &self.shared_terms {
            let rep = self.equality_classes.find(term);
            classes.entry(rep).or_default().push(term);
        }

        // For each equivalence class with multiple elements
        for (_rep, terms) in classes {
            if terms.len() > 1 {
                // Generate all pairwise equalities, skipping already-propagated ones
                for i in 0..terms.len() {
                    for j in (i + 1)..terms.len() {
                        let key = Self::normalize_pair(terms[i], terms[j]);
                        if !self.propagated_equalities.contains(&key) {
                            implied_equalities.push((terms[i], terms[j]));
                        }
                    }
                }
            }
        }

        implied_equalities
    }

    /// Get statistics.
    pub fn stats(&self) -> &NelsonOppenStats {
        &self.stats
    }

    /// Reset for next SMT check.
    pub fn reset(&mut self) {
        self.shared_terms.clear();
        self.equality_classes = UnionFind::new();
        self.pending_equalities.clear();
        self.propagated_equalities.clear();
        self.theory_assignments.clear();
        self.stats = NelsonOppenStats::default();
        self.fresh_var_counter = 0;
        self.purification_aliases.clear();
    }
}

impl Default for NelsonOppenCombiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Union-Find data structure for equivalence classes.
#[derive(Debug, Clone)]
struct UnionFind {
    parent: FxHashMap<TermId, TermId>,
    rank: FxHashMap<TermId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    fn make_set(&mut self, x: TermId) {
        self.parent.insert(x, x);
        self.rank.insert(x, 0);
    }

    fn find(&self, x: TermId) -> TermId {
        let mut current = x;
        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        x // Not found, return itself
    }

    fn union(&mut self, x: TermId, y: TermId) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root == y_root {
            return;
        }

        let x_rank = *self.rank.get(&x_root).unwrap_or(&0);
        let y_rank = *self.rank.get(&y_root).unwrap_or(&0);

        if x_rank < y_rank {
            self.parent.insert(x_root, y_root);
        } else if x_rank > y_rank {
            self.parent.insert(y_root, x_root);
        } else {
            self.parent.insert(y_root, x_root);
            self.rank.insert(x_root, x_rank + 1);
        }
    }
}

// Placeholder types (these would be defined elsewhere in the codebase)
// Note: Using types from oxiz_core::ast instead
// #[derive(Debug, Clone)]
// struct Term {
//     kind: TermKind,
//     sort: SortId,
// }
//
// #[derive(Debug, Clone)]
// enum TermKind {
//     Var(String),
//     App(FuncId, Vec<TermId>),
//     Const(ConstId),
// }

type SortId = usize;
type FuncId = usize;
type ConstId = usize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelson_oppen_creation() {
        let combiner = NelsonOppenCombiner::new();
        assert_eq!(combiner.stats.shared_terms_count, 0);
    }

    #[test]
    fn test_register_shared_term() {
        let mut combiner = NelsonOppenCombiner::new();
        let term_id = TermId(0);

        combiner.register_shared_term(term_id, TheoryId(0), TheoryId(1));

        assert_eq!(combiner.stats.shared_terms_count, 1);
        assert!(combiner.shared_terms.contains(&term_id));
    }

    #[test]
    fn test_assert_equality() {
        let mut combiner = NelsonOppenCombiner::new();
        let t1 = TermId(0);
        let t2 = TermId(1);

        combiner.register_shared_term(t1, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t2, TheoryId(0), TheoryId(1));

        assert!(combiner.assert_equality(t1, t2).is_ok());
        assert!(combiner.are_equal(t1, t2));
        assert_eq!(combiner.stats.equalities_propagated, 1);
    }

    // ===== Purification regression tests =====
    //
    // Before `purify_term` was rewritten it matched only `TermKind::Apply` and
    // fell through with `_ => Ok(term_id)`; on top of that its
    // `needs_purification` predicate could never be true, so *every* input was
    // returned unpurified while `stats.purifications` counted work that had not
    // happened, and its single definitional equality was discarded by
    // `let _ = self.assert_equality(..)`.

    /// True when no subterm of `term` is an argument of an operator owned by a
    /// different theory -- i.e. `term` is pure in the theory of its own top
    /// symbol. This is the invariant purification must establish.
    fn is_pure(term: TermId, tm: &TermManager) -> bool {
        for parent in collect_subterms(term, tm) {
            let Some(parent_term) = tm.get(parent) else {
                return false;
            };
            let parent_theory = NelsonOppenCombiner::theory_of(&parent_term.kind);
            if parent_theory == TermTheory::Shared {
                continue;
            }
            for child in get_children(&parent_term.kind) {
                let Some(child_term) = tm.get(child) else {
                    return false;
                };
                let child_theory = NelsonOppenCombiner::theory_of(&child_term.kind);
                if child_theory != TermTheory::Shared && child_theory != parent_theory {
                    return false;
                }
            }
        }
        true
    }

    /// `(f (bvadd x y))` mixes UF with bit-vectors and must be split.
    #[test]
    fn purify_splits_uf_over_bitvector() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let bv_sum = tm.mk_bv_add(x, y);
        let mixed = tm.mk_apply("f", [bv_sum], int_sort);

        assert!(!is_pure(mixed, &tm), "the input really is mixed");

        let mut combiner = NelsonOppenCombiner::new();
        let purified = combiner
            .purify_term(mixed, &mut tm)
            .expect("purification should succeed");

        assert_ne!(purified, mixed, "a mixed term must not be returned as-is");
        assert!(is_pure(purified, &tm), "result must be pure");

        // The bit-vector subterm is gone from the purified term, and its
        // definition was recorded as a shared equality.
        assert!(
            !collect_subterms(purified, &tm).contains(&bv_sum),
            "the foreign subterm must be replaced by its alias"
        );
        let pending = combiner.get_pending_equalities();
        assert_eq!(pending.len(), 1, "exactly one definitional equality");
        let (alias, defined) = pending[0];
        assert_eq!(defined, bv_sum);
        assert!(
            matches!(tm.get(alias).map(|t| &t.kind), Some(TermKind::Var(_))),
            "the alias must be a fresh variable"
        );
        assert!(combiner.are_equal(alias, bv_sum));
        assert_eq!(combiner.stats().purifications, 1);
    }

    /// `(+ (f (bvadd x y)) 1)` needs two nested aliases, and the definition of
    /// the outer one must itself be purified.
    #[test]
    fn purify_splits_nested_mixed_term() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let bv_sum = tm.mk_bv_add(x, y);
        let fapp = tm.mk_apply("f", [bv_sum], int_sort);
        let one = tm.mk_int(1);
        let mixed = tm.mk_add([fapp, one]);

        let mut combiner = NelsonOppenCombiner::new();
        let purified = combiner
            .purify_term(mixed, &mut tm)
            .expect("purification should succeed");

        assert!(is_pure(purified, &tm));
        assert_eq!(combiner.stats().purifications, 2);

        let pending = combiner.get_pending_equalities();
        assert_eq!(pending.len(), 2);
        for (alias, defined) in pending {
            assert!(
                matches!(tm.get(alias).map(|t| &t.kind), Some(TermKind::Var(_))),
                "alias must be a variable"
            );
            assert!(
                is_pure(defined, &tm),
                "every definition must itself be pure"
            );
        }
    }

    /// Arithmetic over an array select: the old code's `_ => Ok(term_id)` arm.
    #[test]
    fn purify_splits_arithmetic_over_array() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let arr = tm.mk_var("a", int_sort);
        let i = tm.mk_var("i", int_sort);
        let sel = tm.mk_select(arr, i);
        let one = tm.mk_int(1);
        let mixed = tm.mk_add([sel, one]);

        let mut combiner = NelsonOppenCombiner::new();
        let purified = combiner
            .purify_term(mixed, &mut tm)
            .expect("purification should succeed");

        assert_ne!(purified, mixed);
        assert!(is_pure(purified, &tm));
        assert_eq!(combiner.get_pending_equalities().len(), 1);
    }

    /// A term already pure in one theory is returned unchanged, with no fresh
    /// variables and no equalities.
    #[test]
    fn purify_leaves_a_pure_term_alone() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let pure = tm.mk_bv_add(x, y);

        let mut combiner = NelsonOppenCombiner::new();
        assert_eq!(combiner.purify_term(pure, &mut tm), Ok(pure));
        assert!(combiner.get_pending_equalities().is_empty());
        assert_eq!(combiner.stats().purifications, 0);
    }

    /// The Boolean skeleton and `=` are shared, so a bit-vector literal under
    /// `and` is not a purification boundary.
    #[test]
    fn purify_does_not_split_the_boolean_skeleton() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let sum = tm.mk_bv_add(x, y);
        let eq = tm.mk_eq(sum, x);
        let ult = tm.mk_bv_ult(x, y);
        let formula = tm.mk_and([eq, ult]);

        let mut combiner = NelsonOppenCombiner::new();
        assert_eq!(combiner.purify_term(formula, &mut tm), Ok(formula));
        assert!(combiner.get_pending_equalities().is_empty());
    }

    /// Repeated calls share one alias per subterm rather than minting a new one
    /// (and a new definitional equality) each time.
    #[test]
    fn purify_reuses_aliases_across_calls() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bv8 = tm.sorts.bitvec(8);
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let bv_sum = tm.mk_bv_add(x, y);
        let f = tm.mk_apply("f", [bv_sum], int_sort);
        let g = tm.mk_apply("g", [bv_sum], int_sort);

        let mut combiner = NelsonOppenCombiner::new();
        combiner.purify_term(f, &mut tm).expect("purify f");
        combiner.purify_term(g, &mut tm).expect("purify g");

        assert_eq!(
            combiner.stats().purifications,
            1,
            "one alias for the one shared bit-vector subterm"
        );
    }

    /// Purification of a binder is an error, not a silent no-op.
    #[test]
    fn purify_rejects_binders() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bool_sort = tm.sorts.bool_sort;
        let p = tm.mk_var("p", bool_sort);
        let quantified = tm.mk_forall([("x", int_sort)], p);

        let mut combiner = NelsonOppenCombiner::new();
        let err = combiner
            .purify_term(quantified, &mut tm)
            .expect_err("a binder must be reported, not silently returned");
        assert!(err.contains("binding form"), "unexpected message: {err}");
    }

    /// An absent term is an error rather than a defaulted result.
    #[test]
    fn purify_reports_missing_terms() {
        let mut tm = TermManager::new();
        let mut combiner = NelsonOppenCombiner::new();
        assert!(combiner.purify_term(TermId::from(9999), &mut tm).is_err());
    }

    /// Every `TermKind` reachable through the builders gets a classification;
    /// the exhaustive `match` in `theory_of` is what guarantees this for the
    /// rest.
    #[test]
    fn theory_of_classifies_representative_kinds() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let bv8 = tm.sorts.bitvec(8);
        let i = tm.mk_var("i", int_sort);
        let x = tm.mk_var("x", bv8);
        let one = tm.mk_int(1);

        let cases: Vec<(TermId, TermTheory)> = vec![
            (i, TermTheory::Shared),
            (one, TermTheory::Shared),
            (tm.mk_add([i, one]), TermTheory::Arithmetic),
            (tm.mk_bv_add(x, x), TermTheory::BitVector),
            (tm.mk_select(i, one), TermTheory::Array),
            (tm.mk_apply("f", [i], int_sort), TermTheory::Uf),
            (tm.mk_dt_selector("head", i, int_sort), TermTheory::Datatype),
            (tm.mk_str_len(i), TermTheory::String),
            (tm.mk_fp_abs(i), TermTheory::FloatingPoint),
        ];

        for (term, expected) in cases {
            let kind = &tm.get(term).expect("term exists").kind;
            assert_eq!(
                NelsonOppenCombiner::theory_of(kind),
                expected,
                "wrong theory for {kind:?}"
            );
        }
    }

    #[test]
    fn test_convexity_closure() {
        let mut combiner = NelsonOppenCombiner::new();
        let t1 = TermId(0);
        let t2 = TermId(1);
        let t3 = TermId(2);

        combiner.register_shared_term(t1, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t2, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t3, TheoryId(0), TheoryId(1));

        combiner
            .assert_equality(t1, t2)
            .expect("test operation should succeed");
        combiner
            .assert_equality(t2, t3)
            .expect("test operation should succeed");

        let implied = combiner.convexity_closure();
        assert!(!implied.is_empty());
    }
}
