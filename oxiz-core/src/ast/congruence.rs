//! Congruence closure for efficient equality reasoning
//!
//! This module implements an advanced congruence closure data structure for maintaining
//! and reasoning about equalities between terms. It's a fundamental component
//! for equality reasoning in SMT solvers.
//!
//! Features:
//! - Backtrackable union-find for incremental solving (push/pop)
//! - Explanation tracking for proof generation
//! - Efficient worklist-based propagation
//! - Disequality reasoning and conflict detection

use crate::ast::traversal::get_children;
use crate::ast::{RoundingMode, TermId, TermKind, TermManager};
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use core::mem::Discriminant;
use smallvec::SmallVec;

/// Operator identity of a term, with all child terms erased.
///
/// Two terms are congruent exactly when their `OpKey`s are equal and their
/// argument lists are pairwise equivalent. The key therefore has to capture
/// *everything* about a term except its children: the `TermKind`
/// discriminant plus any non-child payload (function symbol, constructor
/// name, extraction bounds, rounding mode, format widths).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OpKey {
    /// Operator fully determined by its `TermKind` discriminant.
    Plain(Discriminant<TermKind>),
    /// Uninterpreted function application `f(...)`.
    Func(Spur),
    /// Datatype constructor application.
    Constructor(Spur),
    /// Datatype tester `is-C`.
    Tester(Spur),
    /// Datatype selector.
    Selector(Spur),
    /// Bit-vector extraction, which carries its bit range.
    Extract {
        /// High bit index.
        high: u32,
        /// Low bit index.
        low: u32,
    },
    /// Floating-point operator carrying a rounding mode.
    Rounded(Discriminant<TermKind>, RoundingMode),
    /// Floating-point conversion carrying a target format.
    Format {
        /// Discriminant of the conversion operator.
        op: Discriminant<TermKind>,
        /// Rounding mode of the conversion.
        rm: RoundingMode,
        /// Target exponent width (or bit-vector width for `fp.to_?bv`).
        eb: u32,
        /// Target significand width (`0` for bit-vector targets).
        sb: u32,
    },
}

/// Congruence signature of a term: its operator identity and its arguments.
///
/// Returns `None` for terms that must not participate in congruence closure:
///
/// * nullary terms (constants, variables, string/FP literals) — they have no
///   arguments, so congruence over them degenerates to syntactic identity,
///   which the hash-consed term manager already provides; and
/// * binders (`Forall`, `Exists`, `Let`, `Match`) — congruence over a body
///   that mentions bound variables is unsound, because equality of the
///   bodies is only meaningful relative to the binding context.
///
/// Every `TermKind` variant is listed explicitly so that adding a variant is
/// a compile error rather than a silently dropped congruence.
fn congruence_signature(kind: &TermKind) -> Option<(OpKey, SmallVec<[TermId; 4]>)> {
    let op = match kind {
        // Nullary: nothing to be congruent about.
        TermKind::True
        | TermKind::False
        | TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::StringLit(_)
        | TermKind::Var(_)
        | TermKind::FpLit { .. }
        | TermKind::FpPlusInfinity { .. }
        | TermKind::FpMinusInfinity { .. }
        | TermKind::FpPlusZero { .. }
        | TermKind::FpMinusZero { .. }
        | TermKind::FpNaN { .. } => return None,

        // Binders: congruence below a binder is not sound.
        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => return None,

        // Operators whose identity is exactly their discriminant.
        TermKind::Not(_)
        | TermKind::And(_)
        | TermKind::Or(_)
        | TermKind::Xor(_, _)
        | TermKind::Implies(_, _)
        | TermKind::Ite(_, _, _)
        | TermKind::Eq(_, _)
        | TermKind::Distinct(_)
        | TermKind::Neg(_)
        | TermKind::Add(_)
        | TermKind::Sub(_, _)
        | TermKind::Mul(_)
        | TermKind::Div(_, _)
        | TermKind::Mod(_, _)
        | TermKind::Lt(_, _)
        | TermKind::Le(_, _)
        | TermKind::Gt(_, _)
        | TermKind::Ge(_, _)
        | TermKind::BvConcat(_, _)
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
        | TermKind::BvSle(_, _)
        | TermKind::Select(_, _)
        | TermKind::Store(_, _, _)
        | TermKind::StrConcat(_, _)
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
        | TermKind::StrFromCode(_)
        | TermKind::FpAbs(_)
        | TermKind::FpNeg(_)
        | TermKind::FpRem(_, _)
        | TermKind::FpMin(_, _)
        | TermKind::FpMax(_, _)
        | TermKind::FpLeq(_, _)
        | TermKind::FpLt(_, _)
        | TermKind::FpGeq(_, _)
        | TermKind::FpGt(_, _)
        | TermKind::FpEq(_, _)
        | TermKind::FpIsNormal(_)
        | TermKind::FpIsSubnormal(_)
        | TermKind::FpIsZero(_)
        | TermKind::FpIsInfinite(_)
        | TermKind::FpIsNaN(_)
        | TermKind::FpIsNegative(_)
        | TermKind::FpIsPositive(_)
        | TermKind::FpToReal(_) => OpKey::Plain(core::mem::discriminant(kind)),

        TermKind::BvExtract { high, low, .. } => OpKey::Extract {
            high: *high,
            low: *low,
        },

        TermKind::Apply { func, .. } => OpKey::Func(*func),
        TermKind::DtConstructor { constructor, .. } => OpKey::Constructor(*constructor),
        TermKind::DtTester { constructor, .. } => OpKey::Tester(*constructor),
        TermKind::DtSelector { selector, .. } => OpKey::Selector(*selector),

        TermKind::FpSqrt(rm, _)
        | TermKind::FpRoundToIntegral(rm, _)
        | TermKind::FpAdd(rm, _, _)
        | TermKind::FpSub(rm, _, _)
        | TermKind::FpMul(rm, _, _)
        | TermKind::FpDiv(rm, _, _)
        | TermKind::FpFma(rm, _, _, _) => OpKey::Rounded(core::mem::discriminant(kind), *rm),

        TermKind::FpToFp { rm, eb, sb, .. }
        | TermKind::RealToFp { rm, eb, sb, .. }
        | TermKind::SBVToFp { rm, eb, sb, .. }
        | TermKind::UBVToFp { rm, eb, sb, .. } => OpKey::Format {
            op: core::mem::discriminant(kind),
            rm: *rm,
            eb: *eb,
            sb: *sb,
        },

        TermKind::FpToSBV { rm, width, .. } | TermKind::FpToUBV { rm, width, .. } => {
            OpKey::Format {
                op: core::mem::discriminant(kind),
                rm: *rm,
                eb: *width,
                sb: 0,
            }
        }
    };

    Some((op, get_children(kind)))
}

/// Explanation for why two terms are equal
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Explanation {
    /// Given equality (asserted)
    Given,
    /// Congruence: f(a1,...,an) = f(b1,...,bn) because ai = bi for all i
    Congruence(Vec<(TermId, TermId)>),
    /// Transitivity: a = c via b (a = b and b = c)
    Transitivity(TermId),
}

/// Undo operation for backtracking
#[derive(Debug, Clone)]
enum UndoOp {
    /// Undo a merge operation: restore parent[child] = old_parent
    Merge { child: TermId, old_parent: TermId },
    /// Undo a lookup insertion
    LookupInsert { key: (OpKey, Vec<TermId>) },
    /// Undo a use list insertion
    UseListInsert { arg: TermId, parent: TermId },
    /// Undo a disequality insertion (only recorded when the pair was newly
    /// inserted, i.e. not already asserted by an outer scope).
    DiseqInsert { pair: (TermId, TermId) },
    /// Undo a union-by-rank increment.
    RankChange { term: TermId, old_rank: usize },
    /// Undo an explanation insertion, restoring whatever (if anything) was
    /// there before.
    ExplanationInsert {
        key: (TermId, TermId),
        old_value: Option<Explanation>,
    },
}

/// Congruence closure data structure with advanced features
///
/// Maintains equivalence classes of terms under the congruence relation.
/// Two terms are congruent if they have the same function symbol and their
/// arguments are pairwise equivalent.
#[derive(Debug, Clone)]
pub struct CongruenceClosure {
    /// Union-find parent pointers
    parent: FxHashMap<TermId, TermId>,
    /// Rank for union-by-rank heuristic
    rank: FxHashMap<TermId, usize>,
    /// Explanations for why terms are equal
    explanations: FxHashMap<(TermId, TermId), Explanation>,
    /// Lookup table for congruence: maps (operator, normalized args) to the
    /// representative term registered for that signature.
    ///
    /// The key deliberately does **not** contain the term itself: a key that
    /// includes the term can never collide with another term's key, so the
    /// table could never detect a congruence at all.
    lookup: FxHashMap<(OpKey, Vec<TermId>), TermId>,
    /// Use list: for each term, which terms use it as an argument
    use_list: FxHashMap<TermId, Vec<TermId>>,
    /// Worklist for pending propagations
    worklist: Vec<TermId>,
    /// Disequalities: set of pairs (a, b) where a ≠ b
    diseqs: FxHashSet<(TermId, TermId)>,
    /// Undo trail for backtracking
    undo_trail: Vec<UndoOp>,
    /// Scope levels for push/pop
    scope_levels: Vec<usize>,
}

impl CongruenceClosure {
    /// Create a new empty congruence closure
    #[must_use]
    pub fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
            explanations: FxHashMap::default(),
            lookup: FxHashMap::default(),
            use_list: FxHashMap::default(),
            worklist: Vec::new(),
            diseqs: FxHashSet::default(),
            undo_trail: Vec::new(),
            scope_levels: vec![0],
        }
    }

    /// Find the representative of a term's equivalence class (without path compression for backtracking)
    pub fn find(&mut self, term: TermId) -> TermId {
        if let crate::prelude::hash_map::Entry::Vacant(e) = self.parent.entry(term) {
            e.insert(term);
            self.rank.insert(term, 0);
            return term;
        }

        // Iterative find to avoid path compression (which would break undo trail)
        let mut current = term;
        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        current
    }

    /// Find with path halving (lighter compression that's easier to undo)
    #[allow(dead_code)]
    fn find_with_halving(&mut self, term: TermId) -> TermId {
        let mut current = term;
        loop {
            let parent = match self.parent.get(&current) {
                Some(&p) if p != current => p,
                _ => return current,
            };

            // Path halving: make current point to grandparent
            if let Some(&grandparent) = self.parent.get(&parent)
                && grandparent != parent
            {
                self.parent.insert(current, grandparent);
            }

            current = parent;
        }
    }

    /// Check if two terms are in the same equivalence class
    pub fn are_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.find(a) == self.find(b)
    }

    /// Push a new scope for backtracking
    pub fn push(&mut self) {
        self.scope_levels.push(self.undo_trail.len());
    }

    /// Pop the most recent scope, undoing all operations since the last push
    pub fn pop(&mut self) {
        if self.scope_levels.len() <= 1 {
            return; // Cannot pop base level
        }

        let target_level = self
            .scope_levels
            .pop()
            .expect("scope_levels has elements after length check");

        // Undo all operations back to the target level
        while self.undo_trail.len() > target_level {
            if let Some(op) = self.undo_trail.pop() {
                match op {
                    UndoOp::Merge { child, old_parent } => {
                        self.parent.insert(child, old_parent);
                    }
                    UndoOp::LookupInsert { key } => {
                        self.lookup.remove(&key);
                    }
                    UndoOp::UseListInsert { arg, parent } => {
                        if let Some(list) = self.use_list.get_mut(&arg) {
                            list.retain(|&p| p != parent);
                        }
                    }
                    UndoOp::DiseqInsert { pair } => {
                        self.diseqs.remove(&pair);
                    }
                    UndoOp::RankChange { term, old_rank } => {
                        self.rank.insert(term, old_rank);
                    }
                    UndoOp::ExplanationInsert { key, old_value } => match old_value {
                        Some(e) => {
                            self.explanations.insert(key, e);
                        }
                        None => {
                            self.explanations.remove(&key);
                        }
                    },
                }
            }
        }

        // Clear worklist
        self.worklist.clear();
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.explanations.clear();
        self.lookup.clear();
        self.use_list.clear();
        self.worklist.clear();
        self.diseqs.clear();
        self.undo_trail.clear();
        self.scope_levels = vec![0];
    }

    /// Add a disequality constraint: a ≠ b
    /// Returns None if no conflict, or Some((a, b)) if this creates a conflict
    pub fn assert_diseq(&mut self, a: TermId, b: TermId) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        // Conflict: asserting a ≠ b but they're already equal
        if a_root == b_root {
            return Some((a, b));
        }

        // Normalize the pair
        let pair = if a_root.0 < b_root.0 {
            (a_root, b_root)
        } else {
            (b_root, a_root)
        };

        // Only record an undo entry when the pair is genuinely new: if an
        // outer (still-active) scope already asserted this disequality,
        // popping this scope must not erase it.
        if self.diseqs.insert(pair) {
            self.undo_trail.push(UndoOp::DiseqInsert { pair });
        }
        None
    }

    /// Check if asserting a = b would violate any disequality
    fn check_diseq_conflict(&mut self, a: TermId, b: TermId) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        let pair = if a_root.0 < b_root.0 {
            (a_root, b_root)
        } else {
            (b_root, a_root)
        };

        if self.diseqs.contains(&pair) {
            Some(pair)
        } else {
            None
        }
    }

    /// Get explanation for why two terms are equal
    #[must_use]
    pub fn get_explanation(&self, a: TermId, b: TermId) -> Option<Explanation> {
        let key = if a.0 < b.0 { (a, b) } else { (b, a) };
        self.explanations.get(&key).cloned()
    }

    /// Add a term and all of its subterms to the congruence closure.
    ///
    /// Subterms are visited with an explicit worklist (never by recursion),
    /// so an arbitrarily deep term cannot overflow the stack. Every term with
    /// a congruence signature — including uninterpreted function
    /// applications `f(a1..an)` and datatype constructors/selectors/testers —
    /// is registered in the signature table and in the use-lists of its
    /// arguments, which is what makes congruence propagation see it at all.
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) {
        let mut stack = vec![term];
        let mut seen = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !seen.insert(current) {
                continue;
            }

            // Initialize the term if not present
            if let crate::prelude::hash_map::Entry::Vacant(e) = self.parent.entry(current) {
                e.insert(current);
                self.rank.insert(current, 0);
            }

            let Some(t) = manager.get(current) else {
                continue;
            };
            let Some((op, args)) = congruence_signature(&t.kind) else {
                // Nullary terms and binders carry no congruence signature;
                // they are registered as singleton classes above and that is
                // all that is required of them.
                continue;
            };

            // Normalize args by finding representatives
            let normalized_args: Vec<_> = args.iter().map(|&a| self.find(a)).collect();
            let key = (op, normalized_args);

            // Check for congruence against an already-registered term with
            // the same signature.
            match self.lookup.get(&key).copied() {
                Some(existing) if existing != current => {
                    let arg_pairs = self.get_argument_pairs(existing, current, manager);
                    let _ = self.merge(existing, current, Explanation::Congruence(arg_pairs));
                }
                Some(_) => {}
                None => {
                    self.lookup.insert(key.clone(), current);
                    self.undo_trail.push(UndoOp::LookupInsert { key });
                }
            }

            // Update use lists and queue the arguments themselves.
            for &arg in &args {
                let list = self.use_list.entry(arg).or_default();
                // Only record an undo entry for a genuinely new use-list
                // edge: re-adding the same term must not let a pop erase the
                // edge an outer scope established.
                if !list.contains(&current) {
                    list.push(current);
                    self.undo_trail.push(UndoOp::UseListInsert {
                        arg,
                        parent: current,
                    });
                }
                stack.push(arg);
            }
        }
    }

    /// Merge two equivalence classes with explanation
    /// Returns Some(conflict) if this merge violates a disequality, None otherwise
    pub fn merge(
        &mut self,
        a: TermId,
        b: TermId,
        explanation: Explanation,
    ) -> Option<(TermId, TermId)> {
        let a_root = self.find(a);
        let b_root = self.find(b);

        if a_root == b_root {
            return None; // Already in same class
        }

        // Check for disequality conflict
        if let Some(conflict) = self.check_diseq_conflict(a_root, b_root) {
            return Some(conflict);
        }

        // Union by rank
        let a_rank = self.rank.get(&a_root).copied().unwrap_or(0);
        let b_rank = self.rank.get(&b_root).copied().unwrap_or(0);

        let (child, parent) = if a_rank < b_rank {
            (a_root, b_root)
        } else if a_rank > b_rank {
            (b_root, a_root)
        } else {
            // Equal ranks: increase parent's rank
            self.undo_trail.push(UndoOp::RankChange {
                term: a_root,
                old_rank: a_rank,
            });
            self.rank.insert(a_root, a_rank + 1);
            (b_root, a_root)
        };

        // Record undo operation
        let old_parent = self.parent.get(&child).copied().unwrap_or(child);
        self.undo_trail.push(UndoOp::Merge { child, old_parent });

        // Perform the merge
        self.parent.insert(child, parent);

        // Store explanation
        let key = if a.0 < b.0 { (a, b) } else { (b, a) };
        let old_explanation = self.explanations.insert(key, explanation);
        self.undo_trail.push(UndoOp::ExplanationInsert {
            key,
            old_value: old_explanation,
        });

        // Add merged terms to worklist for propagation
        self.worklist.push(a_root);
        self.worklist.push(b_root);

        None
    }

    /// Merge without explanation (for internal use)
    fn merge_internal(&mut self, a: TermId, b: TermId) {
        let _ = self.merge(a, b, Explanation::Given);
    }

    /// Process all pending merges and propagate congruences using worklist
    pub fn close(&mut self, manager: &TermManager) {
        let mut processed = FxHashSet::default();

        while let Some(term) = self.worklist.pop() {
            let root = self.find(term);

            // Skip if already processed this root
            if !processed.insert(root) {
                continue;
            }

            // Gather use-lists from *every* term currently in this
            // equivalence class, not just `term` (the worklist entry) and
            // `root` (the current representative). A term can join the
            // class via an earlier merge without ever being pushed to the
            // worklist itself or becoming the representative — e.g. class
            // {x, y, z} with root r: a use of `y` as an argument
            // (`use_list[y]`) would otherwise never be inspected when only
            // `term`/`root`'s own use-lists are consulted, silently
            // missing a congruence between a parent application over `y`
            // and one over `x`/`z`/`r`.
            let all_terms: Vec<TermId> = self.parent.keys().copied().collect();
            let class_members: Vec<TermId> = all_terms
                .into_iter()
                .filter(|&t| self.find(t) == root)
                .collect();

            let mut seen_parents = FxHashSet::default();
            let all_parents: Vec<_> = class_members
                .iter()
                .flat_map(|t| self.use_list.get(t).cloned().unwrap_or_default())
                .filter(|&p| seen_parents.insert(p))
                .collect();

            // Check for congruent parents
            for i in 0..all_parents.len() {
                for j in (i + 1)..all_parents.len() {
                    let parent_a = all_parents[i];
                    let parent_b = all_parents[j];

                    // Check if parent_a and parent_b are congruent
                    if self.are_congruent(parent_a, parent_b, manager) {
                        let pa_root = self.find(parent_a);
                        let pb_root = self.find(parent_b);

                        if pa_root != pb_root {
                            // Merge congruent terms
                            let arg_pairs = self.get_argument_pairs(parent_a, parent_b, manager);
                            let explanation = Explanation::Congruence(arg_pairs);
                            self.merge_internal(pa_root, pb_root);

                            // Store explanation for the original terms
                            let key = if parent_a.0 < parent_b.0 {
                                (parent_a, parent_b)
                            } else {
                                (parent_b, parent_a)
                            };
                            let old_value = self.explanations.insert(key, explanation);
                            self.undo_trail
                                .push(UndoOp::ExplanationInsert { key, old_value });
                        }
                    }
                }
            }
        }
    }

    /// Get the pairs of arguments that justify congruence.
    ///
    /// Returns an empty vector when the two terms do not share an operator,
    /// which is the only case in which no argument pair can justify
    /// anything.
    fn get_argument_pairs(
        &mut self,
        a: TermId,
        b: TermId,
        manager: &TermManager,
    ) -> Vec<(TermId, TermId)> {
        let (Some(ta), Some(tb)) = (manager.get(a), manager.get(b)) else {
            return Vec::new();
        };
        let (Some((op_a, args_a)), Some((op_b, args_b))) = (
            congruence_signature(&ta.kind),
            congruence_signature(&tb.kind),
        ) else {
            return Vec::new();
        };
        if op_a != op_b || args_a.len() != args_b.len() {
            return Vec::new();
        }
        args_a
            .iter()
            .zip(args_b.iter())
            .map(|(&x, &y)| (x, y))
            .collect()
    }

    /// Check if two terms are congruent (same operator, equivalent arguments)
    fn are_congruent(&mut self, a: TermId, b: TermId, manager: &TermManager) -> bool {
        let (Some(ta), Some(tb)) = (manager.get(a), manager.get(b)) else {
            return false;
        };
        let (Some((op_a, args_a)), Some((op_b, args_b))) = (
            congruence_signature(&ta.kind),
            congruence_signature(&tb.kind),
        ) else {
            return false;
        };
        if op_a != op_b || args_a.len() != args_b.len() {
            return false;
        }
        args_a
            .iter()
            .zip(args_b.iter())
            .all(|(&x, &y)| self.are_equal(x, y))
    }

    /// Get all terms in the same equivalence class as the given term
    #[must_use]
    pub fn get_class(&mut self, term: TermId) -> Vec<TermId> {
        let root = self.find(term);
        let terms: Vec<_> = self.parent.keys().copied().collect();
        terms
            .into_iter()
            .filter(|&t| self.find(t) == root)
            .collect()
    }

    /// Get the number of equivalence classes
    #[must_use]
    pub fn num_classes(&mut self) -> usize {
        let terms: Vec<_> = self.parent.keys().copied().collect();
        let mut roots: Vec<_> = terms.iter().map(|&t| self.find(t)).collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    }
}

impl Default for CongruenceClosure {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_closure() {
        let cc = CongruenceClosure::new();
        assert_eq!(cc.parent.len(), 0);
    }

    #[test]
    fn test_find_creates_class() {
        let mut cc = CongruenceClosure::new();
        let term = TermId(1);
        let root = cc.find(term);
        assert_eq!(root, term);
        assert_eq!(cc.find(term), root);
    }

    #[test]
    fn test_merge_same_class() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));
    }

    #[test]
    fn test_transitivity() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        cc.merge(a, b, Explanation::Given);
        cc.merge(b, c, Explanation::Given);

        // a = b and b = c implies a = c
        assert!(cc.are_equal(a, c));
    }

    #[test]
    fn test_get_class() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        cc.merge(a, b, Explanation::Given);
        cc.merge(b, c, Explanation::Given);

        let class = cc.get_class(a);
        assert_eq!(class.len(), 3);
        assert!(class.contains(&a));
        assert!(class.contains(&b));
        assert!(class.contains(&c));
    }

    #[test]
    fn test_num_classes() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);
        let d = TermId(4);

        cc.find(a);
        cc.find(b);
        cc.find(c);
        cc.find(d);
        assert_eq!(cc.num_classes(), 4);

        cc.merge(a, b, Explanation::Given);
        assert_eq!(cc.num_classes(), 3);

        cc.merge(c, d, Explanation::Given);
        assert_eq!(cc.num_classes(), 2);

        cc.merge(a, c, Explanation::Given);
        assert_eq!(cc.num_classes(), 1);
    }

    #[test]
    fn test_add_term_simple() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        cc.add_term(x, &manager);

        assert!(cc.parent.contains_key(&x));
    }

    #[test]
    fn test_basic_usage() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        // Create some simple terms
        let a = manager.mk_var("a", manager.sorts.int_sort);
        let b = manager.mk_var("b", manager.sorts.int_sort);
        let c = manager.mk_var("c", manager.sorts.int_sort);

        cc.add_term(a, &manager);
        cc.add_term(b, &manager);
        cc.add_term(c, &manager);

        // Initially all different
        assert!(!cc.are_equal(a, b));
        assert!(!cc.are_equal(b, c));

        // Merge a and b
        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));

        // Merge b and c
        cc.merge(b, c, Explanation::Given);
        assert!(cc.are_equal(b, c));
        assert!(cc.are_equal(a, c)); // Transitivity
    }

    #[test]
    fn test_push_pop() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);
        let c = TermId(3);

        // Initial scope
        cc.merge(a, b, Explanation::Given);
        assert!(cc.are_equal(a, b));

        // Push new scope
        cc.push();
        cc.merge(b, c, Explanation::Given);
        assert!(cc.are_equal(a, c));

        // Pop scope - should undo b = c but keep a = b
        cc.pop();
        assert!(cc.are_equal(a, b));
        assert!(!cc.are_equal(b, c));
    }

    #[test]
    fn test_diseq_conflict() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        // Assert a != b
        assert!(cc.assert_diseq(a, b).is_none());

        // Try to merge a and b - should create conflict
        let conflict = cc.merge(a, b, Explanation::Given);
        assert!(conflict.is_some());
    }

    #[test]
    fn test_diseq_after_merge() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        // Merge a and b first
        cc.merge(a, b, Explanation::Given);

        // Try to assert a != b - should create conflict
        let conflict = cc.assert_diseq(a, b);
        assert!(conflict.is_some());
    }

    #[test]
    fn test_congruence_over_uninterpreted_apply() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let b = manager.mk_var("b", int_sort);
        let fa = manager.mk_apply("f", [a], int_sort);
        let fb = manager.mk_apply("f", [b], int_sort);

        cc.add_term(fa, &manager);
        cc.add_term(fb, &manager);

        // Without a = b, f(a) and f(b) are distinct classes.
        assert!(!cc.are_equal(fa, fb));

        cc.merge(a, b, Explanation::Given);
        cc.close(&manager);

        // Congruence: a = b implies f(a) = f(b).
        assert!(cc.are_equal(fa, fb));
    }

    #[test]
    fn test_apply_different_symbols_not_congruent() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let b = manager.mk_var("b", int_sort);
        let fa = manager.mk_apply("f", [a], int_sort);
        let gb = manager.mk_apply("g", [b], int_sort);

        cc.add_term(fa, &manager);
        cc.add_term(gb, &manager);
        cc.merge(a, b, Explanation::Given);
        cc.close(&manager);

        assert!(!cc.are_equal(fa, gb));
    }

    #[test]
    fn test_add_term_registers_subterms() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let b = manager.mk_var("b", int_sort);
        let inner = manager.mk_apply("f", [a, b], int_sort);
        let outer = manager.mk_apply("g", [inner], int_sort);

        cc.add_term(outer, &manager);

        for t in [outer, inner, a, b] {
            assert!(cc.parent.contains_key(&t), "subterm not registered");
        }
    }

    #[test]
    fn test_nested_congruence_propagates_upwards() {
        let mut manager = TermManager::new();
        let mut cc = CongruenceClosure::new();

        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let b = manager.mk_var("b", int_sort);
        let fa = manager.mk_apply("f", [a], int_sort);
        let fb = manager.mk_apply("f", [b], int_sort);
        let gfa = manager.mk_apply("g", [fa], int_sort);
        let gfb = manager.mk_apply("g", [fb], int_sort);

        cc.add_term(gfa, &manager);
        cc.add_term(gfb, &manager);
        cc.merge(a, b, Explanation::Given);
        cc.close(&manager);

        assert!(cc.are_equal(fa, fb));
        assert!(cc.are_equal(gfa, gfb));
    }

    #[test]
    fn test_add_term_deep_nesting_does_not_overflow() {
        // A left-nested chain f(f(...f(a)...)) far beyond any recursion
        // budget: the assertion is simply that `add_term` returns.
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let mut term = manager.mk_var("a", int_sort);
                for _ in 0..7_500 {
                    term = manager.mk_apply("f", [term], int_sort);
                }

                let mut cc = CongruenceClosure::new();
                cc.add_term(term, &manager);
                cc.num_classes()
            })
            .expect("thread spawn should succeed");

        let classes = handle.join().expect("deep add_term must not overflow");
        assert_eq!(classes, 7_501);
    }

    #[test]
    fn test_explanation() {
        let mut cc = CongruenceClosure::new();
        let a = TermId(1);
        let b = TermId(2);

        cc.merge(a, b, Explanation::Given);

        // Should have an explanation for this merge
        let exp = cc.get_explanation(a, b);
        assert!(exp.is_some());
        assert_eq!(
            exp.expect("test operation should succeed"),
            Explanation::Given
        );
    }
}
