//! BitVector theory implementation
//!
//! Implements a lightweight theory of fixed-width bit-vectors: it tracks the
//! registered bit-vector terms, keeps equality classes over them, folds
//! operations whose arguments are known constants, closes the classes under
//! congruence, and reports a conflict when two different constants are forced
//! into one class.
//!
//! This is the self-contained layer that lives inside `oxiz-core`. It is not
//! the bit-blasting solver used by `oxiz-solver` — that is
//! `oxiz_theories::bv` — and it decides nothing on its own: a run that finds
//! no conflict has not shown the input satisfiable.
//!
//! Reference: Z3's `src/smt/theory_bv.cpp`

use super::combination::{Theory, TheoryResult};
use super::eq_classes::EqClasses;
use crate::ast::traversal::get_children;
use crate::ast::{TermId, TermKind, TermManager, bv_wrap_unsigned};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortKind, SortManager};
use num_bigint::BigInt;

/// BitVector theory axioms and rewrite rules
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BitVectorAxiom {
    /// bvadd identity: bvadd(x, 0) = x
    AddIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvmul identity: bvmul(x, 1) = x
    MulIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvmul by zero: bvmul(x, 0) = 0
    MulZero {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvand identity: bvand(x, all_ones) = x
    AndIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvand zero: bvand(x, 0) = 0
    AndZero {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvor identity: bvor(x, 0) = x
    OrIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvor saturation: bvor(x, all_ones) = all_ones
    OrSaturation {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvxor identity: bvxor(x, 0) = x
    XorIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvxor self: bvxor(x, x) = 0
    XorSelf {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
    },
    /// bvnot involution: bvnot(bvnot(x)) = x
    NotInvolution {
        /// The term this axiom applies to
        term: TermId,
    },
    /// bvneg zero: bvneg(0) = 0
    NegZero {
        /// The bit-width of the bitvector
        width: u32,
    },
    /// Bit-blast rule for comparison: converts bv operations to bit-level operations
    BitBlast {
        /// The term this axiom applies to
        term: TermId,
        /// The bit-width of the bitvector
        width: u32,
        /// The bitvector operation to bit-blast
        operation: BitVectorOp,
    },
}

/// BitVector operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitVectorOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Unsigned division
    UDiv,
    /// Signed division
    SDiv,
    /// Unsigned remainder
    URem,
    /// Signed remainder
    SRem,
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Negation (two's complement)
    Neg,
    /// Left shift
    Shl,
    /// Logical right shift
    LShr,
    /// Arithmetic right shift
    AShr,
}

/// BitVector theory reasoning engine
#[derive(Debug, Clone)]
pub struct BitVectorTheory {
    /// Tracked bitvector terms (maps term to its width)
    bitvectors: FxHashMap<TermId, u32>,
    /// Equality classes over the registered bit-vector terms
    classes: EqClasses,
    /// Equalities already reported by `propagate`, so each is reported once
    reported: FxHashSet<(TermId, TermId)>,
    /// Pending axiom instantiations
    pending_axioms: Vec<BitVectorAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<BitVectorAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

impl Default for BitVectorTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl BitVectorTheory {
    /// Create a new bitvector theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            bitvectors: FxHashMap::default(),
            classes: EqClasses::new(),
            reported: FxHashSet::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a bitvector term with its width
    pub fn register_bitvector(&mut self, term: TermId, width: u32) {
        self.bitvectors.insert(term, width);
        self.classes.add(term);
    }

    /// Add a term to the theory and extract bitvector operations
    ///
    /// Returns `true` when the term has a bit-vector sort, i.e. when this
    /// theory has taken it on.
    pub fn add_term(
        &mut self,
        term: TermId,
        manager: &TermManager,
        sort_manager: &SortManager,
    ) -> bool {
        if let Some(t) = manager.get(term)
            && let Some(sort) = sort_manager.get(t.sort)
            && let SortKind::BitVec(width) = sort.kind
        {
            self.register_bitvector(term, width);

            // Generate axioms based on term structure
            let kind = t.kind.clone();
            self.generate_axioms_for_term(term, &kind, width, manager);
            return true;
        }
        false
    }

    /// Record that two registered bit-vector terms are equal
    ///
    /// Returns `true` when this was new information: both terms are known to
    /// this theory, they have the same width, and they were not already in the
    /// same class. Equalities about unknown terms, and equalities between
    /// terms of different widths (which are not well sorted), are ignored.
    pub fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        let (Some(&width_a), Some(&width_b)) = (self.bitvectors.get(&a), self.bitvectors.get(&b))
        else {
            return false;
        };
        if width_a != width_b {
            return false;
        }

        self.classes.union(a, b)
    }

    /// Whether two terms are currently known to be equal by this theory
    ///
    /// Takes `&mut self` because the lookup compresses the union-find paths.
    pub fn known_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.classes.are_equal(a, b)
    }

    /// Generate axioms for a bitvector term
    fn generate_axioms_for_term(
        &mut self,
        term: TermId,
        kind: &TermKind,
        width: u32,
        manager: &TermManager,
    ) {
        match kind {
            TermKind::BvAdd(lhs, rhs)
                if self.is_bv_zero(*rhs, manager, width)
                    || self.is_bv_zero(*lhs, manager, width) =>
            {
                self.add_axiom(BitVectorAxiom::AddIdentity { term, width });
            }
            TermKind::BvMul(lhs, rhs) => {
                // Check for multiplication by zero or one
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::MulZero { term, width });
                } else if self.is_bv_one(*rhs, manager, width)
                    || self.is_bv_one(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::MulIdentity { term, width });
                }
            }
            TermKind::BvAnd(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::AndZero { term, width });
                } else if self.is_bv_all_ones(*rhs, manager, width)
                    || self.is_bv_all_ones(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::AndIdentity { term, width });
                }
            }
            TermKind::BvOr(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::OrIdentity { term, width });
                } else if self.is_bv_all_ones(*rhs, manager, width)
                    || self.is_bv_all_ones(*lhs, manager, width)
                {
                    self.add_axiom(BitVectorAxiom::OrSaturation { term, width });
                }
            }
            TermKind::BvXor(lhs, rhs) => {
                if self.is_bv_zero(*rhs, manager, width) || self.is_bv_zero(*lhs, manager, width) {
                    self.add_axiom(BitVectorAxiom::XorIdentity { term, width });
                } else if lhs == rhs {
                    self.add_axiom(BitVectorAxiom::XorSelf { term, width });
                }
            }
            TermKind::BvNot(inner) => {
                // Check for double negation
                if let Some(inner_term) = manager.get(*inner)
                    && let TermKind::BvNot(_) = inner_term.kind
                {
                    self.add_axiom(BitVectorAxiom::NotInvolution { term });
                }
            }
            _ => {}
        }
    }

    /// Value of a bit-vector literal, normalised into `[0, 2^width)`
    ///
    /// A literal may be stored with a negative or out-of-range value (`-1` and
    /// `255` are the same 8-bit vector), so every comparison in this module
    /// goes through the normalisation rather than comparing raw values.
    fn bv_literal_value(term: TermId, manager: &TermManager) -> Option<(u32, BigInt)> {
        match &manager.get(term)?.kind {
            TermKind::BitVecConst { value, width } => {
                Some((*width, bv_wrap_unsigned(value, *width)))
            }
            _ => None,
        }
    }

    /// Check if a term is the bit-vector zero of the given width
    fn is_bv_zero(&self, term: TermId, manager: &TermManager, width: u32) -> bool {
        Self::bv_literal_value(term, manager)
            .is_some_and(|(term_width, value)| term_width == width && value == BigInt::ZERO)
    }

    /// Check if a term is the bit-vector one of the given width
    fn is_bv_one(&self, term: TermId, manager: &TermManager, width: u32) -> bool {
        Self::bv_literal_value(term, manager)
            .is_some_and(|(term_width, value)| term_width == width && value == BigInt::from(1u8))
    }

    /// Check if a term is all ones (e.g., 0xFF for width=8)
    fn is_bv_all_ones(&self, term: TermId, manager: &TermManager, width: u32) -> bool {
        Self::bv_literal_value(term, manager).is_some_and(|(term_width, value)| {
            term_width == width && value == crate::ast::bv_fold::all_ones(width)
        })
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: BitVectorAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<BitVectorAxiom> {
        core::mem::take(&mut self.pending_axioms)
    }

    /// Build the equation stated by an axiom
    ///
    /// Returns the equality `axiom_term = simplified_form`, e.g. `bvadd(x, 0)
    /// = x` for [`BitVectorAxiom::AddIdentity`].
    ///
    /// Returns `None` when the axiom cannot be turned into a term:
    ///
    /// * the axiom's term is not in `manager`, or does not have the operator
    ///   the axiom claims (an `AddIdentity` whose term is not a `bvadd`, or
    ///   whose operands are not the constant the axiom names);
    /// * [`BitVectorAxiom::NegZero`], because `bvneg` has no term kind of its
    ///   own here — [`TermManager::mk_bv_neg`] folds `bvneg(0)` straight to
    ///   `0`, so the equation would degenerate to `true`;
    /// * [`BitVectorAxiom::BitBlast`], because this layer has no bit-level
    ///   encoding; bit-blasting lives in `oxiz_theories::bv`.
    pub fn axiom_to_term(
        &self,
        axiom: &BitVectorAxiom,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let (term, replacement) = match axiom {
            BitVectorAxiom::AddIdentity { term, width } => {
                let TermKind::BvAdd(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                let other = self.other_operand(lhs, rhs, manager, *width, Self::is_bv_zero)?;
                (*term, other)
            }
            BitVectorAxiom::MulIdentity { term, width } => {
                let TermKind::BvMul(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                let other = self.other_operand(lhs, rhs, manager, *width, Self::is_bv_one)?;
                (*term, other)
            }
            BitVectorAxiom::MulZero { term, width } => {
                let TermKind::BvMul(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if !self.is_bv_zero(lhs, manager, *width) && !self.is_bv_zero(rhs, manager, *width)
                {
                    return None;
                }
                (*term, manager.mk_bitvec(BigInt::ZERO, *width))
            }
            BitVectorAxiom::AndIdentity { term, width } => {
                let TermKind::BvAnd(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                let other = self.other_operand(lhs, rhs, manager, *width, Self::is_bv_all_ones)?;
                (*term, other)
            }
            BitVectorAxiom::AndZero { term, width } => {
                let TermKind::BvAnd(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if !self.is_bv_zero(lhs, manager, *width) && !self.is_bv_zero(rhs, manager, *width)
                {
                    return None;
                }
                (*term, manager.mk_bitvec(BigInt::ZERO, *width))
            }
            BitVectorAxiom::OrIdentity { term, width } => {
                let TermKind::BvOr(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                let other = self.other_operand(lhs, rhs, manager, *width, Self::is_bv_zero)?;
                (*term, other)
            }
            BitVectorAxiom::OrSaturation { term, width } => {
                let TermKind::BvOr(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if !self.is_bv_all_ones(lhs, manager, *width)
                    && !self.is_bv_all_ones(rhs, manager, *width)
                {
                    return None;
                }
                let all_ones = crate::ast::bv_fold::all_ones(*width);
                (*term, manager.mk_bitvec(all_ones, *width))
            }
            BitVectorAxiom::XorIdentity { term, width } => {
                let TermKind::BvXor(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                let other = self.other_operand(lhs, rhs, manager, *width, Self::is_bv_zero)?;
                (*term, other)
            }
            BitVectorAxiom::XorSelf { term, width } => {
                let TermKind::BvXor(lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if lhs != rhs {
                    return None;
                }
                (*term, manager.mk_bitvec(BigInt::ZERO, *width))
            }
            BitVectorAxiom::NotInvolution { term } => {
                let TermKind::BvNot(inner) = manager.get(*term)?.kind else {
                    return None;
                };
                let TermKind::BvNot(innermost) = manager.get(inner)?.kind else {
                    return None;
                };
                (*term, innermost)
            }
            BitVectorAxiom::NegZero { .. } | BitVectorAxiom::BitBlast { .. } => return None,
        };

        Some(manager.mk_eq(term, replacement))
    }

    /// The operand that is *not* the constant the caller is looking for
    ///
    /// Returns `None` when neither operand is that constant.
    fn other_operand(
        &self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
        width: u32,
        is_constant: fn(&Self, TermId, &TermManager, u32) -> bool,
    ) -> Option<TermId> {
        if is_constant(self, rhs, manager, width) {
            Some(lhs)
        } else if is_constant(self, lhs, manager, width) {
            Some(rhs)
        } else {
            None
        }
    }

    /// Deduce new equalities between the registered bit-vector terms
    ///
    /// Two rules are applied to a fixpoint:
    ///
    /// * *constant folding* — an operation all of whose arguments are known
    ///   equal to literals is equal to the folded literal;
    /// * *congruence* — two applications of the same operator whose arguments
    ///   are pairwise known equal are equal to each other.
    ///
    /// Each deduced equality is merged into this theory's classes and returned
    /// once; later calls return only what is new since the previous one.
    pub fn propagate(&mut self, manager: &mut TermManager) -> Vec<(TermId, TermId)> {
        let mut deduced = Vec::new();

        loop {
            let mut round = self.fold_constants(manager);
            round.extend(self.congruent_pairs(manager));

            let mut progressed = false;
            for (a, b) in round {
                let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                let merged = self.classes.union(a, b);
                if self.reported.insert(key) {
                    deduced.push(key);
                    progressed = true;
                } else if merged {
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }

        self.propagations += deduced.len();
        deduced
    }

    /// One term per class that is a bit-vector literal, keyed by representative
    fn constant_representatives(&mut self, manager: &TermManager) -> FxHashMap<TermId, TermId> {
        let mut constants = FxHashMap::default();

        for class in self.classes.classes() {
            let Some(&representative) = class.first() else {
                continue;
            };
            let representative = self.classes.find(representative);
            for &term in &class {
                if Self::bv_literal_value(term, manager).is_some() {
                    constants.insert(representative, term);
                    break;
                }
            }
        }

        constants
    }

    /// Equalities from folding operations whose arguments are known constants
    fn fold_constants(&mut self, manager: &mut TermManager) -> Vec<(TermId, TermId)> {
        let constants = self.constant_representatives(manager);
        let mut terms: Vec<TermId> = self.bitvectors.keys().copied().collect();
        terms.sort_unstable_by_key(|term| term.0);

        let mut folded_equalities = Vec::new();
        for term in terms {
            let Some(kind) = manager.get(term).map(|t| t.kind.clone()) else {
                continue;
            };
            let children = get_children(&kind);
            if children.is_empty() {
                continue;
            }

            let mut arguments = Vec::with_capacity(children.len());
            let mut all_constant = true;
            for &child in &children {
                let representative = self.classes.find(child);
                match constants.get(&representative) {
                    Some(&constant) => arguments.push(constant),
                    None => {
                        all_constant = false;
                        break;
                    }
                }
            }
            if !all_constant {
                continue;
            }

            let Some(folded) = rebuild_bv_term(&kind, &arguments, manager) else {
                continue;
            };
            if folded == term || Self::bv_literal_value(folded, manager).is_none() {
                continue;
            }

            let width = manager
                .get(folded)
                .and_then(|t| manager.sorts.get(t.sort))
                .and_then(|sort| match sort.kind {
                    SortKind::BitVec(width) => Some(width),
                    _ => None,
                });
            if let Some(width) = width {
                self.register_bitvector(folded, width);
            }

            if !self.classes.are_equal(term, folded) {
                folded_equalities.push((term, folded));
            }
        }

        folded_equalities
    }

    /// Equalities between applications of the same operator to equal arguments
    fn congruent_pairs(&mut self, manager: &TermManager) -> Vec<(TermId, TermId)> {
        let mut terms: Vec<TermId> = self.bitvectors.keys().copied().collect();
        terms.sort_unstable_by_key(|term| term.0);

        let mut seen: FxHashMap<(u16, u32, u32, Vec<TermId>), TermId> = FxHashMap::default();
        let mut congruent = Vec::new();

        for term in terms {
            let Some(kind) = manager.get(term).map(|t| t.kind.clone()) else {
                continue;
            };
            let Some((tag, first_index, second_index)) = bv_operator_key(&kind) else {
                continue;
            };
            let arguments: Vec<TermId> = get_children(&kind)
                .into_iter()
                .map(|child| self.classes.find(child))
                .collect();

            let key = (tag, first_index, second_index, arguments);
            match seen.get(&key) {
                Some(&other) => {
                    if !self.classes.are_equal(term, other) {
                        congruent.push((other, term));
                    }
                }
                None => {
                    seen.insert(key, term);
                }
            }
        }

        congruent
    }

    /// Check for conflicts in the current state
    ///
    /// A class holding two literals with different values cannot be satisfied,
    /// since a bit-vector literal denotes exactly one value. The returned chain
    /// starts at one literal and ends at the other, and every consecutive pair
    /// in it was asserted or deduced equal.
    ///
    /// `None` means "no conflict found by this theory", never "satisfiable":
    /// classes without literals, and everything this layer does not reason
    /// about, are simply not checked.
    pub fn check_for_conflicts(&mut self, manager: &TermManager) -> Option<Vec<TermId>> {
        for class in self.classes.classes() {
            // Bucketed by width: only literals of one width are comparable,
            // and a literal of another width must not shadow the first one of
            // its own, or a real clash further along would be missed.
            let mut witnesses: FxHashMap<u32, (TermId, BigInt)> = FxHashMap::default();

            for term in class {
                let Some((width, value)) = Self::bv_literal_value(term, manager) else {
                    continue;
                };

                match witnesses.get(&width) {
                    Some((first, first_value)) => {
                        if *first_value != value {
                            let first = *first;
                            self.conflicts += 1;
                            let explanation = self.classes.explain(first, term);
                            return Some(if explanation.is_empty() {
                                vec![first, term]
                            } else {
                                explanation
                            });
                        }
                    }
                    None => {
                        witnesses.insert(width, (term, value));
                    }
                }
            }
        }

        None
    }

    /// Reset the theory state (for backtracking)
    pub fn reset(&mut self) {
        self.bitvectors.clear();
        self.classes.reset();
        self.reported.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> BitVectorStatistics {
        BitVectorStatistics {
            num_bitvectors: self.bitvectors.len(),
            num_axioms: self.instantiated.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// Rebuild a bit-vector operation with the given arguments
///
/// `arguments` are in the order [`get_children`] returns them. The
/// `TermManager` constructors fold literal operands, which is what makes this
/// the constant-folding step. Returns `None` for anything that is not a
/// bit-vector operator, or whose arity does not match.
fn rebuild_bv_term(
    kind: &TermKind,
    arguments: &[TermId],
    manager: &mut TermManager,
) -> Option<TermId> {
    let unary = |arguments: &[TermId]| arguments.first().copied();
    let binary = |arguments: &[TermId]| match arguments {
        [lhs, rhs] => Some((*lhs, *rhs)),
        _ => None,
    };

    Some(match kind {
        TermKind::BvNot(_) => manager.mk_bv_not(unary(arguments)?),
        TermKind::BvAnd(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_and(lhs, rhs)
        }
        TermKind::BvOr(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_or(lhs, rhs)
        }
        TermKind::BvXor(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_xor(lhs, rhs)
        }
        TermKind::BvAdd(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_add(lhs, rhs)
        }
        TermKind::BvSub(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_sub(lhs, rhs)
        }
        TermKind::BvMul(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_mul(lhs, rhs)
        }
        TermKind::BvUdiv(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_udiv(lhs, rhs)
        }
        TermKind::BvSdiv(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_sdiv(lhs, rhs)
        }
        TermKind::BvUrem(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_urem(lhs, rhs)
        }
        TermKind::BvSrem(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_srem(lhs, rhs)
        }
        TermKind::BvShl(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_shl(lhs, rhs)
        }
        TermKind::BvLshr(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_lshr(lhs, rhs)
        }
        TermKind::BvAshr(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.mk_bv_ashr(lhs, rhs)
        }
        TermKind::BvConcat(_, _) => {
            let (lhs, rhs) = binary(arguments)?;
            manager.try_mk_bv_concat(lhs, rhs).ok()?
        }
        TermKind::BvExtract { high, low, .. } => {
            manager.mk_bv_extract(*high, *low, unary(arguments)?)
        }
        _ => return None,
    })
}

/// Operator identity of a bit-vector term, for congruence lookup
///
/// The two `u32`s carry the extraction bounds; they are zero for every other
/// operator. Returns `None` for terms that are not bit-vector operations
/// (literals and variables have nothing to be congruent about).
fn bv_operator_key(kind: &TermKind) -> Option<(u16, u32, u32)> {
    Some(match kind {
        TermKind::BvNot(_) => (1, 0, 0),
        TermKind::BvAnd(_, _) => (2, 0, 0),
        TermKind::BvOr(_, _) => (3, 0, 0),
        TermKind::BvXor(_, _) => (4, 0, 0),
        TermKind::BvAdd(_, _) => (5, 0, 0),
        TermKind::BvSub(_, _) => (6, 0, 0),
        TermKind::BvMul(_, _) => (7, 0, 0),
        TermKind::BvUdiv(_, _) => (8, 0, 0),
        TermKind::BvSdiv(_, _) => (9, 0, 0),
        TermKind::BvUrem(_, _) => (10, 0, 0),
        TermKind::BvSrem(_, _) => (11, 0, 0),
        TermKind::BvShl(_, _) => (12, 0, 0),
        TermKind::BvLshr(_, _) => (13, 0, 0),
        TermKind::BvAshr(_, _) => (14, 0, 0),
        TermKind::BvConcat(_, _) => (15, 0, 0),
        TermKind::BvExtract { high, low, .. } => (16, *high, *low),
        _ => return None,
    })
}

impl Theory for BitVectorTheory {
    fn add_term(&mut self, term: TermId, manager: &TermManager) -> bool {
        BitVectorTheory::add_term(self, term, manager, &manager.sorts)
    }

    fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        BitVectorTheory::assert_equality(self, a, b)
    }

    fn check(&mut self, manager: &mut TermManager) -> TheoryResult {
        let deduced = self.propagate(manager);

        if let Some(explanation) = self.check_for_conflicts(manager) {
            return TheoryResult::Unsat { explanation };
        }

        if deduced.is_empty() {
            TheoryResult::Sat
        } else {
            TheoryResult::Propagate(deduced)
        }
    }

    fn name(&self) -> &str {
        "bitvector"
    }

    fn reset(&mut self) {
        BitVectorTheory::reset(self);
    }
}

/// Statistics for bitvector theory
#[derive(Debug, Clone, Copy)]
pub struct BitVectorStatistics {
    /// Number of bitvector terms
    pub num_bitvectors: usize,
    /// Number of axioms instantiated
    pub num_axioms: usize,
    /// Number of equalities deduced by `propagate`
    pub num_propagations: usize,
    /// Number of conflicts detected
    pub num_conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_empty_theory() {
        let theory = BitVectorTheory::new();
        assert_eq!(theory.bitvectors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_bitvector() {
        let mut theory = BitVectorTheory::new();
        let term = TermId(42);
        theory.register_bitvector(term, 32);
        assert_eq!(theory.bitvectors.get(&term), Some(&32));
    }

    #[test]
    fn test_add_term_claims_bitvector_sorts_only() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(32);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let add = manager.mk_bv_add(x, y);

        assert!(theory.add_term(add, &manager, &manager.sorts));
        assert_eq!(theory.bitvectors.get(&add), Some(&32));

        let int_sort = manager.sorts.int_sort;
        let n = manager.mk_var("n", int_sort);
        assert!(!theory.add_term(n, &manager, &manager.sorts));
    }

    #[test]
    fn test_is_bv_zero_normalises_the_literal() {
        let theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let zero = manager.mk_bitvec(0, 8);
        let minus_one = manager.mk_bitvec(-1, 8);
        let all_ones = manager.mk_bitvec(255, 8);

        assert!(theory.is_bv_zero(zero, &manager, 8));
        assert!(!theory.is_bv_zero(minus_one, &manager, 8));
        // -1 and 255 are the same 8-bit vector.
        assert!(theory.is_bv_all_ones(minus_one, &manager, 8));
        assert!(theory.is_bv_all_ones(all_ones, &manager, 8));
        // The same value at a different width is not the same vector.
        assert!(!theory.is_bv_all_ones(all_ones, &manager, 16));
    }

    #[test]
    fn test_negative_and_positive_forms_of_one_literal_are_not_a_conflict() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let minus_one = manager.mk_bitvec(-1, 8);
        let all_ones = manager.mk_bitvec(255, 8);
        theory.register_bitvector(minus_one, 8);
        theory.register_bitvector(all_ones, 8);
        theory.assert_equality(minus_one, all_ones);

        assert!(theory.check_for_conflicts(&manager).is_none());
    }

    #[test]
    fn test_constant_folding_deduces_an_equality() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let sum = manager.mk_bv_add(x, y);
        let two = manager.mk_bitvec(2, 8);
        let three = manager.mk_bitvec(3, 8);

        for term in [x, y, sum, two, three] {
            theory.register_bitvector(term, 8);
        }
        assert!(theory.assert_equality(x, two));
        assert!(theory.assert_equality(y, three));

        let deduced = theory.propagate(&mut manager);
        let five = manager.mk_bitvec(5, 8);
        let expected = if sum.0 < five.0 {
            (sum, five)
        } else {
            (five, sum)
        };
        assert!(
            deduced.contains(&expected),
            "expected x+y = 5 among {deduced:?}"
        );
        assert!(theory.known_equal(sum, five));
    }

    #[test]
    fn test_congruence_deduces_an_equality() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let z = manager.mk_var("z", bv_sort);
        let left = manager.mk_bv_mul(x, z);
        let right = manager.mk_bv_mul(y, z);

        for term in [x, y, z, left, right] {
            theory.register_bitvector(term, 8);
        }
        assert!(theory.assert_equality(x, y));

        let deduced = theory.propagate(&mut manager);
        assert!(!deduced.is_empty(), "congruence should deduce x*z = y*z");
        assert!(theory.known_equal(left, right));
    }

    #[test]
    fn test_conflict_on_clashing_constants() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let zero = manager.mk_bitvec(0, 8);
        let one = manager.mk_bitvec(1, 8);

        for term in [x, y, zero, one] {
            theory.register_bitvector(term, 8);
        }
        theory.assert_equality(x, zero);
        theory.assert_equality(y, one);
        theory.assert_equality(x, y);

        let explanation = theory
            .check_for_conflicts(&manager)
            .expect("clashing constants should be a conflict");
        assert!(explanation.contains(&zero));
        assert!(explanation.contains(&one));
        assert_eq!(theory.statistics().num_conflicts, 1);
    }

    #[test]
    fn test_a_foreign_width_literal_does_not_hide_a_clash() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        // A 16-bit literal sits between the two 8-bit ones in TermId order,
        // so a single-witness scan would latch onto it and miss the clash.
        let word = manager.mk_bitvec(1, 16);
        let zero = manager.mk_bitvec(0, 8);
        let one = manager.mk_bitvec(1, 8);
        theory.register_bitvector(word, 16);
        theory.register_bitvector(zero, 8);
        theory.register_bitvector(one, 8);

        // Merged directly into the classes: assert_equality refuses to relate
        // different widths, which is exactly why the class can hold all three.
        theory.classes.union(word, zero);
        theory.classes.union(word, one);

        let explanation = theory
            .check_for_conflicts(&manager)
            .expect("#x00 and #x01 clash even with a 16-bit literal in the class");
        assert!(explanation.contains(&zero) && explanation.contains(&one));
    }

    #[test]
    fn test_equalities_of_different_widths_are_ignored() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let byte = manager.mk_bitvec(1, 8);
        let word = manager.mk_bitvec(1, 16);
        theory.register_bitvector(byte, 8);
        theory.register_bitvector(word, 16);

        assert!(!theory.assert_equality(byte, word));
        assert!(!theory.known_equal(byte, word));
    }

    #[test]
    fn test_axiom_to_term_builds_the_real_equation() {
        let theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let zero = manager.mk_bitvec(0, 8);
        // The builder folds `x + 0`, so the axiom's term is interned directly.
        let add = manager.intern_term(TermKind::BvAdd(x, zero), bv_sort);

        let axiom = BitVectorAxiom::AddIdentity {
            term: add,
            width: 8,
        };
        let term = theory
            .axiom_to_term(&axiom, &mut manager)
            .expect("bvadd(x, 0) = x should be buildable");

        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Eq(lhs, rhs)) => {
                assert!(
                    (lhs == add && rhs == x) || (lhs == x && rhs == add),
                    "expected the equation bvadd(x, 0) = x"
                );
            }
            other => panic!("expected an equality, got {other:?}"),
        }
    }

    #[test]
    fn test_axiom_to_term_rejects_a_mismatched_term() {
        let theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let add = manager.mk_bv_add(x, y);

        // Neither operand is zero, so there is no identity to state.
        let axiom = BitVectorAxiom::AddIdentity {
            term: add,
            width: 8,
        };
        assert!(theory.axiom_to_term(&axiom, &mut manager).is_none());
    }

    #[test]
    fn test_axiom_to_term_reports_what_it_cannot_build() {
        let theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        assert!(
            theory
                .axiom_to_term(&BitVectorAxiom::NegZero { width: 8 }, &mut manager)
                .is_none()
        );
        assert!(
            theory
                .axiom_to_term(
                    &BitVectorAxiom::BitBlast {
                        term: TermId(0),
                        width: 8,
                        operation: BitVectorOp::Add,
                    },
                    &mut manager
                )
                .is_none()
        );
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = BitVectorTheory::new();
        let axiom = BitVectorAxiom::AddIdentity {
            term: TermId(1),
            width: 32,
        };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = BitVectorTheory::new();
        theory.register_bitvector(TermId(1), 32);
        theory.add_axiom(BitVectorAxiom::AddIdentity {
            term: TermId(1),
            width: 32,
        });

        theory.reset();

        assert_eq!(theory.bitvectors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
        assert_eq!(theory.instantiated.len(), 0);
    }

    #[test]
    fn test_statistics_counts_deduced_equalities() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let one = manager.mk_bitvec(1, 8);
        let negated = manager.intern_term(TermKind::BvNot(x), bv_sort);

        for term in [x, one, negated] {
            theory.register_bitvector(term, 8);
        }
        theory.assert_equality(x, one);
        let deduced = theory.propagate(&mut manager);

        let stats = theory.statistics();
        assert_eq!(stats.num_bitvectors, theory.bitvectors.len());
        assert_eq!(stats.num_propagations, deduced.len());
        assert!(!deduced.is_empty(), "bvnot(1) should fold");
    }

    #[test]
    fn test_propagate_with_nothing_known_deduces_nothing() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let sum = manager.mk_bv_add(x, y);
        for term in [x, y, sum] {
            theory.register_bitvector(term, 8);
        }

        assert!(theory.propagate(&mut manager).is_empty());
    }

    #[test]
    fn test_theory_trait_reports_conflict() {
        let mut theory = BitVectorTheory::new();
        let mut manager = TermManager::new();

        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let zero = manager.mk_bitvec(0, 8);
        let one = manager.mk_bitvec(1, 8);
        for term in [x, zero, one] {
            Theory::add_term(&mut theory, term, &manager);
        }

        assert!(Theory::assert_equality(&mut theory, x, zero));
        assert!(Theory::assert_equality(&mut theory, x, one));

        assert!(matches!(
            Theory::check(&mut theory, &mut manager),
            TheoryResult::Unsat { .. }
        ));
        assert_eq!(Theory::name(&theory), "bitvector");
    }
}
