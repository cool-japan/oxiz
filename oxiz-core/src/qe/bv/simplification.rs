//! BitVector Simplification for QE.
//!
//! Provides simplification techniques for bitvector formulas during
//! quantifier elimination.
//!
//! ## Simplifications
//!
//! - **Constant Folding**: Evaluate constant expressions
//! - **Algebraic Identities**: x + 0 = x, x & x = x, etc.
//! - **Range Analysis**: Track value ranges to eliminate impossible constraints
//! - **Bit-Level Reasoning**: Simplify based on individual bits
//!
//! ## References
//!
//! - "Deciding Bit-Vector Arithmetic with Abstraction" (Bryant et al., 2007)
//! - Z3's `qe/qe_bv.cpp`

/// Variable identifier.
#[allow(unused_imports)]
use crate::prelude::*;
/// Variable identifier for BV simplification.
pub type VarId = usize;

/// Bitvector term.
///
/// # Depth invariant
///
/// There is deliberately no bound on how deep a `BvTerm` may be: the variants
/// are public and take `Box<Self>` children, so callers build values of any
/// depth directly. Every walk over this type is therefore iterative —
/// [`Clone`], [`Drop`], [`PartialEq`], [`std::hash::Hash`] and
/// [`BvSimplifier::simplify`]. Do **not** replace any of them with a
/// `derive`: a derived `Clone`/`PartialEq`/`Hash` (and the compiler-generated
/// recursive `drop_in_place`) walks one native frame per nesting level and
/// aborts the process on a deep term, with no error channel to report it.
///
/// The one exception is the derived [`std::fmt::Debug`], which is still
/// recursive: it is a diagnostics-only formatter and hand-writing it would
/// change `{:#?}` output. This mirrors `oxiz-proof`'s `InterpolantTerm`.
#[derive(Debug)]
pub enum BvTerm {
    /// Constant.
    Const(u64, u32), // value, width
    /// Variable.
    Var(VarId, u32), // id, width
    /// Addition.
    Add(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise AND.
    And(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise OR.
    Or(Box<BvTerm>, Box<BvTerm>),
    /// Bitwise XOR.
    Xor(Box<BvTerm>, Box<BvTerm>),
    /// Negation (two's complement).
    Neg(Box<BvTerm>),
}

impl BvTerm {
    /// Stable per-variant tag mixed into [`std::hash::Hash`].
    ///
    /// Written out rather than derived from `mem::discriminant` so that the
    /// value is a fixed, exhaustively-checked constant per variant.
    const fn hash_tag(&self) -> u8 {
        match self {
            Self::Const(_, _) => 0,
            Self::Var(_, _) => 1,
            Self::Add(_, _) => 2,
            Self::And(_, _) => 3,
            Self::Or(_, _) => 4,
            Self::Xor(_, _) => 5,
            Self::Neg(_) => 6,
        }
    }

    /// The children of this node, left to right.
    fn children(&self) -> [Option<&Self>; 2] {
        match self {
            Self::Const(_, _) | Self::Var(_, _) => [None, None],
            Self::Add(a, b) | Self::And(a, b) | Self::Or(a, b) | Self::Xor(a, b) => {
                [Some(a.as_ref()), Some(b.as_ref())]
            }
            Self::Neg(a) => [Some(a.as_ref()), None],
        }
    }
}

impl PartialEq for BvTerm {
    /// Iterative structural equality.
    ///
    /// The pairs still to be compared live on the heap rather than in native
    /// call frames; the relation itself is exactly the derived one. The match
    /// is exhaustive over `self`'s variants on purpose, so that a new variant
    /// must be handled explicitly instead of silently falling into a catch-all
    /// that reports "not equal".
    fn eq(&self, other: &Self) -> bool {
        let mut worklist = vec![(self, other)];

        while let Some((a, b)) = worklist.pop() {
            match a {
                Self::Const(v1, w1) => {
                    let Self::Const(v2, w2) = b else { return false };
                    if v1 != v2 || w1 != w2 {
                        return false;
                    }
                }
                Self::Var(i1, w1) => {
                    let Self::Var(i2, w2) = b else { return false };
                    if i1 != i2 || w1 != w2 {
                        return false;
                    }
                }
                Self::Add(x1, x2) => {
                    let Self::Add(y1, y2) = b else { return false };
                    worklist.push((x2.as_ref(), y2.as_ref()));
                    worklist.push((x1.as_ref(), y1.as_ref()));
                }
                Self::And(x1, x2) => {
                    let Self::And(y1, y2) = b else { return false };
                    worklist.push((x2.as_ref(), y2.as_ref()));
                    worklist.push((x1.as_ref(), y1.as_ref()));
                }
                Self::Or(x1, x2) => {
                    let Self::Or(y1, y2) = b else { return false };
                    worklist.push((x2.as_ref(), y2.as_ref()));
                    worklist.push((x1.as_ref(), y1.as_ref()));
                }
                Self::Xor(x1, x2) => {
                    let Self::Xor(y1, y2) = b else { return false };
                    worklist.push((x2.as_ref(), y2.as_ref()));
                    worklist.push((x1.as_ref(), y1.as_ref()));
                }
                Self::Neg(x) => {
                    let Self::Neg(y) = b else { return false };
                    worklist.push((x.as_ref(), y.as_ref()));
                }
            }
        }

        true
    }
}

impl Eq for BvTerm {}

impl core::hash::Hash for BvTerm {
    /// Iterative structural hashing, consistent with the [`PartialEq`] above.
    ///
    /// Each node contributes its variant tag and then its non-child payload;
    /// children are visited in the same left-to-right order `eq` compares them
    /// in, which is what keeps `a == b` implying equal hashes.
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        let mut stack = vec![self];

        while let Some(node) = stack.pop() {
            state.write_u8(node.hash_tag());
            match node {
                Self::Const(value, width) => {
                    state.write_u64(*value);
                    state.write_u32(*width);
                }
                Self::Var(id, width) => {
                    state.write_usize(*id);
                    state.write_u32(*width);
                }
                Self::Add(_, _)
                | Self::And(_, _)
                | Self::Or(_, _)
                | Self::Xor(_, _)
                | Self::Neg(_) => {
                    // Children are pushed right-to-left so they are visited
                    // left-to-right, matching `eq`'s traversal order.
                    for child in node.children().into_iter().flatten().rev() {
                        stack.push(child);
                    }
                }
            }
        }
    }
}

/// The shape of a node being rebuilt by the iterative [`Clone`] impl.
enum CloneShape {
    /// `Add`, two children.
    Add,
    /// `And`, two children.
    And,
    /// `Or`, two children.
    Or,
    /// `Xor`, two children.
    Xor,
    /// `Neg`, one child.
    Neg,
}

/// Work item for the iterative [`Clone`] impl.
enum CloneTask<'a> {
    /// Clone this subterm.
    Visit(&'a BvTerm),
    /// Rebuild a node from the already-cloned children on the result stack.
    Rebuild(CloneShape),
}

impl Clone for BvTerm {
    /// Iterative clone.
    ///
    /// Structurally identical to the derived clone; only the recursion is
    /// gone. `simplify` clones sub-terms constantly, so this is the walk that
    /// a deep term hits first.
    fn clone(&self) -> Self {
        /// Pop the two most recent results as `(left, right)`.
        ///
        /// Both are always present: a `Rebuild` is only ever scheduled below
        /// the `Visit`s that produce its operands. A starved stack would mean
        /// a bug in this function rather than bad input, so it falls back to a
        /// zero-width zero constant rather than panicking.
        fn pair(results: &mut Vec<BvTerm>) -> (Box<BvTerm>, Box<BvTerm>) {
            let right = results.pop().unwrap_or(BvTerm::Const(0, 0));
            let left = results.pop().unwrap_or(BvTerm::Const(0, 0));
            (Box::new(left), Box::new(right))
        }

        let mut tasks = vec![CloneTask::Visit(self)];
        let mut results: Vec<Self> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                CloneTask::Visit(term) => match term {
                    Self::Const(value, width) => results.push(Self::Const(*value, *width)),
                    Self::Var(id, width) => results.push(Self::Var(*id, *width)),
                    Self::Add(a, b) | Self::And(a, b) | Self::Or(a, b) | Self::Xor(a, b) => {
                        let shape = match term {
                            Self::Add(_, _) => CloneShape::Add,
                            Self::And(_, _) => CloneShape::And,
                            Self::Or(_, _) => CloneShape::Or,
                            _ => CloneShape::Xor,
                        };
                        tasks.push(CloneTask::Rebuild(shape));
                        tasks.push(CloneTask::Visit(b));
                        tasks.push(CloneTask::Visit(a));
                    }
                    Self::Neg(inner) => {
                        tasks.push(CloneTask::Rebuild(CloneShape::Neg));
                        tasks.push(CloneTask::Visit(inner));
                    }
                },
                CloneTask::Rebuild(shape) => {
                    let rebuilt = match shape {
                        CloneShape::Add => {
                            let (a, b) = pair(&mut results);
                            Self::Add(a, b)
                        }
                        CloneShape::And => {
                            let (a, b) = pair(&mut results);
                            Self::And(a, b)
                        }
                        CloneShape::Or => {
                            let (a, b) = pair(&mut results);
                            Self::Or(a, b)
                        }
                        CloneShape::Xor => {
                            let (a, b) = pair(&mut results);
                            Self::Xor(a, b)
                        }
                        CloneShape::Neg => {
                            let inner = results.pop().unwrap_or(Self::Const(0, 0));
                            Self::Neg(Box::new(inner))
                        }
                    };
                    results.push(rebuilt);
                }
            }
        }

        results.pop().unwrap_or(Self::Const(0, 0))
    }
}

impl Drop for BvTerm {
    /// Iterative drop.
    ///
    /// With every other walk over this type made iterative, the
    /// compiler-generated recursive `drop_in_place` would be the one remaining
    /// way for a deep term to abort the process — at scope exit, with no
    /// diagnostic. Each node is dismantled into a shallow shell before being
    /// released.
    fn drop(&mut self) {
        /// Detach a node's children, leaving a shell that drops trivially.
        fn dismantle(node: &mut BvTerm, out: &mut Vec<BvTerm>) {
            /// Replace a boxed child with a leaf and hand the child over.
            fn take(slot: &mut Box<BvTerm>, out: &mut Vec<BvTerm>) {
                out.push(core::mem::replace(slot.as_mut(), BvTerm::Const(0, 0)));
            }

            match node {
                BvTerm::Const(_, _) | BvTerm::Var(_, _) => {}
                BvTerm::Add(a, b) | BvTerm::And(a, b) | BvTerm::Or(a, b) | BvTerm::Xor(a, b) => {
                    take(a, out);
                    take(b, out);
                }
                BvTerm::Neg(a) => take(a, out),
            }
        }

        let mut pending = Vec::new();
        dismantle(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            dismantle(&mut node, &mut pending);
        }
    }
}

/// Configuration for BV simplification.
#[derive(Debug, Clone)]
pub struct BvSimplificationConfig {
    /// Enable constant folding.
    pub enable_constant_folding: bool,
    /// Enable algebraic identities.
    pub enable_algebraic_identities: bool,
    /// Enable range analysis.
    pub enable_range_analysis: bool,
    /// Enable bit-level reasoning.
    pub enable_bit_reasoning: bool,
}

impl Default for BvSimplificationConfig {
    fn default() -> Self {
        Self {
            enable_constant_folding: true,
            enable_algebraic_identities: true,
            enable_range_analysis: true,
            enable_bit_reasoning: true,
        }
    }
}

/// Statistics for BV simplification.
#[derive(Debug, Clone, Default)]
pub struct BvSimplificationStats {
    /// Constant folding applications.
    pub constant_foldings: u64,
    /// Algebraic simplifications.
    pub algebraic_simplifications: u64,
    /// Range-based eliminations.
    pub range_eliminations: u64,
    /// Bit-level simplifications.
    pub bit_simplifications: u64,
}

/// Compute the all-ones mask for a bitvector of the given width.
///
/// `1u64 << width` is undefined-ish for `width >= 64` (panics on the
/// shift-overflow debug check; wraps to a shift-by-`width % 64` — silently
/// producing the *wrong* mask, e.g. `0` instead of `u64::MAX` for
/// `width == 64` — in release). 64-bit bitvectors are a common width, so
/// this guard is not just a defensive edge case.
#[inline]
fn width_mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// BV simplifier.
#[derive(Debug)]
pub struct BvSimplifier {
    /// Variable ranges (var_id -> (min, max)).
    ranges: FxHashMap<VarId, (u64, u64)>,
    /// Configuration.
    config: BvSimplificationConfig,
    /// Statistics.
    stats: BvSimplificationStats,
}

impl BvSimplifier {
    /// Create a new BV simplifier.
    pub fn new(config: BvSimplificationConfig) -> Self {
        Self {
            ranges: FxHashMap::default(),
            config,
            stats: BvSimplificationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BvSimplificationConfig::default())
    }

    /// Set a variable's range.
    pub fn set_range(&mut self, var: VarId, min: u64, max: u64) {
        self.ranges.insert(var, (min, max));
    }

    /// Simplify a BV term.
    ///
    /// Iterative (explicit heap stack) bottom-up rewriting: results are
    /// identical to the recursive formulation — each node is still rewritten
    /// only after both of its operands have been — but the walk no longer
    /// consumes one native frame per nesting level, which a `BvTerm` of
    /// caller-controlled depth could exhaust.
    pub fn simplify(&mut self, term: &BvTerm) -> BvTerm {
        /// Which binary rewrite to apply to the top two results.
        enum BinOp {
            /// `Add`.
            Add,
            /// `And`.
            And,
            /// `Or`.
            Or,
            /// `Xor`.
            Xor,
        }

        /// Work item for the iterative simplifier.
        enum SimplifyTask<'a> {
            /// Simplify this subterm.
            Enter(&'a BvTerm),
            /// Rewrite a binary node from the top two results.
            BuildBinary(BinOp),
            /// Rewrite a `Neg` from the single result on top.
            BuildNeg,
        }

        let mut tasks = vec![SimplifyTask::Enter(term)];
        let mut results: Vec<BvTerm> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                SimplifyTask::Enter(current) => match current {
                    BvTerm::Const(_, _) | BvTerm::Var(_, _) => results.push(current.clone()),
                    BvTerm::Add(left, right)
                    | BvTerm::And(left, right)
                    | BvTerm::Or(left, right)
                    | BvTerm::Xor(left, right) => {
                        let op = match current {
                            BvTerm::Add(_, _) => BinOp::Add,
                            BvTerm::And(_, _) => BinOp::And,
                            BvTerm::Or(_, _) => BinOp::Or,
                            _ => BinOp::Xor,
                        };
                        tasks.push(SimplifyTask::BuildBinary(op));
                        tasks.push(SimplifyTask::Enter(right));
                        tasks.push(SimplifyTask::Enter(left));
                    }
                    BvTerm::Neg(inner) => {
                        tasks.push(SimplifyTask::BuildNeg);
                        tasks.push(SimplifyTask::Enter(inner));
                    }
                },
                SimplifyTask::BuildBinary(op) => {
                    // Both operands are always present: a `Build*` task is only
                    // ever scheduled below the `Enter`s that produce them.
                    let (Some(right), Some(left)) = (results.pop(), results.pop()) else {
                        return term.clone();
                    };
                    let rebuilt = match op {
                        BinOp::Add => self.simplify_add(&left, &right),
                        BinOp::And => self.simplify_and(&left, &right),
                        BinOp::Or => self.simplify_or(&left, &right),
                        BinOp::Xor => self.simplify_xor(&left, &right),
                    };
                    results.push(rebuilt);
                }
                SimplifyTask::BuildNeg => {
                    let Some(inner) = results.pop() else {
                        return term.clone();
                    };
                    let rebuilt = self.simplify_neg(&inner);
                    results.push(rebuilt);
                }
            }
        }

        // Exactly one result is produced for the single root task.
        results.pop().unwrap_or_else(|| term.clone())
    }

    /// Simplify addition.
    fn simplify_add(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            let mask = width_mask(*w1);
            return BvTerm::Const((v1.wrapping_add(*v2)) & mask, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x + 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
            if let BvTerm::Const(0, _) = left {
                self.stats.algebraic_simplifications += 1;
                return right.clone();
            }
        }

        BvTerm::Add(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise AND.
    fn simplify_and(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 & v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x & x = x
            if left == right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }

            // x & 0 = 0
            if let BvTerm::Const(0, w) = right {
                self.stats.algebraic_simplifications += 1;
                return BvTerm::Const(0, *w);
            }

            // x & ~0 = x
            if let BvTerm::Const(v, w) = right {
                let all_ones = width_mask(*w);
                if *v == all_ones {
                    self.stats.algebraic_simplifications += 1;
                    return left.clone();
                }
            }
        }

        BvTerm::And(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise OR.
    fn simplify_or(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 | v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x | x = x
            if left == right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }

            // x | 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
        }

        BvTerm::Or(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify bitwise XOR.
    fn simplify_xor(&mut self, left: &BvTerm, right: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let (BvTerm::Const(v1, w1), BvTerm::Const(v2, w2)) = (left, right)
            && w1 == w2
        {
            self.stats.constant_foldings += 1;
            return BvTerm::Const(v1 ^ v2, *w1);
        }

        // Algebraic identities
        if self.config.enable_algebraic_identities {
            // x ^ x = 0
            if left == right {
                self.stats.algebraic_simplifications += 1;
                if let BvTerm::Var(_, w) = left {
                    return BvTerm::Const(0, *w);
                }
            }

            // x ^ 0 = x
            if let BvTerm::Const(0, _) = right {
                self.stats.algebraic_simplifications += 1;
                return left.clone();
            }
        }

        BvTerm::Xor(Box::new(left.clone()), Box::new(right.clone()))
    }

    /// Simplify negation.
    fn simplify_neg(&mut self, inner: &BvTerm) -> BvTerm {
        // Constant folding
        if self.config.enable_constant_folding
            && let BvTerm::Const(v, w) = inner
        {
            self.stats.constant_foldings += 1;
            let mask = width_mask(*w);
            let negated = (!(v.wrapping_sub(1))) & mask;
            return BvTerm::Const(negated, *w);
        }

        BvTerm::Neg(Box::new(inner.clone()))
    }

    /// Get statistics.
    pub fn stats(&self) -> &BvSimplificationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = BvSimplificationStats::default();
    }
}

impl Default for BvSimplifier {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplifier_creation() {
        let simp = BvSimplifier::default_config();
        assert_eq!(simp.stats().constant_foldings, 0);
    }

    #[test]
    fn test_constant_folding_add() {
        let mut simp = BvSimplifier::default_config();

        let term = BvTerm::Add(Box::new(BvTerm::Const(3, 8)), Box::new(BvTerm::Const(5, 8)));

        let result = simp.simplify(&term);

        assert_eq!(result, BvTerm::Const(8, 8));
        assert_eq!(simp.stats().constant_foldings, 1);
    }

    #[test]
    fn test_algebraic_identity_add() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::Add(Box::new(var.clone()), Box::new(BvTerm::Const(0, 8)));

        let result = simp.simplify(&term);

        assert_eq!(result, var);
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }

    #[test]
    fn test_and_self() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::And(Box::new(var.clone()), Box::new(var.clone()));

        let result = simp.simplify(&term);

        assert_eq!(result, var);
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }

    /// Run `body` on a worker thread with a deliberately small (128 KiB) stack,
    /// so a recursive walk over a deep `BvTerm` would abort instead of getting
    /// away with the main thread's much larger stack.
    fn run_with_small_stack<F>(body: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(body)
            .expect("thread spawn should succeed")
            .join()
            .expect("deep-nesting walk must not overflow the stack");
    }

    /// Build `Neg(Neg(... Var(0, 8) ...))` nested `depth` levels deep.
    fn deep_term(depth: usize) -> BvTerm {
        let mut term = BvTerm::Var(0, 8);
        for _ in 0..depth {
            term = BvTerm::Neg(Box::new(term));
        }
        term
    }

    #[test]
    fn test_deep_term_clone_eq_hash_and_drop_are_iterative() {
        run_with_small_stack(|| {
            use core::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;

            const DEPTH: usize = 6_250;

            let term = deep_term(DEPTH);
            let copy = term.clone();

            assert!(term == copy);

            let mut h1 = DefaultHasher::new();
            term.hash(&mut h1);
            let mut h2 = DefaultHasher::new();
            copy.hash(&mut h2);
            assert_eq!(h1.finish(), h2.finish());

            // A structurally different term of the same depth must compare
            // unequal without recursing either.
            let mut other = deep_term(DEPTH - 1);
            other = BvTerm::Add(Box::new(other), Box::new(BvTerm::Const(0, 8)));
            assert!(term != other);

            // Both terms are dropped here, on this small stack.
        });
    }

    #[test]
    fn test_deep_term_simplify_is_iterative() {
        run_with_small_stack(|| {
            const DEPTH: usize = 6_250;

            let mut simp = BvSimplifier::default_config();

            // A 50k-deep `Neg` chain over a constant: every level folds, so the
            // rewrite stays linear and the only thing under test is whether the
            // walk itself survives the depth.
            let mut term = BvTerm::Const(1, 8);
            for _ in 0..DEPTH {
                term = BvTerm::Neg(Box::new(term));
            }

            // Two's-complement negation is an involution, and DEPTH is even.
            let result = simp.simplify(&term);
            assert_eq!(result, BvTerm::Const(1, 8));
            assert_eq!(simp.stats().constant_foldings, DEPTH as u64);
        });
    }

    #[test]
    fn test_hash_distinguishes_shape() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn digest(term: &BvTerm) -> u64 {
            let mut hasher = DefaultHasher::new();
            term.hash(&mut hasher);
            hasher.finish()
        }

        let a = BvTerm::Var(0, 8);
        let b = BvTerm::Var(1, 8);
        let add = BvTerm::Add(Box::new(a.clone()), Box::new(b.clone()));
        let xor = BvTerm::Xor(Box::new(a.clone()), Box::new(b));

        assert_ne!(digest(&add), digest(&xor));
        assert_eq!(digest(&add), digest(&add.clone()));
        assert_ne!(digest(&a), digest(&BvTerm::Var(0, 16)));
    }

    #[test]
    fn test_clone_and_eq_are_structural() {
        let term = BvTerm::Or(
            Box::new(BvTerm::And(
                Box::new(BvTerm::Var(3, 32)),
                Box::new(BvTerm::Const(7, 32)),
            )),
            Box::new(BvTerm::Neg(Box::new(BvTerm::Var(4, 32)))),
        );

        let copy = term.clone();
        assert_eq!(term, copy);
        assert_ne!(term, BvTerm::Const(0, 32));
    }

    #[test]
    fn test_xor_self() {
        let mut simp = BvSimplifier::default_config();

        let var = BvTerm::Var(0, 8);
        let term = BvTerm::Xor(Box::new(var.clone()), Box::new(var));

        let result = simp.simplify(&term);

        assert_eq!(result, BvTerm::Const(0, 8));
        assert_eq!(simp.stats().algebraic_simplifications, 1);
    }
}
