//! Cooper's Algorithm for Presburger Arithmetic QE.
//!
//! Implements Cooper's method for quantifier elimination in linear
//! integer arithmetic (Presburger arithmetic).
//!
//! Given `∃x. φ(x)` with φ quantifier-free over linear integer arithmetic,
//! the procedure returns an equivalent quantifier-free formula in which `x`
//! no longer occurs. The construction follows the classic "minus infinity"
//! elimination (Cooper 1972; see also Bradley & Manna, *The Calculus of
//! Computation*, §7.3):
//!
//! 1. Normalise the matrix to negation normal form, tracking polarity.
//! 2. Isolate `x` in every literal and scale coefficients to a common
//!    absolute value `L` (`lcm` of all coefficients of `x`), introducing the
//!    global divisibility constraint `L | x` for the renamed unit-coefficient
//!    variable.
//! 3. Compute `δ = lcm` of all divisibility moduli.
//! 4. Return `⋁_{j=1}^{δ} φ_{-∞}(j)  ∨  ⋁_{b∈B} ⋁_{j=1}^{δ} φ(b + j)` where
//!    `φ_{-∞}` replaces each bound literal by its truth value as `x → -∞`,
//!    and `B` is the set of lower-bound boundary terms.
//!
//! Formulae that fall outside the supported linear-integer fragment (a
//! non-linear occurrence of `x`, `x` under an uninterpreted function, a real
//! sort, etc.) are reported as an explicit `Err` rather than silently
//! returning a wrong or unchanged result.

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::rc::Rc;

/// Maximum period `δ` before Cooper elimination gives up (returns `Err`).
const MAX_DELTA: i64 = 100_000;
/// Maximum number of generated disjuncts before giving up (returns `Err`).
const MAX_DISJUNCTS: i64 = 500_000;

/// Cooper's algorithm QE engine.
pub struct CooperEliminator {
    /// Statistics
    stats: CooperStats,
}

/// Cooper elimination statistics.
#[derive(Debug, Clone, Default)]
pub struct CooperStats {
    /// Number of quantifiers eliminated
    pub quantifiers_eliminated: usize,
    /// Number of boundary test cases generated
    pub test_cases: usize,
    /// Number of infinity (minus-infinity period) tests generated
    pub infinity_tests: usize,
}

/// A relation `expr REL 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpRel {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpRel {
    /// Logical negation of the relation.
    fn negate(self) -> Self {
        match self {
            CmpRel::Lt => CmpRel::Ge,
            CmpRel::Le => CmpRel::Gt,
            CmpRel::Gt => CmpRel::Le,
            CmpRel::Ge => CmpRel::Lt,
            CmpRel::Eq => CmpRel::Ne,
            CmpRel::Ne => CmpRel::Eq,
        }
    }

    /// Swap the roles of `<`/`>` and `≤`/`≥` (used when the coefficient of the
    /// eliminated variable is negative and the inequality is multiplied by -1).
    fn flip_lg(self) -> Self {
        match self {
            CmpRel::Lt => CmpRel::Gt,
            CmpRel::Le => CmpRel::Ge,
            CmpRel::Gt => CmpRel::Lt,
            CmpRel::Ge => CmpRel::Le,
            CmpRel::Eq => CmpRel::Eq,
            CmpRel::Ne => CmpRel::Ne,
        }
    }
}

/// A linear expression `x_coeff·x + Σ others + constant`, where `x` is the
/// variable being eliminated and `others` are opaque sub-terms (other
/// variables or `x`-free compound terms) keyed by their `TermId`.
#[derive(Debug, Clone)]
struct LinearForm {
    x_coeff: BigInt,
    others: FxHashMap<TermId, BigInt>,
    constant: BigInt,
}

impl LinearForm {
    fn zero() -> Self {
        Self {
            x_coeff: BigInt::zero(),
            others: FxHashMap::default(),
            constant: BigInt::zero(),
        }
    }

    fn constant_val(c: BigInt) -> Self {
        Self {
            x_coeff: BigInt::zero(),
            others: FxHashMap::default(),
            constant: c,
        }
    }

    fn x() -> Self {
        Self {
            x_coeff: BigInt::one(),
            others: FxHashMap::default(),
            constant: BigInt::zero(),
        }
    }

    fn atom(id: TermId) -> Self {
        let mut others = FxHashMap::default();
        others.insert(id, BigInt::one());
        Self {
            x_coeff: BigInt::zero(),
            others,
            constant: BigInt::zero(),
        }
    }

    fn is_constant(&self) -> bool {
        self.x_coeff.is_zero() && self.others.values().all(BigInt::is_zero)
    }

    fn neg(mut self) -> Self {
        self.x_coeff = -self.x_coeff;
        self.constant = -self.constant;
        for v in self.others.values_mut() {
            *v = -core::mem::take(v);
        }
        self
    }

    fn add(mut self, other: Self) -> Self {
        self.x_coeff += other.x_coeff;
        self.constant += other.constant;
        for (k, v) in other.others {
            let entry = self.others.entry(k).or_insert_with(BigInt::zero);
            *entry += v;
        }
        self
    }

    fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    fn scale(mut self, k: &BigInt) -> Self {
        self.x_coeff *= k;
        self.constant *= k;
        for v in self.others.values_mut() {
            *v *= k;
        }
        self
    }
}

/// Intermediate boolean DAG, produced before coefficient normalisation.
///
/// Children are reference-counted so that the `Xor`/`Ite` expansions — which
/// mention each operand under two different parents — *share* one sub-DAG per
/// `(sub-formula, polarity)` pair instead of duplicating it. With a tree the
/// expansion doubles the node count at every nesting level, i.e. a chain of
/// `n` nested `Xor`s builds `2ⁿ` nodes; with sharing it builds `O(n)`.
enum Raw {
    And(Vec<Rc<Raw>>),
    Or(Vec<Rc<Raw>>),
    Const(bool),
    /// An `x`-free literal, kept verbatim.
    Free(TermId),
    /// A comparison `form REL 0` where `form` mentions `x`.
    Cmp {
        form: LinearForm,
        rel: CmpRel,
    },
    /// A divisibility `modulus | form` (or its negation), `form` mentions `x`.
    Divis {
        modulus: BigInt,
        form: LinearForm,
        negated: bool,
    },
}

/// Iterative teardown of a [`Raw`] DAG.
///
/// The derived drop glue recurses once per level, which would overflow the
/// stack on exactly the deeply nested formulae the iterative builders above
/// were written to support. Dismantling the DAG with an explicit stack — each
/// uniquely owned node has its children moved out before it is released —
/// keeps teardown flat.
impl Drop for Raw {
    fn drop(&mut self) {
        let mut stack: Vec<Rc<Raw>> = Vec::new();
        take_raw_children(self, &mut stack);
        while let Some(node) = stack.pop() {
            if let Ok(mut owned) = Rc::try_unwrap(node) {
                take_raw_children(&mut owned, &mut stack);
                // `owned` is released here with no children left to recurse on.
            }
        }
    }
}

/// Move a [`Raw`] node's children onto `out`, leaving it childless.
fn take_raw_children(raw: &mut Raw, out: &mut Vec<Rc<Raw>>) {
    if let Raw::And(subs) | Raw::Or(subs) = raw {
        out.append(&mut core::mem::take(subs));
    }
}

/// Iterative teardown of a [`Node`] DAG, for the same reason as [`Raw`]'s.
impl Drop for Node {
    fn drop(&mut self) {
        let mut stack: Vec<Rc<Node>> = Vec::new();
        take_node_children(self, &mut stack);
        while let Some(node) = stack.pop() {
            if let Ok(mut owned) = Rc::try_unwrap(node) {
                take_node_children(&mut owned, &mut stack);
            }
        }
    }
}

/// Move a [`Node`]'s children onto `out`, leaving it childless.
fn take_node_children(node: &mut Node, out: &mut Vec<Rc<Node>>) {
    if let Node::And(subs) | Node::Or(subs) = node {
        out.append(&mut core::mem::take(subs));
    }
}

/// Identity key of a node in a reference-counted DAG.
///
/// Every node of a DAG being walked is kept alive by its root for the whole
/// walk, so its address uniquely identifies it and cannot be recycled
/// underneath the walker.
fn node_key<T>(rc: &Rc<T>) -> usize {
    Rc::as_ptr(rc) as usize
}

/// A normalised literal (coefficient of the eliminated variable is `±1`).
enum NLit {
    /// `bound < x`
    Lower(TermId),
    /// `x < bound`
    Upper(TermId),
    /// `modulus | (x + off)`
    Div { modulus: BigInt, off: TermId },
    /// `¬(modulus | (x + off))`
    NotDiv { modulus: BigInt, off: TermId },
    /// `x`-free literal.
    Free(TermId),
    /// Boolean constant.
    Const(bool),
}

/// Normalised boolean DAG over [`NLit`], sharing sub-DAGs exactly as [`Raw`]
/// does (one [`Node`] per distinct [`Raw`] node).
enum Node {
    And(Vec<Rc<Node>>),
    Or(Vec<Rc<Node>>),
    Lit(NLit),
}

/// How to instantiate the eliminated variable when materialising a [`Node`].
enum XVal<'a> {
    /// Behaviour as `x → -∞`, with divisibilities evaluated at `x = j`.
    MinusInf(&'a BigInt),
    /// Substitute the (x-free) term `v` for `x`.
    At(TermId),
}

impl CooperEliminator {
    /// Create a new Cooper eliminator.
    pub fn new() -> Self {
        Self {
            stats: CooperStats::default(),
        }
    }

    /// Eliminate an existential quantifier: `∃var. formula(var)`.
    ///
    /// On success returns a quantifier-free formula equivalent to
    /// `∃var. formula` in which `var` does not occur. Returns `Err` for
    /// formulae outside the supported linear-integer fragment (soundness is
    /// preserved: no wrong or `var`-containing result is ever returned as
    /// `Ok`).
    pub fn eliminate_exists(
        &mut self,
        var: String,
        formula: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        let x_spur = tm.intern_str(&var);

        // If x does not occur, ∃x.φ ≡ φ.
        if !self.mentions_x(formula, x_spur, tm) {
            self.stats.quantifiers_eliminated += 1;
            return Ok(formula);
        }

        // x must be of integer sort: the ±1 boundary offsets are only valid
        // over the integers.
        if !self.var_is_int(formula, x_spur, tm) {
            return Err("cooper: eliminated variable is not of integer sort".to_string());
        }

        // Pass 1: build the polarity-resolved boolean tree.
        let raw = self.build_raw(formula, x_spur, true, tm)?;

        // Compute L = lcm of |coefficient of x| over all x-literals.
        let mut lcm_coeff = BigInt::one();
        Self::collect_x_coeff_lcm(&raw, &mut lcm_coeff);

        // Pass 2: normalise coefficients to ±1 and build the Node tree.
        let mut node = self.convert(&raw, &lcm_coeff, tm)?;

        // Renaming L·x → u introduces the constraint L | u (off = 0).
        if lcm_coeff > BigInt::one() {
            let zero = tm.mk_int(0);
            node = Rc::new(Node::And(vec![
                node,
                Rc::new(Node::Lit(NLit::Div {
                    modulus: lcm_coeff.clone(),
                    off: zero,
                })),
            ]));
        }

        // δ = lcm of all divisibility moduli.
        let mut delta = BigInt::one();
        Self::collect_moduli_lcm(&node, &mut delta);

        if delta > BigInt::from(MAX_DELTA) {
            return Err("cooper: divisibility period too large to eliminate".to_string());
        }
        let delta_i64 = delta
            .to_i64()
            .ok_or_else(|| "cooper: divisibility period overflow".to_string())?;

        // Lower-bound boundary set.
        let mut bset = Vec::new();
        Self::collect_lower_bounds(&node, &mut bset);

        let total = delta_i64
            .checked_mul(bset.len() as i64 + 1)
            .ok_or_else(|| "cooper: elimination too large".to_string())?;
        if total > MAX_DISJUNCTS {
            return Err("cooper: elimination too large".to_string());
        }

        self.stats.quantifiers_eliminated += 1;

        let mut disjuncts: Vec<TermId> = Vec::new();

        // Minus-infinity part: ⋁_{j=1}^{δ} φ_{-∞}(j).
        for j in 1..=delta_i64 {
            let jb = BigInt::from(j);
            let t = self.materialize(&node, &XVal::MinusInf(&jb), tm)?;
            disjuncts.push(t);
            self.stats.infinity_tests += 1;
        }

        // Boundary part: ⋁_{b∈B} ⋁_{j=1}^{δ} φ(b + j).
        for &b in &bset {
            for j in 1..=delta_i64 {
                let jt = tm.mk_int(j);
                let v = tm.mk_add(vec![b, jt]);
                let t = self.materialize(&node, &XVal::At(v), tm)?;
                disjuncts.push(t);
                self.stats.test_cases += 1;
            }
        }

        Ok(tm.mk_or(disjuncts))
    }

    /// Whether `id` syntactically contains the eliminated variable.
    ///
    /// Iterative, with a visited set. `bool` offers no error channel, so a
    /// depth cap could only answer "does not mention `x`" for a term it
    /// never finished inspecting — which would silently drop a constraint
    /// from the elimination. The visited set also prevents the re-walk of
    /// shared sub-terms; this predicate is called at the top of several
    /// other walks.
    fn mentions_x(&self, id: TermId, x_spur: crate::interner::Spur, tm: &TermManager) -> bool {
        let mut stack = vec![id];
        let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(term) = tm.get(current) else {
                continue;
            };
            if let TermKind::Var(s) = term.kind {
                if s == x_spur {
                    return true;
                }
                continue;
            }
            stack.extend(crate::ast::traversal::get_children(&term.kind));
        }

        false
    }

    /// Whether the first occurrence of `x` has integer sort.
    fn var_is_int(&self, id: TermId, x_spur: crate::interner::Spur, tm: &mut TermManager) -> bool {
        let int_sort = {
            let z = tm.mk_int(0);
            tm.get(z).map(|t| t.sort)
        };
        let Some(int_sort) = int_sort else {
            return false;
        };
        self.find_var_sort(id, x_spur, tm) == Some(int_sort)
    }

    /// Sort of the first syntactic occurrence of `x`, found with an explicit
    /// stack that preserves the left-to-right order of the recursive walk.
    fn find_var_sort(
        &self,
        id: TermId,
        x_spur: crate::interner::Spur,
        tm: &TermManager,
    ) -> Option<crate::sort::SortId> {
        let mut stack = vec![id];
        let mut visited: crate::prelude::FxHashSet<TermId> = crate::prelude::FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(term) = tm.get(current) else {
                continue;
            };
            if let TermKind::Var(s) = term.kind {
                if s == x_spur {
                    return Some(term.sort);
                }
                continue;
            }
            let children = crate::ast::traversal::get_children(&term.kind);
            stack.extend(children.iter().rev().copied());
        }

        None
    }

    /// Parse `id` into a [`LinearForm`], or `None` if `x` occurs non-linearly.
    fn to_linear(
        &self,
        id: TermId,
        x_spur: crate::interner::Spur,
        tm: &TermManager,
    ) -> Option<LinearForm> {
        /// Work item of the iterative linear-form parser.
        enum Step {
            /// Classify a term and schedule its operands.
            Enter(TermId),
            /// Fold already-parsed operands into this term's linear form.
            Build(TermId),
        }

        // Explicit stack plus a memo keyed on `TermId`: the recursive form
        // had one `LinearForm`-carrying frame per level of arithmetic
        // nesting and re-parsed every shared sub-term once per occurrence.
        let mut memo: FxHashMap<TermId, LinearForm> = FxHashMap::default();
        let mut stack = vec![Step::Enter(id)];

        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(current) => {
                    if memo.contains_key(&current) {
                        continue;
                    }
                    let term = tm.get(current)?;
                    match &term.kind {
                        TermKind::IntConst(n) => {
                            memo.insert(current, LinearForm::constant_val(n.clone()));
                        }
                        TermKind::Var(s) => {
                            let form = if *s == x_spur {
                                LinearForm::x()
                            } else {
                                LinearForm::atom(current)
                            };
                            memo.insert(current, form);
                        }
                        TermKind::Neg(a) => {
                            stack.push(Step::Build(current));
                            stack.push(Step::Enter(*a));
                        }
                        TermKind::Add(args) | TermKind::Mul(args) => {
                            stack.push(Step::Build(current));
                            for &a in args.iter() {
                                stack.push(Step::Enter(a));
                            }
                        }
                        TermKind::Sub(a, b) => {
                            stack.push(Step::Build(current));
                            stack.push(Step::Enter(*a));
                            stack.push(Step::Enter(*b));
                        }
                        _ => {
                            // Any other term: acceptable only if it does not
                            // mention x (then it is an opaque atom);
                            // otherwise x occurs non-linearly.
                            if self.mentions_x(current, x_spur, tm) {
                                return None;
                            }
                            memo.insert(current, LinearForm::atom(current));
                        }
                    }
                }
                Step::Build(current) => {
                    let term = tm.get(current)?;
                    let form = match &term.kind {
                        TermKind::Neg(a) => memo.get(a)?.clone().neg(),
                        TermKind::Add(args) => {
                            let mut acc = LinearForm::zero();
                            for a in args.iter() {
                                acc = acc.add(memo.get(a)?.clone());
                            }
                            acc
                        }
                        TermKind::Sub(a, b) => {
                            let la = memo.get(a)?.clone();
                            let lb = memo.get(b)?.clone();
                            la.sub(lb)
                        }
                        TermKind::Mul(args) => {
                            let mut const_prod = BigInt::one();
                            let mut nonconst: Option<LinearForm> = None;
                            for a in args.iter() {
                                let lf = memo.get(a)?.clone();
                                if lf.is_constant() {
                                    const_prod *= lf.constant;
                                } else if nonconst.is_none() {
                                    nonconst = Some(lf);
                                } else {
                                    // product of two non-constant factors → non-linear
                                    return None;
                                }
                            }
                            match nonconst {
                                Some(lf) => lf.scale(&const_prod),
                                None => LinearForm::constant_val(const_prod),
                            }
                        }
                        // `Build` is only ever scheduled for the kinds above.
                        _ => return None,
                    };
                    memo.insert(current, form);
                }
            }
        }

        memo.remove(&id)
    }

    /// Build the polarity-resolved [`Raw`] DAG.
    ///
    /// Explicit work stack with a memo keyed on `(sub-formula, polarity)`.
    /// The construction reads no other context — the polarity is the only
    /// thing pushed down the walk, and the term manager is only ever asked
    /// for hash-consed terms — so two requests for the same pair must build
    /// the same sub-DAG, and returning the first one is exact.
    ///
    /// The memo is what makes `Xor` and `Ite` affordable: each expands into
    /// *four* sub-results (both polarities of both operands, resp. of the
    /// condition), so the recursive form had a call tree that doubled per
    /// nesting level — a 30-deep `Xor` chain meant ~2³⁰ calls. Here each pair
    /// is built once and shared, so the same chain is linear.
    fn build_raw(
        &self,
        id: TermId,
        x_spur: crate::interner::Spur,
        positive: bool,
        tm: &mut TermManager,
    ) -> Result<Rc<Raw>, String> {
        /// Work item of the iterative [`Raw`] builder.
        enum Task {
            /// Classify a sub-formula and schedule the sub-results it needs.
            Enter(TermId, bool),
            /// Assemble a node from its already-built sub-results.
            Build(TermId, bool),
        }

        /// Fetch an already-built sub-result.
        fn sub(
            memo: &FxHashMap<(TermId, bool), Rc<Raw>>,
            id: TermId,
            positive: bool,
        ) -> Result<Rc<Raw>, String> {
            memo.get(&(id, positive))
                .map(Rc::clone)
                .ok_or_else(|| "cooper: internal error: unresolved sub-formula".to_string())
        }

        let mut memo: FxHashMap<(TermId, bool), Rc<Raw>> = FxHashMap::default();
        let mut stack = vec![Task::Enter(id, positive)];

        while let Some(task) = stack.pop() {
            match task {
                Task::Enter(current, pos) => {
                    if memo.contains_key(&(current, pos)) {
                        continue;
                    }
                    let kind = match tm.get(current) {
                        Some(t) => t.kind.clone(),
                        None => return Err("cooper: term not found".to_string()),
                    };
                    match kind {
                        TermKind::True => {
                            memo.insert((current, pos), Rc::new(Raw::Const(pos)));
                        }
                        TermKind::False => {
                            memo.insert((current, pos), Rc::new(Raw::Const(!pos)));
                        }
                        TermKind::Not(a) => {
                            stack.push(Task::Build(current, pos));
                            stack.push(Task::Enter(a, !pos));
                        }
                        TermKind::And(args) | TermKind::Or(args) => {
                            stack.push(Task::Build(current, pos));
                            for &a in args.iter() {
                                stack.push(Task::Enter(a, pos));
                            }
                        }
                        TermKind::Implies(a, b) => {
                            // a → b  ≡  ¬a ∨ b
                            stack.push(Task::Build(current, pos));
                            stack.push(Task::Enter(a, !pos));
                            stack.push(Task::Enter(b, pos));
                        }
                        TermKind::Xor(a, b) => {
                            // Both expansions need both polarities of both
                            // operands.
                            stack.push(Task::Build(current, pos));
                            stack.push(Task::Enter(a, true));
                            stack.push(Task::Enter(a, false));
                            stack.push(Task::Enter(b, true));
                            stack.push(Task::Enter(b, false));
                        }
                        TermKind::Ite(c, t, e) => {
                            stack.push(Task::Build(current, pos));
                            stack.push(Task::Enter(c, true));
                            stack.push(Task::Enter(c, false));
                            stack.push(Task::Enter(t, pos));
                            stack.push(Task::Enter(e, pos));
                        }
                        TermKind::Lt(a, b) => {
                            let r =
                                self.classify_cmp(current, a, b, CmpRel::Lt, x_spur, pos, tm)?;
                            memo.insert((current, pos), r);
                        }
                        TermKind::Le(a, b) => {
                            let r =
                                self.classify_cmp(current, a, b, CmpRel::Le, x_spur, pos, tm)?;
                            memo.insert((current, pos), r);
                        }
                        TermKind::Gt(a, b) => {
                            let r =
                                self.classify_cmp(current, a, b, CmpRel::Gt, x_spur, pos, tm)?;
                            memo.insert((current, pos), r);
                        }
                        TermKind::Ge(a, b) => {
                            let r =
                                self.classify_cmp(current, a, b, CmpRel::Ge, x_spur, pos, tm)?;
                            memo.insert((current, pos), r);
                        }
                        TermKind::Eq(a, b) => {
                            let r =
                                self.classify_cmp(current, a, b, CmpRel::Eq, x_spur, pos, tm)?;
                            memo.insert((current, pos), r);
                        }
                        _ => {
                            // Any other atom.
                            if self.mentions_x(current, x_spur, tm) {
                                return Err(
                                    "cooper: unsupported term mentioning the eliminated variable"
                                        .to_string(),
                                );
                            }
                            let lit = if pos { current } else { tm.mk_not(current) };
                            memo.insert((current, pos), Rc::new(Raw::Free(lit)));
                        }
                    }
                }
                Task::Build(current, pos) => {
                    let kind = match tm.get(current) {
                        Some(t) => t.kind.clone(),
                        None => return Err("cooper: term not found".to_string()),
                    };
                    let built = match kind {
                        TermKind::Not(a) => sub(&memo, a, !pos)?,
                        TermKind::And(args) => {
                            let mut subs = Vec::with_capacity(args.len());
                            for &a in args.iter() {
                                subs.push(sub(&memo, a, pos)?);
                            }
                            Rc::new(if pos { Raw::And(subs) } else { Raw::Or(subs) })
                        }
                        TermKind::Or(args) => {
                            let mut subs = Vec::with_capacity(args.len());
                            for &a in args.iter() {
                                subs.push(sub(&memo, a, pos)?);
                            }
                            Rc::new(if pos { Raw::Or(subs) } else { Raw::And(subs) })
                        }
                        TermKind::Implies(a, b) => {
                            let parts = vec![sub(&memo, a, !pos)?, sub(&memo, b, pos)?];
                            Rc::new(if pos { Raw::Or(parts) } else { Raw::And(parts) })
                        }
                        TermKind::Xor(a, b) => {
                            let (at, af) = (sub(&memo, a, true)?, sub(&memo, a, false)?);
                            let (bt, bf) = (sub(&memo, b, true)?, sub(&memo, b, false)?);
                            let (l, r) = if pos {
                                // (a ∧ ¬b) ∨ (¬a ∧ b)
                                (vec![at, bf], vec![af, bt])
                            } else {
                                // a ↔ b  ≡  (a ∧ b) ∨ (¬a ∧ ¬b)
                                (vec![at, bt], vec![af, bf])
                            };
                            Rc::new(Raw::Or(vec![Rc::new(Raw::And(l)), Rc::new(Raw::And(r))]))
                        }
                        TermKind::Ite(c, t, e) => {
                            // (c ∧ t) ∨ (¬c ∧ e), polarity applied to the branches.
                            let l = vec![sub(&memo, c, true)?, sub(&memo, t, pos)?];
                            let r = vec![sub(&memo, c, false)?, sub(&memo, e, pos)?];
                            Rc::new(Raw::Or(vec![Rc::new(Raw::And(l)), Rc::new(Raw::And(r))]))
                        }
                        // `Build` is only ever scheduled for the kinds above.
                        _ => {
                            return Err(
                                "cooper: internal error: unexpected rebuild target".to_string()
                            );
                        }
                    };
                    memo.insert((current, pos), built);
                }
            }
        }

        sub(&memo, id, positive)
    }

    /// Classify a comparison atom `lhs REL rhs` (with `REL = rel0`).
    #[allow(clippy::too_many_arguments)]
    fn classify_cmp(
        &self,
        atom: TermId,
        lhs: TermId,
        rhs: TermId,
        rel0: CmpRel,
        x_spur: crate::interner::Spur,
        positive: bool,
        tm: &mut TermManager,
    ) -> Result<Rc<Raw>, String> {
        // x-free comparison: keep verbatim (respecting polarity).
        if !self.mentions_x(atom, x_spur, tm) {
            let lit = if positive { atom } else { tm.mk_not(atom) };
            return Ok(Rc::new(Raw::Free(lit)));
        }

        // Divisibility pattern: (mod E d) = 0  with d a positive constant.
        if rel0 == CmpRel::Eq
            && let Some((modulus, inner)) = self.match_mod_zero(lhs, rhs, tm)
        {
            let form = self
                .to_linear(inner, x_spur, tm)
                .ok_or_else(|| "cooper: non-linear divisibility argument".to_string())?;
            if form.x_coeff.is_zero() {
                // x cancelled inside the divisibility → x-free literal.
                let lit = if positive { atom } else { tm.mk_not(atom) };
                return Ok(Rc::new(Raw::Free(lit)));
            }
            return Ok(Rc::new(Raw::Divis {
                modulus,
                form,
                negated: !positive,
            }));
        }

        let lf_l = self
            .to_linear(lhs, x_spur, tm)
            .ok_or_else(|| "cooper: non-linear comparison operand".to_string())?;
        let lf_r = self
            .to_linear(rhs, x_spur, tm)
            .ok_or_else(|| "cooper: non-linear comparison operand".to_string())?;
        let form = lf_l.sub(lf_r);
        let rel = if positive { rel0 } else { rel0.negate() };

        if form.x_coeff.is_zero() {
            // x cancelled out: materialise the x-free comparison directly.
            let t = self.mk_linear_term(&form.others, &form.constant, tm);
            let zero = tm.mk_int(0);
            let lit = match rel {
                CmpRel::Lt => tm.mk_lt(t, zero),
                CmpRel::Le => tm.mk_le(t, zero),
                CmpRel::Gt => tm.mk_gt(t, zero),
                CmpRel::Ge => tm.mk_ge(t, zero),
                CmpRel::Eq => tm.mk_eq(t, zero),
                CmpRel::Ne => {
                    let e = tm.mk_eq(t, zero);
                    tm.mk_not(e)
                }
            };
            return Ok(Rc::new(Raw::Free(lit)));
        }

        Ok(Rc::new(Raw::Cmp { form, rel }))
    }

    /// Recognise `(mod E d) = 0` (in either argument order) with `d > 0`.
    fn match_mod_zero(&self, a: TermId, b: TermId, tm: &TermManager) -> Option<(BigInt, TermId)> {
        let try_side = |m: TermId, z: TermId| -> Option<(BigInt, TermId)> {
            let mt = tm.get(m)?;
            let TermKind::Mod(inner, d) = &mt.kind else {
                return None;
            };
            let dt = tm.get(*d)?;
            let TermKind::IntConst(dv) = &dt.kind else {
                return None;
            };
            if dv <= &BigInt::zero() {
                return None;
            }
            let zt = tm.get(z)?;
            if let TermKind::IntConst(zv) = &zt.kind
                && zv.is_zero()
            {
                return Some((dv.clone(), *inner));
            }
            None
        };
        try_side(a, b).or_else(|| try_side(b, a))
    }

    /// Materialise a linear combination `Σ others + constant` as a term (the
    /// eliminated variable is guaranteed absent from `others`).
    fn mk_linear_term(
        &self,
        others: &FxHashMap<TermId, BigInt>,
        constant: &BigInt,
        tm: &mut TermManager,
    ) -> TermId {
        let mut entries: Vec<(TermId, BigInt)> = others
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        entries.sort_by_key(|(k, _)| k.0);

        let mut parts: Vec<TermId> = Vec::new();
        for (atom, coeff) in entries {
            if coeff.is_one() {
                parts.push(atom);
            } else {
                let c = tm.mk_int(coeff);
                parts.push(tm.mk_mul(vec![c, atom]));
            }
        }
        if !constant.is_zero() || parts.is_empty() {
            let c = tm.mk_int(constant.clone());
            parts.push(c);
        }
        if parts.len() == 1 {
            parts[0]
        } else {
            tm.mk_add(parts)
        }
    }

    /// `lcm` accumulation over all `x`-coefficients in a [`Raw`] DAG.
    ///
    /// Explicit stack; the `seen` set keeps a shared sub-DAG from being
    /// re-walked once per parent. `lcm` is idempotent, so visiting a node once
    /// gives the same accumulator as visiting it many times.
    fn collect_x_coeff_lcm(raw: &Rc<Raw>, acc: &mut BigInt) {
        let mut seen: FxHashSet<usize> = FxHashSet::default();
        let mut stack = vec![Rc::clone(raw)];

        while let Some(current) = stack.pop() {
            if !seen.insert(node_key(&current)) {
                continue;
            }
            match &*current {
                Raw::And(subs) | Raw::Or(subs) => stack.extend(subs.iter().map(Rc::clone)),
                Raw::Cmp { form, .. } | Raw::Divis { form, .. } => {
                    if !form.x_coeff.is_zero() {
                        *acc = acc.lcm(&form.x_coeff.abs());
                    }
                }
                Raw::Const(_) | Raw::Free(_) => {}
            }
        }
    }

    /// Convert a [`Raw`] DAG into a normalised [`Node`] DAG, scaling every
    /// `x`-literal so the coefficient of `x` becomes `±1` (with `|x_coeff| = L`
    /// renamed to the unit variable `u`).
    ///
    /// Explicit stack with a memo keyed on node identity: the conversion of a
    /// node depends only on the node and on `lcm_coeff` (fixed for the call),
    /// so a shared sub-DAG converts once and stays shared in the result.
    fn convert(
        &self,
        raw: &Rc<Raw>,
        lcm_coeff: &BigInt,
        tm: &mut TermManager,
    ) -> Result<Rc<Node>, String> {
        /// Work item of the iterative converter.
        enum Task {
            /// Classify a [`Raw`] node and schedule its children.
            Enter(Rc<Raw>),
            /// Assemble a [`Node`] from its already-converted children.
            Build(Rc<Raw>),
        }

        /// Fetch an already-converted child.
        fn converted(
            memo: &FxHashMap<usize, Rc<Node>>,
            child: &Rc<Raw>,
        ) -> Result<Rc<Node>, String> {
            memo.get(&node_key(child))
                .map(Rc::clone)
                .ok_or_else(|| "cooper: internal error: unconverted sub-formula".to_string())
        }

        let mut memo: FxHashMap<usize, Rc<Node>> = FxHashMap::default();
        let mut stack = vec![Task::Enter(Rc::clone(raw))];

        while let Some(task) = stack.pop() {
            match task {
                Task::Enter(current) => {
                    let key = node_key(&current);
                    if memo.contains_key(&key) {
                        continue;
                    }
                    let converted_leaf = match &*current {
                        Raw::And(subs) | Raw::Or(subs) => {
                            let children: Vec<Rc<Raw>> = subs.iter().map(Rc::clone).collect();
                            stack.push(Task::Build(Rc::clone(&current)));
                            stack.extend(children.into_iter().map(Task::Enter));
                            continue;
                        }
                        Raw::Const(b) => Rc::new(Node::Lit(NLit::Const(*b))),
                        Raw::Free(t) => Rc::new(Node::Lit(NLit::Free(*t))),
                        Raw::Cmp { form, rel } => self.convert_cmp(form, *rel, lcm_coeff, tm)?,
                        Raw::Divis {
                            modulus,
                            form,
                            negated,
                        } => self.convert_divis(modulus, form, *negated, lcm_coeff, tm)?,
                    };
                    memo.insert(key, converted_leaf);
                }
                Task::Build(current) => {
                    let key = node_key(&current);
                    let built = match &*current {
                        Raw::And(subs) => {
                            let mut out = Vec::with_capacity(subs.len());
                            for s in subs {
                                out.push(converted(&memo, s)?);
                            }
                            Rc::new(Node::And(out))
                        }
                        Raw::Or(subs) => {
                            let mut out = Vec::with_capacity(subs.len());
                            for s in subs {
                                out.push(converted(&memo, s)?);
                            }
                            Rc::new(Node::Or(out))
                        }
                        // `Build` is only ever scheduled for the kinds above.
                        _ => {
                            return Err(
                                "cooper: internal error: unexpected conversion target".to_string()
                            );
                        }
                    };
                    memo.insert(key, built);
                }
            }
        }

        converted(&memo, raw)
    }

    fn convert_cmp(
        &self,
        form: &LinearForm,
        rel: CmpRel,
        lcm_coeff: &BigInt,
        tm: &mut TermManager,
    ) -> Result<Rc<Node>, String> {
        let c = &form.x_coeff;
        let m = lcm_coeff / c.abs();
        let scaled = form.clone().scale(&m);
        let sign_pos = !c.is_negative();

        // rest' = scaled form without the x term (materialised).
        let rest = self.mk_linear_term(&scaled.others, &scaled.constant, tm);

        // base = threshold value for x, eff = orientation relation.
        let (base, eff) = if sign_pos {
            (tm.mk_neg(rest), rel)
        } else {
            (rest, rel.flip_lg())
        };

        let node = match eff {
            CmpRel::Lt => Node::Lit(NLit::Upper(base)),
            CmpRel::Le => {
                let b = self.shift(base, 1, tm);
                Node::Lit(NLit::Upper(b))
            }
            CmpRel::Gt => Node::Lit(NLit::Lower(base)),
            CmpRel::Ge => {
                let b = self.shift(base, -1, tm);
                Node::Lit(NLit::Lower(b))
            }
            CmpRel::Eq => {
                let lo = self.shift(base, -1, tm);
                let hi = self.shift(base, 1, tm);
                Node::And(vec![
                    Rc::new(Node::Lit(NLit::Lower(lo))),
                    Rc::new(Node::Lit(NLit::Upper(hi))),
                ])
            }
            CmpRel::Ne => Node::Or(vec![
                Rc::new(Node::Lit(NLit::Upper(base))),
                Rc::new(Node::Lit(NLit::Lower(base))),
            ]),
        };
        Ok(Rc::new(node))
    }

    fn convert_divis(
        &self,
        modulus: &BigInt,
        form: &LinearForm,
        negated: bool,
        lcm_coeff: &BigInt,
        tm: &mut TermManager,
    ) -> Result<Rc<Node>, String> {
        let c = &form.x_coeff;
        let m = lcm_coeff / c.abs();
        let scaled = form.clone().scale(&m);
        let new_modulus = modulus * &m;
        let sign_pos = !c.is_negative();

        // off = sign · rest'   (so that new_modulus | (u + off)).
        let rest = self.mk_linear_term(&scaled.others, &scaled.constant, tm);
        let off = if sign_pos { rest } else { tm.mk_neg(rest) };

        Ok(Rc::new(Node::Lit(if negated {
            NLit::NotDiv {
                modulus: new_modulus,
                off,
            }
        } else {
            NLit::Div {
                modulus: new_modulus,
                off,
            }
        })))
    }

    /// Build `term + k` (with `k` a small integer offset).
    fn shift(&self, term: TermId, k: i64, tm: &mut TermManager) -> TermId {
        if k == 0 {
            return term;
        }
        let kt = tm.mk_int(k);
        tm.mk_add(vec![term, kt])
    }

    /// `lcm` accumulation over all divisibility moduli in a [`Node`] DAG.
    ///
    /// Explicit stack; `lcm` is idempotent, so visiting each shared node once
    /// yields the same period `δ` as re-walking it per parent.
    fn collect_moduli_lcm(node: &Rc<Node>, acc: &mut BigInt) {
        let mut seen: FxHashSet<usize> = FxHashSet::default();
        let mut stack = vec![Rc::clone(node)];

        while let Some(current) = stack.pop() {
            if !seen.insert(node_key(&current)) {
                continue;
            }
            match &*current {
                Node::And(subs) | Node::Or(subs) => stack.extend(subs.iter().map(Rc::clone)),
                Node::Lit(NLit::Div { modulus, .. }) | Node::Lit(NLit::NotDiv { modulus, .. }) => {
                    *acc = acc.lcm(modulus);
                }
                Node::Lit(_) => {}
            }
        }
    }

    /// Collect all lower-bound boundary terms.
    ///
    /// Explicit stack. `B` is a *set* of boundary terms — the elimination
    /// takes a disjunction over it — so a repeated bound (from a shared
    /// sub-DAG, or from two literals with the same boundary term) is dropped:
    /// it would only add a disjunct identical to one already present.
    fn collect_lower_bounds(node: &Rc<Node>, out: &mut Vec<TermId>) {
        let mut seen_nodes: FxHashSet<usize> = FxHashSet::default();
        let mut seen_bounds: FxHashSet<TermId> = FxHashSet::default();
        let mut stack = vec![Rc::clone(node)];

        while let Some(current) = stack.pop() {
            if !seen_nodes.insert(node_key(&current)) {
                continue;
            }
            match &*current {
                Node::And(subs) | Node::Or(subs) => stack.extend(subs.iter().map(Rc::clone)),
                Node::Lit(NLit::Lower(b)) => {
                    if seen_bounds.insert(*b) {
                        out.push(*b);
                    }
                }
                Node::Lit(_) => {}
            }
        }
    }

    /// Materialise a [`Node`] DAG under a given instantiation of `x`.
    ///
    /// Explicit stack with a memo keyed on node identity: with `xval` fixed
    /// for the call, a node always materialises to the same term, so a shared
    /// sub-DAG is materialised once.
    ///
    /// A missing child term cannot happen (every child is materialised before
    /// its parent is assembled) and is reported as an error rather than
    /// patched with a boolean constant: either constant would silently change
    /// the strength of a disjunct of the eliminated formula.
    fn materialize(
        &self,
        node: &Rc<Node>,
        xval: &XVal,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        /// Work item of the iterative materialiser.
        enum Task {
            /// Classify a node and schedule its children.
            Enter(Rc<Node>),
            /// Assemble a term from its already-materialised children.
            Build(Rc<Node>),
        }

        /// Fetch the already-materialised terms of a node's children.
        fn child_terms(
            memo: &FxHashMap<usize, TermId>,
            subs: &[Rc<Node>],
        ) -> Result<Vec<TermId>, String> {
            subs.iter()
                .map(|s| {
                    memo.get(&node_key(s)).copied().ok_or_else(|| {
                        "cooper: internal error: unmaterialised sub-node".to_string()
                    })
                })
                .collect()
        }

        let mut memo: FxHashMap<usize, TermId> = FxHashMap::default();
        let mut stack = vec![Task::Enter(Rc::clone(node))];

        while let Some(task) = stack.pop() {
            match task {
                Task::Enter(current) => {
                    let key = node_key(&current);
                    if memo.contains_key(&key) {
                        continue;
                    }
                    match &*current {
                        Node::And(subs) | Node::Or(subs) => {
                            let children: Vec<Rc<Node>> = subs.iter().map(Rc::clone).collect();
                            stack.push(Task::Build(Rc::clone(&current)));
                            stack.extend(children.into_iter().map(Task::Enter));
                        }
                        Node::Lit(lit) => {
                            let t = self.materialize_lit(lit, xval, tm);
                            memo.insert(key, t);
                        }
                    }
                }
                Task::Build(current) => {
                    let key = node_key(&current);
                    let built = match &*current {
                        Node::And(subs) => {
                            let parts = child_terms(&memo, subs)?;
                            tm.mk_and(parts)
                        }
                        Node::Or(subs) => {
                            let parts = child_terms(&memo, subs)?;
                            tm.mk_or(parts)
                        }
                        Node::Lit(lit) => self.materialize_lit(lit, xval, tm),
                    };
                    memo.insert(key, built);
                }
            }
        }

        memo.get(&node_key(node))
            .copied()
            .ok_or_else(|| "cooper: internal error: node was not materialised".to_string())
    }

    fn materialize_lit(&self, lit: &NLit, xval: &XVal, tm: &mut TermManager) -> TermId {
        match lit {
            NLit::Lower(b) => match xval {
                XVal::MinusInf(_) => tm.mk_false(),
                XVal::At(v) => tm.mk_lt(*b, *v),
            },
            NLit::Upper(a) => match xval {
                XVal::MinusInf(_) => tm.mk_true(),
                XVal::At(v) => tm.mk_lt(*v, *a),
            },
            NLit::Div { modulus, off } => self.materialize_div(modulus, *off, xval, false, tm),
            NLit::NotDiv { modulus, off } => self.materialize_div(modulus, *off, xval, true, tm),
            NLit::Free(t) => *t,
            NLit::Const(b) => tm.mk_bool(*b),
        }
    }

    /// Materialise `modulus | (x + off)` (or its negation) at the given `x`.
    fn materialize_div(
        &self,
        modulus: &BigInt,
        off: TermId,
        xval: &XVal,
        negated: bool,
        tm: &mut TermManager,
    ) -> TermId {
        let x_term = match xval {
            XVal::MinusInf(j) => tm.mk_int((*j).clone()),
            XVal::At(v) => *v,
        };
        let arg = tm.mk_add(vec![x_term, off]);
        let m = tm.mk_int(modulus.clone());
        let modterm = tm.mk_mod(arg, m);
        let zero = tm.mk_int(0);
        let eq = tm.mk_eq(modterm, zero);
        if negated { tm.mk_not(eq) } else { eq }
    }

    /// Get statistics.
    pub fn stats(&self) -> &CooperStats {
        &self.stats
    }
}

impl Default for CooperEliminator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_var(tm: &mut TermManager, name: &str) -> TermId {
        let z = tm.mk_int(0);
        let int_sort = tm.get(z).expect("int const has a sort").sort;
        tm.mk_var(name, int_sort)
    }

    #[test]
    fn test_cooper_eliminator() {
        let eliminator = CooperEliminator::new();
        assert_eq!(eliminator.stats.quantifiers_eliminated, 0);
    }

    #[test]
    fn test_result_is_x_free_even_predicate() {
        // ∃x. 2*x = y   ≡   y even
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let two = tm.mk_int(2);
        let two_x = tm.mk_mul(vec![two, x]);
        let phi = tm.mk_eq(two_x, y);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");

        // The quantified variable must no longer occur.
        let x_spur = tm.intern_str("x");
        assert!(
            !elim.mentions_x(result, x_spur, &tm),
            "eliminated variable x still occurs in the result"
        );
        // And it must not simply be the input formula.
        assert_ne!(result, phi, "result equals the input (unsound stub)");
    }

    /// Minimal ground evaluator over the fragment Cooper emits (no free vars).
    pub(super) fn eval_ground(tm: &TermManager, id: TermId) -> bool {
        fn int(tm: &TermManager, id: TermId) -> i128 {
            match &tm.get(id).expect("term").kind {
                TermKind::IntConst(n) => n.to_string().parse::<i128>().expect("fits"),
                TermKind::Neg(a) => -int(tm, *a),
                TermKind::Add(args) => args.iter().map(|&a| int(tm, a)).sum(),
                TermKind::Sub(a, b) => int(tm, *a) - int(tm, *b),
                TermKind::Mul(args) => args.iter().map(|&a| int(tm, a)).product(),
                TermKind::Mod(a, b) => {
                    // `rem_euclid` panics on a zero divisor, and an opaque
                    // arithmetic panic here would read as a bug in this
                    // evaluator rather than in what it is evaluating. Every
                    // `mod` Cooper emits is a divisibility test whose modulus
                    // is a least common multiple of the coefficients it just
                    // collected, so it is positive by construction; naming
                    // that invariant is what makes a violation of it legible.
                    let divisor = int(tm, *b);
                    assert!(
                        divisor != 0,
                        "Cooper emits `mod` only against a positive lcm modulus, \
                         so a zero divisor means the elimination produced a term \
                         it should not have"
                    );
                    int(tm, *a).rem_euclid(divisor)
                }
                other => panic!("unexpected int term {other:?}"),
            }
        }
        match &tm.get(id).expect("term").kind {
            TermKind::True => true,
            TermKind::False => false,
            TermKind::Not(a) => !eval_ground(tm, *a),
            TermKind::And(args) => args.iter().all(|&a| eval_ground(tm, a)),
            TermKind::Or(args) => args.iter().any(|&a| eval_ground(tm, a)),
            TermKind::Lt(a, b) => int(tm, *a) < int(tm, *b),
            TermKind::Le(a, b) => int(tm, *a) <= int(tm, *b),
            TermKind::Gt(a, b) => int(tm, *a) > int(tm, *b),
            TermKind::Ge(a, b) => int(tm, *a) >= int(tm, *b),
            TermKind::Eq(a, b) => int(tm, *a) == int(tm, *b),
            other => panic!("unexpected bool term {other:?}"),
        }
    }

    #[test]
    fn test_bounded_true() {
        // ∃x. (2 < x) ∧ (x < 4)  is true (x = 3).
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let two = tm.mk_int(2);
        let four = tm.mk_int(4);
        let c1 = tm.mk_lt(two, x);
        let c2 = tm.mk_lt(x, four);
        let phi = tm.mk_and(vec![c1, c2]);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");
        let x_spur = tm.intern_str("x");
        assert!(!elim.mentions_x(result, x_spur, &tm), "x still present");
        assert!(eval_ground(&tm, result), "expected the result to be true");
    }

    #[test]
    fn test_bounded_false() {
        // ∃x. (4 < x) ∧ (x < 4)  is false (empty interval).
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let four = tm.mk_int(4);
        let c1 = tm.mk_lt(four, x);
        let c2 = tm.mk_lt(x, four);
        let phi = tm.mk_and(vec![c1, c2]);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");
        let x_spur = tm.intern_str("x");
        assert!(!elim.mentions_x(result, x_spur, &tm), "x still present");
        assert!(!eval_ground(&tm, result), "expected the result to be false");
    }

    #[test]
    fn test_nonlinear_is_rejected() {
        // ∃x. x*x = y  is outside the linear fragment → honest Err.
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let xx = tm.mk_mul(vec![x, x]);
        let phi = tm.mk_eq(xx, y);

        let mut elim = CooperEliminator::new();
        let result = elim.eliminate_exists("x".to_string(), phi, &mut tm);
        assert!(
            result.is_err(),
            "non-linear input must be rejected, not faked"
        );
    }

    #[test]
    fn test_x_free_formula_returned() {
        // ∃x. (y < 3)  with x absent ≡ (y < 3).
        let mut tm = TermManager::new();
        let y = int_var(&mut tm, "y");
        let three = tm.mk_int(3);
        let phi = tm.mk_lt(y, three);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");
        assert_eq!(result, phi);
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;

    #[test]
    fn test_mentions_x_shared_dag_is_fast() {
        // Two-strand DAG, 55 levels: 2^55 nodes without a visited set.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let (mut a, mut b) = (x, y);
        for _ in 0..55 {
            let next_a = tm.mk_sub(a, b);
            let next_b = tm.mk_add([b, a]);
            a = next_a;
            b = next_b;
        }
        let x_spur = tm.intern_str("x");
        let z_spur = tm.intern_str("z");

        let elim = CooperEliminator::new();
        assert!(elim.mentions_x(a, x_spur, &tm));
        assert!(!elim.mentions_x(a, z_spur, &tm));
        assert_eq!(elim.find_var_sort(a, x_spur, &tm), Some(int_sort));
    }

    fn int_var(tm: &mut TermManager, name: &str) -> TermId {
        let int_sort = tm.sorts.int_sort;
        tm.mk_var(name, int_sort)
    }

    /// `levels` alternating `And`/`Or` nestings around an `x` atom. Alternating
    /// the connective defeats the n-ary flattening in `mk_and`/`mk_or`, so the
    /// boolean skeleton really is `levels` deep.
    fn deep_bool_formula(tm: &mut TermManager, levels: usize) -> TermId {
        let x = int_var(tm, "x");
        let y = int_var(tm, "y");
        let zero = tm.mk_int(0);
        let atom_x = tm.mk_lt(zero, x);
        let atom_y = tm.mk_lt(zero, y);
        let mut f = atom_x;
        for _ in 0..levels / 2 {
            f = tm.mk_and([f, atom_y]);
            f = tm.mk_or([f, atom_y]);
        }
        f
    }

    #[test]
    fn test_eliminate_exists_deep_nesting_does_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let f = deep_bool_formula(&mut tm, 6_250);
                let mut elim = CooperEliminator::new();
                let outcome = elim.eliminate_exists("x".to_string(), f, &mut tm);
                let x_spur = tm.intern_str("x");
                let still_mentions = outcome
                    .as_ref()
                    .ok()
                    .map(|&t| elim.mentions_x(t, x_spur, &tm))
                    .unwrap_or(true);
                (outcome.is_ok(), still_mentions)
            })
            .expect("thread spawn should succeed");

        let (ok, still_mentions) = handle
            .join()
            .expect("cooper elimination must not overflow the stack");
        assert!(ok, "deep elimination failed");
        assert!(!still_mentions, "x survived the elimination");
    }

    #[test]
    fn test_nested_xor_elimination_is_not_exponential() {
        // Each `Xor` level needs both polarities of both operands: the
        // recursive expansion doubled the work per level, so 30 levels meant
        // ~2³⁰ calls (and a 2³⁰-node intermediate tree).
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let y = int_var(&mut tm, "y");
        let zero = tm.mk_int(0);
        let mut f = tm.mk_lt(zero, x);
        let atom_y = tm.mk_lt(zero, y);
        for _ in 0..30 {
            f = tm.mk_xor(f, atom_y);
        }

        let start = oxiz_time::Instant::now();
        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), f, &mut tm)
            .expect("elimination should succeed");
        let elapsed = start.elapsed();

        let x_spur = tm.intern_str("x");
        assert!(!elim.mentions_x(result, x_spur, &tm), "x still present");
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "nested-Xor elimination took {elapsed:?}: the sharing memo regressed"
        );
    }

    #[test]
    fn test_xor_expansion_semantics() {
        // ∃x. (x > 0) xor (x ≥ 1): equivalent literals over the integers, so
        // the xor is false for every x and the elimination must be false.
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let zero = tm.mk_int(0);
        let one = tm.mk_int(1);
        let gt0 = tm.mk_gt(x, zero);
        let ge1 = tm.mk_ge(x, one);
        let phi = tm.mk_xor(gt0, ge1);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");
        assert!(
            !super::tests::eval_ground(&tm, result),
            "xor of equivalent literals must eliminate to false"
        );

        // ∃x. (x > 0) xor (x > 5) holds (any x in 1..=5).
        let five = tm.mk_int(5);
        let gt5 = tm.mk_gt(x, five);
        let psi = tm.mk_xor(gt0, gt5);
        let result = elim
            .eliminate_exists("x".to_string(), psi, &mut tm)
            .expect("elimination should succeed");
        assert!(
            super::tests::eval_ground(&tm, result),
            "xor with a strictly weaker literal is satisfiable"
        );
    }

    #[test]
    fn test_ite_expansion_semantics() {
        // ∃x. ite(x > 0, x > 5, x < -10) holds (x = 6 takes the then branch,
        // x = -11 the else branch).
        let mut tm = TermManager::new();
        let x = int_var(&mut tm, "x");
        let zero = tm.mk_int(0);
        let five = tm.mk_int(5);
        let minus_ten = tm.mk_int(-10);
        let c = tm.mk_gt(x, zero);
        let t = tm.mk_gt(x, five);
        let e = tm.mk_lt(x, minus_ten);
        let phi = tm.mk_ite(c, t, e);

        let mut elim = CooperEliminator::new();
        let result = elim
            .eliminate_exists("x".to_string(), phi, &mut tm)
            .expect("elimination should succeed");
        assert!(
            super::tests::eval_ground(&tm, result),
            "both branches of the ite are satisfiable"
        );

        // ∃x. ite(x > 0, x < 0, x < 0 ∧ x > 0) is unsatisfiable in both
        // branches: `(c ∧ x<0) ∨ (¬c ∧ x<0 ∧ c)`.
        let neg = tm.mk_lt(x, zero);
        let contradiction = tm.mk_and([neg, c]);
        let psi = tm.mk_ite(c, neg, contradiction);
        let result = elim
            .eliminate_exists("x".to_string(), psi, &mut tm)
            .expect("elimination should succeed");
        assert!(
            !super::tests::eval_ground(&tm, result),
            "both branches are unsatisfiable"
        );
    }

    #[test]
    fn test_cooper_walks_deep_nesting_do_not_overflow() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut tm = TermManager::new();
                let int_sort = tm.sorts.int_sort;
                let x = tm.mk_var("x", int_sort);
                let one = tm.mk_int(1);
                let mut term = x;
                for _ in 0..7_500 {
                    term = tm.mk_add([term, one]);
                }
                let x_spur = tm.intern_str("x");

                let elim = CooperEliminator::new();
                let mentions = elim.mentions_x(term, x_spur, &tm);
                let linear = elim.to_linear(term, x_spur, &tm);
                (mentions, linear.map(|f| f.x_coeff))
            })
            .expect("thread spawn should succeed");

        let (mentions, x_coeff) = handle.join().expect("deep walks must not overflow");
        assert!(mentions);
        assert_eq!(x_coeff, Some(BigInt::one()));
    }
}
