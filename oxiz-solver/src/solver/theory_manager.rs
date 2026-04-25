//! Theory manager that bridges the SAT solver with theory solvers

#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_sat::{Lit, TheoryCallback, TheoryCheckResult, Var};
use oxiz_theories::arithmetic::ArithSolver;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::euf::EufSolver;
use oxiz_theories::{EqualityNotification, Theory, TheoryCombination};
use smallvec::SmallVec;

use super::types::{
    ArithConstraintType, Constraint, ParsedArithConstraint, Statistics, TheoryMode,
};

/// Theory decision hint
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TheoryDecision {
    /// The variable to branch on
    pub var: Var,
    /// Suggested value (true = positive, false = negative)
    pub value: bool,
    /// Priority (higher = more important)
    pub priority: i32,
}

/// Theory manager that bridges the SAT solver with theory solvers
pub(crate) struct TheoryManager<'a> {
    /// Reference to the term manager
    manager: &'a TermManager,
    /// Reference to the EUF solver
    euf: &'a mut EufSolver,
    /// Reference to the arithmetic solver
    arith: &'a mut ArithSolver,
    /// Reference to the bitvector solver
    bv: &'a mut BvSolver,
    /// Bitvector terms (for identifying BV variables)
    bv_terms: &'a FxHashSet<TermId>,
    /// Mapping from SAT variables to constraints
    var_to_constraint: &'a FxHashMap<Var, Constraint>,
    /// Mapping from SAT variables to parsed arithmetic constraints
    var_to_parsed_arith: &'a FxHashMap<Var, ParsedArithConstraint>,
    /// Mapping from terms to SAT variables (for conflict clause generation)
    term_to_var: &'a FxHashMap<TermId, Var>,
    /// Reverse mapping from SAT variables to terms (for EUF merge reasons)
    var_to_term: &'a Vec<TermId>,
    /// Current decision level stack for backtracking
    level_stack: Vec<usize>,
    /// Number of processed assignments
    processed_count: usize,
    /// Theory checking mode
    theory_mode: TheoryMode,
    /// Pending assignments for lazy theory checking
    pending_assignments: Vec<(Lit, bool)>,
    /// Theory decision hints for branching
    #[allow(dead_code)]
    decision_hints: Vec<TheoryDecision>,
    /// Pending equality notifications for Nelson-Oppen
    pending_equalities: Vec<EqualityNotification>,
    /// Processed equalities (to avoid duplicates)
    processed_equalities: FxHashMap<(TermId, TermId), bool>,
    /// Reference to solver statistics (for tracking)
    statistics: &'a mut Statistics,
    /// Maximum conflicts allowed (0 = unlimited)
    max_conflicts: u64,
    /// Maximum decisions allowed (0 = unlimited)
    #[allow(dead_code)]
    max_decisions: u64,
    /// Whether formula contains BV arithmetic operations (division/remainder)
    has_bv_arith_ops: bool,
    /// Canonical EUF node for each distinct integer constant value.
    ///
    /// Maps an integer literal value (i64) to the canonical EUF node that
    /// represents it.  When a new `IntConst(v)` term is first encountered for a
    /// value `v`, we create its EUF node, assert pairwise disequalities against
    /// every canonical node of a different value, and record it here.
    ///
    /// If the same value `v` appears again (e.g., as a fresh TermId created
    /// during MBQI instantiation), we merge the new node with the existing
    /// canonical node rather than appending another entry.  This keeps the
    /// number of distinct entries — and therefore the number of pairwise
    /// disequality edges — bounded by the number of *distinct* integer literal
    /// values in the original formula, not by the total number of term IDs
    /// created across all MBQI iterations (which grows without bound).
    interned_int_constants: FxHashMap<i64, u32>,
    /// Canonical EUF nodes for Boolean true and false values.
    /// Used to track Bool-valued function applications in EUF:
    /// when `f(x)` is assigned true by the SAT solver, we merge its EUF node
    /// with `bool_true_node`; when assigned false, with `bool_false_node`.
    /// A disequality `true != false` is asserted so that congruence closure
    /// detects conflicts (e.g., f(a)=true, f(b)=false, but a=b).
    bool_true_node: Option<u32>,
    bool_false_node: Option<u32>,
}

impl<'a> TheoryManager<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        manager: &'a TermManager,
        euf: &'a mut EufSolver,
        arith: &'a mut ArithSolver,
        bv: &'a mut BvSolver,
        bv_terms: &'a FxHashSet<TermId>,
        var_to_constraint: &'a FxHashMap<Var, Constraint>,
        var_to_parsed_arith: &'a FxHashMap<Var, ParsedArithConstraint>,
        term_to_var: &'a FxHashMap<TermId, Var>,
        var_to_term: &'a Vec<TermId>,
        theory_mode: TheoryMode,
        statistics: &'a mut Statistics,
        max_conflicts: u64,
        max_decisions: u64,
        has_bv_arith_ops: bool,
    ) -> Self {
        Self {
            manager,
            euf,
            arith,
            bv,
            bv_terms,
            var_to_constraint,
            var_to_parsed_arith,
            term_to_var,
            var_to_term,
            level_stack: vec![0],
            processed_count: 0,
            theory_mode,
            pending_assignments: Vec::new(),
            decision_hints: Vec::new(),
            pending_equalities: Vec::new(),
            processed_equalities: FxHashMap::default(),
            statistics,
            max_conflicts,
            max_decisions,
            has_bv_arith_ops,
            interned_int_constants: FxHashMap::default(),
            bool_true_node: None,
            bool_false_node: None,
        }
    }

    /// Process Nelson-Oppen equality sharing
    /// Propagates equalities between theories until a fixed point is reached
    #[allow(dead_code)]
    fn propagate_equalities(&mut self) -> TheoryCheckResult {
        // Process all pending equalities
        while let Some(eq) = self.pending_equalities.pop() {
            // Avoid processing the same equality twice
            let key = if eq.lhs < eq.rhs {
                (eq.lhs, eq.rhs)
            } else {
                (eq.rhs, eq.lhs)
            };

            if self.processed_equalities.contains_key(&key) {
                continue;
            }
            self.processed_equalities.insert(key, true);

            // Notify EUF theory
            let lhs_node = self.euf.intern(eq.lhs);
            let rhs_node = self.euf.intern(eq.rhs);
            if let Err(_e) = self
                .euf
                .merge(lhs_node, rhs_node, eq.reason.unwrap_or(eq.lhs))
            {
                // Merge failed - should not happen
                continue;
            }

            // Check for conflicts after merging
            if let Some(conflict_terms) = self.euf.check_conflicts() {
                let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                return TheoryCheckResult::Conflict(conflict_lits);
            }

            // Notify arithmetic theory
            self.arith.notify_equality(eq);
        }

        TheoryCheckResult::Sat
    }

    /// Propagate EUF-derived equalities to the arithmetic solver.
    ///
    /// When EUF fires congruence closure and derives `f(x) = f(y)` because
    /// `x = y` was asserted, the arithmetic solver is unaware of this equality.
    /// This method gathers all arithmetic terms from `var_to_parsed_arith`,
    /// looks each one up in EUF (via `term_to_node`), and for any pair whose
    /// EUF nodes are in the same equivalence class asserts `t1 - t2 = 0` into
    /// the arithmetic solver.
    ///
    /// Note: `euf.intern(t)` uses the `term_to_node` map first, so it correctly
    /// returns the shared node index even when two distinct term IDs (e.g.
    /// `f_x_term` and `f_y_term`) were mapped to the same node via congruence
    /// during `intern_app`.
    fn propagate_euf_equalities_to_arith(&mut self) -> TheoryCheckResult {
        // Collect every unique term ID that appears in any parsed arithmetic
        // constraint.  These are the terms the arithmetic solver knows about.
        let mut arith_terms: Vec<TermId> = Vec::new();
        for parsed in self.var_to_parsed_arith.values() {
            for &(term, _coef) in &parsed.terms {
                if !arith_terms.contains(&term) {
                    arith_terms.push(term);
                }
            }
        }

        // For each pair of arith terms, check if they are EUF-equal.
        // `euf.intern(t)` looks up `term_to_node` first, so two terms that
        // share the same EUF node (via congruence at intern-time) correctly
        // return the same node index.
        for i in 0..arith_terms.len() {
            for j in (i + 1)..arith_terms.len() {
                let t1 = arith_terms[i];
                let t2 = arith_terms[j];
                if t1 == t2 {
                    continue;
                }
                // Only consider terms that have been registered in EUF.
                let Some(n1) = self.euf.term_to_node(t1) else {
                    continue;
                };
                let Some(n2) = self.euf.term_to_node(t2) else {
                    continue;
                };
                if self.euf.are_equal(n1, n2) {
                    // EUF has derived t1 = t2.  Assert this equality into the
                    // arithmetic solver as `1*t1 + (-1)*t2 = 0`.
                    // Use t1 as the reason term for conflict clause generation.
                    let reason = t1;
                    self.arith.assert_eq(
                        &[
                            (t1, Rational64::from_integer(1)),
                            (t2, Rational64::from_integer(-1)),
                        ],
                        Rational64::from_integer(0),
                        reason,
                    );

                    // Check ArithSolver for conflicts after each new equality.
                    use oxiz_theories::Theory;
                    use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                    if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) = self.arith.check() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
        }

        TheoryCheckResult::Sat
    }

    /// Model-based theory combination
    /// Detects conflicts where EUF has derived an equality between two terms
    /// but the arithmetic solver assigns them different values.
    fn model_based_combination(&mut self) -> TheoryCheckResult {
        // Check: EUF equality vs arith disagreement
        let shared_terms: Vec<TermId> = self.term_to_var.keys().copied().collect();
        for i in 0..shared_terms.len() {
            for j in (i + 1)..shared_terms.len() {
                let t1 = shared_terms[i];
                let t2 = shared_terms[j];

                let t1_node = self.euf.intern(t1);
                let t2_node = self.euf.intern(t2);

                if self.euf.are_equal(t1_node, t2_node) {
                    let t1_value = self.arith.value(t1);
                    let t2_value = self.arith.value(t2);
                    if let (Some(v1), Some(v2)) = (t1_value, t2_value)
                        && v1 != v2
                    {
                        let conflict_lits = self.terms_to_conflict_clause(&[t1, t2]);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
        }

        TheoryCheckResult::Sat
    }

    /// Add an equality to be shared between theories
    #[allow(dead_code)]
    fn add_shared_equality(&mut self, lhs: TermId, rhs: TermId, reason: Option<TermId>) {
        self.pending_equalities
            .push(EqualityNotification { lhs, rhs, reason });
    }

    /// Get theory decision hints for branching
    /// Returns suggested variables to branch on, ordered by priority
    #[allow(dead_code)]
    fn get_decision_hints(&mut self) -> &[TheoryDecision] {
        // Clear old hints
        self.decision_hints.clear();

        // Collect hints from theory solvers
        // For now, we can suggest branching on variables that appear in
        // unsatisfied constraints or pending equalities

        // EUF hints: suggest branching on disequalities that might conflict
        // Arithmetic hints: suggest branching on bounds that are close to being violated

        // This is a placeholder - full implementation would query theory solvers
        // for their preferred branching decisions

        &self.decision_hints
    }

    /// Sentinel function ID used for array `select(array, index)` in EUF.
    ///
    /// `Spur::into_inner()` always returns a `NonZeroU32` (>= 1), so 0 is safe
    /// to use as a special, collision-free function ID for the built-in select
    /// operation.  By interning `select(a, i)` as `intern_app(term, SELECT_FUNC_ID,
    /// [a_node, i_node])`, the EUF congruence closure engine treats select like any
    /// other binary function application and will automatically derive
    /// `select(a, x) = select(a, y)` whenever `x = y` is merged.
    const SELECT_FUNC_ID: u32 = 0;

    /// Intern a term into EUF, using `intern_app` for Apply terms and
    /// `TermKind::Select` terms so that congruence closure works correctly.
    ///
    /// Plain `intern` creates opaque nodes with no function-symbol or argument
    /// information, which prevents the congruence closure algorithm from firing
    /// when argument classes are merged.
    ///
    /// `Select(array, index)` is treated as a binary function application with
    /// the special function ID `SELECT_FUNC_ID` (0).  This ensures that when
    /// `x = y` causes their EUF nodes to merge, congruence automatically
    /// derives `select(a, x) = select(a, y)`, which in turn allows further
    /// congruence steps (e.g., `f(select(a,x)) = f(select(a,y))`).
    fn intern_term_deep(&mut self, term: TermId, manager: &TermManager) -> u32 {
        if let Some(idx) = self.euf.term_to_node(term) {
            return idx;
        }
        if let Some(t) = manager.get(term) {
            match &t.kind {
                TermKind::Apply { func, args, .. } => {
                    let func_id = func.into_inner().get();
                    let arg_nodes: SmallVec<[u32; 4]> = args
                        .iter()
                        .map(|&a| self.intern_term_deep(a, manager))
                        .collect();
                    return self.euf.intern_app(term, func_id, arg_nodes);
                }
                TermKind::Select(array, index) => {
                    // Intern both sub-terms first (recursively), then register
                    // `select` as a binary function application so that EUF
                    // congruence closure fires when the index (or array) args
                    // become equal.
                    let array_node = self.intern_term_deep(*array, manager);
                    let index_node = self.intern_term_deep(*index, manager);
                    return self.euf.intern_app(
                        term,
                        Self::SELECT_FUNC_ID,
                        [array_node, index_node],
                    );
                }
                TermKind::IntConst(n) => {
                    // Intern the integer constant as an EUF node and maintain
                    // pairwise disequalities between *distinct* integer values.
                    //
                    // EUF has no built-in notion of numeric inequality.  Without
                    // explicit disequality edges, a congruence chain equating a
                    // node merged with `10` and one merged with `20` would not
                    // produce a conflict.  We therefore assert `10 ≠ 20` etc.
                    //
                    // Performance: we track one *canonical* EUF node per unique
                    // integer value.  When the same value appears again (e.g. as a
                    // fresh TermId created during MBQI instantiation) we merge the
                    // new node into the canonical one.  This bounds the number of
                    // entries — and therefore of pairwise disequality edges — to the
                    // number of *distinct* literal values in the formula, preventing
                    // the O(n²) blowup that arises when MBQI creates many fresh
                    // TermIds for the same integer literal across iterations.
                    if let Some(val) = n.to_i64() {
                        let new_node = self.euf.intern(term);
                        if let Some(&canonical) = self.interned_int_constants.get(&val) {
                            // This value already has a canonical node.  Merge the
                            // new term's node into it so that congruence closure
                            // treats them as equal (they represent the same number).
                            // Ignore merge errors: the nodes may already be in the
                            // same class if this term was interned before.
                            let _ = self.euf.merge(new_node, canonical, term);
                            return canonical;
                        }
                        // First time we see this value: register the canonical node
                        // and assert disequality against every other distinct value.
                        let diseq_targets: Vec<u32> =
                            self.interned_int_constants.values().copied().collect();
                        for other_node in diseq_targets {
                            self.euf.assert_diseq(new_node, other_node, term);
                        }
                        self.interned_int_constants.insert(val, new_node);
                        return new_node;
                    }
                    // BigInt too large for i64 -- fall through to plain intern.
                }
                _ => {}
            }
        }
        self.euf.intern(term)
    }

    /// Intern a term into EUF for congruence closure, using `intern_app` for
    /// Apply and Select terms so that congruence fires correctly.
    ///
    /// Unlike `intern_term_deep`, this variant does NOT add IntConst pairwise
    /// disequality edges.  Those edges are necessary for conflict detection when
    /// numeric constants are compared via the EUF layer, but they cause spurious
    /// UNSAT in SAT cases where the ArithSolver is the one tracking numeric
    /// inequalities.  This function is used exclusively inside
    /// `process_constraint` for equality/disequality assertions so that
    /// `f(a)=f(b)` congruence works while arithmetic stays in the ArithSolver.
    fn intern_term_for_congruence(&mut self, term: TermId, manager: &TermManager) -> u32 {
        if let Some(idx) = self.euf.term_to_node(term) {
            return idx;
        }
        if let Some(t) = manager.get(term) {
            match &t.kind {
                TermKind::Apply { func, args, .. } => {
                    let func_id = func.into_inner().get();
                    let arg_nodes: SmallVec<[u32; 4]> = args
                        .iter()
                        .map(|&a| self.intern_term_for_congruence(a, manager))
                        .collect();
                    return self.euf.intern_app(term, func_id, arg_nodes);
                }
                TermKind::Select(array, index) => {
                    let array_node = self.intern_term_for_congruence(*array, manager);
                    let index_node = self.intern_term_for_congruence(*index, manager);
                    return self.euf.intern_app(
                        term,
                        Self::SELECT_FUNC_ID,
                        [array_node, index_node],
                    );
                }
                _ => {}
            }
        }
        self.euf.intern(term)
    }

    /// Ensure canonical EUF nodes for Boolean true/false exist, with a
    /// disequality between them.  Returns `(true_node, false_node)`.
    fn ensure_bool_nodes(&mut self) -> (u32, u32) {
        if let (Some(t), Some(f)) = (self.bool_true_node, self.bool_false_node) {
            return (t, f);
        }
        // Use sentinel TermIds that will never collide with real terms.
        // TermId(u32::MAX) and TermId(u32::MAX - 1) are reserved for this.
        let true_term = TermId::new(u32::MAX);
        let false_term = TermId::new(u32::MAX - 1);
        let t = self.euf.intern(true_term);
        let f = self.euf.intern(false_term);
        self.euf.assert_diseq(t, f, true_term);
        self.bool_true_node = Some(t);
        self.bool_false_node = Some(f);
        (t, f)
    }

    /// Look up the term ID for a SAT variable.
    /// Returns a sentinel zero TermId if not found.
    #[inline]
    fn term_for_var(&self, var: Var) -> TermId {
        self.var_to_term
            .get(var.index())
            .copied()
            .unwrap_or_else(|| TermId::new(0))
    }

    /// Convert a list of term IDs to a conflict clause
    /// Each term ID should correspond to a constraint that was asserted
    fn terms_to_conflict_clause(&self, terms: &[TermId]) -> SmallVec<[Lit; 8]> {
        let mut conflict = SmallVec::new();
        for &term in terms {
            if let Some(&var) = self.term_to_var.get(&term) {
                conflict.push(Lit::neg(var));
            }
        }
        conflict
    }

    /// Process a theory constraint
    fn process_constraint(
        &mut self,
        var: Var,
        constraint: Constraint,
        is_positive: bool,
        manager: &TermManager,
    ) -> TheoryCheckResult {
        match constraint {
            Constraint::Eq(lhs, rhs) => {
                if is_positive {
                    // Positive assignment: a = b, tell EUF to merge.
                    // Use the constraint term (which has a SAT variable) as the
                    // merge reason so that conflict clause generation can find it
                    // in term_to_var.
                    let constraint_term = self.term_for_var(var);
                    // Use intern_term_for_congruence so that Apply/Select terms are
                    // registered with intern_app, enabling EUF congruence closure
                    // (e.g., a=b → f(a)=f(b)).  This variant does NOT add IntConst
                    // pairwise disequality edges, keeping arithmetic reasoning in the
                    // ArithSolver and avoiding spurious UNSAT in SAT cases.
                    let lhs_node = self.intern_term_for_congruence(lhs, manager);
                    let rhs_node = self.intern_term_for_congruence(rhs, manager);
                    if let Err(_e) = self.euf.merge(lhs_node, rhs_node, constraint_term) {
                        // Merge failed - should not happen in normal operation
                        return TheoryCheckResult::Sat;
                    }

                    // Check for immediate conflicts
                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        // Convert term IDs to literals for conflict clause
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }

                    // For arithmetic equalities, also send to ArithSolver
                    // Use pre-parsed constraint if available
                    if let Some(parsed) = self.var_to_parsed_arith.get(&var) {
                        let terms: Vec<(TermId, Rational64)> =
                            parsed.terms.iter().copied().collect();
                        let constant = parsed.constant;
                        let reason = parsed.reason_term;

                        // For equality, use assert_eq which has GCD-based infeasibility detection
                        // This is critical for LIA: e.g., 2x + 2y = 7 is unsatisfiable because
                        // gcd(2,2) = 2 doesn't divide 7
                        self.arith.assert_eq(&terms, constant, reason);

                        // Check ArithSolver for conflicts
                        use oxiz_theories::Theory;
                        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                        if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) = self.arith.check()
                        {
                            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                            return TheoryCheckResult::Conflict(conflict_lits);
                        }
                    }

                    // For bitvector equalities, also send to BvSolver
                    // Handle variables, constants, and BV operations
                    // Check if terms have BV sort (not just if they're in bv_terms)
                    let lhs_is_bv = manager
                        .get(lhs)
                        .and_then(|t| manager.sorts.get(t.sort))
                        .is_some_and(|s| s.is_bitvec());
                    let rhs_is_bv = manager
                        .get(rhs)
                        .and_then(|t| manager.sorts.get(t.sort))
                        .is_some_and(|s| s.is_bitvec());

                    if lhs_is_bv || rhs_is_bv {
                        let mut did_assert = false;

                        // Helper to extract BV constant info
                        let get_bv_const = |term_id: TermId| -> Option<(u64, u32)> {
                            manager.get(term_id).and_then(|t| match &t.kind {
                                TermKind::BitVecConst { value, width } => {
                                    let val_u64 = value.iter_u64_digits().next().unwrap_or(0);
                                    Some((val_u64, *width))
                                }
                                _ => None,
                            })
                        };

                        // Helper to get BV width from term's sort
                        let get_bv_width = |term_id: TermId| -> Option<u32> {
                            manager.get(term_id).and_then(|t| {
                                manager.sorts.get(t.sort).and_then(|s| s.bitvec_width())
                            })
                        };

                        // Helper to check if term is a simple variable
                        let is_var = |term_id: TermId| -> bool {
                            manager
                                .get(term_id)
                                .is_some_and(|t| matches!(t.kind, TermKind::Var(_)))
                        };

                        // Helper to encode a BV operation and return the result term
                        // This ensures operands have BV variables created
                        let encode_bv_op =
                            |bv: &mut BvSolver, op_term: TermId, mgr: &TermManager| -> bool {
                                let term = match mgr.get(op_term) {
                                    Some(t) => t,
                                    None => return false,
                                };
                                let width = mgr.sorts.get(term.sort).and_then(|s| s.bitvec_width());
                                let width = match width {
                                    Some(w) => w,
                                    None => return false,
                                };

                                match &term.kind {
                                    TermKind::BvAdd(a, b) => {
                                        // Ensure operands have BV variables
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_add(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvMul(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_mul(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvSub(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_sub(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvAnd(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_and(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvOr(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_or(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvXor(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_xor(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvNot(a) => {
                                        bv.new_bv(*a, width);
                                        bv.bv_not(op_term, *a);
                                        true
                                    }
                                    TermKind::BvUdiv(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_udiv(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvSdiv(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_sdiv(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvUrem(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_urem(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::BvSrem(a, b) => {
                                        bv.new_bv(*a, width);
                                        bv.new_bv(*b, width);
                                        bv.bv_srem(op_term, *a, *b);
                                        true
                                    }
                                    TermKind::Var(_) => {
                                        // Simple variable - just ensure it has BV var
                                        bv.new_bv(op_term, width);
                                        true
                                    }
                                    _ => false,
                                }
                            };

                        // Check for BV operations and encode them
                        let lhs_term = manager.get(lhs);
                        let rhs_term = manager.get(rhs);

                        // Helper to check if a term is a BV operation
                        let is_bv_op = |t: &oxiz_core::ast::Term| {
                            matches!(
                                t.kind,
                                TermKind::BvAdd(_, _)
                                    | TermKind::BvMul(_, _)
                                    | TermKind::BvSub(_, _)
                                    | TermKind::BvAnd(_, _)
                                    | TermKind::BvOr(_, _)
                                    | TermKind::BvXor(_, _)
                                    | TermKind::BvNot(_)
                                    | TermKind::BvUdiv(_, _)
                                    | TermKind::BvSdiv(_, _)
                                    | TermKind::BvUrem(_, _)
                                    | TermKind::BvSrem(_, _)
                            )
                        };

                        let lhs_is_op = lhs_term.is_some_and(is_bv_op);
                        let rhs_is_op = rhs_term.is_some_and(is_bv_op);

                        let lhs_const_info = get_bv_const(lhs);
                        let rhs_const_info = get_bv_const(rhs);
                        let lhs_is_var = is_var(lhs);
                        let rhs_is_var = is_var(rhs);

                        // Track whether the current constraint involves a BV arithmetic op
                        // (division/remainder). We only run the full BV SAT check when an
                        // arithmetic op constraint is fully encoded. Running it on simple
                        // var=const constraints (before the op encoding is complete) can
                        // cause false UNSAT because intermediate states are partially encoded.
                        let mut has_arith_op_in_constraint = false;

                        // Case 1: BV operation = constant (e.g., (= (bvmul x y) #x0c))
                        if lhs_is_op {
                            if let Some(width) = get_bv_width(lhs) {
                                // Encode the LHS operation
                                let _encoded = encode_bv_op(self.bv, lhs, manager);
                                has_arith_op_in_constraint = true;

                                if let Some((val, _)) = rhs_const_info {
                                    // Assert operation result = constant
                                    self.bv.assert_const(lhs, val, width);
                                    did_assert = true;
                                } else if rhs_is_var && self.bv_terms.contains(&rhs) {
                                    // Assert operation result = variable
                                    self.bv.new_bv(rhs, width);
                                    self.bv.assert_eq(lhs, rhs);
                                    did_assert = true;
                                }
                            }
                        }
                        // Case 2: constant = BV operation
                        else if rhs_is_op {
                            if let Some(width) = get_bv_width(rhs) {
                                // Encode the RHS operation
                                encode_bv_op(self.bv, rhs, manager);
                                has_arith_op_in_constraint = true;

                                if let Some((val, _)) = lhs_const_info {
                                    // Assert operation result = constant
                                    self.bv.assert_const(rhs, val, width);
                                    did_assert = true;
                                } else if lhs_is_var && self.bv_terms.contains(&lhs) {
                                    // Assert variable = operation result
                                    self.bv.new_bv(lhs, width);
                                    self.bv.assert_eq(lhs, rhs);
                                    did_assert = true;
                                }
                            }
                        }
                        // Case 3: Simple variable = constant
                        else if lhs_is_var && self.bv_terms.contains(&lhs) {
                            if let Some((val, width)) = rhs_const_info {
                                self.bv.assert_const(lhs, val, width);
                                did_assert = true;
                            }
                        }
                        // Case 4: constant = simple variable
                        else if rhs_is_var && self.bv_terms.contains(&rhs) {
                            if let Some((val, width)) = lhs_const_info {
                                self.bv.assert_const(rhs, val, width);
                                did_assert = true;
                            }
                        }
                        // Case 5: Both simple variables
                        else if lhs_is_var
                            && rhs_is_var
                            && self.bv_terms.contains(&lhs)
                            && self.bv_terms.contains(&rhs)
                            && let Some(width) = get_bv_width(lhs)
                        {
                            self.bv.new_bv(lhs, width);
                            self.bv.new_bv(rhs, width);
                            self.bv.assert_eq(lhs, rhs);
                            did_assert = true;
                        }

                        // Only run the BV SAT check when a BV arithmetic operation
                        // (e.g. bvmul, bvudiv) was fully encoded in this constraint.
                        // Simple var=const constraints are intermediate states; running
                        // check() on them before the op encoding is complete produces
                        // false UNSAT because the solver sees partial constraints.
                        if did_assert && has_arith_op_in_constraint {
                            use oxiz_theories::Theory;
                            use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                            if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) =
                                self.bv.check()
                            {
                                let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                                return TheoryCheckResult::Conflict(conflict_lits);
                            }
                        }
                    }
                } else {
                    // Negative assignment: a != b, tell EUF about disequality.
                    // Use the constraint term as the reason (it has a SAT variable).
                    let constraint_term = self.term_for_var(var);
                    let lhs_node = self.intern_term_for_congruence(lhs, manager);
                    let rhs_node = self.intern_term_for_congruence(rhs, manager);
                    self.euf.assert_diseq(lhs_node, rhs_node, constraint_term);

                    // Check for immediate conflicts (if a = b was already derived)
                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
            Constraint::Diseq(lhs, rhs) => {
                if is_positive {
                    // Positive assignment: a != b.
                    // Use the constraint term as the reason for EUF disequality.
                    let constraint_term = self.term_for_var(var);
                    let lhs_node = self.intern_term_for_congruence(lhs, manager);
                    let rhs_node = self.intern_term_for_congruence(rhs, manager);
                    self.euf.assert_diseq(lhs_node, rhs_node, constraint_term);

                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                } else {
                    // Negative assignment: ~(a != b) means a = b.
                    // Use the constraint term as the merge reason.
                    let constraint_term = self.term_for_var(var);
                    let lhs_node = self.intern_term_for_congruence(lhs, manager);
                    let rhs_node = self.intern_term_for_congruence(rhs, manager);
                    if let Err(_e) = self.euf.merge(lhs_node, rhs_node, constraint_term) {
                        return TheoryCheckResult::Sat;
                    }

                    if let Some(conflict_terms) = self.euf.check_conflicts() {
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        return TheoryCheckResult::Conflict(conflict_lits);
                    }
                }
            }
            // Arithmetic constraints - use parsed linear expressions
            Constraint::Lt(lhs, rhs)
            | Constraint::Le(lhs, rhs)
            | Constraint::Gt(lhs, rhs)
            | Constraint::Ge(lhs, rhs) => {
                // Intern both sides into EUF with congruence support so that
                // Apply/Select terms are registered for congruence closure.
                self.intern_term_for_congruence(lhs, manager);
                self.intern_term_for_congruence(rhs, manager);

                // Check if this is a BV comparison
                let lhs_is_bv = self.bv_terms.contains(&lhs);
                let rhs_is_bv = self.bv_terms.contains(&rhs);

                // Handle BV comparisons
                if lhs_is_bv || rhs_is_bv {
                    // Get BV width
                    let width = manager
                        .get(lhs)
                        .and_then(|t| manager.sorts.get(t.sort).and_then(|s| s.bitvec_width()));

                    if let Some(width) = width {
                        // Ensure both operands have BV variables
                        self.bv.new_bv(lhs, width);
                        self.bv.new_bv(rhs, width);

                        // Derive signedness from the original TermKind stored for
                        // the SAT variable.  Both BvSlt and BvUlt encode to
                        // Constraint::Lt(lhs, rhs) during formula encoding (encode.rs),
                        // so the distinction is only recoverable by inspecting the term
                        // that the SAT variable was created for.
                        let constraint_term_id = self.term_for_var(var);
                        let is_signed = manager.get(constraint_term_id).is_some_and(|t| {
                            matches!(t.kind, TermKind::BvSlt(_, _) | TermKind::BvSle(_, _))
                        });

                        if is_positive {
                            // Positive assignment: constraint holds
                            match constraint {
                                Constraint::Lt(a, b) => {
                                    if is_signed {
                                        self.bv.assert_slt(a, b);
                                    } else {
                                        self.bv.assert_ult(a, b);
                                    }
                                }
                                Constraint::Le(a, b) if is_signed => {
                                    self.bv.assert_sle(a, b);
                                }
                                Constraint::Le(..) => {
                                    // a <= b is equivalent to NOT(b < a) in BV
                                    // For now, skip or encode differently
                                    // We'll focus on strict comparisons first
                                }
                                _ => {}
                            }
                        }

                        // Check BV solver for conflicts
                        use oxiz_theories::Theory;
                        use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                        if let Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) = self.bv.check() {
                            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                            return TheoryCheckResult::Conflict(conflict_lits);
                        }
                    }
                }

                // Look up the pre-parsed linear constraint for arithmetic
                if let Some(parsed) = self.var_to_parsed_arith.get(&var) {
                    // Add constraint to ArithSolver
                    let terms: Vec<(TermId, Rational64)> = parsed.terms.iter().copied().collect();
                    let reason = parsed.reason_term;
                    let constant = parsed.constant;

                    if is_positive {
                        // Positive assignment: constraint holds
                        match parsed.constraint_type {
                            ArithConstraintType::Lt => {
                                // lhs - rhs < 0, i.e., sum of terms < constant
                                self.arith.assert_lt(&terms, constant, reason);
                            }
                            ArithConstraintType::Le => {
                                // lhs - rhs <= 0
                                self.arith.assert_le(&terms, constant, reason);
                            }
                            ArithConstraintType::Gt => {
                                // lhs - rhs > 0, i.e., sum of terms > constant
                                self.arith.assert_gt(&terms, constant, reason);
                            }
                            ArithConstraintType::Ge => {
                                // lhs - rhs >= 0
                                self.arith.assert_ge(&terms, constant, reason);
                            }
                        }
                    } else {
                        // Negative assignment: negation of constraint holds
                        // ~(a < b) => a >= b
                        // ~(a <= b) => a > b
                        // ~(a > b) => a <= b
                        // ~(a >= b) => a < b
                        match parsed.constraint_type {
                            ArithConstraintType::Lt => {
                                // ~(lhs < rhs) => lhs >= rhs
                                self.arith.assert_ge(&terms, constant, reason);
                            }
                            ArithConstraintType::Le => {
                                // ~(lhs <= rhs) => lhs > rhs
                                self.arith.assert_gt(&terms, constant, reason);
                            }
                            ArithConstraintType::Gt => {
                                // ~(lhs > rhs) => lhs <= rhs
                                self.arith.assert_le(&terms, constant, reason);
                            }
                            ArithConstraintType::Ge => {
                                // ~(lhs >= rhs) => lhs < rhs
                                self.arith.assert_lt(&terms, constant, reason);
                            }
                        }
                    }

                    // Check ArithSolver for conflicts
                    use oxiz_theories::Theory;
                    use oxiz_theories::TheoryCheckResult as TheoryCheckResultEnum;
                    let arith_result = self.arith.check();
                    match arith_result {
                        Ok(TheoryCheckResultEnum::Unsat(conflict_terms)) => {
                            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                            return TheoryCheckResult::Conflict(conflict_lits);
                        }
                        Ok(TheoryCheckResultEnum::Sat) => {}
                        other => {
                            let _ = other;
                        }
                    }
                }
            }
            Constraint::BoolApp(app_term) => {
                // Bool-valued function application (e.g., `t(m)`).
                // Intern the application in EUF so that congruence closure
                // can fire.  Then merge its EUF node with the canonical
                // true or false node depending on the SAT assignment.
                let app_node = self.intern_term_for_congruence(app_term, manager);
                let (true_node, false_node) = self.ensure_bool_nodes();
                let merge_target = if is_positive { true_node } else { false_node };
                let constraint_term = self.term_for_var(var);
                if let Err(_e) = self.euf.merge(app_node, merge_target, constraint_term) {
                    // Merge error (should not happen in normal operation)
                    return TheoryCheckResult::Sat;
                }

                // Check for immediate conflicts
                if let Some(conflict_terms) = self.euf.check_conflicts() {
                    let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                    return TheoryCheckResult::Conflict(conflict_lits);
                }
            }
        }
        TheoryCheckResult::Sat
    }
}

impl TheoryCallback for TheoryManager<'_> {
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
        let var = lit.var();
        let is_positive = !lit.is_neg();

        // Track propagation
        self.statistics.propagations += 1;

        // In lazy mode, just collect assignments for batch processing
        if self.theory_mode == TheoryMode::Lazy {
            // Check if this variable has a theory constraint
            if self.var_to_constraint.contains_key(&var) {
                self.pending_assignments.push((lit, is_positive));
            }
            return TheoryCheckResult::Sat;
        }

        // Eager mode: process immediately
        // Check if this variable has a theory constraint
        let Some(constraint) = self.var_to_constraint.get(&var).cloned() else {
            return TheoryCheckResult::Sat;
        };

        self.processed_count += 1;
        self.statistics.theory_propagations += 1;

        let result = self.process_constraint(var, constraint, is_positive, self.manager);

        // Track theory conflicts
        if matches!(result, TheoryCheckResult::Conflict(_)) {
            self.statistics.theory_conflicts += 1;
            self.statistics.conflicts += 1;

            // Check conflict limit
            if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                return TheoryCheckResult::Sat; // Return Sat to signal resource exhaustion
            }
        }

        result
    }

    fn final_check(&mut self) -> TheoryCheckResult {
        // In lazy mode, process all pending assignments now
        if self.theory_mode == TheoryMode::Lazy {
            for &(lit, is_positive) in &self.pending_assignments.clone() {
                let var = lit.var();
                let Some(constraint) = self.var_to_constraint.get(&var).cloned() else {
                    continue;
                };

                self.statistics.theory_propagations += 1;

                // Process the constraint (same logic as eager mode)
                let result = self.process_constraint(var, constraint, is_positive, self.manager);
                if let TheoryCheckResult::Conflict(conflict) = result {
                    self.statistics.theory_conflicts += 1;
                    self.statistics.conflicts += 1;

                    // Check conflict limit
                    if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                        return TheoryCheckResult::Sat; // Signal resource exhaustion
                    }

                    return TheoryCheckResult::Conflict(conflict);
                }
            }
            // Clear pending assignments after processing
            self.pending_assignments.clear();
        }

        // Check EUF for conflicts
        if let Some(conflict_terms) = self.euf.check_conflicts() {
            // Convert TermIds to Lits for the conflict clause
            let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
            self.statistics.theory_conflicts += 1;
            self.statistics.conflicts += 1;

            // Check conflict limit
            if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts {
                return TheoryCheckResult::Sat; // Signal resource exhaustion
            }

            return TheoryCheckResult::Conflict(conflict_lits);
        }

        // Propagate EUF-derived equalities into the arithmetic solver.
        // When EUF fires congruence closure and derives f(x) = f(y) because
        // x = y was asserted, the arithmetic solver is unaware of this equality.
        // We must propagate it so the arithmetic solver can detect contradictions.
        let eq_result = self.propagate_euf_equalities_to_arith();
        if let TheoryCheckResult::Conflict(_) = eq_result {
            self.statistics.theory_conflicts += 1;
            self.statistics.conflicts += 1;
            return eq_result;
        }

        // Check arithmetic
        match self.arith.check() {
            Ok(result) => {
                match result {
                    oxiz_theories::TheoryCheckResult::Sat => {
                        // Arithmetic is consistent, now check model-based theory combination
                        // This ensures that different theories agree on shared terms
                        self.model_based_combination()
                    }
                    oxiz_theories::TheoryCheckResult::Unsat(conflict_terms) => {
                        // Arithmetic conflict detected - convert to SAT conflict clause
                        let conflict_lits = self.terms_to_conflict_clause(&conflict_terms);
                        self.statistics.theory_conflicts += 1;
                        self.statistics.conflicts += 1;

                        // Check conflict limit
                        if self.max_conflicts > 0 && self.statistics.conflicts >= self.max_conflicts
                        {
                            return TheoryCheckResult::Sat; // Signal resource exhaustion
                        }

                        TheoryCheckResult::Conflict(conflict_lits)
                    }
                    oxiz_theories::TheoryCheckResult::Propagate(_) => {
                        // Propagations should be handled in on_assignment
                        self.model_based_combination()
                    }
                    oxiz_theories::TheoryCheckResult::Unknown => {
                        // Theory is incomplete, be conservative
                        TheoryCheckResult::Sat
                    }
                }
            }
            Err(_error) => {
                // Internal error in the arithmetic solver
                // For now, be conservative and return Sat
                TheoryCheckResult::Sat
            }
        }
    }

    fn on_new_level(&mut self, level: u32) {
        // Push theory state when a new decision level is created
        // Ensure we have enough levels in the stack
        while self.level_stack.len() < (level as usize + 1) {
            self.level_stack.push(self.processed_count);
            self.euf.push();
            self.arith.push();
            self.bv.push();
        }
    }

    fn on_backtrack(&mut self, level: u32) {
        // Pop EUF, Arith, and BV states if needed
        while self.level_stack.len() > (level as usize + 1) {
            self.level_stack.pop();
            self.euf.pop();
            self.arith.pop();
            self.bv.pop();
        }
        self.processed_count = *self.level_stack.last().unwrap_or(&0);

        // Evict stale integer-constant canonicals whose EUF nodes were removed
        // by the preceding pop().  After truncation, any node index >=
        // euf.node_count() is invalid; keeping such entries would cause an
        // out-of-bounds access in `intern_term_deep` when `merge` is called
        // against the stale canonical.  Evicting them forces re-registration
        // (and fresh disequality assertions) the next time those values appear.
        let live_nodes = self.euf.node_count();
        self.interned_int_constants
            .retain(|_val, &mut canonical| (canonical as usize) < live_nodes);

        // Evict stale Boolean canonical nodes
        if let Some(t) = self.bool_true_node {
            if (t as usize) >= live_nodes {
                self.bool_true_node = None;
            }
        }
        if let Some(f) = self.bool_false_node {
            if (f as usize) >= live_nodes {
                self.bool_false_node = None;
            }
        }

        // Clear pending assignments on backtrack (in lazy mode)
        if self.theory_mode == TheoryMode::Lazy {
            self.pending_assignments.clear();
        }
    }
}

/// Result from parallel theory checking
#[cfg(feature = "parallel-theories")]
#[derive(Debug, Clone)]
pub enum ParallelTheoryResult {
    /// All theories report SAT
    AllSat,
    /// At least one theory found a conflict
    Conflict(SmallVec<[Lit; 8]>),
}

/// Parallel theory checking support.
#[cfg(feature = "parallel-theories")]
pub struct ParallelTheoryChecker;

#[cfg(feature = "parallel-theories")]
impl ParallelTheoryChecker {
    /// Check multiple independent theory assertions in parallel.
    pub fn check_parallel(
        assertions: &[(Var, Constraint, bool)],
        _term_to_var: &FxHashMap<TermId, Var>,
    ) -> ParallelTheoryResult {
        use rayon::prelude::*;

        let mut euf_assertions = Vec::new();
        let mut arith_assertions = Vec::new();
        let bv_assertions = Vec::new();

        for (var, constraint, is_positive) in assertions {
            match constraint {
                Constraint::Eq(_, _) | Constraint::Diseq(_, _) => {
                    euf_assertions.push((*var, constraint.clone(), *is_positive));
                }
                Constraint::Le(_, _)
                | Constraint::Lt(_, _)
                | Constraint::Ge(_, _)
                | Constraint::Gt(_, _) => {
                    arith_assertions.push((*var, constraint.clone(), *is_positive));
                }
                Constraint::BoolApp(_) => {
                    euf_assertions.push((*var, constraint.clone(), *is_positive));
                }
            }
        }

        let results: Vec<Option<SmallVec<[Lit; 8]>>> =
            [&euf_assertions, &arith_assertions, &bv_assertions]
                .par_iter()
                .map(|domain| Self::check_domain_contradictions(domain))
                .collect();

        if let Some(conflict) = results.into_iter().flatten().next() {
            return ParallelTheoryResult::Conflict(conflict);
        }

        ParallelTheoryResult::AllSat
    }

    fn check_domain_contradictions(
        assertions: &[(Var, Constraint, bool)],
    ) -> Option<SmallVec<[Lit; 8]>> {
        for i in 0..assertions.len() {
            for j in (i + 1)..assertions.len() {
                let (var_i, constraint_i, pos_i) = &assertions[i];
                let (var_j, constraint_j, pos_j) = &assertions[j];
                if Self::are_contradictory(constraint_i, *pos_i, constraint_j, *pos_j) {
                    let mut conflict = SmallVec::new();
                    conflict.push(Lit::neg(*var_i));
                    conflict.push(Lit::neg(*var_j));
                    return Some(conflict);
                }
            }
        }
        None
    }

    fn are_contradictory(c1: &Constraint, pos1: bool, c2: &Constraint, pos2: bool) -> bool {
        match (c1, c2) {
            (Constraint::Eq(a1, b1), Constraint::Eq(a2, b2)) => {
                a1 == a2 && b1 == b2 && pos1 != pos2
            }
            (Constraint::Eq(a1, b1), Constraint::Diseq(a2, b2))
            | (Constraint::Diseq(a2, b2), Constraint::Eq(a1, b1)) => {
                a1 == a2 && b1 == b2 && pos1 && pos2
            }
            _ => false,
        }
    }
}
