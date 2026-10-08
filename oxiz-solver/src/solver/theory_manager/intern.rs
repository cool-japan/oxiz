//! Iterative EUF interning for the theory manager: turning a `TermId` into a
//! congruence-closure node, with `Apply`/`Select` structure preserved so
//! congruence can actually fire, and an explicit frame stack in place of
//! native recursion.
//!
//! Split out of the parent module so `theory_manager.rs` stays under the
//! workspace 2000-line limit -- see `conflict_clause.rs` for the identical
//! precedent (a self-contained concern lifted into its own child module,
//! `impl TheoryManager<'_>` reopened here rather than in the parent file).

use super::TheoryManager;
use num_traits::ToPrimitive;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use smallvec::SmallVec;

/// One pending application of an iterative EUF interning walk
/// ([`TheoryManager::intern_term_deep`] and
/// [`TheoryManager::intern_term_for_congruence`]).
///
/// The frame owns the application's operand list and the EUF nodes of the
/// operands already interned, so the walk never needs the native call stack
/// and never needs to re-borrow a half-finished parent.
pub(super) struct InternFrame {
    /// The application term whose operands are being interned.
    term: TermId,
    /// EUF function symbol of the application (`SELECT_FUNC_ID` for `select`).
    func_id: u32,
    /// The application's operands, in order.
    operands: SmallVec<[TermId; 4]>,
    /// Index of the next operand to descend into.
    next: usize,
    /// EUF nodes of the operands interned so far, in order.
    nodes: SmallVec<[u32; 4]>,
}

impl TheoryManager<'_> {
    /// Sentinel function ID used for array `select(array, index)` in EUF.
    ///
    /// `Spur::into_inner()` always returns a `NonZeroU32` (>= 1), so 0 is safe
    /// to use as a special, collision-free function ID for the built-in select
    /// operation.  By interning `select(a, i)` as `intern_app(term, SELECT_FUNC_ID,
    /// [a_node, i_node])`, the EUF congruence closure engine treats select like any
    /// other binary function application and will automatically derive
    /// `select(a, x) = select(a, y)` whenever `x = y` is merged.
    pub(super) const SELECT_FUNC_ID: u32 = 0;

    /// Sentinel function ID used for array `store(array, index, value)` in EUF.
    ///
    /// `store` is a function of the array theory exactly as `select` is, so
    /// interning it as a ternary application gives congruence closure the one
    /// fact it otherwise lacks: two writes that agree argument-wise are the
    /// same array.  Without it, `store(a, i, 5)` and `store(a, i, w)` stayed
    /// two unrelated opaque leaves *even with `5 = w` merged*, and so did the
    /// reads over them — which is how `#P2b-33`'s QF_AUFLIA row survived the
    /// guard fix: `Solver::purify_numeric_uf_args` rewrites a numeric literal
    /// under an uninterpreted application into a fresh proxy variable
    /// throughout that assertion, so the `5` inside the *store* of
    /// `(distinct (g (select (store arr i 5) i)) (g 5))` became a proxy `w`
    /// while the array-axiom instantiator (which walks `Solver::assertions`,
    /// the pre-purification terms) kept reasoning about the literal copy.  The
    /// two halves only meet through congruence over `store`.
    ///
    /// The value is `u32::MAX - 1` because `u32::MAX` is `ENode::NO_FUNC`, the
    /// E-graph's own "this node is a leaf" sentinel (`euf/solver.rs`), and 0 is
    /// [`Self::SELECT_FUNC_ID`].  Real function symbols are `Spur::into_inner()`
    /// values, handed out from 1 upwards in interning order, so neither
    /// sentinel can collide with one.
    pub(super) const STORE_FUNC_ID: u32 = u32::MAX - 1;

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
    ///
    /// Iterative: `Apply` arguments and `Select` operands are interned through
    /// an explicit frame stack (post-order, left to right — the recursive
    /// order), so operand nesting depth cannot overflow the native call
    /// stack.  `euf.term_to_node` remains the cross-call memo, so shared
    /// sub-terms of the hash-consed DAG are interned once.
    #[allow(dead_code)]
    pub(super) fn intern_term_deep(&mut self, term: TermId, manager: &TermManager) -> u32 {
        let mut frames: Vec<InternFrame> = Vec::new();
        let mut current = term;
        'open: loop {
            // Intern `current`, descending into application operands first.
            let mut value: u32 = loop {
                if let Some(idx) = self.euf.term_to_node(current) {
                    break idx;
                }
                match Self::intern_operands(current, manager) {
                    Some((func_id, operands)) => match operands.first().copied() {
                        Some(first) => {
                            frames.push(InternFrame {
                                term: current,
                                func_id,
                                operands,
                                next: 1,
                                nodes: SmallVec::new(),
                            });
                            current = first;
                        }
                        None => {
                            break self.euf.intern_app(
                                current,
                                func_id,
                                SmallVec::<[u32; 4]>::new(),
                            );
                        }
                    },
                    None => break self.intern_leaf_deep(current, manager),
                }
            };

            // Hand the finished operand node to the innermost application.
            loop {
                let Some(mut frame) = frames.pop() else {
                    return value;
                };
                frame.nodes.push(value);
                if let Some(&child) = frame.operands.get(frame.next) {
                    frame.next += 1;
                    frames.push(frame);
                    current = child;
                    continue 'open;
                }
                value = self.euf.intern_app(frame.term, frame.func_id, frame.nodes);
            }
        }
    }

    /// The application structure of `term` for EUF interning: `Apply` uses its
    /// function symbol, `Select(array, index)` is a binary application of the
    /// sentinel [`Self::SELECT_FUNC_ID`] so that congruence closure fires when
    /// the index (or array) arguments become equal, and `Store(array, index,
    /// value)` a ternary application of [`Self::STORE_FUNC_ID`] for the same
    /// reason one level up.  Everything else is a leaf.
    pub(super) fn intern_operands(
        term: TermId,
        manager: &TermManager,
    ) -> Option<(u32, SmallVec<[TermId; 4]>)> {
        match manager.get(term).map(|t| &t.kind) {
            Some(TermKind::Apply { func, args, .. }) => {
                Some((func.into_inner().get(), args.clone()))
            }
            Some(TermKind::Select(array, index)) => Some((
                Self::SELECT_FUNC_ID,
                SmallVec::from_slice(&[*array, *index]),
            )),
            Some(TermKind::Store(array, index, value)) => Some((
                Self::STORE_FUNC_ID,
                SmallVec::from_slice(&[*array, *index, *value]),
            )),
            _ => None,
        }
    }

    /// Intern a non-application term for [`Self::intern_term_deep`]: integer
    /// constants get a canonical node plus pairwise disequalities, everything
    /// else a plain opaque node.
    pub(super) fn intern_leaf_deep(&mut self, term: TermId, manager: &TermManager) -> u32 {
        if let Some(t) = manager.get(term) {
            if let TermKind::IntConst(n) = &t.kind {
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
                    // Both the merge and the disequalities below carry `term`
                    // as their reason and `term` names no literal; they are
                    // true in every model.  Declaring that keeps
                    // `terms_to_conflict_clause` able to distinguish "omitted
                    // because tautological" from "justification lost".
                    self.tautological_reasons.insert(term);
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
    ///
    /// Iterative: `Apply` arguments and `Select` operands are interned through
    /// an explicit [`InternFrame`] stack in post-order, left to right — exactly
    /// the order the recursive version used, which matters because
    /// `intern_app` assigns node indices in creation order.  Operand nesting
    /// depth is therefore bounded by memory rather than by the native call
    /// stack.  `euf.term_to_node` remains the memo, so shared sub-terms of the
    /// hash-consed DAG are interned once.
    pub(super) fn intern_term_for_congruence(
        &mut self,
        term: TermId,
        manager: &TermManager,
    ) -> u32 {
        let mut frames: Vec<InternFrame> = Vec::new();
        let mut current = term;
        'open: loop {
            // Intern `current`, descending into application operands first.
            let mut value: u32 = loop {
                if let Some(idx) = self.euf.term_to_node(current) {
                    break idx;
                }
                match Self::intern_operands(current, manager) {
                    Some((func_id, operands)) => match operands.first().copied() {
                        Some(first) => {
                            frames.push(InternFrame {
                                term: current,
                                func_id,
                                operands,
                                next: 1,
                                nodes: SmallVec::new(),
                            });
                            current = first;
                        }
                        None => {
                            break self.euf.intern_app(
                                current,
                                func_id,
                                SmallVec::<[u32; 4]>::new(),
                            );
                        }
                    },
                    None => break self.intern_leaf_for_congruence(current, manager),
                }
            };

            // Hand the finished operand node to the innermost application.
            loop {
                let Some(mut frame) = frames.pop() else {
                    return value;
                };
                frame.nodes.push(value);
                if let Some(&child) = frame.operands.get(frame.next) {
                    frame.next += 1;
                    frames.push(frame);
                    current = child;
                    continue 'open;
                }
                value = self.euf.intern_app(frame.term, frame.func_id, frame.nodes);
            }
        }
    }

    /// Intern a non-application term for [`Self::intern_term_for_congruence`]:
    /// bit-vector constants get a canonical node plus pairwise disequalities
    /// against the other distinct constants of the same width, everything else
    /// a plain opaque node.  Unlike [`Self::intern_leaf_deep`], integer
    /// constants get **no** disequality edges (see the caller's docs).
    pub(super) fn intern_leaf_for_congruence(
        &mut self,
        term: TermId,
        manager: &TermManager,
    ) -> u32 {
        if let Some(t) = manager.get(term) {
            if let TermKind::BitVecConst { value, width } = &t.kind {
                // Register the BV constant as an EUF node and maintain pairwise
                // disequalities between *distinct* same-width constant values.
                //
                // EUF has no built-in notion that two different bit-vector
                // literals are unequal.  Without explicit disequality edges, a
                // congruence chain that equates a node merged with `#x00` and one
                // merged with `#x01` (e.g. `g(a)=#x00`, `g(b)=#x01`, `a=b`) would
                // not produce a conflict.  We therefore assert `#x00 ≠ #x01` etc.
                //
                // As with `interned_int_constants`, we keep one canonical EUF
                // node per distinct `(value, width)` pair: when the same value
                // reappears (a fresh TermId) we merge it into the canonical node,
                // bounding the number of pairwise edges by the count of distinct
                // BV literals rather than the total number of term IDs.
                //
                // The key carries every limb of the value.  Truncating it to
                // the low 64 bits made `0` and `2^64` the *same* key at width
                // 128, so the two constants were merged into one EUF class —
                // and the merge was recorded as tautological, which is exactly
                // what it was not.  `(distinct (g a) (g b))` over those two
                // constants was then reported `unsat`.
                let key = (
                    value.iter_u64_digits().collect::<SmallVec<[u64; 2]>>(),
                    *width,
                );
                let new_node = self.euf.intern(term);
                // Every edge asserted from here carries `term` as its reason
                // and `term` names no literal: two ids for the same constant
                // really are equal and two distinct constants really are
                // unequal, in every model.  Declare that so a conflict clause
                // can omit it *knowingly*.
                self.tautological_reasons.insert(term);
                if let Some(&canonical) = self.interned_bv_constants.get(&key) {
                    let _ = self.euf.merge(new_node, canonical, term);
                    return canonical;
                }
                // First time we see this value: assert disequality against every
                // other distinct constant of the SAME width (different widths are
                // different sorts and are never merged), then register it.
                let diseq_targets: Vec<u32> = self
                    .interned_bv_constants
                    .iter()
                    .filter_map(|(&(_, w), &node)| (w == *width).then_some(node))
                    .collect();
                for other_node in diseq_targets {
                    self.euf.assert_diseq(new_node, other_node, term);
                }
                self.interned_bv_constants.insert(key, new_node);
                return new_node;
            }
        }
        self.euf.intern(term)
    }
}
