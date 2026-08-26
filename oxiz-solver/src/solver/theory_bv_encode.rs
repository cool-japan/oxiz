//! Standalone bit-vector term bit-blasting helpers for the theory manager.
//!
//! These are pure free functions (no `TheoryManager` state) extracted from
//! `theory_manager.rs` to keep that file under the workspace 2000-line limit.
//! They recursively encode BV-sorted terms into a [`BvSolver`]'s bit-blasted
//! circuits so the embedded SAT solver can reason about them.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_theories::bv::BvSolver;

#[cfg(debug_assertions)]
use crate::prelude::FxHashMap;
use crate::prelude::FxHashSet;

/// Post-order, memoised BV term encoding.
///
/// Bit-blast every BV-sorted operand reachable through a boolean condition
/// `cond` (the kind that appears as an `ite` selector). Walks the boolean
/// connective/comparison structure and bit-blasts the BV terms underneath the
/// `Eq`/comparison leaves, so that `BvSolver::encode_bool_node` can look them
/// up. Returns `false` if any BV operand fails to encode.
///
/// Iterative (explicit work stack), so the boolean nesting of the condition is
/// bounded by memory rather than by the native call stack.  The walk is a pure
/// conjunction over the condition's boolean structure, and the traversal order
/// and the short-circuit are preserved exactly: children are pushed in reverse
/// so they are processed left to right, and the first failing sub-term returns
/// `false` immediately without blasting anything further.  `done` skips a
/// boolean sub-term that already succeeded, so a shared sub-condition of the
/// hash-consed DAG is blasted once instead of once per path.
fn bit_blast_cond_operands(bv: &mut BvSolver, cond: TermId, mgr: &TermManager) -> bool {
    let mut done: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = vec![cond];
    while let Some(cond) = stack.pop() {
        if !done.insert(cond) {
            continue;
        }
        let Some(term) = mgr.get(cond) else {
            return false;
        };
        match &term.kind {
            // Boolean structure: descend into the operands.
            TermKind::Not(inner) => stack.push(*inner),
            TermKind::And(args) | TermKind::Or(args) => {
                stack.extend(args.iter().rev().copied());
            }
            // Comparison/equality leaves: their operands are BV terms.
            TermKind::Eq(lhs, rhs)
            | TermKind::BvUlt(lhs, rhs)
            | TermKind::BvUle(lhs, rhs)
            | TermKind::BvSlt(lhs, rhs)
            | TermKind::BvSle(lhs, rhs) => {
                let mut encoded: FxHashSet<TermId> = FxHashSet::default();
                let lhs_ok = encode_bv_term_recursive(bv, *lhs, mgr, &mut encoded) || {
                    if let Some(w) = mgr
                        .get(*lhs)
                        .and_then(|t| mgr.sorts.get(t.sort))
                        .and_then(|s| s.bitvec_width())
                    {
                        bv.new_bv(*lhs, w);
                        true
                    } else {
                        false
                    }
                };
                let rhs_ok = encode_bv_term_recursive(bv, *rhs, mgr, &mut encoded) || {
                    if let Some(w) = mgr
                        .get(*rhs)
                        .and_then(|t| mgr.sorts.get(t.sort))
                        .and_then(|s| s.bitvec_width())
                    {
                        bv.new_bv(*rhs, w);
                        true
                    } else {
                        false
                    }
                };
                if !(lhs_ok && rhs_ok) {
                    return false;
                }
            }
            // A bare boolean variable / constant has no BV operands to blast.
            TermKind::Var(_) | TermKind::True | TermKind::False => {}
            // Anything else is outside the supported condition fragment.
            _ => return false,
        }
    }
    true
}

/// Declared bit-width of `tid`, or `None` when it is not bit-vector sorted.
fn bv_width(mgr: &TermManager, tid: TermId) -> Option<u32> {
    mgr.get(tid)
        .and_then(|t| mgr.sorts.get(t.sort))
        .and_then(|s| s.bitvec_width())
}

/// Whether every operand in `operands` is bit-vector sorted at exactly `width`.
///
/// The term builder does not currently reject `(bvadd x8 y16)` — it interns the
/// application at the *left* operand's sort — so a mixed-width application can
/// reach the bit-blaster.  There is no circuit for it: wiring the bits that
/// happen to line up would encode a different operation than the term denotes,
/// and the widths must therefore be checked here, before a single gate is
/// emitted.  A `false` makes the whole term unencodable for this solver.
fn operands_have_width(mgr: &TermManager, operands: &[TermId], width: u32) -> bool {
    operands
        .iter()
        .all(|&operand| bv_width(mgr, operand) == Some(width))
}

/// If `tid` is a `BitVecConst` whose value is a positive power of two, return
/// the exponent (shift amount).  Returns `None` for zero, non-powers-of-two,
/// and non-constant terms.
fn bitvec_const_pow2_shift(mgr: &TermManager, tid: TermId) -> Option<u32> {
    let term = mgr.get(tid)?;
    if let TermKind::BitVecConst { value, .. } = &term.kind {
        let digits: Vec<u64> = value.iter_u64_digits().collect();
        let set_bits: u32 = digits.iter().map(|&d| d.count_ones()).sum();
        if set_bits != 1 {
            return None;
        }
        for (chunk, &d) in digits.iter().enumerate() {
            if d != 0 {
                return Some(chunk as u32 * 64 + d.trailing_zeros());
            }
        }
    }
    None
}

/// Recursively encodes every sub-term of `root` into the BV solver using an
/// explicit work-stack so that arbitrarily deep nesting is handled without
/// overflowing the call stack.  A `FxHashSet<TermId>` memo prevents duplicate
/// encoding when the same sub-term appears in multiple branches of the DAG.
///
/// Returns `true` when `root` was fully encoded, `false` when an unrecognised
/// TermKind is encountered.
pub(super) fn encode_bv_term_recursive(
    bv: &mut BvSolver,
    root: TermId,
    mgr: &TermManager,
    encoded: &mut FxHashSet<TermId>,
) -> bool {
    // Work-stack entry: (term_id, children_pushed)
    // We push a term twice: first time to push children, second time to
    // encode the term itself (post-order).
    let mut stack: Vec<(TermId, bool)> = vec![(root, false)];

    while let Some((tid, children_done)) = stack.pop() {
        if encoded.contains(&tid) {
            continue;
        }
        // If the BV solver already has a circuit for this term (encoded in a
        // previous call to encode_bv_term_recursive), skip both the child-push
        // and the encoding phases.  This makes the function globally idempotent:
        // calling it a second time for the same sub-tree is a no-op, which
        // prevents the adder/multiplier circuits from being duplicated across
        // CDCL restarts (each duplicate brings ~33 fresh carry SAT variables and
        // hundreds of new clauses, causing the embedded BV SAT to blow up).
        // Leaves (Var, BitVecConst) are already idempotent via `new_bv`'s
        // `or_insert_with`, so this guard is only strictly necessary for
        // compound operations, but checking unconditionally is correct and safe.
        if bv.get_bv(tid).is_some() {
            encoded.insert(tid);
            continue;
        }

        let term = match mgr.get(tid) {
            Some(t) => t,
            None => return false,
        };

        let width = match mgr.sorts.get(term.sort).and_then(|s| s.bitvec_width()) {
            Some(w) => w,
            None => return false,
        };

        if !children_done {
            // Re-push this node as "children done" so we encode it after children
            stack.push((tid, true));

            // Push children (they will be encoded first)
            match &term.kind {
                TermKind::BvAdd(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b) => {
                    if !encoded.contains(a) {
                        stack.push((*a, false));
                    }
                    if !encoded.contains(b) {
                        stack.push((*b, false));
                    }
                }
                TermKind::BvNot(a) => {
                    if !encoded.contains(a) {
                        stack.push((*a, false));
                    }
                }
                // Shifts: value and shift-amount operands (same width).
                TermKind::BvShl(a, b) | TermKind::BvLshr(a, b) | TermKind::BvAshr(a, b) => {
                    if !encoded.contains(a) {
                        stack.push((*a, false));
                    }
                    if !encoded.contains(b) {
                        stack.push((*b, false));
                    }
                }
                // Concatenation: both operands (their own widths).
                TermKind::BvConcat(a, b) => {
                    if !encoded.contains(a) {
                        stack.push((*a, false));
                    }
                    if !encoded.contains(b) {
                        stack.push((*b, false));
                    }
                }
                // Extraction: single source operand (its own width).
                TermKind::BvExtract { arg, .. } => {
                    if !encoded.contains(arg) {
                        stack.push((*arg, false));
                    }
                }
                // ITE over BV: bit-blast both branches; the condition's BV
                // operands are bit-blasted separately just before encoding.
                TermKind::Ite(_cond, then_t, else_t) => {
                    if !encoded.contains(then_t) {
                        stack.push((*then_t, false));
                    }
                    if !encoded.contains(else_t) {
                        stack.push((*else_t, false));
                    }
                }
                // Leaves: Var, BitVecConst — no children to push
                TermKind::Var(_) | TermKind::BitVecConst { .. } => {}
                // Unknown term kind — cannot encode, abort
                _ => return false,
            }
        } else {
            // Encode this node (children already encoded).
            //
            // Every same-width operation first checks that its operands really
            // are `width` bits wide and then propagates the encoder's own
            // verdict: a `false` from the BV solver means no circuit was built,
            // and continuing would leave `tid` as free bits that satisfy
            // anything — a false `sat`.  Reporting the term as unencodable is
            // the honest alternative.
            match &term.kind {
                TermKind::BvAdd(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_add(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvMul(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    let encoded_ok = if let Some(shift) = bitvec_const_pow2_shift(mgr, *b) {
                        bv.new_bv(*a, width);
                        bv.bv_shl_const(tid, *a, shift, width)
                    } else if let Some(shift) = bitvec_const_pow2_shift(mgr, *a) {
                        bv.new_bv(*b, width);
                        bv.bv_shl_const(tid, *b, shift, width)
                    } else {
                        bv.new_bv(*a, width);
                        bv.new_bv(*b, width);
                        bv.bv_mul(tid, *a, *b)
                    };
                    if !encoded_ok {
                        return false;
                    }
                }
                TermKind::BvSub(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_sub(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvAnd(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_and(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvOr(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_or(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvXor(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_xor(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvNot(a) => {
                    if !operands_have_width(mgr, &[*a], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    if !bv.bv_not(tid, *a) {
                        return false;
                    }
                }
                TermKind::BvShl(a, b) => {
                    // Operands and result share `width`.
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_shl(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvLshr(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_lshr(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvAshr(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_ashr(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvConcat(a, b) => {
                    // Operands keep their own (possibly differing) widths; the
                    // result width is their sum (already `width` here).
                    let (Some(aw), Some(bw)) = (bv_width(mgr, *a), bv_width(mgr, *b)) else {
                        return false;
                    };
                    // The result of a concatenation is exactly as wide as its
                    // two operands together; anything else is a malformed term.
                    if aw.checked_add(bw) != Some(width) {
                        return false;
                    }
                    bv.new_bv(*a, aw);
                    bv.new_bv(*b, bw);
                    // BvConcat(high, low) — `a` is the high (most-significant) part.
                    if !bv.concat(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvExtract { high, low, arg } => {
                    let Some(arg_w) = bv_width(mgr, *arg) else {
                        return false;
                    };
                    bv.new_bv(*arg, arg_w);
                    if !bv.extract(tid, *arg, *high, *low) {
                        return false;
                    }
                }
                TermKind::Ite(cond, then_t, else_t) => {
                    // Branches are already bit-blasted (pushed as children). The
                    // condition's BV operands must be bit-blasted before the
                    // condition itself is encoded inside `bv_ite`.
                    if !operands_have_width(mgr, &[*then_t, *else_t], width) {
                        return false;
                    }
                    if !bit_blast_cond_operands(bv, *cond, mgr) {
                        return false;
                    }
                    bv.bv_ite(tid, *cond, *then_t, *else_t, mgr);
                }
                TermKind::BvUdiv(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_udiv(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvSdiv(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_sdiv(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvUrem(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_urem(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::BvSrem(a, b) => {
                    if !operands_have_width(mgr, &[*a, *b], width) {
                        return false;
                    }
                    bv.new_bv(*a, width);
                    bv.new_bv(*b, width);
                    if !bv.bv_srem(tid, *a, *b) {
                        return false;
                    }
                }
                TermKind::Var(_) => {
                    // Leaf variable: just ensure a BV variable exists.
                    bv.new_bv(tid, width);
                }
                TermKind::BitVecConst { value, .. } => {
                    // Leaf constant: create the BV variable AND pin its bits to
                    // the concrete value.  Without this the constant operand of a
                    // bit-blasted op (e.g. the `#x02` in `(bvmul #x02 x)`) would be
                    // an unconstrained free variable, silently weakening the
                    // encoding and causing false SAT for constant-folded identities.
                    //
                    // The *whole* value is pinned.  Keeping only its low 64 bits
                    // (`iter_u64_digits().next()`) pinned the wrong constant for
                    // every wider bit-vector — `x = 2^64` at width 128 became
                    // `x = 0`, which made `x <u 1` satisfiable and answered
                    // `sat` to an unsatisfiable query.
                    let Some(magnitude) = value.to_biguint() else {
                        // A negative literal is not a bit-vector constant; pinning
                        // its magnitude would assert a different value.
                        return false;
                    };
                    if !bv.assert_const_big(tid, &magnitude, width) {
                        return false;
                    }
                }
                _ => return false,
            }
            encoded.insert(tid);
        }
    }

    true
}

/// Release build: the model-validity net compiles away entirely.
#[cfg(not(debug_assertions))]
#[inline]
pub(super) fn debug_verify_bv_circuits(_bv: &BvSolver, _root: TermId, _mgr: &TermManager) {}

/// All-ones mask for `width` bits (saturating at 64).
#[cfg(debug_assertions)]
fn width_mask(width: u32) -> u64 {
    if width >= u64::BITS {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Whether `value`, read as a two's-complement `width`-bit vector, is negative.
#[cfg(debug_assertions)]
fn bv_is_negative(value: u64, width: u32) -> bool {
    width > 0 && (value >> (width - 1)) & 1 == 1
}

/// Two's-complement negation of `value` at `width` bits.
#[cfg(debug_assertions)]
fn bv_negate(value: u64, width: u32) -> u64 {
    value.wrapping_neg() & width_mask(width)
}

/// Magnitude of `value` read as a signed `width`-bit vector.
#[cfg(debug_assertions)]
fn bv_magnitude(value: u64, width: u32) -> u64 {
    if bv_is_negative(value, width) {
        bv_negate(value, width)
    } else {
        value
    }
}

/// Reference (non-circuit) semantics of a binary BV operation, mirroring the
/// totalised SMT-LIB `FixedSizeBitVectors` definitions — including the
/// divide-by-zero conventions, which are where the bit-blasted circuits are
/// most likely to drift from the constant folder.
#[cfg(debug_assertions)]
fn eval_bv_binop(kind: &TermKind, lhs: u64, rhs: u64, width: u32) -> Option<u64> {
    let mask = width_mask(width);
    let wrap = |v: u64| v & mask;
    let shift_amount = rhs;
    let value = match kind {
        TermKind::BvAnd(_, _) => lhs & rhs,
        TermKind::BvOr(_, _) => lhs | rhs,
        TermKind::BvXor(_, _) => lhs ^ rhs,
        TermKind::BvAdd(_, _) => wrap(lhs.wrapping_add(rhs)),
        TermKind::BvSub(_, _) => wrap(lhs.wrapping_sub(rhs)),
        TermKind::BvMul(_, _) => wrap(lhs.wrapping_mul(rhs)),
        // `(bvudiv s 0)` is all ones, `(bvurem s 0)` is `s`.
        TermKind::BvUdiv(_, _) => lhs.checked_div(rhs).unwrap_or(mask),
        TermKind::BvUrem(_, _) => lhs.checked_rem(rhs).unwrap_or(lhs),
        // `(bvsdiv s 0)` is `-1` for a non-negative `s` and `1` for a negative
        // one; `(bvsrem s 0)` is `s` for both signs.
        TermKind::BvSdiv(_, _) => {
            if rhs == 0 {
                if bv_is_negative(lhs, width) {
                    wrap(1)
                } else {
                    mask
                }
            } else {
                let quotient = bv_magnitude(lhs, width) / bv_magnitude(rhs, width);
                if bv_is_negative(lhs, width) == bv_is_negative(rhs, width) {
                    wrap(quotient)
                } else {
                    bv_negate(quotient, width)
                }
            }
        }
        TermKind::BvSrem(_, _) => {
            if rhs == 0 {
                lhs
            } else {
                let rem = bv_magnitude(lhs, width) % bv_magnitude(rhs, width);
                if bv_is_negative(lhs, width) {
                    bv_negate(rem, width)
                } else {
                    wrap(rem)
                }
            }
        }
        // Shifts by at least the bit-width saturate (to zero, or to the sign
        // for the arithmetic right shift) rather than wrapping the amount.
        TermKind::BvShl(_, _) => {
            if shift_amount >= u64::from(width) {
                0
            } else {
                wrap(lhs << shift_amount)
            }
        }
        TermKind::BvLshr(_, _) => {
            if shift_amount >= u64::from(width) {
                0
            } else {
                lhs >> shift_amount
            }
        }
        TermKind::BvAshr(_, _) => {
            let negative = bv_is_negative(lhs, width);
            if shift_amount >= u64::from(width) {
                if negative { mask } else { 0 }
            } else {
                let shifted = lhs >> shift_amount;
                if negative {
                    // Re-introduce the sign bits the logical shift dropped.
                    wrap(shifted | !width_mask(width - shift_amount as u32))
                } else {
                    shifted
                }
            }
        }
        _ => return None,
    };
    Some(value)
}

/// Debug-only model-validity net for the bit-blasted BV encoding.
///
/// After the embedded SAT solver reports `Sat`, every term reachable from
/// `root` that the encoder claims to model must satisfy its own definition
/// *concretely*: the model value the circuit produced for a node has to equal
/// the reference semantics of that node's operation applied to the model values
/// of its operands.  Two independent implementations of every BV operation
/// (the bit-blasted circuit, and [`eval_bv_binop`] here) therefore have to
/// agree on every satisfying assignment the solver ever hands back.
///
/// This is exactly the check that turns a silent false `sat` into a loud
/// failure: `(bvsdiv s 0)` was bit-blasted to all-ones for *both* signs of `s`
/// while SMT-LIB (and this crate's constant folder) give `1` for a negative
/// `s`, and the disagreement only ever surfaced as a wrong verdict on formulas
/// with a symbolic dividend.
///
/// Deliberately *tolerant* where the encoding is honestly an abstraction: a
/// node whose kind is not modelled (an uninterpreted `Apply` returning a
/// bit-vector, say) or that is wider than 64 bits yields no reference value,
/// and its ancestors are skipped rather than flagged.  Compiles to nothing in
/// release builds.
#[cfg(debug_assertions)]
pub(super) fn debug_verify_bv_circuits(bv: &BvSolver, root: TermId, mgr: &TermManager) {
    // `None` marks a node with no comparable model value, which suppresses the
    // check for every ancestor that depends on it.
    let mut values: FxHashMap<TermId, Option<u64>> = FxHashMap::default();
    let mut stack: Vec<(TermId, bool)> = vec![(root, false)];

    while let Some((tid, children_done)) = stack.pop() {
        if values.contains_key(&tid) {
            continue;
        }
        let Some(term) = mgr.get(tid) else {
            values.insert(tid, None);
            continue;
        };

        if !children_done {
            stack.push((tid, true));
            match &term.kind {
                TermKind::BvAdd(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b)
                | TermKind::BvShl(a, b)
                | TermKind::BvLshr(a, b)
                | TermKind::BvAshr(a, b)
                | TermKind::BvConcat(a, b) => {
                    stack.push((*a, false));
                    stack.push((*b, false));
                }
                TermKind::BvNot(a) | TermKind::BvExtract { arg: a, .. } => {
                    stack.push((*a, false));
                }
                TermKind::Ite(_, then_t, else_t) => {
                    stack.push((*then_t, false));
                    stack.push((*else_t, false));
                }
                _ => {}
            }
            continue;
        }

        let width = mgr.sorts.get(term.sort).and_then(|s| s.bitvec_width());
        // The circuit's own answer for this node, as read back from the model.
        let modelled = bv.get_value(tid);
        let child = |t: &TermId| values.get(t).copied().flatten();

        let expected = match (&term.kind, width) {
            (TermKind::BvNot(a), Some(w)) => child(a).map(|v| !v & width_mask(w)),
            (TermKind::BvConcat(high, low), Some(w)) => {
                let low_width = mgr
                    .get(*low)
                    .and_then(|t| mgr.sorts.get(t.sort))
                    .and_then(|s| s.bitvec_width());
                match (child(high), child(low), low_width) {
                    (Some(h), Some(l), Some(lw)) if w <= u64::BITS => {
                        Some(((h << lw) | l) & width_mask(w))
                    }
                    _ => None,
                }
            }
            (TermKind::BvExtract { high, low, arg }, _) => child(arg)
                .map(|v| (v >> low) & width_mask(high.saturating_sub(*low).saturating_add(1))),
            // The `ite` selector is a boolean node inside the BV solver, so read
            // its truth value from the same model rather than re-deriving it.
            (TermKind::Ite(cond, then_t, else_t), _) => match bv.bool_value(*cond) {
                Some(true) => child(then_t),
                Some(false) => child(else_t),
                None => None,
            },
            (kind, Some(w)) => match binop_operands(kind) {
                Some((a, b)) => match (child(&a), child(&b)) {
                    (Some(x), Some(y)) => eval_bv_binop(kind, x, y, w),
                    _ => None,
                },
                None => None,
            },
            _ => None,
        };

        if let (Some(actual), Some(expected), Some(w)) = (modelled, expected, width) {
            debug_assert_eq!(
                actual,
                expected & width_mask(w),
                "bit-blasted BV circuit disagrees with the reference semantics \
                 of {:?} at width {w}: the model says {actual:#x} but the \
                 operation evaluates to {expected:#x} on its operands' model \
                 values — the circuit admits assignments the operation forbids, \
                 which surfaces as a false `sat`",
                term.kind
            );
        }
        values.insert(tid, modelled);
    }
}

/// Operand pair of a binary BV `TermKind`, or `None` for any other kind.
#[cfg(debug_assertions)]
fn binop_operands(kind: &TermKind) -> Option<(TermId, TermId)> {
    match kind {
        TermKind::BvAdd(a, b)
        | TermKind::BvSub(a, b)
        | TermKind::BvMul(a, b)
        | TermKind::BvAnd(a, b)
        | TermKind::BvOr(a, b)
        | TermKind::BvXor(a, b)
        | TermKind::BvUdiv(a, b)
        | TermKind::BvSdiv(a, b)
        | TermKind::BvUrem(a, b)
        | TermKind::BvSrem(a, b)
        | TermKind::BvShl(a, b)
        | TermKind::BvLshr(a, b)
        | TermKind::BvAshr(a, b) => Some((*a, *b)),
        _ => None,
    }
}

#[cfg(test)]
mod s8_iterative_tests {
    use super::*;
    use oxiz_core::ast::TermManager;

    /// Nesting depth that would overflow the native stack under the previous
    /// recursive walk; the assertion is that the call **returns**.
    ///
    /// This depth and [`SMALL_STACK`] were scaled down together by a factor
    /// of 8 (from 60 000 on 1 MiB): what these tests pin is the ~17 bytes of
    /// stack per level, which no native frame can fit into. `mk_and` flattens
    /// its arguments (so `acc = mk_and([acc, leaf])` never actually nests,
    /// and is quadratic to boot) and `mk_not` folds double negation (so
    /// `acc = mk_not(acc)` oscillates between depth 0 and 1 and never nests
    /// either), so both the `and`- and `not`-nesting tests below build their
    /// chains with `TermManager::intern_term` directly, which neither
    /// flattens/folds nor re-interns already-built prefixes. Never raise one
    /// of `DEEP`/`SMALL_STACK` without the other.
    const DEEP: usize = 7_500;

    /// Worker stack for the deep-nesting tests; see [`DEEP`].
    const SMALL_STACK: usize = 1 << 17;

    #[test]
    fn s8_bit_blast_cond_operands_deep_not_chain_returns() {
        let handle = std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(|| {
                let mut tm = TermManager::new();
                let bool_sort = tm.sorts.bool_sort;
                let bv8 = tm.sorts.bitvec(8);
                let x = tm.mk_var("x", bv8);
                let y = tm.mk_var("y", bv8);
                let mut cond = tm.mk_eq(x, y);
                // `mk_not` folds double negation (`not(not(p)) == p`), so
                // looping `cond = tm.mk_not(cond)` oscillates between depth 0
                // and 1 and never actually nests -- at `DEEP` even, `cond`
                // ends up identical to the un-negated `eq`. Intern the `Not`
                // nodes directly so the chain really is `DEEP` deep.
                for _ in 0..DEEP {
                    cond = tm.intern_term(TermKind::Not(cond), bool_sort);
                }
                let mut bv = BvSolver::new();
                bit_blast_cond_operands(&mut bv, cond, &tm)
            })
            .expect("spawn deep-nesting worker");
        assert_eq!(handle.join().ok(), Some(true));
    }

    /// Deeply left-nested `and`: the boolean structure is descended
    /// iteratively, so this returns instead of aborting the process.
    #[test]
    fn s8_bit_blast_cond_operands_deep_and_nesting_returns() {
        let handle = std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(|| {
                let mut tm = TermManager::new();
                let bool_sort = tm.sorts.bool_sort;
                let bv8 = tm.sorts.bitvec(8);
                let x = tm.mk_var("x", bv8);
                let y = tm.mk_var("y", bv8);
                let leaf = tm.mk_eq(x, y);
                let mut cond = leaf;
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the chain really is
                // `DEEP` deep.
                for _ in 0..DEEP {
                    cond = tm.intern_term(TermKind::And(vec![cond, leaf].into()), bool_sort);
                }
                let mut bv = BvSolver::new();
                bit_blast_cond_operands(&mut bv, cond, &tm)
            })
            .expect("spawn deep-nesting worker");
        assert_eq!(handle.join().ok(), Some(true));
    }

    /// Semantic pins: the supported fragment still returns `true`, and an
    /// unsupported sub-term still fails the whole condition.
    #[test]
    fn s8_bit_blast_cond_operands_verdicts_preserved() {
        let mut tm = TermManager::new();
        let bv8 = tm.sorts.bitvec(8);
        let bool_sort = tm.sorts.bool_sort;
        let x = tm.mk_var("x", bv8);
        let y = tm.mk_var("y", bv8);
        let eq = tm.mk_eq(x, y);
        let ult = tm.mk_bv_ult(x, y);
        let p = tm.mk_var("p", bool_sort);
        let supported = tm.mk_and(vec![eq, ult, p]);
        let mut bv = BvSolver::new();
        assert!(bit_blast_cond_operands(&mut bv, supported, &tm));

        // An arithmetic comparison is outside the supported condition
        // fragment: the walk must report failure rather than silently succeed.
        let int_sort = tm.sorts.int_sort;
        let i = tm.mk_var("i", int_sort);
        let j = tm.mk_var("j", int_sort);
        let unsupported_leaf = tm.mk_lt(i, j);
        let unsupported = tm.mk_and(vec![eq, unsupported_leaf]);
        let mut bv2 = BvSolver::new();
        assert!(!bit_blast_cond_operands(&mut bv2, unsupported, &tm));
    }
}
