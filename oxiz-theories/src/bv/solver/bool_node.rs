//! Boolean nodes of the bit-blasted circuit: the truth variables that stand
//! for `ite` selectors and the connectives/comparisons underneath them, the
//! values the enclosing CDCL(T) search pins into them, and the record of
//! those pins that every conflict explanation must name.
//!
//! Split out of `solver.rs` to keep that file under the workspace 2000-line
//! limit; it is the same `impl BvSolver` block.

use super::BvSolver;
use oxiz_core::ast::{TermId, TermManager};
use oxiz_sat::{Lit, Var};

impl BvSolver {
    /// Fix a Bool-sorted term's truth value from the *enclosing* CDCL(T) search.
    ///
    /// A bit-blasted `ite` selector that is a bare boolean variable has no
    /// circuit of its own: [`Self::encode_bool_node`] gives it a fresh, free SAT
    /// variable inside the embedded solver. Free means the embedded search may
    /// pick the branch the outer solver has *ruled out*, so
    /// `(= (ite c #x01 #x02) x) ∧ ¬c ∧ (= x #x01)` looked satisfiable: the outer
    /// solver knows `c` is false, the BV solver did not, and each considered its
    /// own half consistent.
    ///
    /// The theory manager therefore replays every atom assignment here. The unit
    /// lands on the embedded solver's trail at the current level, which is kept
    /// in lockstep with the outer decision levels, so it is retracted on
    /// backtrack exactly like the (dis)equality and comparison assertions. The
    /// value is also remembered so a selector that is *first encoded later*
    /// still picks it up — the outer assignment and the bit-blasting can happen
    /// in either order.
    ///
    /// Returns `true` when `term` already has a boolean node, i.e. the value
    /// was pinned into a live circuit right now.  The caller must then run a
    /// `Theory::check`: the pin is an assertion the embedded solver can
    /// refute, and one that reaches it *after* the last constraint was
    /// checked would otherwise go unexamined — `(= x (ite p 1 2)) ∧ (= x 2)
    /// ∧ p`, asserted in that order, ended the search with `p` pinned and no
    /// check run, and only the model gate noticed (`unknown` for an `unsat`;
    /// `#P2b-24`).  `false` means the value was merely remembered for a node
    /// that does not exist yet; `encode_bool_node` applies it on creation,
    /// and the check that follows that encoding covers it.
    pub fn assert_bool_value(&mut self, term: TermId, value: bool) -> bool {
        let previous = self.outer_bool.insert(term, value);
        self.outer_bool_journal.push((term, previous));
        if let Some(&var) = self.bool_node.get(&term) {
            self.pin_bool_var(term, var, value);
            return true;
        }
        false
    }

    /// Add the unit clause forcing `var` — the boolean node of the outer atom
    /// `term` — to `value`, and record `term` as a conflict hypothesis for as
    /// long as that unit is live (see [`Self::pinned_terms`]).
    fn pin_bool_var(&mut self, term: TermId, var: Var, value: bool) {
        let lit = if value { Lit::pos(var) } else { Lit::neg(var) };
        self.sat.add_clause([lit]);
        if self.pinned_set.insert(term) {
            self.pinned_terms.push(term);
        }
    }

    /// The outer Boolean atoms whose values are currently pinned into the
    /// circuit, in pin order — the hypotheses every conflict explanation
    /// names besides the recorded constraint terms.  Exposed for tests.
    #[must_use]
    pub fn pinned_terms(&self) -> &[TermId] {
        &self.pinned_terms
    }

    /// Encode a Bool-sorted term into a single SAT truth variable over
    /// already bit-blasted BV operands.  Returns `None` for boolean shapes
    /// outside the supported fragment.
    ///
    /// Supported: bool `Var`, `True`/`False`, `Not`, `And`, `Or`, `Xor`,
    /// `Implies`, Bool-sorted `Ite`, `Eq` over BV operands (bit equality) and
    /// over Bool operands (`iff`), `Distinct` over either (pairwise), and the
    /// BV comparisons `BvUlt`/`BvUle`/`BvSlt`/`BvSle`.
    ///
    /// `Xor`, `Implies`, the Bool `Ite`, the Bool `Eq` and `Distinct` are the
    /// `#P2b-24` additions: cargo-formal's printer emits every one of them
    /// inside `ite` selectors, and a selector outside the fragment made the
    /// whole term unencodable — which the `oxiz-solver` encoder then replaced
    /// with a *free* bit-vector, a false `sat` that only its debug-build
    /// circuit self-check noticed.
    pub fn encode_bool_node(&mut self, term: TermId, manager: &TermManager) -> Option<Var> {
        use oxiz_core::ast::TermKind;
        if let Some(&v) = self.bool_node.get(&term) {
            // Re-apply any outer truth value: the node may have been created
            // below a decision level that has since been popped, which retracts
            // the unit clause but not the cached variable.
            if let Some(&value) = self.outer_bool.get(&term) {
                self.pin_bool_var(term, v, value);
            }
            return Some(v);
        }
        let kind = manager.get(term)?.kind.clone();
        let out = match kind {
            TermKind::Var(_) => {
                // Free boolean variable: a single fresh SAT var stands for it.
                self.sat.new_var()
            }
            TermKind::True => {
                let v = self.sat.new_var();
                self.sat.add_clause([Lit::pos(v)]);
                v
            }
            TermKind::False => {
                let v = self.sat.new_var();
                self.sat.add_clause([Lit::neg(v)]);
                v
            }
            TermKind::Not(inner) => {
                let iv = self.encode_bool_node(inner, manager)?;
                let v = self.sat.new_var();
                self.encode_not(v, iv);
                v
            }
            TermKind::Xor(lhs, rhs) => {
                let lv = self.encode_bool_node(lhs, manager)?;
                let rv = self.encode_bool_node(rhs, manager)?;
                let v = self.sat.new_var();
                self.encode_xor(v, lv, rv);
                v
            }
            TermKind::Implies(lhs, rhs) => {
                // `lhs => rhs` is `not(lhs) or rhs`.
                let lv = self.encode_bool_node(lhs, manager)?;
                let rv = self.encode_bool_node(rhs, manager)?;
                let not_lhs = self.sat.new_var();
                self.encode_not(not_lhs, lv);
                let v = self.sat.new_var();
                self.encode_or(v, not_lhs, rv);
                v
            }
            TermKind::Ite(cond, then_t, else_t) => {
                // Only a Bool-sorted `ite` is a boolean node; a BV-sorted one
                // is a term, bit-blasted by `bv_ite`.
                if !Self::is_bool_sorted(manager, term) {
                    return None;
                }
                let cv = self.encode_bool_node(cond, manager)?;
                let tv = self.encode_bool_node(then_t, manager)?;
                let ev = self.encode_bool_node(else_t, manager)?;
                let v = self.sat.new_var();
                self.encode_mux(v, cv, tv, ev);
                v
            }
            TermKind::Distinct(ref args) => {
                // Pairwise: every pair of operands differs.  A pair of Bool
                // operands differs iff their truth values do (`xor`); a pair
                // of BV operands iff their bit equality is false.
                let mut acc: Option<Var> = None;
                for i in 0..args.len() {
                    for j in (i + 1)..args.len() {
                        let differ = if Self::is_bool_sorted(manager, args[i]) {
                            let lv = self.encode_bool_node(args[i], manager)?;
                            let rv = self.encode_bool_node(args[j], manager)?;
                            let v = self.sat.new_var();
                            self.encode_xor(v, lv, rv);
                            v
                        } else {
                            let eq = self.bool_bv_eq(args[i], args[j])?;
                            let v = self.sat.new_var();
                            self.encode_not(v, eq);
                            v
                        };
                        acc = Some(match acc {
                            None => differ,
                            Some(prev) => {
                                let v = self.sat.new_var();
                                self.encode_and(v, prev, differ);
                                v
                            }
                        });
                    }
                }
                match acc {
                    Some(v) => v,
                    None => {
                        // `(distinct)` / `(distinct x)` is vacuously true.
                        let v = self.sat.new_var();
                        self.sat.add_clause([Lit::pos(v)]);
                        v
                    }
                }
            }
            TermKind::And(ref args) => {
                // Conjunction of all operands.
                let mut acc: Option<Var> = None;
                for &arg in args {
                    let av = self.encode_bool_node(arg, manager)?;
                    acc = Some(match acc {
                        None => av,
                        Some(prev) => {
                            let v = self.sat.new_var();
                            self.encode_and(v, prev, av);
                            v
                        }
                    });
                }
                match acc {
                    Some(v) => v,
                    None => {
                        // Empty conjunction is `true`.
                        let v = self.sat.new_var();
                        self.sat.add_clause([Lit::pos(v)]);
                        v
                    }
                }
            }
            TermKind::Or(ref args) => {
                let mut acc: Option<Var> = None;
                for &arg in args {
                    let av = self.encode_bool_node(arg, manager)?;
                    acc = Some(match acc {
                        None => av,
                        Some(prev) => {
                            let v = self.sat.new_var();
                            self.encode_or(v, prev, av);
                            v
                        }
                    });
                }
                match acc {
                    Some(v) => v,
                    None => {
                        // Empty disjunction is `false`.
                        let v = self.sat.new_var();
                        self.sat.add_clause([Lit::neg(v)]);
                        v
                    }
                }
            }
            TermKind::Eq(lhs, rhs) if Self::is_bool_sorted(manager, lhs) => {
                // Bool equality is `iff`: `not(lhs xor rhs)`.
                let lv = self.encode_bool_node(lhs, manager)?;
                let rv = self.encode_bool_node(rhs, manager)?;
                let xor = self.sat.new_var();
                self.encode_xor(xor, lv, rv);
                let v = self.sat.new_var();
                self.encode_not(v, xor);
                v
            }
            TermKind::Eq(lhs, rhs) => self.bool_bv_eq(lhs, rhs)?,
            TermKind::BvUlt(lhs, rhs) => self.bool_ult(lhs, rhs, manager, false)?,
            TermKind::BvUle(lhs, rhs) => self.bool_ule(lhs, rhs, manager, false)?,
            TermKind::BvSlt(lhs, rhs) => self.bool_ult(lhs, rhs, manager, true)?,
            TermKind::BvSle(lhs, rhs) => self.bool_ule(lhs, rhs, manager, true)?,
            _ => return None,
        };
        if self.bool_node.insert(term, out).is_none() {
            self.bool_node_journal.push(term);
        }
        // Honour an outer assignment recorded before this node existed.
        if let Some(&value) = self.outer_bool.get(&term) {
            self.pin_bool_var(term, out, value);
        }
        Some(out)
    }

    /// Assert the clause `(∨_i differ_i.0 ≠ differ_i.1) ∨ (∨_j equal_j.0 =
    /// equal_j.1)` over already bit-blasted, pairwise equal-width operands.
    ///
    /// This is the shape of the lemmas the bit-vector / EUF equality
    /// exchange derives (`#P2b-29`): congruence closure refutes a
    /// *partition* of the circuit's model — "these argument pairs equal and
    /// those leaves apart is impossible" — and hands the circuit the
    /// disjunction that fact entails, so the next model must move.  Every
    /// disjunct is a fresh truth variable defined by the same XOR/AND
    /// circuits [`Self::encode_bool_node`] uses for a bit equality.
    ///
    /// Returns `false` — asserting nothing — when the clause would be empty
    /// or any operand is missing or ill-matched in width.
    #[must_use]
    pub fn assert_any(&mut self, differ: &[(TermId, TermId)], equal: &[(TermId, TermId)]) -> bool {
        let mut lits: Vec<Lit> = Vec::with_capacity(differ.len() + equal.len());
        for &(lhs, rhs) in differ {
            let Some(eq) = self.bool_bv_eq(lhs, rhs) else {
                return false;
            };
            lits.push(Lit::neg(eq));
        }
        for &(lhs, rhs) in equal {
            let Some(eq) = self.bool_bv_eq(lhs, rhs) else {
                return false;
            };
            lits.push(Lit::pos(eq));
        }
        if lits.is_empty() {
            return false;
        }
        self.sat.add_clause(lits);
        true
    }

    /// Whether `term` is Bool-sorted in `manager`.
    fn is_bool_sorted(manager: &TermManager, term: TermId) -> bool {
        manager
            .get(term)
            .is_some_and(|t| t.sort == manager.sorts.bool_sort)
    }

    /// The truth variable of the bit equality `lhs = rhs` over two
    /// pre-bit-blasted, equal-width operands: `out <=> AND_i (lhs[i] <=>
    /// rhs[i])`.  `None` when either is missing or the widths differ.
    ///
    /// Memoised per unordered pair in `eq_cache` (journalled like
    /// `ult_cache`, so the entry lives exactly as long as its clauses): the
    /// bit-vector / EUF exchange asks for the same pairs round after round,
    /// and re-encoding them made every round's instance — and search —
    /// bigger than the last (see the field's documentation).
    fn bool_bv_eq(&mut self, lhs: TermId, rhs: TermId) -> Option<Var> {
        let key = if lhs.raw() <= rhs.raw() {
            super::ComparisonKey { a: lhs, b: rhs }
        } else {
            super::ComparisonKey { a: rhs, b: lhs }
        };
        if let Some(&cached) = self.eq_cache.get(&key) {
            return Some(cached);
        }
        let (va, vb) = match (
            self.term_to_bv.get(&lhs).cloned(),
            self.term_to_bv.get(&rhs).cloned(),
        ) {
            (Some(va), Some(vb)) if va.width == vb.width => (va, vb),
            _ => return None,
        };
        let mut acc: Option<Var> = None;
        for i in 0..va.width as usize {
            // bit_eq <=> (a[i] <=> b[i])
            let bit_eq = self.sat.new_var();
            let xor = self.sat.new_var();
            self.encode_xor(xor, va.bits[i], vb.bits[i]);
            self.encode_not(bit_eq, xor);
            acc = Some(match acc {
                None => bit_eq,
                Some(prev) => {
                    let v = self.sat.new_var();
                    self.encode_and(v, prev, bit_eq);
                    v
                }
            });
        }
        let out = match acc {
            Some(v) => v,
            None => {
                // Two zero-width vectors are trivially equal.
                let v = self.sat.new_var();
                self.sat.add_clause([Lit::pos(v)]);
                v
            }
        };
        self.eq_cache.insert(key.clone(), out);
        self.eq_cache_journal.push(key);
        Some(out)
    }

    /// Encode a strict less-than (signed or unsigned) comparison result var.
    /// Operands are assumed already bit-blasted by the caller.
    fn bool_ult(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        _manager: &TermManager,
        signed: bool,
    ) -> Option<Var> {
        let (va, vb) = match (
            self.term_to_bv.get(&lhs).cloned(),
            self.term_to_bv.get(&rhs).cloned(),
        ) {
            (Some(va), Some(vb)) if va.width == vb.width => (va, vb),
            _ => return None,
        };
        let width = va.width as usize;
        let result = self.sat.new_var();
        if signed {
            // Signed: if sign bits differ, lhs<rhs iff sign_lhs=1; else unsigned.
            let sign_a = va.bits[width - 1];
            let sign_b = vb.bits[width - 1];
            let diff_sign = self.sat.new_var();
            self.encode_xor(diff_sign, sign_a, sign_b);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::neg(sign_a), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::neg(diff_sign), Lit::pos(sign_a), Lit::neg(result)]);
            let ult = self.sat.new_var();
            self.encode_ult_result(&va.bits, &vb.bits, ult);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::neg(ult), Lit::pos(result)]);
            self.sat
                .add_clause([Lit::pos(diff_sign), Lit::pos(ult), Lit::neg(result)]);
        } else {
            self.encode_ult_result(&va.bits, &vb.bits, result);
        }
        Some(result)
    }

    /// Encode a less-than-or-equal (signed or unsigned) comparison result var
    /// as `not(rhs < lhs)`.
    fn bool_ule(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
        signed: bool,
    ) -> Option<Var> {
        // a <= b  ≡  not(b < a).
        let gt = self.bool_ult(rhs, lhs, manager, signed)?;
        let v = self.sat.new_var();
        self.encode_not(v, gt);
        Some(v)
    }
}
