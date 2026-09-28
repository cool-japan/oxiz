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
    /// The theory manager therefore replays every atom assignment here.  The
    /// value is recorded in `outer_bool` (journalled, so the matching `pop`
    /// retracts it) and every check assumes it on the term's boolean node —
    /// whether the node existed when the value arrived or was encoded later,
    /// and however many `pop`s the node itself has survived (see `scope.rs`:
    /// the node is a definition, the pin an assertion).
    ///
    /// Returns `true` when `term` already has a boolean node, i.e. the value
    /// is pinned onto a live circuit right now.  The caller must then run a
    /// `Theory::check`: the pin is an assertion the embedded solver can
    /// refute, and one that reaches it *after* the last constraint was
    /// checked would otherwise go unexamined — `(= x (ite p 1 2)) ∧ (= x 2)
    /// ∧ p`, asserted in that order, ended the search with `p` pinned and no
    /// check run, and only the model gate noticed (`unknown` for an `unsat`;
    /// `#P2b-24`).  `false` means the value was merely remembered for a node
    /// that does not exist yet; the check that follows that node's encoding
    /// covers it.
    pub fn assert_bool_value(&mut self, term: TermId, value: bool) -> bool {
        let previous = self.outer_bool.insert(term, value);
        self.outer_bool_journal.push((term, previous));
        self.bool_node.contains_key(&term)
    }

    /// The outer Boolean atoms whose values are currently pinned onto the
    /// circuit, in term order — the hypotheses a whole-scope conflict
    /// explanation names besides the recorded constraint terms.  Exposed for
    /// tests.
    #[must_use]
    pub fn pinned_terms(&self) -> Vec<TermId> {
        self.current_pins()
            .into_iter()
            .map(|(term, _)| term)
            .collect()
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
                self.define([Lit::pos(v)]);
                v
            }
            TermKind::False => {
                let v = self.sat.new_var();
                self.define([Lit::neg(v)]);
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
                        self.define([Lit::pos(v)]);
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
                        self.define([Lit::pos(v)]);
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
                        self.define([Lit::neg(v)]);
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
        // A definition, permanent (`scope.rs`).  An outer value recorded
        // before the node existed is picked up by the next check's pins.
        self.bool_node.insert(term, out);
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
        // The disjunction is *defined* once, behind a selector, and the
        // selector is asserted — so a lemma derived again reuses its clause,
        // and a `pop` retracts the assertion without deleting a clause.
        let selector = self.clause_selector(&lits);
        self.assert_lit(selector);
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
    /// Memoised per unordered pair in `eq_cache`, and — a definition — never
    /// retracted (`scope.rs`): the bit-vector / EUF exchange asks for the same
    /// pairs round after round, `assert_eq` / `assert_neq` assert it on every
    /// trail assignment of an atom, and re-encoding it made every round's
    /// instance — and search — bigger than the last.
    pub(super) fn bool_bv_eq(&mut self, lhs: TermId, rhs: TermId) -> Option<Var> {
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
        // `out <=> AND_i (a[i] <=> b[i])`, in as few variables as the
        // definition allows, because a root definition is live in every later
        // check of the round (`scope.rs`): one XNOR for a single bit, and for
        // a wider pair `out` plus one "this bit differs" witness per bit —
        // `w + 1` variables where the XOR/NOT/AND chain this replaced took
        // `3w`.  Both directions are encoded, so `out` is a function of the
        // operands and asserting either polarity is exact:
        //   out  -> a[i] = b[i]            (two binary clauses per bit)
        //   !out -> some d[i]              (one clause)
        //   d[i] -> a[i] != b[i]           (two clauses per bit)
        let width = va.width as usize;
        let out = match width {
            0 => self.const_var(true),
            1 => {
                let out = self.sat.new_var();
                self.encode_xnor(out, va.bits[0], vb.bits[0]);
                out
            }
            _ => {
                let out = self.sat.new_var();
                let mut differs: Vec<Lit> = Vec::with_capacity(width + 1);
                differs.push(Lit::pos(out));
                for i in 0..width {
                    let (a, b) = (va.bits[i], vb.bits[i]);
                    self.define([Lit::neg(out), Lit::neg(a), Lit::pos(b)]);
                    self.define([Lit::neg(out), Lit::pos(a), Lit::neg(b)]);
                    let d = self.sat.new_var();
                    self.define([Lit::neg(d), Lit::pos(a), Lit::pos(b)]);
                    self.define([Lit::neg(d), Lit::neg(a), Lit::neg(b)]);
                    differs.push(Lit::pos(d));
                }
                self.define(differs);
                out
            }
        };
        self.eq_cache.insert(key, out);
        Some(out)
    }

    /// The strict less-than (signed or unsigned) gate over two bit-blasted
    /// operands — the memoised definition `assert_ult` / `assert_slt` assert.
    fn bool_ult(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        _manager: &TermManager,
        signed: bool,
    ) -> Option<Var> {
        if signed {
            self.slt_gate(lhs, rhs)
        } else {
            self.ult_gate(lhs, rhs)
        }
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
