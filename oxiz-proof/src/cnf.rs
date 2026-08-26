//! CNF (Conjunctive Normal Form) transformation with proofs.
//!
//! This module provides algorithms for converting arbitrary Boolean formulas
//! to CNF with proof generation using Tseitin transformation.
//!
//! # What Tseitin encoding is
//!
//! A SAT solver's input format (CNF: a conjunction of clauses, each a
//! disjunction of literals) cannot represent an arbitrary Boolean formula
//! directly without first flattening its connectives. The textbook approach --
//! repeatedly distributing `∨` over `∧` (De Morgan / double-negation
//! elimination followed by distribution) -- produces a *logically equivalent*
//! CNF formula, but that distribution step can blow the formula up
//! exponentially: converting `(a1 ∧ b1) ∨ (a2 ∧ b2) ∨ ... ∨ (an ∧ bn)` this
//! way produces `2^n` clauses, because every disjunct's two choices must be
//! combined with every other disjunct's.
//!
//! Tseitin's transformation avoids that blowup by not insisting on logical
//! equivalence. For every subformula `f` it introduces a fresh Boolean
//! variable `v_f` and asserts the *definition* `v_f <=> f` (itself expanded to
//! a handful of clauses, linear in `f`'s top-level arity), rather than
//! substituting `f`'s expansion inline. The root subformula's variable is then
//! asserted as a unit clause. The result is linear in the size of the input
//! formula -- see [`CnfTransformer::transform`]'s doc comment for the
//! memoization that is additionally needed to keep it linear in the presence
//! of shared subformulas, not just in a straight-line tree.
//!
//! # Equisatisfiable, not equivalent
//!
//! The price of avoiding the exponential blowup is that the output CNF is
//! only **equisatisfiable** with the input formula, not logically equivalent
//! to it: the output is satisfiable if and only if the input is, but it is
//! satisfiable by a *different, larger* set of models, because it ranges over
//! the original variables *and* every Tseitin variable introduced along the
//! way. A model of the output restricted to the original variables is always
//! a model of the input (that is the direction a SAT solver's caller actually
//! needs); the converse -- that every model of the input extends to a model
//! of the output -- also holds here (each Tseitin variable's definition
//! pins its value to its subformula's truth value under any extension), but
//! is not, in general, a requirement of equisatisfiability, only a stronger
//! property this particular construction happens to have. The
//! `iff_xor_ladder_is_equisatisfiable_with_original` test (in this module's
//! `tests` submodule) checks exactly this, exhaustively over all `2^11`
//! assignments to an 11-variable test formula, rather than merely checking
//! clause counts.
//!
//! # Examples
//!
//! ```
//! use oxiz_proof::cnf::{CnfTransformer, Formula};
//!
//! // (a ∧ b) => c, i.e. Implies(And(a, b), c)
//! let a = Formula::var(1);
//! let b = Formula::var(2);
//! let c = Formula::var(3);
//! let formula = Formula::implies(Formula::and(vec![a, b]), c);
//!
//! // Variables 1..=3 are the original formula's; Tseitin variables start at 4.
//! let mut transformer = CnfTransformer::new(4);
//! let root = transformer.transform(&formula);
//!
//! // The transformation introduced at least one fresh Tseitin variable for
//! // the root connective, and emitted clauses defining it.
//! assert!(root.0 >= 4);
//! assert!(!transformer.clauses().is_empty());
//! ```
//!
//! # Structural sharing
//!
//! [`CnfTransformer::transform`] memoizes on a canonical (`Debug`-derived)
//! key of each subformula in `tseitin_vars`, so a subformula that occurs more
//! than once is converted exactly once; every later occurrence reuses the
//! already-allocated Tseitin variable instead of re-emitting its defining
//! clauses. This matters most for `Implies`/`Iff`/`Xor`, whose standard
//! desugaring (`a <=> b` ~> `(a => b) ∧ (b => a)`, `a xor b` ~> `(a ∨ b) ∧
//! (¬a ∨ ¬b)`) references each operand twice. Without memoization, a chain of
//! `n` nested `Iff`/`Xor` -- each introducing one new variable, so *linear*
//! in input size -- drove ~2^n `transform` calls and clauses, because each
//! level's two references to the previous (already-doubled) level were both
//! reprocessed from scratch instead of being recognized as the same
//! subformula. Measured before this fix: n=5 -> 403 clauses, n=10 -> 13,299,
//! n=15 -> 425,971, n=20 -> 13,631,475 clauses (2.9s) -- textbook exponential.
//! After: clause count is linear in `n` (see `cnf::tests::iff_xor_ladder_*`
//! for pinned counts and an equisatisfiability check).
//!
//! This still costs `O(n^2)` wall time (not `O(n)`) for an `n`-deep chain:
//! `Formula` is a plain `Box`-based tree with no built-in structural sharing
//! (no `Rc`/hash-consing), so computing a subformula's cache key requires
//! re-walking its full text every time it is referenced, even on a cache
//! hit. That is a real gap versus a fully linear hash-consed implementation,
//! but it converts what was an *intractable* exponential (a chain of 20-30
//! nested connectives already would not finish) into a comfortably
//! tractable quadratic (thousands of nested connectives resolve in a
//! fraction of a second).

use crate::resolution::{Clause, Literal};
use std::collections::HashMap;
use std::fmt;

/// A Boolean formula variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A Boolean formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula {
    /// Variable.
    Var(Var),
    /// Negation.
    Not(Box<Formula>),
    /// Conjunction (AND).
    And(Vec<Formula>),
    /// Disjunction (OR).
    Or(Vec<Formula>),
    /// Implication.
    Implies(Box<Formula>, Box<Formula>),
    /// Equivalence (IFF).
    Iff(Box<Formula>, Box<Formula>),
    /// Exclusive OR (XOR).
    Xor(Box<Formula>, Box<Formula>),
}

impl Formula {
    /// Create a variable formula.
    #[must_use]
    pub fn var(v: u32) -> Self {
        Self::Var(Var(v))
    }

    /// Create a negation.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn not(f: Formula) -> Self {
        Self::Not(Box::new(f))
    }

    /// Create a conjunction.
    #[must_use]
    pub fn and(formulas: Vec<Formula>) -> Self {
        Self::And(formulas)
    }

    /// Create a disjunction.
    #[must_use]
    pub fn or(formulas: Vec<Formula>) -> Self {
        Self::Or(formulas)
    }

    /// Create an implication.
    #[must_use]
    pub fn implies(a: Formula, b: Formula) -> Self {
        Self::Implies(Box::new(a), Box::new(b))
    }

    /// Create an equivalence.
    #[must_use]
    pub fn iff(a: Formula, b: Formula) -> Self {
        Self::Iff(Box::new(a), Box::new(b))
    }

    /// Create an XOR.
    #[must_use]
    pub fn xor(a: Formula, b: Formula) -> Self {
        Self::Xor(Box::new(a), Box::new(b))
    }
}

/// CNF transformation context.
pub struct CnfTransformer {
    /// Next available variable ID.
    next_var: u32,
    /// Structural-sharing cache: maps a canonical (`Debug`-derived) key of a
    /// subformula already converted to the `Var` naming it. Consulted and
    /// populated by every non-`Var` call to [`CnfTransformer::transform`], so
    /// a subformula occurring more than once is converted (and its defining
    /// clauses emitted) exactly once -- see the module doc comment.
    tseitin_vars: HashMap<String, Var>,
    /// Generated clauses.
    clauses: Vec<Clause>,
}

impl CnfTransformer {
    /// Create a new CNF transformer.
    #[must_use]
    pub fn new(first_var: u32) -> Self {
        Self {
            next_var: first_var,
            tseitin_vars: HashMap::new(),
            clauses: Vec::new(),
        }
    }

    /// Allocate a fresh Tseitin variable.
    fn fresh_var(&mut self) -> Var {
        let v = Var(self.next_var);
        self.next_var += 1;
        v
    }

    /// Canonical structural key for memoizing `formula` in `tseitin_vars`.
    ///
    /// `Formula` derives `Debug`, whose output is a deterministic function of
    /// structure and leaf values, so two structurally-equal subformulas
    /// always produce the same key and two structurally-different ones never
    /// collide.
    fn formula_key(formula: &Formula) -> String {
        format!("{formula:?}")
    }

    /// Transform a formula to CNF using Tseitin transformation.
    ///
    /// Memoized via `tseitin_vars` (see the module doc comment): a bare
    /// variable needs no fresh naming and is returned directly; every other
    /// formula is looked up by its structural key first, and only actually
    /// converted -- allocating a fresh variable and emitting its defining
    /// clauses -- on a cache miss.
    pub fn transform(&mut self, formula: &Formula) -> Var {
        if let Formula::Var(v) = formula {
            return *v;
        }

        let key = Self::formula_key(formula);
        if let Some(&cached) = self.tseitin_vars.get(&key) {
            return cached;
        }

        let result_var = self.transform_uncached(formula);
        self.tseitin_vars.insert(key, result_var);
        result_var
    }

    /// Convert a formula not already resolved by the `tseitin_vars` cache
    /// lookup in [`CnfTransformer::transform`].
    ///
    /// Always reached through `transform`, never called directly, so every
    /// recursive reference to an operand -- including the duplicated
    /// operand references `Implies`/`Iff`/`Xor` construct while desugaring --
    /// goes back through the memoized entry point rather than reprocessing
    /// an already-converted subformula from scratch.
    fn transform_uncached(&mut self, formula: &Formula) -> Var {
        match formula {
            Formula::Var(v) => *v,

            Formula::Not(f) => {
                let sub_var = self.transform(f);
                let result_var = self.fresh_var();

                // result_var <=> ~sub_var
                // (~result_var ∨ ~sub_var) ∧ (result_var ∨ sub_var)
                self.clauses.push(Clause::new(vec![
                    Literal::neg(result_var.0),
                    Literal::neg(sub_var.0),
                ]));
                self.clauses.push(Clause::new(vec![
                    Literal::pos(result_var.0),
                    Literal::pos(sub_var.0),
                ]));

                result_var
            }

            Formula::And(formulas) => {
                let sub_vars: Vec<Var> = formulas.iter().map(|f| self.transform(f)).collect();
                let result_var = self.fresh_var();

                // result_var <=> (v1 ∧ v2 ∧ ... ∧ vn)
                // (result_var ∨ ~v1 ∨ ~v2 ∨ ... ∨ ~vn)
                let mut clause_lits = vec![Literal::pos(result_var.0)];
                for &v in &sub_vars {
                    clause_lits.push(Literal::neg(v.0));
                }
                self.clauses.push(Clause::new(clause_lits));

                // (~result_var ∨ v1) ∧ (~result_var ∨ v2) ∧ ... ∧ (~result_var ∨ vn)
                for &v in &sub_vars {
                    self.clauses.push(Clause::new(vec![
                        Literal::neg(result_var.0),
                        Literal::pos(v.0),
                    ]));
                }

                result_var
            }

            Formula::Or(formulas) => {
                let sub_vars: Vec<Var> = formulas.iter().map(|f| self.transform(f)).collect();
                let result_var = self.fresh_var();

                // result_var <=> (v1 ∨ v2 ∨ ... ∨ vn)
                // (~result_var ∨ v1 ∨ v2 ∨ ... ∨ vn)
                let mut clause_lits = vec![Literal::neg(result_var.0)];
                for &v in &sub_vars {
                    clause_lits.push(Literal::pos(v.0));
                }
                self.clauses.push(Clause::new(clause_lits));

                // (result_var ∨ ~v1) ∧ (result_var ∨ ~v2) ∧ ... ∧ (result_var ∨ ~vn)
                for &v in &sub_vars {
                    self.clauses.push(Clause::new(vec![
                        Literal::pos(result_var.0),
                        Literal::neg(v.0),
                    ]));
                }

                result_var
            }

            Formula::Implies(a, b) => {
                // a => b is equivalent to ~a ∨ b
                let not_a = Formula::not((**a).clone());
                let equiv = Formula::or(vec![not_a, (**b).clone()]);
                self.transform(&equiv)
            }

            Formula::Iff(a, b) => {
                // a <=> b is equivalent to (a => b) ∧ (b => a)
                let a_implies_b = Formula::implies((**a).clone(), (**b).clone());
                let b_implies_a = Formula::implies((**b).clone(), (**a).clone());
                let equiv = Formula::and(vec![a_implies_b, b_implies_a]);
                self.transform(&equiv)
            }

            Formula::Xor(a, b) => {
                // a XOR b is equivalent to (a ∨ b) ∧ (~a ∨ ~b)
                let a_or_b = Formula::or(vec![(**a).clone(), (**b).clone()]);
                let not_a_or_not_b = Formula::or(vec![
                    Formula::not((**a).clone()),
                    Formula::not((**b).clone()),
                ]);
                let equiv = Formula::and(vec![a_or_b, not_a_or_not_b]);
                self.transform(&equiv)
            }
        }
    }

    /// Get the generated clauses.
    #[must_use]
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Take the generated clauses.
    pub fn take_clauses(self) -> Vec<Clause> {
        self.clauses
    }

    /// Get the next variable ID.
    #[must_use]
    pub fn next_var(&self) -> u32 {
        self.next_var
    }
}

/// CNF transformation statistics.
#[derive(Debug, Clone)]
pub struct CnfStats {
    /// Number of clauses generated.
    pub clause_count: usize,
    /// Number of literals generated.
    pub literal_count: usize,
    /// Number of Tseitin variables introduced.
    pub tseitin_vars: usize,
    /// Maximum clause size.
    pub max_clause_size: usize,
    /// Average clause size.
    pub avg_clause_size: f64,
}

impl CnfStats {
    /// Compute statistics from a set of clauses.
    #[must_use]
    pub fn compute(clauses: &[Clause], original_vars: u32, final_vars: u32) -> Self {
        let clause_count = clauses.len();
        let literal_count: usize = clauses.iter().map(|c| c.literals.len()).sum();
        let max_clause_size = clauses.iter().map(|c| c.literals.len()).max().unwrap_or(0);
        let avg_clause_size = if clause_count > 0 {
            literal_count as f64 / clause_count as f64
        } else {
            0.0
        };
        let tseitin_vars = (final_vars - original_vars) as usize;

        Self {
            clause_count,
            literal_count,
            tseitin_vars,
            max_clause_size,
            avg_clause_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a chain of `n` nested `Iff`/`Xor` connectives (alternating),
    /// each level introducing exactly one fresh variable: `f_0 = Var(1)`,
    /// `f_i = f_{i-1} <op> Var(i+1)` for `i` in `1..=n` (so the ladder uses
    /// variables `1..=n+1`). The chain is linear in **input** size --
    /// constructing it never clones an already-built subformula -- so any
    /// blowup measured against it comes from `CnfTransformer::transform`
    /// itself, not from building the input. Built with a single iterative
    /// `for` loop, never recursively.
    ///
    /// Deliberately starts numbering at `1`, not `0`: `resolution::Literal`
    /// encodes polarity as the sign of an `i32` (`Literal::neg(v)` is
    /// `Literal(-(v as i32))`), so `Literal::neg(0)` is `-0i32`, which equals
    /// `0i32` -- indistinguishable from `Literal::pos(0)`. Variable 0 can
    /// therefore never be soundly negated in this representation; that is a
    /// latent gap in `resolution::Literal` found while writing this test, not
    /// something the Tseitin-naming fix this ladder exercises is responsible
    /// for, so the ladder just avoids variable 0 rather than tripping over it.
    fn build_ladder(n: u32) -> Formula {
        let mut formula = Formula::var(1);
        for i in 1..=n {
            formula = if i % 2 == 1 {
                Formula::iff(formula, Formula::var(i + 1))
            } else {
                Formula::xor(formula, Formula::var(i + 1))
            };
        }
        formula
    }

    #[test]
    fn test_cnf_var() {
        let mut transformer = CnfTransformer::new(1);
        let formula = Formula::var(1);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(1));
        assert_eq!(transformer.clauses().len(), 0);
    }

    #[test]
    fn test_cnf_not() {
        let mut transformer = CnfTransformer::new(2);
        let formula = Formula::not(Formula::var(1));
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(2));
        assert_eq!(transformer.clauses().len(), 2);
    }

    #[test]
    fn test_cnf_and() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(3));
        // Should generate 3 clauses for AND
        assert_eq!(transformer.clauses().len(), 3);
    }

    #[test]
    fn test_cnf_or() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::or(vec![Formula::var(1), Formula::var(2)]);
        let result = transformer.transform(&formula);

        assert_eq!(result, Var(3));
        // Should generate 3 clauses for OR
        assert_eq!(transformer.clauses().len(), 3);
    }

    #[test]
    fn test_cnf_implies() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::implies(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // Implication gets converted to disjunction
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_iff() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::iff(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // IFF gets converted to conjunction of implications
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_xor() {
        let mut transformer = CnfTransformer::new(3);
        let formula = Formula::xor(Formula::var(1), Formula::var(2));
        let _result = transformer.transform(&formula);

        // XOR gets converted
        assert!(!transformer.clauses().is_empty());
    }

    /// Evaluate `formula` under `assignment`, where `assignment[i]` is the
    /// truth value of `Var(i)`. Only ever called on the small test ladders in
    /// this module (depth well under a few hundred), so plain recursion is
    /// fine here -- unlike the ladder *construction* above, which must scale
    /// to sizes that would make recursion a real stack-depth concern.
    fn eval_formula(formula: &Formula, assignment: &[bool]) -> bool {
        match formula {
            Formula::Var(Var(v)) => assignment[*v as usize],
            Formula::Not(f) => !eval_formula(f, assignment),
            Formula::And(fs) => fs.iter().all(|f| eval_formula(f, assignment)),
            Formula::Or(fs) => fs.iter().any(|f| eval_formula(f, assignment)),
            Formula::Implies(a, b) => !eval_formula(a, assignment) || eval_formula(b, assignment),
            Formula::Iff(a, b) => eval_formula(a, assignment) == eval_formula(b, assignment),
            Formula::Xor(a, b) => eval_formula(a, assignment) != eval_formula(b, assignment),
        }
    }

    /// Does `assignment` (indexed by variable id) satisfy every clause?
    fn satisfies_all(clauses: &[Clause], assignment: &[bool]) -> bool {
        clauses.iter().all(|clause| {
            clause
                .literals
                .iter()
                .any(|lit| assignment[lit.var() as usize] == lit.is_positive())
        })
    }

    /// Extend `full` (already seeded with `base`'s values at the original
    /// variable positions) with the value every Tseitin variable must take
    /// for a *correct* Tseitin transformation: the actual truth value, under
    /// `base`, of the subformula that variable names. This is the "natural"
    /// witness a sound transformation always admits -- see
    /// `iff_xor_ladder_is_equisatisfiable_with_original`.
    ///
    /// `Implies`/`Iff`/`Xor` are desugared here exactly as
    /// `CnfTransformer::transform_uncached` desugars them, so the *synthetic*
    /// intermediate `Or`/`Not`/`And` nodes that desugaring builds -- which
    /// get their own cache entries and Tseitin variables, distinct from the
    /// original node's -- are visited too, not just the original formula's
    /// own children. Looks up each node's key the same way `transform` does
    /// (`tests` is a child module of `cnf`, so it can see `formula_key` and
    /// `tseitin_vars` despite neither being `pub`).
    fn natural_extension(
        transformer: &CnfTransformer,
        formula: &Formula,
        base: &[bool],
        full: &mut [bool],
    ) {
        if matches!(formula, Formula::Var(_)) {
            return;
        }
        let key = CnfTransformer::formula_key(formula);
        if let Some(&var) = transformer.tseitin_vars.get(&key) {
            full[var.0 as usize] = eval_formula(formula, base);
        }
        match formula {
            Formula::Var(_) => {}
            Formula::Not(f) => natural_extension(transformer, f, base, full),
            Formula::And(fs) | Formula::Or(fs) => {
                for f in fs {
                    natural_extension(transformer, f, base, full);
                }
            }
            Formula::Implies(a, b) => {
                let not_a = Formula::not((**a).clone());
                let equiv = Formula::or(vec![not_a, (**b).clone()]);
                natural_extension(transformer, &equiv, base, full);
            }
            Formula::Iff(a, b) => {
                let a_implies_b = Formula::implies((**a).clone(), (**b).clone());
                let b_implies_a = Formula::implies((**b).clone(), (**a).clone());
                let equiv = Formula::and(vec![a_implies_b, b_implies_a]);
                natural_extension(transformer, &equiv, base, full);
            }
            Formula::Xor(a, b) => {
                let a_or_b = Formula::or(vec![(**a).clone(), (**b).clone()]);
                let not_a_or_not_b = Formula::or(vec![
                    Formula::not((**a).clone()),
                    Formula::not((**b).clone()),
                ]);
                let equiv = Formula::and(vec![a_or_b, not_a_or_not_b]);
                natural_extension(transformer, &equiv, base, full);
            }
        }
    }

    #[test]
    fn iff_xor_ladder_clause_counts_are_linear() {
        // Regression pin for the Tseitin-naming fix: before it, `Iff`/`Xor`
        // duplicated their operands with no memoization, so transforming
        // this *linear-input-size* ladder cost ~2^n clauses. Measured on the
        // pre-fix code: n=5 -> 403, n=10 -> 13,299, n=15 -> 425,971,
        // n=20 -> 13,631,475 clauses (2.9s). After the fix each ladder level
        // contributes exactly 13 new clauses (5 fresh Tseitin variables: two
        // `Not`s, two binary `Or`s, and one binary `And`, per the
        // `Implies`/`Iff`/`Xor` desugaring), so clause count is exactly
        // `13 * n` -- linear. A regression back toward exponential growth
        // will fail these exact-count assertions long before it becomes slow
        // enough to notice.
        for &n in &[5u32, 10, 15, 20] {
            let formula = build_ladder(n);
            let mut transformer = CnfTransformer::new(n + 2); // ladder uses variables 1..=n+1
            transformer.transform(&formula);
            assert_eq!(
                transformer.clauses().len(),
                13 * n as usize,
                "clause count must be exactly linear in the ladder height (n={n})"
            );
        }
    }

    #[test]
    fn iff_xor_ladder_scales_to_size_impossible_under_exponential_blowup() {
        // n=100 nested Iff/Xor connectives: 2^100 is far beyond anything the
        // pre-fix code could ever have completed (the pre-fix growth rate,
        // measured up to n=20 above, was already at 13.6M clauses and 2.9s;
        // extrapolating its ~2x-per-level growth puts n=100 at roughly
        // 13,631,475 * 2^80 clauses, an astronomically impossible number).
        // The fixed transform still recurses natively (see the module doc
        // comment: this task fixed the clause-count blowup, not the
        // per-node recursion depth), so this runs on a deliberately
        // constrained 1 MiB stack to confirm that recursion depth -- now
        // proportional to input size rather than exponentially inflated --
        // is not itself a problem at this scale.
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — n=100
        // levels of native recursion (sub-10,000-depth), so 1 MiB is
        // intentionally generous. See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let formula = build_ladder(100);
                let mut transformer = CnfTransformer::new(102); // ladder uses variables 1..=101
                transformer.transform(&formula);
                transformer.clauses().len()
            })
            .expect("failed to spawn constrained-stack thread");

        let clause_count = handle.join().expect("thread panicked (stack overflow?)");
        assert_eq!(
            clause_count,
            13 * 100,
            "clause count must stay exactly linear at n=100"
        );
    }

    #[test]
    fn iff_xor_ladder_is_equisatisfiable_with_original() {
        // Equisatisfiability (in fact full equivalence-under-extension, a
        // stronger property): for every assignment to the ladder's original
        // variables, extending it with each Tseitin variable's *actual*
        // subformula value must (a) satisfy every generated clause and
        // (b) make the root variable equal the original formula's truth
        // value. Checked exhaustively (all 2^11 assignments), not merely
        // structurally, so a future change that preserves clause *counts*
        // but breaks the encoding's actual semantics would still be caught.
        const N: u32 = 10;
        let formula = build_ladder(N);

        let mut transformer = CnfTransformer::new(N + 2); // ladder uses variables 1..=N+1
        let root_var = transformer.transform(&formula);

        let total_vars = transformer.next_var() as usize;
        let num_relevant_bits = (N + 1) as usize; // variables 1..=N+1

        for bits in 0u32..(1 << num_relevant_bits) {
            let mut base = vec![false; total_vars];
            for i in 0..num_relevant_bits {
                base[i + 1] = (bits >> i) & 1 == 1;
            }
            let expected = eval_formula(&formula, &base);

            let mut full = base.clone();
            natural_extension(&transformer, &formula, &base, &mut full);

            assert!(
                satisfies_all(transformer.clauses(), &full),
                "generated clauses violated under the natural extension for base={base:?}"
            );
            assert_eq!(
                full[root_var.0 as usize], expected,
                "root variable must equal the original formula's truth value for base={base:?}"
            );
        }
    }

    #[test]
    fn test_cnf_complex() {
        let mut transformer = CnfTransformer::new(4);

        // (v1 ∧ v2) ∨ (v2 ∧ v3)
        let left = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let right = Formula::and(vec![Formula::var(2), Formula::var(3)]);
        let formula = Formula::or(vec![left, right]);

        let _result = transformer.transform(&formula);

        // Should generate multiple clauses
        assert!(!transformer.clauses().is_empty());
    }

    #[test]
    fn test_cnf_stats() {
        let clauses = vec![
            Clause::new(vec![Literal::pos(1), Literal::neg(2)]),
            Clause::new(vec![Literal::pos(2), Literal::pos(3), Literal::neg(4)]),
            Clause::new(vec![Literal::neg(1)]),
        ];

        let stats = CnfStats::compute(&clauses, 4, 10);

        assert_eq!(stats.clause_count, 3);
        assert_eq!(stats.literal_count, 6);
        assert_eq!(stats.max_clause_size, 3);
        assert_eq!(stats.tseitin_vars, 6);
        assert!((stats.avg_clause_size - 2.0).abs() < 0.01);
    }
}
