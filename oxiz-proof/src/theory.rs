//! Theory proof generation for SMT solvers.
//!
//! This module provides infrastructure for generating machine-checkable proofs
//! from theory solver inferences. It bridges the gap between theory-specific
//! reasoning and proof formats like Alethe and LFSC.
//!
//! ## Supported Theories
//!
//! - **EUF**: Equality with Uninterpreted Functions (congruence, transitivity)
//! - **LRA/LIA**: Linear Real/Integer Arithmetic (Farkas lemmas)
//! - **BV**: BitVectors (bit-level reasoning)
//! - **Arrays**: Read-over-write axioms
//!
//! ## Proof Structure
//!
//! Theory proofs consist of:
//! - **Axiom applications**: Theory axioms (e.g., reflexivity, array extensionality)
//! - **Inference rules**: Theory-specific deduction rules
//! - **Lemma steps**: Derived facts used in the proof

use std::fmt;
use std::io::{self, Write};

/// Unique identifier for a theory proof step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheoryStepId(pub u32);

impl fmt::Display for TheoryStepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// A term reference (opaque identifier for formula/term)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofTerm(pub String);

impl fmt::Display for ProofTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<S: Into<String>> From<S> for ProofTerm {
    fn from(s: S) -> Self {
        ProofTerm(s.into())
    }
}

/// Theory-specific proof rules
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoryRule {
    // === EUF Rules ===
    /// Reflexivity: ⊢ t = t
    Refl,
    /// Symmetry: t1 = t2 ⊢ t2 = t1
    Symm,
    /// Transitivity: t1 = t2, t2 = t3 ⊢ t1 = t3
    Trans,
    /// Congruence: f(a1,...,an) = f(b1,...,bn) when ai = bi
    Cong,
    /// Function application equality
    FuncEq,
    /// Distinctness axiom: distinct(t1,...,tn) ⊢ ti ≠ tj for i ≠ j
    Distinct,

    // === Linear Arithmetic Rules ===
    /// Linear combination of inequalities (Farkas lemma)
    LaGeneric,
    /// Multiplication of inequality by positive constant
    LaMult,
    /// Addition of two inequalities
    LaAdd,
    /// Tightening of integer bounds: x < c ⊢ x ≤ c-1
    LaTighten,
    /// Division by constant
    LaDiv,
    /// Integer totality: x ≤ c ∨ x ≥ c+1
    LaTotality,
    /// Disequality split: x ≠ c ⊢ x < c ∨ x > c
    LaDiseq,

    // === BitVector Rules ===
    /// Bit-blasting equality
    BvBlastEq,
    /// Bit extraction
    BvExtract,
    /// Concatenation
    BvConcat,
    /// Zero extension
    BvZeroExtend,
    /// Sign extension
    BvSignExtend,
    /// Bitwise operations (AND, OR, XOR, NOT)
    BvBitwise,
    /// Arithmetic operations
    BvArith,
    /// Comparison operations
    BvCompare,
    /// Overflow detection
    BvOverflow,

    // === Array Rules ===
    /// Read-over-write (same index): (select (store a i v) i) = v
    ArrReadWrite1,
    /// Read-over-write (different index): i ≠ j → (select (store a i v) j) = (select a j)
    ArrReadWrite2,
    /// Extensionality: (∀i. (select a i) = (select b i)) → a = b
    ArrExt,
    /// Const array: (select (const v) i) = v
    ArrConst,

    // === Quantifier Rules ===
    /// Forall elimination: ∀x. φ(x) ⊢ φ(t) for any term t
    ForallElim,
    /// Exists introduction: φ(t) ⊢ ∃x. φ(x) for any term t
    ExistsIntro,
    /// Skolemization: ∃x. φ(x) ⊢ φ(sk) where sk is fresh
    Skolemize,
    /// Quantifier instantiation with pattern matching
    QuantInst,
    /// α-equivalence (renaming bound variables)
    AlphaEquiv,

    // === General SMT Rules ===
    /// Theory conflict (unsat core from theory)
    TheoryConflict,
    /// Theory propagation
    TheoryProp,
    /// ITE elimination
    IteElim,
    /// Boolean simplification
    BoolSimp,
    /// Custom rule with name
    Custom(String),
}

impl fmt::Display for TheoryRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refl => write!(f, "refl"),
            Self::Symm => write!(f, "symm"),
            Self::Trans => write!(f, "trans"),
            Self::Cong => write!(f, "cong"),
            Self::FuncEq => write!(f, "func_eq"),
            Self::Distinct => write!(f, "distinct"),
            Self::LaGeneric => write!(f, "la_generic"),
            Self::LaMult => write!(f, "la_mult"),
            Self::LaAdd => write!(f, "la_add"),
            Self::LaTighten => write!(f, "la_tighten"),
            Self::LaDiv => write!(f, "la_div"),
            Self::LaTotality => write!(f, "la_totality"),
            Self::LaDiseq => write!(f, "la_diseq"),
            Self::BvBlastEq => write!(f, "bv_blast_eq"),
            Self::BvExtract => write!(f, "bv_extract"),
            Self::BvConcat => write!(f, "bv_concat"),
            Self::BvZeroExtend => write!(f, "bv_zero_extend"),
            Self::BvSignExtend => write!(f, "bv_sign_extend"),
            Self::BvBitwise => write!(f, "bv_bitwise"),
            Self::BvArith => write!(f, "bv_arith"),
            Self::BvCompare => write!(f, "bv_compare"),
            Self::BvOverflow => write!(f, "bv_overflow"),
            Self::ArrReadWrite1 => write!(f, "arr_read_write_1"),
            Self::ArrReadWrite2 => write!(f, "arr_read_write_2"),
            Self::ArrExt => write!(f, "arr_ext"),
            Self::ArrConst => write!(f, "arr_const"),
            Self::ForallElim => write!(f, "forall_elim"),
            Self::ExistsIntro => write!(f, "exists_intro"),
            Self::Skolemize => write!(f, "skolemize"),
            Self::QuantInst => write!(f, "quant_inst"),
            Self::AlphaEquiv => write!(f, "alpha_equiv"),
            Self::TheoryConflict => write!(f, "theory_conflict"),
            Self::TheoryProp => write!(f, "theory_prop"),
            Self::IteElim => write!(f, "ite_elim"),
            Self::BoolSimp => write!(f, "bool_simp"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// A single step in a theory proof
#[derive(Debug, Clone)]
pub struct TheoryStep {
    /// Unique identifier for this step
    pub id: TheoryStepId,
    /// The rule applied
    pub rule: TheoryRule,
    /// Premises (indices of previous steps)
    pub premises: Vec<TheoryStepId>,
    /// Arguments to the rule (theory-specific)
    pub args: Vec<ProofTerm>,
    /// The conclusion (what this step proves)
    pub conclusion: ProofTerm,
}

impl fmt::Display for TheoryStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} [", self.id, self.conclusion)?;
        if !self.premises.is_empty() {
            for (i, premise) in self.premises.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", premise)?;
            }
            write!(f, " | ")?;
        }
        write!(f, "{}", self.rule)?;
        for arg in &self.args {
            write!(f, " {}", arg)?;
        }
        write!(f, "]")
    }
}

/// Theory proof builder
#[derive(Debug, Default)]
pub struct TheoryProof {
    /// Steps in the proof
    steps: Vec<TheoryStep>,
    /// Next step ID
    next_id: u32,
}

impl TheoryProof {
    /// Create a new empty theory proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            next_id: 0,
        }
    }

    /// Allocate a new step ID
    fn alloc_id(&mut self) -> TheoryStepId {
        let id = TheoryStepId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add an axiom step (no premises)
    pub fn add_axiom(
        &mut self,
        rule: TheoryRule,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let id = self.alloc_id();
        self.steps.push(TheoryStep {
            id,
            rule,
            premises: Vec::new(),
            args: Vec::new(),
            conclusion: conclusion.into(),
        });
        id
    }

    /// Add an inference step with premises
    pub fn add_step(
        &mut self,
        rule: TheoryRule,
        premises: Vec<TheoryStepId>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let id = self.alloc_id();
        self.steps.push(TheoryStep {
            id,
            rule,
            premises,
            args: Vec::new(),
            conclusion: conclusion.into(),
        });
        id
    }

    /// Add an inference step with premises and arguments
    pub fn add_step_with_args(
        &mut self,
        rule: TheoryRule,
        premises: Vec<TheoryStepId>,
        args: Vec<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let id = self.alloc_id();
        self.steps.push(TheoryStep {
            id,
            rule,
            premises,
            args,
            conclusion: conclusion.into(),
        });
        id
    }

    // === EUF Proof Helpers ===

    /// Add reflexivity step: ⊢ t = t
    pub fn refl(&mut self, term: impl Into<ProofTerm>) -> TheoryStepId {
        let t = term.into();
        let conclusion = format!("(= {} {})", t.0, t.0);
        self.add_axiom(TheoryRule::Refl, conclusion)
    }

    /// Add symmetry step: t1 = t2 ⊢ t2 = t1
    pub fn symm(
        &mut self,
        premise: TheoryStepId,
        t1: impl Into<ProofTerm>,
        t2: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let conclusion = format!("(= {} {})", t2.into().0, t1.into().0);
        self.add_step(TheoryRule::Symm, vec![premise], conclusion)
    }

    /// Add transitivity step: t1 = t2, t2 = t3 ⊢ t1 = t3
    pub fn trans(
        &mut self,
        p1: TheoryStepId,
        p2: TheoryStepId,
        t1: impl Into<ProofTerm>,
        t3: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let conclusion = format!("(= {} {})", t1.into().0, t3.into().0);
        self.add_step(TheoryRule::Trans, vec![p1, p2], conclusion)
    }

    /// Add congruence step
    pub fn cong(
        &mut self,
        premises: Vec<TheoryStepId>,
        func: impl Into<ProofTerm>,
        args1: &[ProofTerm],
        args2: &[ProofTerm],
    ) -> TheoryStepId {
        let f = func.into();
        let lhs = if args1.is_empty() {
            f.0.clone()
        } else {
            format!(
                "({} {})",
                f.0,
                args1
                    .iter()
                    .map(|a| a.0.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        let rhs = if args2.is_empty() {
            f.0.clone()
        } else {
            format!(
                "({} {})",
                f.0,
                args2
                    .iter()
                    .map(|a| a.0.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        };
        let conclusion = format!("(= {} {})", lhs, rhs);
        self.add_step(TheoryRule::Cong, premises, conclusion)
    }

    // === Arithmetic Proof Helpers ===

    /// Add a Farkas lemma step (linear combination proving unsatisfiability)
    pub fn farkas(
        &mut self,
        premises: Vec<TheoryStepId>,
        coefficients: &[ProofTerm],
    ) -> TheoryStepId {
        let conclusion = "false".to_string();
        self.add_step_with_args(
            TheoryRule::LaGeneric,
            premises,
            coefficients.to_vec(),
            conclusion,
        )
    }

    /// Add linear arithmetic addition step
    pub fn la_add(
        &mut self,
        p1: TheoryStepId,
        p2: TheoryStepId,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step(TheoryRule::LaAdd, vec![p1, p2], conclusion)
    }

    /// Add linear arithmetic multiplication step
    pub fn la_mult(
        &mut self,
        premise: TheoryStepId,
        coefficient: impl Into<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step_with_args(
            TheoryRule::LaMult,
            vec![premise],
            vec![coefficient.into()],
            conclusion,
        )
    }

    // === Array Proof Helpers ===

    /// Add read-over-write-same step
    pub fn read_write_same(
        &mut self,
        array: impl Into<ProofTerm>,
        index: impl Into<ProofTerm>,
        value: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let a = array.into();
        let i = index.into();
        let v = value.into();
        let conclusion = format!(
            "(= (select (store {} {} {}) {}) {})",
            a.0, i.0, v.0, i.0, v.0
        );
        self.add_axiom(TheoryRule::ArrReadWrite1, conclusion)
    }

    /// Add read-over-write-different step
    pub fn read_write_diff(
        &mut self,
        premise: TheoryStepId,
        array: impl Into<ProofTerm>,
        i: impl Into<ProofTerm>,
        j: impl Into<ProofTerm>,
        v: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let a = array.into();
        let idx_i = i.into();
        let idx_j = j.into();
        let val = v.into();
        let conclusion = format!(
            "(= (select (store {} {} {}) {}) (select {} {}))",
            a.0, idx_i.0, val.0, idx_j.0, a.0, idx_j.0
        );
        self.add_step(TheoryRule::ArrReadWrite2, vec![premise], conclusion)
    }

    // === BitVector Proof Helpers ===

    /// Add bit-blasting equality step
    pub fn bv_blast_eq(
        &mut self,
        premises: Vec<TheoryStepId>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step(TheoryRule::BvBlastEq, premises, conclusion)
    }

    // === Quantifier Proof Helpers ===

    /// Add forall elimination step: ∀x. φ(x) ⊢ φ(t)
    pub fn forall_elim(
        &mut self,
        premise: TheoryStepId,
        variable: impl Into<ProofTerm>,
        instantiation: impl Into<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step_with_args(
            TheoryRule::ForallElim,
            vec![premise],
            vec![variable.into(), instantiation.into()],
            conclusion,
        )
    }

    /// Add exists introduction step: φ(t) ⊢ ∃x. φ(x)
    pub fn exists_intro(
        &mut self,
        premise: TheoryStepId,
        witness: impl Into<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step_with_args(
            TheoryRule::ExistsIntro,
            vec![premise],
            vec![witness.into()],
            conclusion,
        )
    }

    /// Add skolemization step: ∃x. φ(x) ⊢ φ(sk)
    pub fn skolemize(
        &mut self,
        premise: TheoryStepId,
        skolem_constant: impl Into<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.add_step_with_args(
            TheoryRule::Skolemize,
            vec![premise],
            vec![skolem_constant.into()],
            conclusion,
        )
    }

    /// Add quantifier instantiation with pattern matching
    pub fn quant_inst(
        &mut self,
        quantified_formula: TheoryStepId,
        pattern: impl Into<ProofTerm>,
        instantiations: Vec<ProofTerm>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        let mut args = vec![pattern.into()];
        args.extend(instantiations);
        self.add_step_with_args(
            TheoryRule::QuantInst,
            vec![quantified_formula],
            args,
            conclusion,
        )
    }

    // === General Methods ===

    /// Get the number of steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get all steps
    #[must_use]
    pub fn steps(&self) -> &[TheoryStep] {
        &self.steps
    }

    /// Get a step by ID
    #[must_use]
    pub fn get_step(&self, id: TheoryStepId) -> Option<&TheoryStep> {
        self.steps.get(id.0 as usize)
    }

    /// Clear the proof
    pub fn clear(&mut self) {
        self.steps.clear();
        self.next_id = 0;
    }

    /// Write the proof in human-readable format
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "; Theory proof generated by OxiZ")?;
        writeln!(writer)?;

        for step in &self.steps {
            writeln!(writer, "{}", step)?;
        }

        Ok(())
    }

    /// Convert to string
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        let mut buf = Vec::new();
        self.write(&mut buf)
            .expect("writing to Vec should not fail");
        String::from_utf8(buf).expect("output is UTF-8")
    }
}

/// EUF-specific proof recorder
#[derive(Debug, Default)]
pub struct EufProofRecorder {
    /// Underlying proof
    proof: TheoryProof,
    /// Mapping from equality pairs to step IDs
    equality_steps: rustc_hash::FxHashMap<(u32, u32), TheoryStepId>,
}

impl EufProofRecorder {
    /// Create a new EUF proof recorder
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: TheoryProof::new(),
            equality_steps: rustc_hash::FxHashMap::default(),
        }
    }

    /// Record an equality assertion
    pub fn record_equality(&mut self, a: u32, b: u32, term: impl Into<ProofTerm>) -> TheoryStepId {
        let id = self
            .proof
            .add_axiom(TheoryRule::Custom("assert".to_string()), term);
        self.equality_steps.insert((a.min(b), a.max(b)), id);
        id
    }

    /// Record a congruence step
    pub fn record_congruence(
        &mut self,
        func: impl Into<ProofTerm>,
        arg_equalities: Vec<TheoryStepId>,
        result: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.proof
            .add_step_with_args(TheoryRule::Cong, arg_equalities, vec![func.into()], result)
    }

    /// Record a transitivity chain
    pub fn record_transitivity(
        &mut self,
        steps: Vec<TheoryStepId>,
        conclusion: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.proof.add_step(TheoryRule::Trans, steps, conclusion)
    }

    /// Get the recorded proof
    #[must_use]
    pub fn proof(&self) -> &TheoryProof {
        &self.proof
    }

    /// Take the proof
    pub fn take_proof(&mut self) -> TheoryProof {
        std::mem::take(&mut self.proof)
    }
}

/// Arithmetic-specific proof recorder
#[derive(Debug, Default)]
pub struct ArithProofRecorder {
    /// Underlying proof
    proof: TheoryProof,
    /// Constraint step IDs
    constraint_steps: Vec<TheoryStepId>,
}

impl ArithProofRecorder {
    /// Create a new arithmetic proof recorder
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: TheoryProof::new(),
            constraint_steps: Vec::new(),
        }
    }

    /// Record a bound constraint
    pub fn record_bound(&mut self, constraint: impl Into<ProofTerm>) -> TheoryStepId {
        let id = self
            .proof
            .add_axiom(TheoryRule::Custom("bound".to_string()), constraint);
        self.constraint_steps.push(id);
        id
    }

    /// Record a Farkas conflict
    pub fn record_farkas_conflict(&mut self, reasons: &[u32]) -> TheoryStepId {
        let premises: Vec<TheoryStepId> = reasons
            .iter()
            .filter_map(|&r| self.constraint_steps.get(r as usize).copied())
            .collect();
        self.proof
            .add_step(TheoryRule::LaGeneric, premises, "false")
    }

    /// Get the recorded proof
    #[must_use]
    pub fn proof(&self) -> &TheoryProof {
        &self.proof
    }

    /// Take the proof
    pub fn take_proof(&mut self) -> TheoryProof {
        std::mem::take(&mut self.proof)
    }
}

/// Array-specific proof recorder
#[derive(Debug, Default)]
pub struct ArrayProofRecorder {
    /// Underlying proof
    proof: TheoryProof,
}

impl ArrayProofRecorder {
    /// Create a new array proof recorder
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: TheoryProof::new(),
        }
    }

    /// Record a select-store axiom (same index)
    pub fn record_select_store_same(
        &mut self,
        array: impl Into<ProofTerm>,
        index: impl Into<ProofTerm>,
        value: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.proof.read_write_same(array, index, value)
    }

    /// Record a select-store axiom (different index)
    pub fn record_select_store_diff(
        &mut self,
        neq_proof: TheoryStepId,
        array: impl Into<ProofTerm>,
        i: impl Into<ProofTerm>,
        j: impl Into<ProofTerm>,
        v: impl Into<ProofTerm>,
    ) -> TheoryStepId {
        self.proof.read_write_diff(neq_proof, array, i, j, v)
    }

    /// Get the recorded proof
    #[must_use]
    pub fn proof(&self) -> &TheoryProof {
        &self.proof
    }

    /// Take the proof
    pub fn take_proof(&mut self) -> TheoryProof {
        std::mem::take(&mut self.proof)
    }
}

/// Trait for theory solvers that support proof generation
pub trait TheoryProofProducer {
    /// Enable proof recording
    fn enable_proof(&mut self);

    /// Disable proof recording
    fn disable_proof(&mut self);

    /// Check if proof recording is enabled
    fn proof_enabled(&self) -> bool;

    /// Get the theory proof (if available)
    fn get_theory_proof(&self) -> Option<&TheoryProof>;

    /// Take the theory proof
    fn take_theory_proof(&mut self) -> Option<TheoryProof>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theory_step_id_display() {
        let id = TheoryStepId(42);
        assert_eq!(format!("{}", id), "t42");
    }

    #[test]
    fn test_theory_rule_display() {
        assert_eq!(format!("{}", TheoryRule::Refl), "refl");
        assert_eq!(format!("{}", TheoryRule::Trans), "trans");
        assert_eq!(format!("{}", TheoryRule::LaGeneric), "la_generic");
        assert_eq!(format!("{}", TheoryRule::ArrReadWrite1), "arr_read_write_1");
    }

    #[test]
    fn test_theory_proof_reflexivity() {
        let mut proof = TheoryProof::new();

        let step = proof.refl("x");
        assert_eq!(step.0, 0);

        let s = proof.get_step(step).unwrap();
        assert_eq!(s.conclusion.0, "(= x x)");
        assert_eq!(s.rule, TheoryRule::Refl);
        assert!(s.premises.is_empty());
    }

    #[test]
    fn test_theory_proof_transitivity() {
        let mut proof = TheoryProof::new();

        // Step 1: a = b (axiom)
        let s1 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");

        // Step 2: b = c (axiom)
        let s2 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= b c)");

        // Step 3: a = c (by transitivity)
        let s3 = proof.trans(s1, s2, "a", "c");

        let step = proof.get_step(s3).unwrap();
        assert_eq!(step.conclusion.0, "(= a c)");
        assert_eq!(step.rule, TheoryRule::Trans);
        assert_eq!(step.premises.len(), 2);
    }

    #[test]
    fn test_theory_proof_farkas() {
        let mut proof = TheoryProof::new();

        // x >= 10
        let s1 = proof.add_axiom(TheoryRule::Custom("bound".into()), "(>= x 10)");

        // x <= 5
        let s2 = proof.add_axiom(TheoryRule::Custom("bound".into()), "(<= x 5)");

        // Conflict by Farkas lemma with coefficients [1, 1]
        let s3 = proof.farkas(
            vec![s1, s2],
            &[ProofTerm("1".into()), ProofTerm("1".into())],
        );

        let step = proof.get_step(s3).unwrap();
        assert_eq!(step.conclusion.0, "false");
        assert_eq!(step.rule, TheoryRule::LaGeneric);
    }

    #[test]
    fn test_theory_proof_array_read_write() {
        let mut proof = TheoryProof::new();

        let step = proof.read_write_same("a", "i", "v");

        let s = proof.get_step(step).unwrap();
        assert!(s.conclusion.0.contains("select"));
        assert!(s.conclusion.0.contains("store"));
        assert_eq!(s.rule, TheoryRule::ArrReadWrite1);
    }

    #[test]
    fn test_euf_proof_recorder() {
        let mut recorder = EufProofRecorder::new();

        let s1 = recorder.record_equality(0, 1, "(= a b)");
        let s2 = recorder.record_equality(1, 2, "(= b c)");
        let _s3 = recorder.record_transitivity(vec![s1, s2], "(= a c)");

        assert_eq!(recorder.proof().len(), 3);
    }

    #[test]
    fn test_arith_proof_recorder() {
        let mut recorder = ArithProofRecorder::new();

        recorder.record_bound("(>= x 10)");
        recorder.record_bound("(<= x 5)");
        let conflict = recorder.record_farkas_conflict(&[0, 1]);

        let step = recorder.proof().get_step(conflict).unwrap();
        assert_eq!(step.conclusion.0, "false");
    }

    #[test]
    fn test_theory_proof_clear() {
        let mut proof = TheoryProof::new();
        proof.refl("x");
        proof.refl("y");

        assert_eq!(proof.len(), 2);
        proof.clear();
        assert!(proof.is_empty());
    }

    #[test]
    fn test_theory_proof_write() {
        let mut proof = TheoryProof::new();
        proof.refl("x");

        let output = proof.to_string_repr();
        assert!(output.contains("Theory proof"));
        assert!(output.contains("(= x x)"));
    }

    #[test]
    fn test_quantifier_forall_elim() {
        let mut proof = TheoryProof::new();

        // ∀x. P(x)
        let forall_step = proof.add_axiom(TheoryRule::Custom("axiom".into()), "(forall x (P x))");

        // ∀x. P(x) ⊢ P(5)
        let inst_step = proof.forall_elim(forall_step, "x", "5", "(P 5)");

        let step = proof.get_step(inst_step).unwrap();
        assert_eq!(step.rule, TheoryRule::ForallElim);
        assert_eq!(step.premises.len(), 1);
        assert_eq!(step.args.len(), 2);
    }

    #[test]
    fn test_quantifier_exists_intro() {
        let mut proof = TheoryProof::new();

        // P(5)
        let concrete = proof.add_axiom(TheoryRule::Custom("axiom".into()), "(P 5)");

        // P(5) ⊢ ∃x. P(x)
        let exists_step = proof.exists_intro(concrete, "5", "(exists x (P x))");

        let step = proof.get_step(exists_step).unwrap();
        assert_eq!(step.rule, TheoryRule::ExistsIntro);
        assert_eq!(step.args.len(), 1);
    }

    #[test]
    fn test_quantifier_skolemization() {
        let mut proof = TheoryProof::new();

        // ∃x. P(x)
        let exists_step = proof.add_axiom(TheoryRule::Custom("axiom".into()), "(exists x (P x))");

        // ∃x. P(x) ⊢ P(sk_1)
        let skolem_step = proof.skolemize(exists_step, "sk_1", "(P sk_1)");

        let step = proof.get_step(skolem_step).unwrap();
        assert_eq!(step.rule, TheoryRule::Skolemize);
        assert!(step.conclusion.0.contains("sk_1"));
    }

    #[test]
    fn test_quantifier_inst_pattern() {
        let mut proof = TheoryProof::new();

        // ∀x. (P x) => (Q x)
        let forall_step = proof.add_axiom(
            TheoryRule::Custom("axiom".into()),
            "(forall x (=> (P x) (Q x)))",
        );

        // Instantiate with pattern matching
        let inst_step = proof.quant_inst(
            forall_step,
            "(P x)",
            vec![ProofTerm("a".into())],
            "(=> (P a) (Q a))",
        );

        let step = proof.get_step(inst_step).unwrap();
        assert_eq!(step.rule, TheoryRule::QuantInst);
        assert!(!step.args.is_empty());
    }

    #[test]
    fn test_quantifier_rule_display() {
        assert_eq!(format!("{}", TheoryRule::ForallElim), "forall_elim");
        assert_eq!(format!("{}", TheoryRule::ExistsIntro), "exists_intro");
        assert_eq!(format!("{}", TheoryRule::Skolemize), "skolemize");
        assert_eq!(format!("{}", TheoryRule::QuantInst), "quant_inst");
    }
}
