//! Alethe proof format for SMT proofs.
//!
//! Alethe is the emerging standard for SMT proof output,
//! designed to be checkable by proof assistants and verified tools.
//!
//! ## Format
//!
//! Alethe proofs consist of:
//! - **Assume steps**: Introduce assertions from the original problem
//! - **Step steps**: Apply proof rules to derive new conclusions
//! - **Anchor steps**: Define local scopes for subproofs
//!
//! ## Proof Rules
//!
//! Alethe supports various proof rules including:
//! - SAT rules (resolution, unit propagation)
//! - Theory rules (arithmetic, arrays, etc.)
//! - Structural rules (scope, subproof)

use std::fmt;
use std::io::{self, Write};

/// An index to a proof step
pub type StepIndex = u32;

/// A term reference (using SMT-LIB term representation)
pub type TermRef = String;

/// An Alethe proof step
#[derive(Debug, Clone)]
pub enum AletheStep {
    /// Assume: introduce a premise from the original problem
    /// `(assume <index> <term>)`
    Assume { index: StepIndex, term: TermRef },

    /// Step: apply a proof rule
    /// `(step <index> <clause> :rule <rule> [:premises (<premise>*)] [:args (<arg>*)])`
    Step {
        index: StepIndex,
        clause: Vec<TermRef>,
        rule: AletheRule,
        premises: Vec<StepIndex>,
        args: Vec<TermRef>,
    },

    /// Anchor: define a local scope for subproofs
    /// `(anchor :step <index> [:args (<arg>*)])`
    Anchor {
        step: StepIndex,
        args: Vec<(String, TermRef)>,
    },

    /// Define: define a local function
    /// `(define-fun <name> (<args>*) <sort> <term>)`
    DefineFun {
        name: String,
        args: Vec<(String, TermRef)>,
        return_sort: TermRef,
        body: TermRef,
    },
}

/// Alethe proof rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AletheRule {
    // SAT Rules
    /// Assumption
    Assume,
    /// Boolean resolution
    Resolution,
    /// Transitivity of equality
    Trans,
    /// Congruence
    Cong,
    /// Reflexivity
    Refl,
    /// Symmetry
    Symm,
    /// Negation elimination
    NotNot,
    /// And elimination
    AndPos,
    /// And negative
    AndNeg,
    /// Or positive
    OrPos,
    /// Or elimination
    OrNeg,
    /// Implication positive 1
    ImpliesPos1,
    /// Implication positive 2
    ImpliesPos2,
    /// Implication negative 1
    ImpliesNeg1,
    /// Implication negative 2
    ImpliesNeg2,
    /// Equivalence positive 1
    EquivPos1,
    /// Equivalence positive 2
    EquivPos2,
    /// Equivalence negative 1
    EquivNeg1,
    /// Equivalence negative 2
    EquivNeg2,
    /// ITE positive 1
    ItePos1,
    /// ITE positive 2
    ItePos2,
    /// ITE negative 1
    IteNeg1,
    /// ITE negative 2
    IteNeg2,
    /// XOR positive 1
    XorPos1,
    /// XOR positive 2
    XorPos2,
    /// XOR negative 1
    XorNeg1,
    /// XOR negative 2
    XorNeg2,

    // Equality Rules
    /// Equality reflexivity
    EqRefl,
    /// Equality symmetry
    EqSymm,
    /// Equality transitivity
    EqTrans,
    /// Equality congruence
    EqCong,

    // Arithmetic Rules
    /// Linear arithmetic
    LaGeneric,
    /// Disequality
    LaDisequality,
    /// Totality
    LaTotality,
    /// Tightening
    LaTightening,

    // Array Rules
    /// Array read-over-write same
    ArrayRowSame,
    /// Array read-over-write different
    ArrayRowDiff,
    /// Array extensionality
    ArrayExt,

    // Quantifier Rules
    /// Skolemization
    Skolem,
    /// Forall instantiation
    ForallInst,
    /// Exists introduction
    ExistsIntro,

    // Structural Rules
    /// True introduction
    True,
    /// False elimination
    False,
    /// Contraction (duplicate literal removal)
    Contraction,
    /// Let substitution
    Let,
    /// Bind (scope)
    Bind,

    // Theory-specific
    /// Theory axiom
    ThLemma,
    /// Theory resolution
    ThResolution,

    // Input/Output
    /// Input assertion
    Input,
    /// Subproof
    Subproof,
}

impl fmt::Display for AletheRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Assume => "assume",
            Self::Resolution => "resolution",
            Self::Trans => "trans",
            Self::Cong => "cong",
            Self::Refl => "refl",
            Self::Symm => "symm",
            Self::NotNot => "not_not",
            Self::AndPos => "and_pos",
            Self::AndNeg => "and_neg",
            Self::OrPos => "or_pos",
            Self::OrNeg => "or_neg",
            Self::ImpliesPos1 => "implies_pos1",
            Self::ImpliesPos2 => "implies_pos2",
            Self::ImpliesNeg1 => "implies_neg1",
            Self::ImpliesNeg2 => "implies_neg2",
            Self::EquivPos1 => "equiv_pos1",
            Self::EquivPos2 => "equiv_pos2",
            Self::EquivNeg1 => "equiv_neg1",
            Self::EquivNeg2 => "equiv_neg2",
            Self::ItePos1 => "ite_pos1",
            Self::ItePos2 => "ite_pos2",
            Self::IteNeg1 => "ite_neg1",
            Self::IteNeg2 => "ite_neg2",
            Self::XorPos1 => "xor_pos1",
            Self::XorPos2 => "xor_pos2",
            Self::XorNeg1 => "xor_neg1",
            Self::XorNeg2 => "xor_neg2",
            Self::EqRefl => "eq_refl",
            Self::EqSymm => "eq_symm",
            Self::EqTrans => "eq_trans",
            Self::EqCong => "eq_cong",
            Self::LaGeneric => "la_generic",
            Self::LaDisequality => "la_disequality",
            Self::LaTotality => "la_totality",
            Self::LaTightening => "la_tightening",
            Self::ArrayRowSame => "row_same",
            Self::ArrayRowDiff => "row_diff",
            Self::ArrayExt => "ext",
            Self::Skolem => "skolem",
            Self::ForallInst => "forall_inst",
            Self::ExistsIntro => "exists_intro",
            Self::True => "true",
            Self::False => "false",
            Self::Contraction => "contraction",
            Self::Let => "let",
            Self::Bind => "bind",
            Self::ThLemma => "th_lemma",
            Self::ThResolution => "th_resolution",
            Self::Input => "input",
            Self::Subproof => "subproof",
        };
        write!(f, "{}", name)
    }
}

/// An Alethe proof
#[derive(Debug, Default)]
pub struct AletheProof {
    /// Proof steps
    steps: Vec<AletheStep>,
    /// Next available step index
    next_index: StepIndex,
}

impl AletheProof {
    /// Create a new empty Alethe proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            next_index: 1,
        }
    }

    /// Add an assumption step
    pub fn assume(&mut self, term: impl Into<TermRef>) -> StepIndex {
        let index = self.next_index;
        self.next_index += 1;
        self.steps.push(AletheStep::Assume {
            index,
            term: term.into(),
        });
        index
    }

    /// Add a proof step
    pub fn step(
        &mut self,
        clause: Vec<TermRef>,
        rule: AletheRule,
        premises: Vec<StepIndex>,
        args: Vec<TermRef>,
    ) -> StepIndex {
        let index = self.next_index;
        self.next_index += 1;
        self.steps.push(AletheStep::Step {
            index,
            clause,
            rule,
            premises,
            args,
        });
        index
    }

    /// Add a step with no premises or arguments
    pub fn step_simple(&mut self, clause: Vec<TermRef>, rule: AletheRule) -> StepIndex {
        self.step(clause, rule, Vec::new(), Vec::new())
    }

    /// Add a resolution step
    pub fn resolution(&mut self, clause: Vec<TermRef>, premises: Vec<StepIndex>) -> StepIndex {
        self.step(clause, AletheRule::Resolution, premises, Vec::new())
    }

    /// Add an anchor (scope) step
    pub fn anchor(&mut self, args: Vec<(String, TermRef)>) -> StepIndex {
        let step = self.next_index;
        self.next_index += 1;
        self.steps.push(AletheStep::Anchor { step, args });
        step
    }

    /// Add a define-fun step
    pub fn define_fun(
        &mut self,
        name: impl Into<String>,
        args: Vec<(String, TermRef)>,
        return_sort: impl Into<TermRef>,
        body: impl Into<TermRef>,
    ) {
        self.steps.push(AletheStep::DefineFun {
            name: name.into(),
            args,
            return_sort: return_sort.into(),
            body: body.into(),
        });
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Get the proof steps
    #[must_use]
    pub fn steps(&self) -> &[AletheStep] {
        &self.steps
    }

    /// Clear all proof steps
    pub fn clear(&mut self) {
        self.steps.clear();
        self.next_index = 1;
    }

    /// Write the proof in Alethe format
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "; Alethe proof generated by OxiZ")?;
        writeln!(writer)?;

        for step in &self.steps {
            match step {
                AletheStep::Assume { index, term } => {
                    writeln!(writer, "(assume t{} {})", index, term)?;
                }
                AletheStep::Step {
                    index,
                    clause,
                    rule,
                    premises,
                    args,
                } => {
                    write!(writer, "(step t{} (cl", index)?;
                    for lit in clause {
                        write!(writer, " {}", lit)?;
                    }
                    write!(writer, ") :rule {}", rule)?;

                    if !premises.is_empty() {
                        write!(writer, " :premises (")?;
                        for (i, &p) in premises.iter().enumerate() {
                            if i > 0 {
                                write!(writer, " ")?;
                            }
                            write!(writer, "t{}", p)?;
                        }
                        write!(writer, ")")?;
                    }

                    if !args.is_empty() {
                        write!(writer, " :args (")?;
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 {
                                write!(writer, " ")?;
                            }
                            write!(writer, "{}", arg)?;
                        }
                        write!(writer, ")")?;
                    }

                    writeln!(writer, ")")?;
                }
                AletheStep::Anchor { step, args } => {
                    write!(writer, "(anchor :step t{}", step)?;
                    if !args.is_empty() {
                        write!(writer, " :args (")?;
                        for (i, (name, sort)) in args.iter().enumerate() {
                            if i > 0 {
                                write!(writer, " ")?;
                            }
                            write!(writer, "({} {})", name, sort)?;
                        }
                        write!(writer, ")")?;
                    }
                    writeln!(writer, ")")?;
                }
                AletheStep::DefineFun {
                    name,
                    args,
                    return_sort,
                    body,
                } => {
                    write!(writer, "(define-fun {} (", name)?;
                    for (i, (arg_name, arg_sort)) in args.iter().enumerate() {
                        if i > 0 {
                            write!(writer, " ")?;
                        }
                        write!(writer, "({} {})", arg_name, arg_sort)?;
                    }
                    writeln!(writer, ") {} {})", return_sort, body)?;
                }
            }
        }

        Ok(())
    }

    /// Convert to string
    #[must_use]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let mut buf = Vec::new();
        self.write(&mut buf)
            .expect("writing to Vec should not fail");
        String::from_utf8(buf).expect("Alethe output is UTF-8")
    }
}

/// Trait for solvers that can produce Alethe proofs
pub trait AletheProofProducer {
    /// Enable Alethe proof production
    fn enable_alethe_proof(&mut self);

    /// Disable Alethe proof production
    fn disable_alethe_proof(&mut self);

    /// Get the Alethe proof (if available)
    fn get_alethe_proof(&self) -> Option<&AletheProof>;

    /// Take the Alethe proof, leaving None
    fn take_alethe_proof(&mut self) -> Option<AletheProof>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alethe_assume() {
        let mut proof = AletheProof::new();
        let idx = proof.assume("(= x 5)");

        assert_eq!(idx, 1);
        assert_eq!(proof.len(), 1);

        let output = proof.to_string();
        assert!(output.contains("(assume t1 (= x 5))"));
    }

    #[test]
    fn test_alethe_step() {
        let mut proof = AletheProof::new();
        let a1 = proof.assume("(or p q)");
        let a2 = proof.assume("(not p)");

        let s1 = proof.step(
            vec!["q".to_string()],
            AletheRule::Resolution,
            vec![a1, a2],
            Vec::new(),
        );

        assert_eq!(s1, 3);
        assert_eq!(proof.len(), 3);

        let output = proof.to_string();
        assert!(output.contains(":rule resolution"));
        assert!(output.contains(":premises (t1 t2)"));
    }

    #[test]
    fn test_alethe_empty_clause() {
        let mut proof = AletheProof::new();
        let a1 = proof.assume("p");
        let a2 = proof.assume("(not p)");

        // Empty clause (contradiction)
        proof.step(Vec::new(), AletheRule::Resolution, vec![a1, a2], Vec::new());

        let output = proof.to_string();
        assert!(output.contains("(cl)"));
    }

    #[test]
    fn test_alethe_theory_lemma() {
        let mut proof = AletheProof::new();

        // Theory lemma: x < 5 and x > 10 is unsatisfiable
        proof.step(
            vec!["(not (< x 5))".to_string(), "(not (> x 10))".to_string()],
            AletheRule::LaGeneric,
            Vec::new(),
            Vec::new(),
        );

        let output = proof.to_string();
        assert!(output.contains(":rule la_generic"));
    }

    #[test]
    fn test_alethe_anchor() {
        let mut proof = AletheProof::new();

        let idx = proof.anchor(vec![("x".to_string(), "Int".to_string())]);

        assert_eq!(idx, 1);
        let output = proof.to_string();
        assert!(output.contains("(anchor :step t1 :args ((x Int)))"));
    }

    #[test]
    fn test_alethe_define_fun() {
        let mut proof = AletheProof::new();

        proof.define_fun(
            "f",
            vec![("x".to_string(), "Int".to_string())],
            "Int",
            "(+ x 1)",
        );

        let output = proof.to_string();
        assert!(output.contains("(define-fun f ((x Int)) Int (+ x 1))"));
    }

    #[test]
    fn test_alethe_rule_display() {
        assert_eq!(format!("{}", AletheRule::Resolution), "resolution");
        assert_eq!(format!("{}", AletheRule::LaGeneric), "la_generic");
        assert_eq!(format!("{}", AletheRule::EqTrans), "eq_trans");
    }

    #[test]
    fn test_alethe_clear() {
        let mut proof = AletheProof::new();
        proof.assume("p");
        proof.assume("q");

        assert_eq!(proof.len(), 2);
        proof.clear();
        assert!(proof.is_empty());

        // Next index should reset
        let idx = proof.assume("r");
        assert_eq!(idx, 1);
    }
}
