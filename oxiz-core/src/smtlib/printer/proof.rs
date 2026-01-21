//! Proof printing functionality

use crate::ast::{proof::Proof, proof::ProofRule};
use std::fmt::Write;

use super::basic::Printer;

impl<'a> Printer<'a> {
    /// Print a proof in SMT-LIB2 format
    pub fn print_proof(&self, proof: &Proof) -> String {
        let mut buf = String::new();
        self.write_proof(&mut buf, proof);
        buf
    }

    /// Write a proof in SMT-LIB2 format
    ///
    /// This outputs the proof as a tree of inference steps in a readable format.
    pub fn write_proof(&self, w: &mut impl Write, proof: &Proof) {
        let _ = writeln!(w, "(proof");

        // Get the root node and recursively write the proof tree
        let root_id = proof.root();
        self.write_proof_node(w, proof, root_id, 1);

        let _ = writeln!(w, ")");
    }

    /// Write a single proof node recursively
    fn write_proof_node(
        &self,
        w: &mut impl Write,
        proof: &Proof,
        node_id: crate::ast::proof::ProofId,
        indent: usize,
    ) {
        let Some(node) = proof.get_node(node_id) else {
            return;
        };

        let indent_str = "  ".repeat(indent);

        // Write the proof step
        let _ = write!(w, "{}", indent_str);
        let _ = write!(w, "(step @p{} ", node_id.0);

        // Write the rule
        self.write_proof_rule(w, &node.rule);

        // Write the conclusion
        let _ = write!(w, "\n{}  :conclusion ", indent_str);
        self.write_term(w, node.conclusion);

        // Write premises if any
        if !node.premises.is_empty() {
            let _ = write!(w, "\n{}  :premises (", indent_str);
            for (i, premise_id) in node.premises.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let _ = write!(w, "@p{}", premise_id.0);
            }
            let _ = write!(w, ")");
        }

        // Write metadata if any
        if !node.metadata.is_empty() {
            let _ = write!(w, "\n{}  :metadata (", indent_str);
            for (i, (key, value)) in node.metadata.iter().enumerate() {
                if i > 0 {
                    let _ = write!(w, " ");
                }
                let _ = write!(w, ":{} \"{}\"", key, value);
            }
            let _ = write!(w, ")");
        }

        let _ = writeln!(w, ")");

        // Recursively write premises
        for premise_id in &node.premises {
            self.write_proof_node(w, proof, *premise_id, indent + 1);
        }
    }

    /// Write a proof rule
    fn write_proof_rule(&self, w: &mut impl Write, rule: &ProofRule) {
        match rule {
            ProofRule::Assume { name } => {
                if let Some(n) = name {
                    let _ = write!(w, ":rule assume :name \"{}\"", n);
                } else {
                    let _ = write!(w, ":rule assume");
                }
            }
            ProofRule::Resolution { pivot } => {
                let _ = write!(w, ":rule resolution :pivot ");
                self.write_term(w, *pivot);
            }
            ProofRule::ModusPonens => {
                let _ = write!(w, ":rule modus-ponens");
            }
            ProofRule::Tautology => {
                let _ = write!(w, ":rule tautology");
            }
            ProofRule::ArithInequality => {
                let _ = write!(w, ":rule arith-inequality");
            }
            ProofRule::TheoryLemma { theory } => {
                let _ = write!(w, ":rule theory-lemma :theory \"{}\"", theory);
            }
            ProofRule::Contradiction => {
                let _ = write!(w, ":rule contradiction");
            }
            ProofRule::Rewrite => {
                let _ = write!(w, ":rule rewrite");
            }
            ProofRule::Substitution => {
                let _ = write!(w, ":rule substitution");
            }
            ProofRule::Symmetry => {
                let _ = write!(w, ":rule symmetry");
            }
            ProofRule::Transitivity => {
                let _ = write!(w, ":rule transitivity");
            }
            ProofRule::Congruence => {
                let _ = write!(w, ":rule congruence");
            }
            ProofRule::Reflexivity => {
                let _ = write!(w, ":rule reflexivity");
            }
            ProofRule::Instantiation { terms } => {
                let _ = write!(w, ":rule instantiation :terms (");
                for (i, term_id) in terms.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(w, " ");
                    }
                    self.write_term(w, *term_id);
                }
                let _ = write!(w, ")");
            }
            ProofRule::Custom { name } => {
                let _ = write!(w, ":rule custom :name \"{}\"", name);
            }
        }
    }
}

