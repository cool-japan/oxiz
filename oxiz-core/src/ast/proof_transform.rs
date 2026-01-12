//! Proof transformation utilities
//!
//! Transform proofs between different formats:
//! - Resolution to Natural Deduction
//! - Proof minimization and optimization
//! - Certificate generation for external verifiers
//!
//! Reference: Z3's proof infrastructure and Isabelle/HOL proof formats

use super::proof::{Proof, ProofId, ProofRule};
use crate::ast::{TermId, TermManager};
use rustc_hash::FxHashMap;

#[cfg(test)]
use super::proof::ProofNode;

/// Natural deduction proof rule
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NaturalDeductionRule {
    /// Assumption introduction
    Assume,
    /// Implication introduction (⟹-intro)
    ImpliesIntro {
        /// The hypothesis being discharged
        hypothesis: TermId,
    },
    /// Implication elimination (Modus Ponens)
    ImpliesElim,
    /// And introduction
    AndIntro,
    /// And elimination (left)
    AndElimLeft,
    /// And elimination (right)
    AndElimRight,
    /// Or introduction (left)
    OrIntroLeft,
    /// Or introduction (right)
    OrIntroRight,
    /// Or elimination (case analysis)
    OrElim {
        /// Left case hypothesis
        left_hyp: TermId,
        /// Right case hypothesis
        right_hyp: TermId,
    },
    /// Not introduction
    NotIntro {
        /// The hypothesis being discharged
        hypothesis: TermId,
    },
    /// Not elimination
    NotElim,
    /// Bottom elimination (ex falso quodlibet)
    BottomElim,
    /// Reflexivity of equality
    EqRefl,
    /// Symmetry of equality
    EqSymm,
    /// Transitivity of equality
    EqTrans,
    /// Congruence
    Congruence,
    /// Theory axiom
    TheoryAxiom {
        /// Theory name
        theory: String,
    },
}

/// Natural deduction proof node
#[derive(Debug, Clone)]
pub struct NaturalDeductionNode {
    /// Unique ID
    pub id: ProofId,
    /// The rule used
    pub rule: NaturalDeductionRule,
    /// The conclusion
    pub conclusion: TermId,
    /// Premises
    pub premises: Vec<ProofId>,
}

/// Natural deduction proof
#[derive(Debug, Clone)]
pub struct NaturalDeductionProof {
    /// All nodes
    nodes: FxHashMap<ProofId, NaturalDeductionNode>,
    /// Root node
    root: ProofId,
    /// Next ID
    next_id: u64,
}

impl NaturalDeductionProof {
    /// Create a new empty natural deduction proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            root: ProofId(0),
            next_id: 0,
        }
    }

    /// Generate a new unique proof ID
    pub fn new_id(&mut self) -> ProofId {
        let id = ProofId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a node
    pub fn add_node(&mut self, node: NaturalDeductionNode) {
        self.nodes.insert(node.id, node);
    }

    /// Get a node
    #[must_use]
    pub fn get_node(&self, id: ProofId) -> Option<&NaturalDeductionNode> {
        self.nodes.get(&id)
    }

    /// Set the root
    pub fn set_root(&mut self, root: ProofId) {
        self.root = root;
    }

    /// Get the root
    #[must_use]
    pub fn root(&self) -> ProofId {
        self.root
    }

    /// Get the number of nodes
    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for NaturalDeductionProof {
    fn default() -> Self {
        Self::new()
    }
}

/// Transform a resolution proof to natural deduction
pub fn resolution_to_natural_deduction(
    proof: &Proof,
    manager: &TermManager,
) -> NaturalDeductionProof {
    let mut nd_proof = NaturalDeductionProof::new();
    let mut node_map = FxHashMap::default();

    // Convert each resolution node to natural deduction
    let root = proof.root();
    transform_node(proof, root, &mut nd_proof, &mut node_map, manager);

    nd_proof.set_root(*node_map.get(&root).unwrap_or(&ProofId(0)));
    nd_proof
}

/// Recursively transform a proof node
fn transform_node(
    proof: &Proof,
    node_id: ProofId,
    nd_proof: &mut NaturalDeductionProof,
    node_map: &mut FxHashMap<ProofId, ProofId>,
    _manager: &TermManager,
) -> ProofId {
    // Check if already transformed
    if let Some(&nd_id) = node_map.get(&node_id) {
        return nd_id;
    }

    let node = match proof.get_node(node_id) {
        Some(n) => n,
        None => return ProofId(0),
    };

    // Transform premises first
    let nd_premises: Vec<_> = node
        .premises
        .iter()
        .map(|&p| transform_node(proof, p, nd_proof, node_map, _manager))
        .collect();

    // Convert the rule
    let nd_rule = match &node.rule {
        ProofRule::Assume { .. } => NaturalDeductionRule::Assume,

        ProofRule::Resolution { .. } => {
            // Resolution can be viewed as Or-elimination followed by unit propagation
            // For simplicity, we map it to ImpliesElim
            NaturalDeductionRule::ImpliesElim
        }

        ProofRule::ModusPonens => NaturalDeductionRule::ImpliesElim,

        ProofRule::Contradiction => NaturalDeductionRule::BottomElim,

        ProofRule::Rewrite | ProofRule::Substitution => {
            // Rewrites are combinations of equational reasoning
            NaturalDeductionRule::EqTrans
        }

        ProofRule::Reflexivity => NaturalDeductionRule::EqRefl,

        ProofRule::Symmetry => NaturalDeductionRule::EqSymm,

        ProofRule::Transitivity => NaturalDeductionRule::EqTrans,

        ProofRule::Congruence => NaturalDeductionRule::Congruence,

        ProofRule::TheoryLemma { theory } => NaturalDeductionRule::TheoryAxiom {
            theory: theory.clone(),
        },

        ProofRule::ArithInequality => NaturalDeductionRule::TheoryAxiom {
            theory: "Arithmetic".to_string(),
        },

        ProofRule::Tautology => NaturalDeductionRule::TheoryAxiom {
            theory: "Boolean".to_string(),
        },

        ProofRule::Instantiation { .. } => {
            // Quantifier instantiation is an axiom in natural deduction
            NaturalDeductionRule::TheoryAxiom {
                theory: "Quantifier".to_string(),
            }
        }

        ProofRule::Custom { name } => NaturalDeductionRule::TheoryAxiom {
            theory: name.clone(),
        },
    };

    // Create the natural deduction node
    let nd_id = nd_proof.new_id();
    let nd_node = NaturalDeductionNode {
        id: nd_id,
        rule: nd_rule,
        conclusion: node.conclusion,
        premises: nd_premises,
    };

    nd_proof.add_node(nd_node);
    node_map.insert(node_id, nd_id);

    nd_id
}

/// Proof certificate formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificateFormat {
    /// LFSC format (used by CVC4/CVC5)
    LFSC,
    /// Alethe format (used by veriT, CVC5)
    Alethe,
    /// DRAT format (for SAT proofs)
    DRAT,
    /// Coq proof script
    Coq,
    /// Isabelle/HOL proof
    Isabelle,
}

/// Generate a proof certificate for external verification
pub fn generate_certificate(
    proof: &Proof,
    format: CertificateFormat,
    _manager: &TermManager,
) -> String {
    match format {
        CertificateFormat::LFSC => generate_lfsc_certificate(proof),
        CertificateFormat::Alethe => generate_alethe_certificate(proof),
        CertificateFormat::DRAT => generate_drat_certificate(proof),
        CertificateFormat::Coq => generate_coq_certificate(proof),
        CertificateFormat::Isabelle => generate_isabelle_certificate(proof),
    }
}

/// Generate LFSC certificate
fn generate_lfsc_certificate(proof: &Proof) -> String {
    let mut output = String::new();
    output.push_str("; LFSC proof certificate\n");
    output.push_str("; Generated by OxiZ\n\n");

    // Header
    output.push_str("(check\n");

    // Traverse proof and emit LFSC commands
    for node in proof.nodes().values() {
        output.push_str(&format!("; Step {:?}\n", node.id));
        match &node.rule {
            ProofRule::Assume { name } => {
                output.push_str(&format!(
                    "(% {} (holds cl{:?})\n",
                    name.as_deref().unwrap_or("H"),
                    node.conclusion
                ));
            }
            ProofRule::Resolution { .. } => {
                output.push_str(&format!("(R _ _ _ cl{:?})\n", node.conclusion));
            }
            _ => {
                output.push_str(&format!("; Rule: {:?}\n", node.rule));
            }
        }
    }

    output.push_str(")\n");
    output
}

/// Generate Alethe certificate
fn generate_alethe_certificate(proof: &Proof) -> String {
    let mut output = String::new();
    output.push_str("; Alethe proof certificate\n");
    output.push_str("; Generated by OxiZ\n\n");

    for node in proof.nodes().values() {
        let rule_name = match &node.rule {
            ProofRule::Assume { .. } => "assume",
            ProofRule::Resolution { .. } => "resolution",
            ProofRule::ModusPonens => "implies_elim",
            ProofRule::Contradiction => "false",
            ProofRule::Rewrite => "rewrite",
            ProofRule::Reflexivity => "refl",
            ProofRule::Symmetry => "symm",
            ProofRule::Transitivity => "trans",
            ProofRule::Congruence => "cong",
            _ => "other",
        };

        output.push_str(&format!(
            "(step t{} (cl t{:?}) :rule {}\n",
            node.id.0, node.conclusion, rule_name
        ));

        if !node.premises.is_empty() {
            output.push_str(" :premises (");
            for (i, premise) in node.premises.iter().enumerate() {
                if i > 0 {
                    output.push(' ');
                }
                output.push_str(&format!("t{}", premise.0));
            }
            output.push(')');
        }

        output.push_str(")\n");
    }

    output
}

/// Generate DRAT certificate (for SAT proofs)
fn generate_drat_certificate(proof: &Proof) -> String {
    let mut output = String::new();
    output.push_str("c DRAT proof certificate\n");
    output.push_str("c Generated by OxiZ\n\n");

    for node in proof.nodes().values() {
        if let ProofRule::Resolution { .. } = &node.rule {
            // DRAT format: clause additions and deletions
            output.push_str(&format!("{:?} 0\n", node.conclusion));
        }
    }

    output.push_str("0\n");
    output
}

/// Generate Coq proof script
fn generate_coq_certificate(proof: &Proof) -> String {
    let mut output = String::new();
    output.push_str("(* Coq proof certificate *)\n");
    output.push_str("(* Generated by OxiZ *)\n\n");

    output.push_str("Theorem proof_certificate : False.\n");
    output.push_str("Proof.\n");

    for node in proof.nodes().values() {
        match &node.rule {
            ProofRule::Assume { .. } => {
                output.push_str("  intro.\n");
            }
            ProofRule::ModusPonens => {
                output.push_str("  apply.\n");
            }
            ProofRule::Contradiction => {
                output.push_str("  contradiction.\n");
            }
            ProofRule::Reflexivity => {
                output.push_str("  reflexivity.\n");
            }
            ProofRule::Symmetry => {
                output.push_str("  symmetry.\n");
            }
            ProofRule::Transitivity => {
                output.push_str("  etransitivity.\n");
            }
            _ => {}
        }
    }

    output.push_str("Qed.\n");
    output
}

/// Generate Isabelle/HOL proof
fn generate_isabelle_certificate(proof: &Proof) -> String {
    let mut output = String::new();
    output.push_str("(* Isabelle/HOL proof certificate *)\n");
    output.push_str("(* Generated by OxiZ *)\n\n");

    output.push_str("lemma proof_certificate:\n");
    output.push_str("  shows \"False\"\n");
    output.push_str("proof -\n");

    for node in proof.nodes().values() {
        match &node.rule {
            ProofRule::Assume { name } => {
                output.push_str(&format!(
                    "  assume {}: \"term{:?}\"\n",
                    name.as_deref().unwrap_or("H"),
                    node.conclusion
                ));
            }
            ProofRule::Resolution { .. } => {
                output.push_str("  from this show ?thesis by auto\n");
            }
            _ => {}
        }
    }

    output.push_str("qed\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::proof::ProofRule;

    #[test]
    fn test_empty_nd_proof() {
        let proof = NaturalDeductionProof::new();
        assert_eq!(proof.size(), 0);
    }

    #[test]
    fn test_nd_proof_add_node() {
        let mut proof = NaturalDeductionProof::new();
        let id = proof.new_id();

        let node = NaturalDeductionNode {
            id,
            rule: NaturalDeductionRule::Assume,
            conclusion: TermId(1),
            premises: Vec::new(),
        };

        proof.add_node(node);
        assert_eq!(proof.size(), 1);
    }

    #[test]
    fn test_resolution_to_nd_empty() {
        let res_proof = Proof::new();
        let manager = TermManager::new();

        let nd_proof = resolution_to_natural_deduction(&res_proof, &manager);
        assert_eq!(nd_proof.size(), 0);
    }

    #[test]
    fn test_resolution_to_nd_simple() {
        let mut res_proof = Proof::new();
        let manager = TermManager::new();

        // Create a simple proof with one assumption
        let node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        res_proof.add_node(node);
        res_proof.set_root(ProofId(0));

        let nd_proof = resolution_to_natural_deduction(&res_proof, &manager);
        assert_eq!(nd_proof.size(), 1);

        let nd_node = nd_proof.get_node(ProofId(0)).unwrap();
        assert_eq!(nd_node.rule, NaturalDeductionRule::Assume);
    }

    #[test]
    fn test_generate_lfsc_certificate() {
        let proof = Proof::new();
        let cert = generate_lfsc_certificate(&proof);
        assert!(cert.contains("LFSC"));
        assert!(cert.contains("OxiZ"));
    }

    #[test]
    fn test_generate_alethe_certificate() {
        let proof = Proof::new();
        let cert = generate_alethe_certificate(&proof);
        assert!(cert.contains("Alethe"));
        assert!(cert.contains("OxiZ"));
    }

    #[test]
    fn test_generate_drat_certificate() {
        let proof = Proof::new();
        let cert = generate_drat_certificate(&proof);
        assert!(cert.contains("DRAT"));
    }

    #[test]
    fn test_generate_coq_certificate() {
        let proof = Proof::new();
        let cert = generate_coq_certificate(&proof);
        assert!(cert.contains("Coq"));
        assert!(cert.contains("Theorem"));
    }

    #[test]
    fn test_generate_isabelle_certificate() {
        let proof = Proof::new();
        let cert = generate_isabelle_certificate(&proof);
        assert!(cert.contains("Isabelle"));
        assert!(cert.contains("lemma"));
    }

    #[test]
    fn test_certificate_format_variants() {
        let proof = Proof::new();
        let manager = TermManager::new();

        // Test all formats
        for format in [
            CertificateFormat::LFSC,
            CertificateFormat::Alethe,
            CertificateFormat::DRAT,
            CertificateFormat::Coq,
            CertificateFormat::Isabelle,
        ] {
            let cert = generate_certificate(&proof, format, &manager);
            assert!(!cert.is_empty());
        }
    }
}
