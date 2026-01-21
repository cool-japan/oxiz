//! SMT-LIB2 Printer
//!
//! This module provides two printers:
//! - [`Printer`]: A basic printer that outputs terms on a single line
//! - [`PrettyPrinter`]: A configurable pretty printer with indentation support

mod config;
mod pretty;
mod basic;
mod model;
mod proof;

// Re-export public types
pub use basic::Printer;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermManager, model::Model, proof::{Proof, ProofRule}};
    use config::PrettyConfig;
    use pretty::PrettyPrinter;

    #[test]
    fn test_print_constants() {
        let manager = TermManager::new();
        let printer = Printer::new(&manager);

        assert_eq!(printer.print_term(manager.mk_true()), "true");
        assert_eq!(printer.print_term(manager.mk_false()), "false");
    }

    #[test]
    fn test_print_compound() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let printer = Printer::new(&manager);
        assert_eq!(printer.print_term(and), "(and x y)");
    }

    #[test]
    fn test_roundtrip() {
        let mut manager = TermManager::new();
        let input = "(and (or x y) (not z))";
        let term = crate::smtlib::parse_term(input, &mut manager).unwrap();

        let printer = Printer::new(&manager);
        let output = printer.print_term(term);

        // Note: Output might differ slightly due to canonicalization
        assert!(output.contains("and"));
        assert!(output.contains("or"));
        assert!(output.contains("not"));
    }

    // ==================== PrettyPrinter Tests ====================

    #[test]
    fn test_pretty_config_default() {
        let config = PrettyConfig::default();
        assert_eq!(config.indent_width, 2);
        assert_eq!(config.max_width, 80);
        assert!(!config.use_tabs);
        assert!(!config.print_sorts);
        assert_eq!(config.break_depth, 2);
    }

    #[test]
    fn test_pretty_config_compact() {
        let config = PrettyConfig::compact();
        assert_eq!(config.indent_width, 0);
        assert_eq!(config.max_width, usize::MAX);
        assert_eq!(config.break_depth, usize::MAX);
    }

    #[test]
    fn test_pretty_config_expanded() {
        let config = PrettyConfig::expanded();
        assert_eq!(config.max_width, 40);
        assert_eq!(config.break_depth, 1);
    }

    #[test]
    fn test_pretty_config_builder() {
        let config = PrettyConfig::default()
            .with_indent_width(4)
            .with_max_width(100)
            .with_tabs(true)
            .with_print_sorts(true)
            .with_break_depth(3);

        assert_eq!(config.indent_width, 4);
        assert_eq!(config.max_width, 100);
        assert!(config.use_tabs);
        assert!(config.print_sorts);
        assert_eq!(config.break_depth, 3);
    }

    #[test]
    fn test_pretty_printer_simple_term() {
        let manager = TermManager::new();
        let pretty = PrettyPrinter::new(&manager);

        let output = pretty.print_term(manager.mk_true());
        assert_eq!(output, "true");
    }

    #[test]
    fn test_pretty_printer_compound_term() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let and = manager.mk_and([x, y]);

        let pretty = PrettyPrinter::new(&manager);
        let output = pretty.print_term(and);
        assert_eq!(output, "(and x y)");
    }

    #[test]
    fn test_pretty_printer_compact() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z]);
        let prod = manager.mk_mul([sum, x]);

        let config = PrettyConfig::compact();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(prod);

        // Compact mode should not break lines
        assert!(!output.contains('\n'));
        assert!(output.contains("(* (+ x y z) x)"));
    }

    #[test]
    fn test_pretty_printer_expanded() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);
        let w = manager.mk_var("w", manager.sorts.int_sort);
        let sum = manager.mk_add([x, y, z, w]);

        let config = PrettyConfig::expanded();
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(sum);

        // Expanded mode with many terms should break lines
        // The exact format depends on the width calculation
        assert!(output.contains("+"));
        assert!(output.contains("x"));
        assert!(output.contains("y"));
        assert!(output.contains("z"));
        assert!(output.contains("w"));
    }

    #[test]
    fn test_pretty_printer_nested_ite() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let a = manager.mk_int(1);
        let b = manager.mk_int(2);
        let ite = manager.mk_ite(x, a, b);

        let config = PrettyConfig::default()
            .with_max_width(10)
            .with_break_depth(0);
        let pretty = PrettyPrinter::with_config(&manager, config);
        let output = pretty.print_term(ite);

        // Should break due to small max_width
        assert!(output.contains("ite"));
        assert!(output.contains("x"));
    }

    // ==================== Model Printing Tests ====================

    #[test]
    fn test_print_empty_model() {
        let manager = TermManager::new();
        let model = Model::new();
        let printer = Printer::new(&manager);

        let output = printer.print_model(&model);
        assert!(output.contains("(model"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_bool_assignment() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut model = Model::new();
        model.assign_bool(x, true);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun x () Bool true"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_model_with_int_assignment() {
        let mut manager = TermManager::new();
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_int(y, num_bigint::BigInt::from(42));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("define-fun y () Int 42"));
    }

    #[test]
    fn test_print_model_with_multiple_assignments() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let mut model = Model::new();
        model.assign_bool(x, false);
        model.assign_int(y, num_bigint::BigInt::from(10));

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("x"));
        assert!(output.contains("Bool"));
        assert!(output.contains("false"));
        assert!(output.contains("y"));
        assert!(output.contains("Int"));
        assert!(output.contains("10"));
    }

    #[test]
    fn test_print_model_with_bitvec_assignment() {
        let mut manager = TermManager::new();
        let bv_sort = manager.sorts.bitvec(8);
        let z = manager.mk_var("z", bv_sort);

        let mut model = Model::new();
        model.assign_bitvec(z, 0xFF, 8);

        let printer = Printer::new(&manager);
        let output = printer.print_model(&model);

        assert!(output.contains("(model"));
        assert!(output.contains("z"));
        assert!(output.contains("#xff"));
    }

    // ==================== Proof Printing Tests ====================

    #[test]
    fn test_print_empty_proof() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();
        let false_term = manager.mk_false();

        let root = ProofNode::new(ProofId(0), ProofRule::Contradiction, false_term);
        proof.add_node(root);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("contradiction"));
        assert!(output.contains(")"));
    }

    #[test]
    fn test_print_proof_with_assumption() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        let mut proof = Proof::new();
        let assume_node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            x,
        );
        proof.add_node(assume_node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("assume"));
        assert!(output.contains("H1"));
        assert!(output.contains("conclusion"));
    }

    #[test]
    fn test_print_proof_with_resolution() {
        use crate::ast::proof::*;

        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let p1 = manager.mk_var("p", manager.sorts.bool_sort);
        let p2 = manager.mk_var("q", manager.sorts.bool_sort);

        let mut proof = Proof::new();

        // Add premise nodes
        let node1 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("A1".to_string()),
            },
            p1,
        );
        let node2 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("A2".to_string()),
            },
            p2,
        );

        // Add resolution node
        let resolution_node = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: x },
            manager.mk_true(),
            vec![ProofId(0), ProofId(1)],
        );

        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(resolution_node);
        proof.set_root(ProofId(2));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("(proof"));
        assert!(output.contains("resolution"));
        assert!(output.contains("premises"));
        assert!(output.contains("@p0"));
        assert!(output.contains("@p1"));
    }

    #[test]
    fn test_print_proof_with_metadata() {
        use crate::ast::proof::*;

        let manager = TermManager::new();
        let mut proof = Proof::new();

        let mut node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            manager.mk_false(),
        );
        node.add_metadata("source".to_string(), "farkas".to_string());

        proof.add_node(node);
        proof.set_root(ProofId(0));

        let printer = Printer::new(&manager);
        let output = printer.print_proof(&proof);

        assert!(output.contains("theory-lemma"));
        assert!(output.contains("LIA"));
        assert!(output.contains("metadata"));
        assert!(output.contains("source"));
        assert!(output.contains("farkas"));
    }
}
