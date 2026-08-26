//! Deep-recursion regression tests for the proof-DAG walks and text parsers
//! converted from native recursion to explicit heap stacks.
//!
//! Every test runs on a thread with a deliberately small (128 KiB) stack: a
//! stack overflow aborts the whole process rather than failing a test, so
//! *returning at all* is the assertion. Semantic pins accompany each conversion
//! so the rewrite is proven behaviour-preserving, not merely non-crashing.

use oxiz_proof::craig::InterpolantTerm;
use oxiz_proof::rules::CnfValidator;
use oxiz_proof::theory::{TheoryProof, TheoryRule};
use oxiz_proof::traversal::{
    ConclusionCollector, NodeCounter, TraversalOrder, find_all_paths, topological_order, traverse,
};
use oxiz_proof::{
    CheckResult, CheckerConfig, DetailedProofStats, Proof, ProofChecker, ProofNodeId,
    ProofVisualizer, RuleValidation, VisualizationFormat,
};

/// Stack size for every deep test below (much smaller than the 8 MiB main
/// thread, so a surviving recursion would be caught rather than tolerated).
///
/// `SMALL_STACK` and the depth constants below are scaled *together*, on
/// purpose. What these tests actually pin is the ratio between them -- the
/// per-frame byte budget a still-recursive implementation would have to fit
/// into: `SMALL_STACK / PAREN_DEPTH` is about 21 bytes per frame and
/// `SMALL_STACK / DEPTH` about 10, both far below any real call frame, so a
/// native recursion cannot survive either. The pair was scaled down by 8x from
/// the original 1 MiB / 50_000 because several construction and rendering paths
/// are quadratic in the depth (see the cost table at the top of
/// `src/visualization.rs`: all three indent-by-depth formats emit O(depth^2)
/// bytes, and `AsciiTree` also *retains* O(depth^2) once the proof branches —
/// `write_json_node` used to retain O(depth^2) on any shape, and no longer
/// does); the 8x cut keeps the detection power identical at 1/64th of the
/// memory cost.
///
/// Never raise a depth here without raising `SMALL_STACK` by the same factor
/// (and vice versa): restoring depth 100_000 against a 128 KiB stack would
/// re-create the multi-gigabyte rendering that OOM-killed the test process.
const SMALL_STACK: usize = 1 << 17;

/// Depth for the walks that cost O(depth) per run: proof chains, term nesting,
/// theory dependency chains.
const DEPTH: usize = 12_500;

/// Depth for the visualization test. Lower than [`DEPTH`] because the ASCII
/// tree, indented and JSON renderings all emit a per-line prefix that grows
/// with the depth, i.e. O(depth^2) output bytes.
const VIZ_DEPTH: usize = 7_500;

/// Depth for the parenthesis-nesting parser inputs, which cost one opening and
/// one closing character per level on top of the parse itself.
const PAREN_DEPTH: usize = 6_250;

/// Run `body` on a thread with a deliberately small stack.
fn on_small_stack<F, R>(name: &str, body: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(SMALL_STACK)
        .spawn(body)
        .expect("test thread should spawn");
    handle.join().expect("test thread should not panic")
}

/// A single-premise chain of `depth` inferences over one axiom.
fn deep_chain(depth: usize) -> Proof {
    let mut proof = Proof::new();
    let mut current = proof.add_axiom("p0");
    for level in 1..=depth {
        current = proof.add_inference("step", vec![current], format!("p{level}"));
    }
    proof
}

/// A doubling DAG: every level has two premises, both the *same* node, so a
/// walk without dedup re-expands 2^depth times.
fn shared_dag(levels: usize) -> Proof {
    let mut proof = Proof::new();
    let mut current = proof.add_axiom("leaf");
    for level in 1..=levels {
        current = proof.add_inference("dup", vec![current, current], format!("level{level}"));
    }
    proof
}

#[test]
fn traversal_walks_survive_a_12500_deep_proof() {
    on_small_stack("traversal_deep", || {
        let proof = deep_chain(DEPTH);

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PreOrder);
        assert_eq!(counter.axioms, 1);
        assert_eq!(counter.inferences, DEPTH);

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PostOrder);
        assert_eq!(counter.axioms, 1);
        assert_eq!(counter.inferences, DEPTH);

        let order = topological_order(&proof);
        assert_eq!(order.len(), DEPTH + 1);
        // Leaves first, root last.
        assert_eq!(order[0], ProofNodeId(0));

        let paths = find_all_paths(&proof);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), DEPTH + 1);
    });
}

#[test]
fn traversal_orders_are_preserved_on_a_small_proof() {
    // Semantic pin: the iterative walks must visit in the same order as the
    // recursive ones (root-first for pre-order, leftmost-premise-first).
    let mut proof = Proof::new();
    let p = proof.add_axiom("p");
    let q = proof.add_axiom("q");
    let root = proof.add_inference("and", vec![p, q], "(and p q)");

    let mut collector = ConclusionCollector::default();
    traverse(&proof, &mut collector, TraversalOrder::PreOrder);
    assert_eq!(collector.conclusions, vec!["(and p q)", "p", "q"]);

    let mut collector = ConclusionCollector::default();
    traverse(&proof, &mut collector, TraversalOrder::PostOrder);
    assert_eq!(collector.conclusions, vec!["p", "q", "(and p q)"]);

    let order = topological_order(&proof);
    assert_eq!(order, vec![p, q, root]);

    let paths = find_all_paths(&proof);
    assert_eq!(paths, vec![vec![root, p], vec![root, q]]);
}

#[test]
fn shared_dag_traversal_is_linear_not_exponential() {
    on_small_stack("traversal_shared_dag", || {
        // 60 doubling levels: 2^60 expansions without a visited set.
        let proof = shared_dag(60);

        let mut counter = NodeCounter::default();
        traverse(&proof, &mut counter, TraversalOrder::PreOrder);
        assert_eq!(counter.axioms, 1);
        assert_eq!(counter.inferences, 60);

        assert_eq!(topological_order(&proof).len(), 61);
    });
}

#[test]
fn detailed_stats_survive_a_deep_proof_and_pin_leaf_depth() {
    on_small_stack("stats_deep", || {
        let proof = deep_chain(DEPTH);
        let stats = DetailedProofStats::compute(&proof);
        assert_eq!(stats.total_steps, DEPTH + 1);
        assert_eq!(stats.axioms, 1);
        // The single leaf sits `DEPTH` edges below the root.
        assert!((stats.avg_leaf_depth - DEPTH as f64).abs() < 1e-6);
    });
}

#[test]
fn visualization_formats_survive_a_deep_proof() {
    on_small_stack("visualization_deep", || {
        let proof = deep_chain(VIZ_DEPTH);
        let viz = ProofVisualizer::new();

        for format in [
            VisualizationFormat::Dot,
            VisualizationFormat::AsciiTree,
            VisualizationFormat::IndentedText,
            VisualizationFormat::Json,
        ] {
            let mut out = Vec::new();
            viz.visualize(&proof, format, &mut out)
                .expect("visualization should succeed");
            assert!(!out.is_empty());
        }
    });
}

#[test]
fn dot_visualization_output_is_unchanged() {
    // Semantic pin: node line, then per premise an edge line followed by that
    // premise's own subtree.
    let mut proof = Proof::new();
    let p = proof.add_axiom("p");
    let q = proof.add_axiom("q");
    let _root = proof.add_inference("and", vec![p, q], "(and p q)");

    let mut out = Vec::new();
    ProofVisualizer::new()
        .visualize(&proof, VisualizationFormat::Dot, &mut out)
        .expect("visualization should succeed");
    let dot = String::from_utf8(out).expect("dot output should be utf-8");

    let lines: Vec<&str> = dot.lines().collect();
    assert_eq!(lines[0], "digraph Proof {");
    assert!(lines[3].contains("p2: and"));
    assert_eq!(lines[4], "  0 -> 2;");
    assert!(lines[5].contains("p0: axiom p"));
    assert_eq!(lines[6], "  1 -> 2;");
    assert!(lines[7].contains("p1: axiom q"));
}

#[test]
fn truncated_json_visualization_is_still_parseable() {
    // The depth cap used to emit an empty object with no indication why.
    let proof = deep_chain(5);
    let mut out = Vec::new();
    ProofVisualizer::new()
        .with_max_depth(2)
        .visualize(&proof, VisualizationFormat::Json, &mut out)
        .expect("visualization should succeed");
    let json = String::from_utf8(out).expect("json output should be utf-8");
    assert!(json.contains("\"truncated\": true"), "{json}");
}

#[test]
fn interpolant_term_display_and_walks_survive_deep_nesting() {
    on_small_stack("interpolant_term_deep", || {
        let mut term = InterpolantTerm::var("x");
        for _ in 0..DEPTH {
            // `and` of two elements keeps the tree genuinely deep (`not` would
            // collapse via double-negation elimination).
            term = InterpolantTerm::And(vec![term, InterpolantTerm::var("y")]);
        }

        let rendered = term.to_string();
        assert!(rendered.starts_with("(and (and "));
        assert!(rendered.ends_with(" y) y)"));

        let mut symbols = rustc_hash::FxHashSet::default();
        term.collect_symbols(&mut symbols);
        assert_eq!(symbols.len(), 2);

        // `simplify` honours its depth budget but no longer recurses.
        let simplified = term.simplify();
        assert!(simplified.to_string().starts_with("(and "));
    });
}

#[test]
fn interpolant_term_display_output_is_unchanged() {
    let term = InterpolantTerm::And(vec![
        InterpolantTerm::var("a"),
        InterpolantTerm::not(InterpolantTerm::var("b")),
        InterpolantTerm::Implies(
            Box::new(InterpolantTerm::var("c")),
            Box::new(InterpolantTerm::Bool(false)),
        ),
    ]);
    assert_eq!(term.to_string(), "(and a (not b) (=> c false))");
}

#[test]
fn cnf_validator_survives_deep_negation_and_paren_nesting() {
    on_small_stack("cnf_validator_deep", || {
        // One parser frame per `¬` in the recursive formulation.
        let deep = format!("{}a", "¬".repeat(DEPTH));
        let result = CnfValidator::validate_demorgan_and(&deep, &deep);
        // Not a negated conjunction: an honest verdict, reached without dying.
        assert!(matches!(result, RuleValidation::Invalid(_)));

        // Deeply nested parentheses used to cost several frames per level.
        let nested = format!("{}a{}", "(".repeat(PAREN_DEPTH), ")".repeat(PAREN_DEPTH));
        let result = CnfValidator::validate_demorgan_and(&nested, &nested);
        assert!(matches!(result, RuleValidation::Invalid(_)));
    });
}

#[test]
fn cnf_validator_demorgan_verdicts_are_unchanged() {
    // Semantic pins over parse + Display + multiset-equivalence.
    assert!(matches!(
        CnfValidator::validate_demorgan_and("¬(a ∧ b)", "(¬a ∨ ¬b)"),
        RuleValidation::Valid
    ));
    // Commutativity is accepted (the multiset matching).
    assert!(matches!(
        CnfValidator::validate_demorgan_and("¬(a ∧ b)", "(¬b ∨ ¬a)"),
        RuleValidation::Valid
    ));
    // A genuinely wrong output is still rejected.
    assert!(matches!(
        CnfValidator::validate_demorgan_and("¬(a ∧ b)", "(¬a ∧ ¬b)"),
        RuleValidation::Invalid(_)
    ));
    // Unparseable input is "cannot certify", not a fabricated verdict.
    assert!(matches!(
        CnfValidator::validate_demorgan_and("???", "???"),
        RuleValidation::Unchecked(_)
    ));
}

/// Build a linear theory proof of `depth` steps, each depending on the previous.
fn deep_theory_proof(depth: usize) -> TheoryProof {
    let mut proof = TheoryProof::new();
    let mut current = proof.add_step(TheoryRule::Refl, vec![], "t0");
    for level in 1..=depth {
        current = proof.add_step(TheoryRule::Refl, vec![current], format!("t{level}"));
    }
    proof
}

#[test]
fn theory_proof_checking_survives_a_deep_dependency_chain() {
    on_small_stack("checker_deep", || {
        let proof = deep_theory_proof(DEPTH);
        let mut checker = ProofChecker::new();
        // Reflexivity with a premise is rejected by the rule check; what matters
        // is that the dependency walk reaches that verdict without overflowing.
        let result = checker.check_theory_proof(&proof);
        assert!(matches!(
            result,
            CheckResult::Valid | CheckResult::Invalid { .. } | CheckResult::MultipleErrors(_)
        ));
    });
}

#[test]
fn deeply_nested_sexpr_conclusions_are_rejected_not_fatal() {
    on_small_stack("sexpr_deep", || {
        // Deeply nested parens in a conclusion string reach the s-expression
        // parser once conclusion verification is switched on.
        let deep_term = format!("{}x{}", "(".repeat(DEPTH), ")".repeat(DEPTH));
        let mut proof = TheoryProof::new();
        let lhs = proof.add_step(TheoryRule::Refl, vec![], "x");
        let _ = proof.add_step(TheoryRule::Symm, vec![lhs], deep_term);

        let mut checker = ProofChecker::with_config(CheckerConfig {
            verify_conclusions: true,
            ..CheckerConfig::default()
        });
        // The honest outcome is an error through the existing channel, never a
        // process abort.
        let result = checker.check_theory_proof(&proof);
        assert!(matches!(
            result,
            CheckResult::Valid | CheckResult::Invalid { .. } | CheckResult::MultipleErrors(_)
        ));
    });
}
