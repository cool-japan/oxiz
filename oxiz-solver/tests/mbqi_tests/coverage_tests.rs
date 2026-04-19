use oxiz_solver::mbqi::{Pattern, PatternCoverScorer, PatternSet, TermShape};

#[test]
fn pattern_cover_scorer_orders_by_coverage() {
    let empty_manager = &mut oxiz_core::ast::TermManager::new();

    let set0 = PatternSet::from_patterns(vec![Pattern::new(vec![])], empty_manager);
    let mut set1 = PatternSet::from_patterns(vec![Pattern::new(vec![])], empty_manager);
    let mut set2 = PatternSet::from_patterns(vec![Pattern::new(vec![])], empty_manager);

    set1.covered_shapes = [TermShape::IntConst, TermShape::StrictIneq]
        .into_iter()
        .collect();
    set2.covered_shapes = [TermShape::IntConst].into_iter().collect();

    let scorer = PatternCoverScorer;
    let ranked = scorer.score_cover(
        &[set0, set1, set2],
        &[TermShape::IntConst, TermShape::StrictIneq, TermShape::Apply { arity: 1 }],
    );

    assert_eq!(ranked[0].0, 1);
    assert!(ranked[0].1 > ranked[1].1);
}
