use num_rational::Rational64;
use oxiz_core::ast::TermKind;
use oxiz_core::qe::eliminate_quantifier_vs;

#[test]
fn eliminate_strict_inequality() {
    let mut manager = oxiz_core::ast::TermManager::new();
    let x = manager.mk_var("x", manager.sorts.real_sort);
    let zero = manager.mk_real(Rational64::new(0, 1));
    let one = manager.mk_real(Rational64::new(1, 1));

    let gt_zero = manager.mk_gt(x, zero);
    let lt_one = manager.mk_lt(x, one);
    let sat_formula = manager.mk_and([gt_zero, lt_one]);
    let sat_result = eliminate_quantifier_vs(x, &sat_formula, &mut manager);
    assert!(matches!(manager.get(sat_result).map(|t| &t.kind), Some(TermKind::True)));

    let gt_one = manager.mk_gt(x, one);
    let lt_zero = manager.mk_lt(x, zero);
    let unsat_formula = manager.mk_and([gt_one, lt_zero]);
    let unsat_result = eliminate_quantifier_vs(x, &unsat_formula, &mut manager);
    assert!(matches!(manager.get(unsat_result).map(|t| &t.kind), Some(TermKind::False)));
}
