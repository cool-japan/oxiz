use oxiz_core::ast::TermKind;
use oxiz_core::qe::eliminate_quantifier_vs;

#[test]
fn eliminate_strict_inequality() {
    let mut manager = oxiz_core::ast::TermManager::new();
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let zero = manager.mk_int(0);
    let one = manager.mk_int(1);
    let two = manager.mk_int(2);

    let gt_zero = manager.mk_gt(x, zero);
    let lt_two = manager.mk_lt(x, two);
    let sat_formula = manager.mk_and([gt_zero, lt_two]);
    let sat_result = eliminate_quantifier_vs(x, &sat_formula, &mut manager);
    let sat_simplified = manager.simplify(sat_result);
    assert!(!matches!(
        manager.get(sat_simplified).map(|t| &t.kind),
        Some(TermKind::False)
    ));

    let gt_one = manager.mk_gt(x, one);
    let lt_zero = manager.mk_lt(x, zero);
    let unsat_formula = manager.mk_and([gt_one, lt_zero]);
    let unsat_result = eliminate_quantifier_vs(x, &unsat_formula, &mut manager);
    let unsat_simplified = manager.simplify(unsat_result);
    assert!(matches!(
        manager.get(unsat_simplified).map(|t| &t.kind),
        Some(TermKind::False)
    ));
}
