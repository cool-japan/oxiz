use num_rational::Rational64;
use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::arithmetic::ArithSolver;

#[test]
fn test_gcd_infeasibility_cutting_planes() {
    // Test case: 2x + 2y = 7 with x >= 0, y >= 0
    // This should be UNSAT because gcd(2,2) = 2 doesn't divide 7

    let mut solver = ArithSolver::lia();

    // Create terms for x and y
    let x = TermId(1);
    let y = TermId(2);
    let reason = TermId(0);

    // Add constraint: 2x + 2y = 7
    solver.assert_eq(
        &[
            (x, Rational64::from_integer(2)),
            (y, Rational64::from_integer(2)),
        ],
        Rational64::from_integer(7),
        reason,
    );

    // Add constraints: x >= 0, y >= 0
    solver.assert_ge(
        &[(x, Rational64::from_integer(1))],
        Rational64::from_integer(0),
        reason,
    );
    solver.assert_ge(
        &[(y, Rational64::from_integer(1))],
        Rational64::from_integer(0),
        reason,
    );

    // Check satisfiability - should be UNSAT
    let result = solver.check();
    println!("Result: {:?}", result);

    // The solver should detect infeasibility
    assert!(result.is_ok());
    let theory_result = result.unwrap();

    use oxiz_theories::TheoryCheckResult;
    match theory_result {
        TheoryCheckResult::Unsat(_) => {
            println!("CORRECT: Detected UNSAT via GCD check");
        }
        TheoryCheckResult::Sat => {
            panic!("WRONG: Returned SAT, but should be UNSAT due to GCD infeasibility");
        }
        _ => {
            panic!("Unexpected result");
        }
    }
}
