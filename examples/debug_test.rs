use oxiz_core::TermManager;
use oxiz_solver::{Solver, SolverResult};
use num_bigint::BigInt;

fn main() {
    let mut solver = Solver::new();
    solver.set_logic("QF_LIA");
    let mut tm = TermManager::new();
    
    // Create age variable
    let age = tm.mk_var("age", tm.sorts.int_sort);
    
    // age >= 65
    let val_65 = tm.mk_int(BigInt::from(65));
    let ge_65 = tm.mk_ge(age, val_65);
    
    // age < 18
    let val_18 = tm.mk_int(BigInt::from(18));
    let lt_18 = tm.mk_lt(age, val_18);
    
    // age >= 65 AND age < 18
    let conj = tm.mk_and([ge_65, lt_18]);
    
    solver.assert(conj, &mut tm);
    
    let result = solver.check(&mut tm);
    println!("Result: {:?}", result);
    println!("Expected: Unsat (contradiction)");
    
    match result {
        SolverResult::Sat => println!("PROBLEM: Solver returned SAT for age >= 65 AND age < 18"),
        SolverResult::Unsat => println!("OK: Solver correctly returned UNSAT"),
        SolverResult::Unknown => println!("Solver returned Unknown"),
    }
}
