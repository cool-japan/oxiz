//! Fuzz target for the SMT solver
//!
//! This fuzzer tests the solver with random valid SMT-LIB2 scripts using
//! structured fuzzing to generate well-formed commands.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;
use oxiz::core::smtlib::{Command, parse_script};
use oxiz::{Solver, TermId, TermManager};

/// Represents a structured SMT command
#[derive(Debug, Arbitrary)]
enum SmtCommand {
    /// Declare a boolean constant
    DeclareBool { name_idx: u8 },
    /// Declare an integer constant
    DeclareInt { name_idx: u8 },
    /// Declare a real constant
    DeclareReal { name_idx: u8 },
    /// Assert a simple boolean constraint
    AssertBool { var_idx: u8, is_positive: bool },
    /// Assert an integer comparison
    AssertIntCmp {
        var_idx: u8,
        cmp_type: CmpType,
        value: i16,
    },
    /// Assert an arithmetic constraint
    AssertArith {
        lhs_var: u8,
        rhs_var: u8,
        op: ArithOp,
        result: i16,
    },
    /// Assert an equality between two variables
    AssertEq { lhs_var: u8, rhs_var: u8 },
    /// Check satisfiability
    CheckSat,
    /// Push a scope
    Push,
    /// Pop a scope
    Pop,
    /// Reset the solver
    Reset,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum CmpType {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum ArithOp {
    Add,
    Sub,
    Mul,
}

/// Build a name from an index
fn make_name(prefix: &str, idx: u8) -> String {
    format!("{}_{}", prefix, idx % 8)
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    // Limit the number of commands to prevent OOM and timeouts
    let max_commands = 50;

    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    // Track declared variables
    let mut bool_vars: Vec<(String, TermId)> = Vec::new();
    let mut int_vars: Vec<(String, TermId)> = Vec::new();
    let mut real_vars: Vec<(String, TermId)> = Vec::new();

    // Track push/pop balance
    let mut scope_depth = 0;

    // Generate and execute commands
    for _ in 0..max_commands {
        let cmd: Result<SmtCommand, _> = unstructured.arbitrary();
        let cmd = match cmd {
            Ok(cmd) => cmd,
            Err(_) => break,
        };

        match cmd {
            SmtCommand::DeclareBool { name_idx } => {
                let name = make_name("b", name_idx);
                // Only declare if not already declared
                if !bool_vars.iter().any(|(n, _)| n == &name) {
                    let var = tm.mk_var(&name, tm.sorts.bool_sort);
                    bool_vars.push((name, var));
                }
            }
            SmtCommand::DeclareInt { name_idx } => {
                let name = make_name("i", name_idx);
                if !int_vars.iter().any(|(n, _)| n == &name) {
                    let var = tm.mk_var(&name, tm.sorts.int_sort);
                    int_vars.push((name, var));
                }
            }
            SmtCommand::DeclareReal { name_idx } => {
                let name = make_name("r", name_idx);
                if !real_vars.iter().any(|(n, _)| n == &name) {
                    let var = tm.mk_var(&name, tm.sorts.real_sort);
                    real_vars.push((name, var));
                }
            }
            SmtCommand::AssertBool { var_idx, is_positive } => {
                if !bool_vars.is_empty() {
                    let (_, var) = &bool_vars[var_idx as usize % bool_vars.len()];
                    let term = if is_positive {
                        *var
                    } else {
                        tm.mk_not(*var)
                    };
                    solver.assert(term, &mut tm);
                }
            }
            SmtCommand::AssertIntCmp {
                var_idx,
                cmp_type,
                value,
            } => {
                if !int_vars.is_empty() {
                    let (_, var) = &int_vars[var_idx as usize % int_vars.len()];
                    let const_term = tm.mk_int(BigInt::from(value));
                    let cmp_term = match cmp_type {
                        CmpType::Lt => tm.mk_lt(*var, const_term),
                        CmpType::Le => tm.mk_le(*var, const_term),
                        CmpType::Gt => tm.mk_gt(*var, const_term),
                        CmpType::Ge => tm.mk_ge(*var, const_term),
                        CmpType::Eq => tm.mk_eq(*var, const_term),
                    };
                    solver.assert(cmp_term, &mut tm);
                }
            }
            SmtCommand::AssertArith {
                lhs_var,
                rhs_var,
                op,
                result,
            } => {
                if int_vars.len() >= 2 {
                    let (_, lhs) = &int_vars[lhs_var as usize % int_vars.len()];
                    let (_, rhs) = &int_vars[rhs_var as usize % int_vars.len()];
                    let arith_term = match op {
                        ArithOp::Add => tm.mk_add([*lhs, *rhs]),
                        ArithOp::Sub => tm.mk_sub(*lhs, *rhs),
                        ArithOp::Mul => tm.mk_mul([*lhs, *rhs]),
                    };
                    let result_term = tm.mk_int(BigInt::from(result));
                    let eq_term = tm.mk_eq(arith_term, result_term);
                    solver.assert(eq_term, &mut tm);
                }
            }
            SmtCommand::AssertEq { lhs_var, rhs_var } => {
                if int_vars.len() >= 2 {
                    let (_, lhs) = &int_vars[lhs_var as usize % int_vars.len()];
                    let (_, rhs) = &int_vars[rhs_var as usize % int_vars.len()];
                    let eq_term = tm.mk_eq(*lhs, *rhs);
                    solver.assert(eq_term, &mut tm);
                }
            }
            SmtCommand::CheckSat => {
                // Run the solver - we don't care about the result
                let _ = solver.check(&mut tm);
            }
            SmtCommand::Push => {
                solver.push();
                scope_depth += 1;
            }
            SmtCommand::Pop => {
                if scope_depth > 0 {
                    solver.pop();
                    scope_depth -= 1;
                }
            }
            SmtCommand::Reset => {
                solver.reset();
                bool_vars.clear();
                int_vars.clear();
                real_vars.clear();
                scope_depth = 0;
            }
        }
    }
});

/// Additional fuzzing: parse and check arbitrary SMT-LIB2 scripts
#[allow(dead_code)]
fn fuzz_parse_and_solve(data: &[u8]) {
    if let Ok(input) = std::str::from_utf8(data) {
        let mut tm = TermManager::new();

        // Parse the script
        if let Ok(commands) = parse_script(input, &mut tm) {
            let mut solver = Solver::new();

            // Execute each command
            for cmd in commands {
                match cmd {
                    Command::Assert(term) => {
                        solver.assert(term, &mut tm);
                    }
                    Command::CheckSat => {
                        let _ = solver.check(&mut tm);
                    }
                    Command::Push(n) => {
                        for _ in 0..n.min(10) {
                            solver.push();
                        }
                    }
                    Command::Pop(n) => {
                        for _ in 0..n.min(10) {
                            solver.pop();
                        }
                    }
                    Command::Reset => {
                        solver.reset();
                    }
                    _ => {
                        // Skip other commands
                    }
                }
            }
        }
    }
}
