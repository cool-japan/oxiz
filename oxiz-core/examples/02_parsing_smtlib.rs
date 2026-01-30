//! # SMT-LIB2 Parsing Example
//!
//! This example demonstrates parsing SMT-LIB2 format input.
//! It covers:
//! - Parsing SMT-LIB2 commands
//! - Handling declare-const, assert, check-sat
//! - Converting parsed commands to internal terms
//! - Error handling for malformed input
//!
//! ## SMT-LIB2 Reference
//! See: <http://smtlib.cs.uiowa.edu/language.shtml>
//!
//! ## Complexity
//! - Time: O(n) where n is the input length
//! - Space: O(m) where m is the number of terms created
//!
//! ## See Also
//! - [`parse_script`](oxiz_core::smtlib::parse_script) for the main parser
//! - [`Command`](oxiz_core::smtlib::Command) for SMT-LIB2 commands

use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Command, parse_script};

fn main() {
    println!("=== OxiZ Core: SMT-LIB2 Parsing ===\n");

    // ===== Example 1: Simple Boolean Formula =====
    println!("--- Example 1: Boolean Satisfiability ---");
    let input1 = r#"
        (set-logic QF_UF)
        (declare-const p Bool)
        (declare-const q Bool)
        (assert (and p q))
        (assert (not p))
        (check-sat)
    "#;

    let mut tm = TermManager::new();
    match parse_script(input1, &mut tm) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 2: Integer Arithmetic =====
    println!("\n--- Example 2: Linear Integer Arithmetic ---");
    let input2 = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (>= x 0))
        (assert (<= x 10))
        (assert (= y (* 2 x)))
        (assert (> y 15))
        (check-sat)
        (get-model)
    "#;

    let mut tm2 = TermManager::new();
    match parse_script(input2, &mut tm2) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                match cmd {
                    Command::SetLogic(logic) => {
                        println!("  {}: Set logic to {}", i + 1, logic);
                    }
                    Command::DeclareConst(name, sort) => {
                        println!(
                            "  {}: Declare constant '{}' of sort {:?}",
                            i + 1,
                            name,
                            sort
                        );
                    }
                    Command::Assert(term) => {
                        println!("  {}: Assert formula {:?}", i + 1, term);
                    }
                    Command::CheckSat => {
                        println!("  {}: Check satisfiability", i + 1);
                    }
                    Command::GetModel => {
                        println!("  {}: Get model", i + 1);
                    }
                    _ => {
                        println!("  {}: {:?}", i + 1, cmd);
                    }
                }
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 3: Quantified Formula =====
    println!("\n--- Example 3: Quantified Formula ---");
    let input3 = r#"
        (set-logic LIA)
        (declare-fun f (Int) Int)
        (assert (forall ((x Int)) (>= (f x) 0)))
        (assert (exists ((y Int)) (= (f y) 5)))
        (check-sat)
    "#;

    let mut tm3 = TermManager::new();
    match parse_script(input3, &mut tm3) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 4: Bitvector Logic =====
    println!("\n--- Example 4: Bitvector Arithmetic ---");
    let input4 = r#"
        (set-logic QF_BV)
        (declare-const a (_ BitVec 8))
        (declare-const b (_ BitVec 8))
        (assert (= (bvadd a b) #x80))
        (assert (bvult a b))
        (check-sat)
    "#;

    let mut tm4 = TermManager::new();
    match parse_script(input4, &mut tm4) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 5: Array Theory =====
    println!("\n--- Example 5: Theory of Arrays ---");
    let input5 = r#"
        (set-logic QF_AUFLIA)
        (declare-const arr1 (Array Int Int))
        (declare-const arr2 (Array Int Int))
        (declare-const i Int)
        (declare-const v Int)
        (assert (= arr2 (store arr1 i v)))
        (assert (not (= (select arr2 i) v)))
        (check-sat)
    "#;

    let mut tm5 = TermManager::new();
    match parse_script(input5, &mut tm5) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 6: Error Handling =====
    println!("\n--- Example 6: Error Handling ---");
    let malformed_input = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (>= x))  ; Missing argument - should cause error
        (check-sat)
    "#;

    let mut tm6 = TermManager::new();
    match parse_script(malformed_input, &mut tm6) {
        Ok(commands) => {
            println!("Unexpectedly parsed {} commands", commands.len());
        }
        Err(e) => {
            println!("Expected parse error: {:?}", e);
            println!("Error message: {}", e);
        }
    }

    // ===== Example 7: Incremental Commands =====
    println!("\n--- Example 7: Incremental Solving (Push/Pop) ---");
    let input7 = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (>= x 0))
        (push 1)
        (assert (<= x 5))
        (check-sat)
        (pop 1)
        (assert (>= x 10))
        (check-sat)
    "#;

    let mut tm7 = TermManager::new();
    match parse_script(input7, &mut tm7) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                match cmd {
                    Command::Push(n) => {
                        println!("  {}: Push {} levels", i + 1, n);
                    }
                    Command::Pop(n) => {
                        println!("  {}: Pop {} levels", i + 1, n);
                    }
                    _ => {
                        println!("  {}: {:?}", i + 1, cmd);
                    }
                }
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    // ===== Example 8: Options and Metadata =====
    println!("\n--- Example 8: Options and Metadata ---");
    let input8 = r#"
        (set-info :source "Example from documentation")
        (set-info :category "crafted")
        (set-option :produce-models true)
        (set-option :produce-unsat-cores true)
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 42))
        (check-sat)
    "#;

    let mut tm8 = TermManager::new();
    match parse_script(input8, &mut tm8) {
        Ok(commands) => {
            println!("Parsed {} commands:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {:?}", e);
        }
    }

    println!("\n=== Example Complete ===");
}
