//! Deep-nesting and semantic regression tests for the SMT-LIB parser, the
//! SMT-LIB printer and the model value layer (group C3).
//!
//! Every "deep" test runs on a deliberately small 128 KiB stack. The assertion
//! in each is simply that the call *returns*: a native stack overflow aborts
//! the whole process rather than unwinding, so a test that finishes at all is
//! the proof that the walk under test is no longer recursive.

use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Printer, parse_script};

/// Stack size every deep-nesting test runs under.
const SMALL_STACK: usize = 1 << 17;

/// Run `f` on a thread with a 128 KiB stack and return its value.
fn on_small_stack<T, F>(name: &str, f: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(SMALL_STACK)
        .spawn(f);
    match handle {
        Ok(h) => match h.join() {
            Ok(v) => v,
            Err(_) => panic!("{name}: worker thread panicked"),
        },
        Err(e) => panic!("{name}: could not spawn worker thread: {e}"),
    }
}

// ---------------------------------------------------------------------------
// parser/sorts.rs — self-referential `define-sort`
// ---------------------------------------------------------------------------

/// `(define-sort A () A)` installs `A -> A` in the alias table. Resolving a
/// reference to `A` that goes through `parse_sort` (rather than the
/// short-circuiting `resolve_sort`) used to re-enter `parse_sort_name` once
/// per alias hop, forever, and abort the process.
#[test]
fn self_referential_define_sort_is_rejected_not_infinite() {
    // Each of these detonated the old infinite recursion; all are valid
    // SMT-LIB text as far as the grammar is concerned.
    let bodies = [
        "(declare-const x (Array A Int))",
        "(assert (forall ((x A)) true))",
        "(declare-datatype D ((c (fld (Array A A)))))",
        "(define-sort B () (Array A A))",
    ];
    for body in bodies {
        let script = format!("(define-sort A () A)\n{body}\n");
        let result = on_small_stack("self_referential_define_sort", move || {
            let mut manager = TermManager::new();
            parse_script(&script, &mut manager).map(|cmds| cmds.len())
        });
        assert!(
            result.is_err(),
            "expected a parse error for a self-referential sort alias, got {result:?}"
        );
    }
}

/// A two-step cycle `A -> B -> A` must also terminate.
#[test]
fn mutually_referential_define_sorts_are_rejected() {
    let script = "(define-sort A () B)\n(define-sort B () A)\n(declare-const x (Array A Int))\n";
    let result = on_small_stack("mutual_define_sort", move || {
        let mut manager = TermManager::new();
        parse_script(script, &mut manager).map(|cmds| cmds.len())
    });
    assert!(
        result.is_err(),
        "expected a parse error for a cyclic sort alias, got {result:?}"
    );
}

/// A well-formed alias chain still resolves, so the cycle guard did not just
/// reject everything.
#[test]
fn well_formed_sort_alias_chain_still_resolves() {
    let script = "(define-sort A () Int)\n(define-sort B () A)\n(declare-const x (Array B B))\n\
                  (assert (= (select x 0) 0))\n";
    let mut manager = TermManager::new();
    let commands = parse_script(script, &mut manager).expect("alias chain must resolve");
    assert_eq!(commands.len(), 4);
}

// ---------------------------------------------------------------------------
// parser/commands.rs — unknown-command skipping
// ---------------------------------------------------------------------------

/// Skipping an unrecognized command is a documented feature. It used to be
/// implemented as a tail call into `parse_command`, so a script of N unknown
/// commands consumed N native stack frames.
#[test]
fn twelve_thousand_five_hundred_unknown_commands_do_not_overflow() {
    let mut script = String::new();
    for i in 0..12_500u32 {
        script.push_str(&format!("(vendor-specific-thing {i})\n"));
    }
    script.push_str("(check-sat)\n");

    let count = on_small_stack("unknown_command_flood", move || {
        let mut manager = TermManager::new();
        parse_script(&script, &mut manager).map(|cmds| cmds.len())
    });
    // Every unknown command is skipped; only `(check-sat)` survives.
    assert_eq!(count.ok(), Some(1));
}

// ---------------------------------------------------------------------------
// parser/commands.rs + terms.rs — nullary define-fun inlining accumulates depth
// ---------------------------------------------------------------------------

/// Build `(define-fun a0 () Int 0)` followed by `n` doubling definitions.
fn nullary_define_fun_chain(n: u32) -> String {
    let mut script = String::from("(define-fun a0 () Int 0)\n");
    for i in 1..=n {
        script.push_str(&format!(
            "(define-fun a{i} () Int (+ a{prev} a{prev}))\n",
            prev = i - 1
        ));
    }
    script.push_str(&format!("(assert (= a{n} 0))\n"));
    script
}

/// A short chain is ordinary, valid input and must keep working.
#[test]
fn short_nullary_define_fun_chain_still_parses() {
    let script = nullary_define_fun_chain(100);
    let mut manager = TermManager::new();
    let commands = parse_script(&script, &mut manager)
        .expect("a 100-deep chain is well within the parser's nesting budget");
    assert_eq!(commands.len(), 102);
}

/// Each command in the chain is two parens deep, so the per-parse nesting
/// counter never sees the accumulation — but the *term* grows one level per
/// command. Past the budget the parser must say so honestly instead of
/// handing a 100 000-deep term to whatever consumes it next.
#[test]
fn long_nullary_define_fun_chain_is_an_honest_parse_error() {
    let script = nullary_define_fun_chain(12_500);
    let result = on_small_stack("define_fun_chain", move || {
        let mut manager = TermManager::new();
        parse_script(&script, &mut manager)
            .map(|cmds| cmds.len())
            .map_err(|e| e.to_string())
    });
    match result {
        Err(message) => assert!(
            message.contains("too deep"),
            "expected a nesting-depth error, got: {message}"
        ),
        Ok(n) => panic!("expected a nesting-depth error, parsed {n} commands"),
    }
}

// ---------------------------------------------------------------------------
// printer/basic.rs — write_sort
// ---------------------------------------------------------------------------

/// `SortManager::array` is public and interns in constant stack, so an
/// embedder can build a sort far deeper than any SMT-LIB text could express.
/// Printing it used to recurse once per level with no guard at all —
/// `write_sort` never touched the printer's depth counter.
#[test]
fn printing_a_deeply_nested_array_sort_does_not_overflow() {
    let printed_len = on_small_stack("deep_array_sort_print", || {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let mut sort = int_sort;
        for _ in 0..12_500 {
            sort = manager.sorts.array(int_sort, sort);
        }
        let printer = Printer::new(&manager);
        printer.print_sort(sort).len()
    });
    // "(Array Int " * 100000 + "Int" + ")" * 100000
    assert_eq!(printed_len, 12_500 * 11 + 3 + 12_500);
}

/// Semantic pin: the iterative `write_sort` renders exactly what the
/// recursive one did for the shapes that fit on paper.
#[test]
fn write_sort_output_is_unchanged_for_shallow_sorts() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let bool_sort = manager.sorts.bool_sort;
    let bv = manager.sorts.bitvec(32);
    let inner = manager.sorts.array(int_sort, bool_sort);
    let outer = manager.sorts.array(bv, inner);
    let printer = Printer::new(&manager);
    assert_eq!(printer.print_sort(int_sort), "Int");
    assert_eq!(printer.print_sort(bv), "(_ BitVec 32)");
    assert_eq!(printer.print_sort(inner), "(Array Int Bool)");
    assert_eq!(
        printer.print_sort(outer),
        "(Array (_ BitVec 32) (Array Int Bool))"
    );
}
