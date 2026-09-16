//! `(get-model)` and `(get-value …)` describe **one** model (`#P2b-34`,
//! `#P2b-35`).
//!
//! Four renderers report a model — the constant list, each declared
//! function's `define-fun` interpretation, each array's `store` chain, and
//! `(get-value …)` — and each used to answer "what value does this model give
//! this congruence class?" for itself.  The answers disagreed:
//!
//! * the constant loop synthesised a `@uc_S_n` witness per class over an
//!   uninterpreted sort, while the interpretation's own class walk had none,
//!   so `(= (f a) b)` with `(distinct a b)` printed `b = @uc_U_1` beside
//!   `f = @uc_U_0` — the model falsified its own assertion;
//! * an array printed the *sort* default as its chain base even when its
//!   congruence class contained an `(as const d)` member saying otherwise;
//! * an application `(get-value …)` could not fold echoed itself, while
//!   `(get-model)` printed the same function as a total one with an
//!   else-value.
//!
//! Every test here replays the printed model against the assertions it is
//! supposed to satisfy, which is the property all three defects broke.

use oxiz_solver::Context;

/// Run a script and return its response lines.
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default()
}

/// The `(model …)` block of a script whose first response is `sat`.
fn model_of(script: &str) -> String {
    let output = run(script);
    assert_eq!(
        output.first().map(String::as_str),
        Some("sat"),
        "expected sat: {output:?}"
    );
    output
        .iter()
        .find(|line| line.starts_with("(model"))
        .cloned()
        .unwrap_or_else(|| panic!("no model block: {output:?}"))
}

/// Re-assert a printed model's `define-fun` lines on top of the original
/// script and check the result is still `sat`.
///
/// A model that falsifies its own assertions makes the replay `unsat`, which
/// is the single check every defect in this file fails.
fn replay_is_sat(script: &str, model: &str) -> bool {
    let definitions: String = model
        .lines()
        .filter(|line| line.trim_start().starts_with("(define-fun "))
        .map(|line| format!("{}\n", line.trim()))
        .collect();
    // Declarations must not be repeated, so the replay re-runs the script with
    // the definitions substituted for the declarations of the same names.
    let mut replay = String::new();
    for line in script.lines() {
        let declared = line.trim_start();
        let name = declared
            .strip_prefix("(declare-const ")
            .or_else(|| declared.strip_prefix("(declare-fun "))
            .and_then(|rest| rest.split_whitespace().next());
        match name {
            Some(name)
                if definitions
                    .lines()
                    .any(|d| d.starts_with(&format!("(define-fun {name} "))) =>
            {
                let definition = definitions
                    .lines()
                    .find(|d| d.starts_with(&format!("(define-fun {name} ")))
                    .unwrap_or_default();
                replay.push_str(definition);
                replay.push('\n');
            }
            _ => {
                replay.push_str(line);
                replay.push('\n');
            }
        }
    }
    run(&replay).first().map(String::as_str) == Some("sat")
}

// ---------------------------------------------------------------------------
// (2) one canonical class -> value map
// ---------------------------------------------------------------------------

/// `r3/us/u1_basic`: the interpretation of `f` must name the same witness the
/// constant list gave `b`.
#[test]
fn a_function_over_an_uninterpreted_sort_agrees_with_the_constants() {
    let script = "(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const a U)
(declare-const b U)
(assert (= (f a) b))
(assert (distinct a b))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(
        model.contains("(define-fun b () U @uc_U_1)"),
        "b is the second witness: {model}"
    );
    assert!(
        model.contains("(define-fun f ((x!0 U)) U @uc_U_1)"),
        "f must answer with b's witness, not the sort default: {model}"
    );
}

/// `r3/us/u3_pred`: two argument classes with no value of their own both
/// rendered as the sort's zero-th witness, so the printed predicate tested the
/// *same* guard twice with contradicting answers.
#[test]
fn a_predicate_over_an_uninterpreted_sort_has_distinct_guards() {
    let script = "(set-logic QF_UF)
(declare-sort U 0)
(declare-fun p (U) Bool)
(declare-const a U)
(declare-const b U)
(assert (p a))
(assert (not (p b)))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(
        model.contains("@uc_U_0") && model.contains("@uc_U_1"),
        "both witnesses appear: {model}"
    );
    assert!(
        !model.contains("(ite (= x!0 @uc_U_0) true (ite (= x!0 @uc_U_0)"),
        "the same guard must not be tested twice: {model}"
    );
}

/// An argument the model determines only *structurally* — `(bvsub (f #b10) v)`
/// has no model entry of its own — must be resolved by folding it in the
/// model, not by substituting the argument sort's default.  Substituting
/// `#b00` there let an unrelated application claim the tuple `(#b00)` that
/// `(f w)` needed, and the printed `f` then falsified `(bvslt (f w) #b00)`.
#[test]
fn a_structurally_determined_argument_does_not_steal_another_tuple() {
    let script = "(set-logic QF_AUFBV)
(declare-fun f ((_ BitVec 2)) (_ BitVec 2))
(declare-const v (_ BitVec 2))
(declare-const w (_ BitVec 2))
(assert (not (= (f (bvsub (f #b10) v)) #b01)))
(assert (bvslt (f w) #b00))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(replay_is_sat(script, &model), "{model}");
}

// ---------------------------------------------------------------------------
// (3) an array model per congruence class
// ---------------------------------------------------------------------------

/// `r3/mp/b2`: an array asserted equal to an array constant must print with
/// that constant's default as its base, not the sort default.
#[test]
fn an_array_equal_to_a_constant_prints_that_constants_default() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(assert (= arr ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(
        model.contains("((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)"),
        "the class's own constant is the base: {model}"
    );
    assert!(replay_is_sat(script, &model), "{model}");
}

/// `r3/mp/b3`: an array asserted equal to a `store` prints that store's own
/// base and write.
#[test]
fn an_array_equal_to_a_store_prints_that_store() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 2) (_ BitVec 2)))
(declare-const brr (Array (_ BitVec 2) (_ BitVec 2)))
(assert (= arr (store brr #b01 #b11)))
(assert (= (select arr #b01) #b11))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(replay_is_sat(script, &model), "{model}");
}

/// `r3/mp/b10` and `bench/z3_parity/benchmarks/qf_a/array_07`: the *outer*
/// array of an array of arrays says nothing through the model's assignments —
/// no term denotes a whole array — so its entries come from the array-sorted
/// reads of its class, rendered recursively.
///
/// Asserted on the printed text rather than replayed through the solver:
/// re-running either script with its nested arrays substituted as
/// `define-fun`s is itself a hard goal (12 s on the 0.3.4 base, longer here —
/// recorded as an open item), and what is under test is what the outer array
/// prints.
#[test]
fn an_array_of_arrays_prints_its_inner_rows() {
    let script = "(set-logic QF_ABV)
(declare-const aa (Array (_ BitVec 1) (Array (_ BitVec 1) (_ BitVec 1))))
(declare-const i (_ BitVec 1))
(declare-const j (_ BitVec 1))
(assert (= (select (select aa i) j) #b1))
(check-sat)
(get-model)
";
    let model = model_of(script);
    let outer = model
        .lines()
        .find(|line| line.trim().starts_with("(define-fun aa "))
        .unwrap_or_default()
        .to_string();
    assert!(
        outer.contains("(store") && outer.contains("#b1"),
        "the outer array carries the row it was read at: {model}"
    );

    // `bench/z3_parity/benchmarks/qf_a/array_07`: the outer array used to
    // print as the sort default beside a `row0` saying otherwise.
    let nested = "(set-logic QF_ALIA)
(declare-const matrix (Array Int (Array Int Int)))
(declare-const row0 (Array Int Int))
(assert (= row0 (select matrix 0)))
(assert (= (select row0 0) 42))
(assert (= (select (select matrix 0) 0) 42))
(check-sat)
(get-model)
";
    let nested_model = model_of(nested);
    let matrix_line = nested_model
        .lines()
        .find(|line| line.trim().starts_with("(define-fun matrix "))
        .unwrap_or_default()
        .to_string();
    assert!(
        matrix_line.contains("(store") && matrix_line.contains("42"),
        "matrix must carry row0 at index 0: {nested_model}"
    );
}

/// An array only ever described from the outside — as the base of a `store`
/// the model does pin down — inherits that store's rendering minus the
/// store's own index.
#[test]
fn an_array_pinned_only_through_a_store_over_it_inherits_that_store() {
    let script = "(set-logic QF_ABV)
(declare-const brr (Array (_ BitVec 2) (_ BitVec 2)))
(declare-const k (_ BitVec 2))
(declare-const w (_ BitVec 2))
(assert (= ((as const (Array (_ BitVec 2) (_ BitVec 2))) #b01) (store brr k w)))
(check-sat)
(get-model)
";
    let model = model_of(script);
    assert!(replay_is_sat(script, &model), "{model}");
}

/// Two arrays the solver proved *different* must print differently: the
/// extensionality witness of `#P2b-37` gives their classes a read at an index
/// where they differ, and the rendering shows it.
#[test]
fn arrays_in_different_classes_print_differently() {
    let script = "(set-logic QF_ABV)
(declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))
(declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))
(assert (distinct arr brr))
(check-sat)
(get-model)
";
    let model = model_of(script);
    let value_of = |name: &str| -> String {
        model
            .lines()
            .find(|line| line.trim().starts_with(&format!("(define-fun {name} ")))
            .map(|line| line.trim().to_string())
            .unwrap_or_default()
    };
    let arr = value_of("arr").replacen("arr", "X", 1);
    let brr = value_of("brr").replacen("brr", "X", 1);
    assert!(!arr.is_empty() && !brr.is_empty(), "both printed: {model}");
    assert_ne!(arr, brr, "distinct arrays must print differently: {model}");
    assert!(replay_is_sat(script, &model), "{model}");
}

// ---------------------------------------------------------------------------
// (4) `(get-value)` answers the interpretation's else value
// ---------------------------------------------------------------------------

/// `r3/dt/d3`: a function with no application the model folds is printed as
/// the constant else-value, and `(get-value)` must say the same rather than
/// echo the term.
#[test]
fn get_value_answers_the_else_value_of_an_unapplied_function() {
    let output = run("(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 4)) (_ BitVec 4))
(declare-fun g ((_ BitVec 4) (_ BitVec 4)) Bool)
(declare-const a (_ BitVec 4))
(assert (= a #b0001))
(check-sat)
(get-model)
(get-value ((f a) (g a a)))
");
    let values = output.last().cloned().unwrap_or_default();
    assert!(
        !values.contains("(f a) (f a)") && !values.contains("(g a a) (g a a)"),
        "no echo: {values}"
    );
    assert!(values.contains("((f a) #x0)"), "{values}");
    assert!(values.contains("((g a a) false)"), "{values}");
}

/// `r3/gv2/h1`: an application at an argument tuple matching no entry answers
/// the else-value the printed `define-fun` commits to.
#[test]
fn get_value_answers_the_else_value_at_an_unlisted_argument() {
    let output = run("(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(assert (= (f (bvadd a #x01)) #x07))
(assert (= a #x01))
(check-sat)
(get-model)
(get-value ((f a) (f #x02) (f #x55)))
");
    let values = output.last().cloned().unwrap_or_default();
    for expected in ["((f a) #x07)", "((f #x02) #x07)", "((f #x55) #x07)"] {
        assert!(values.contains(expected), "expected {expected} in {values}");
    }
}

/// `r3/atk/a3`: where the interpretation *does* list the tuple, `(get-value)`
/// answers that entry — the else-value is the fallback, not the answer.
#[test]
fn get_value_answers_a_listed_entry_rather_than_the_else_value() {
    let output = run("(set-logic QF_UFBV)
(declare-fun f ((_ BitVec 8)) (_ BitVec 8))
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (= (f a) #x07))
(assert (= (f b) #x09))
(assert (distinct a b))
(check-sat)
(get-model)
(get-value ((f a) (f b)))
");
    let values = output.last().cloned().unwrap_or_default();
    assert!(values.contains("((f a) #x07)"), "{values}");
    assert!(values.contains("((f b) #x09)"), "{values}");
}

// ---------------------------------------------------------------------------
// (4) the two commands answer the same reads
// ---------------------------------------------------------------------------

/// The value a printed array chain gives at `index`, read the way SMT-LIB
/// reads it: the outermost matching `store` wins, and `((as const S) d)` is
/// `d` everywhere.
///
/// Only the shapes `(get-model)` prints for a bit-vector array are handled —
/// a nested `store` chain over an array constant — which is all these scripts
/// produce.
fn chain_value_at(chain: &str, index: &str) -> Option<String> {
    let chain = chain.trim();
    if let Some(rest) = chain.strip_prefix("(store ") {
        // `(store <base> <index> <value>)`: split off the two trailing atoms.
        let body = rest.strip_suffix(')')?.trim_end();
        let (body, value) = body.rsplit_once(' ')?;
        let (base, written) = body.trim_end().rsplit_once(' ')?;
        if written.trim() == index {
            return Some(value.trim().to_string());
        }
        return chain_value_at(base, index);
    }
    if let Some(rest) = chain.strip_prefix("((as const ") {
        // `((as const <sort>) <default>)`: the default is the last atom.
        let body = rest.strip_suffix(')')?.trim_end();
        let (_, default) = body.rsplit_once(' ')?;
        return Some(default.trim().to_string());
    }
    None
}

/// The `(define-fun <name> () <sort> <body>)` body in a model block.
fn binding_body(model: &str, name: &str) -> Option<String> {
    let needle = format!("(define-fun {name} () ");
    let line = model
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&needle))?;
    let rest = line.strip_prefix(&needle)?.strip_suffix(')')?;
    // Skip the sort, which is either one atom or a parenthesised sort term.
    let rest = rest.trim_start();
    let body = if rest.starts_with('(') {
        let mut depth = 0usize;
        let mut end = 0usize;
        for (position, byte) in rest.bytes().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = position + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        &rest[end..]
    } else {
        rest.split_once(' ').map(|(_, tail)| tail).unwrap_or("")
    };
    Some(body.trim().to_string())
}

/// The answer `(get-value (<key>))` gave, from a response line.
fn answered(response: &str, key: &str) -> Option<String> {
    let needle = format!("({key} ");
    let start = response.find(&needle)? + needle.len();
    let rest = &response[start..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().to_string())
}

/// Every read `(get-value …)` answers must be the entry the printed chain
/// carries at that index — for a plain index, for a compound index term, and
/// for an array the model describes only through an equality.
///
/// This is the invariant decisions (2), (3) and (4) are about: one model, two
/// commands.  `Context::array_class_read` and the array renderer are separate
/// walks over the same three sources, so a change to one that is not made to
/// the other shows up here.
#[test]
fn every_array_read_agrees_with_the_printed_chain() {
    for (script, array, indices) in [
        (
            "(set-logic QF_ABV)\n\
             (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
             (declare-const k (_ BitVec 2))\n\
             (declare-const v (_ BitVec 1))\n\
             (assert (= v (select a (bvadd k #b10))))\n\
             (assert (= v #b1))\n\
             (check-sat)\n",
            "a",
            ["#b00", "#b01", "#b10", "#b11"],
        ),
        (
            "(set-logic QF_ABV)\n\
             (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
             (declare-const b (Array (_ BitVec 2) (_ BitVec 1)))\n\
             (assert (distinct a b))\n\
             (assert (= (select a #b01) #b1))\n\
             (check-sat)\n",
            "a",
            ["#b00", "#b01", "#b10", "#b11"],
        ),
        (
            "(set-logic QF_ABV)\n\
             (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
             (declare-const i (_ BitVec 2))\n\
             (assert (= (store a i #b1) ((as const (Array (_ BitVec 2) (_ BitVec 1))) #b1)))\n\
             (check-sat)\n",
            "a",
            ["#b00", "#b01", "#b10", "#b11"],
        ),
        // A read whose index is an uninterpreted application.  This is the
        // shape `publish_index_leaves` treats differently from the other
        // three: `Apply` is an opaque leaf that is deliberately given no
        // default, because its value belongs to the printed interpretation of
        // `f` rather than to a fabricated circuit leaf.  If that leaves the
        // index unpublished, the renderer drops the entry and the chain no
        // longer covers the whole index sort.
        (
            "(set-logic QF_AUFBV)\n\
             (declare-fun f ((_ BitVec 2)) (_ BitVec 2))\n\
             (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
             (declare-const k (_ BitVec 2))\n\
             (declare-const v (_ BitVec 1))\n\
             (assert (= v (select a (f k))))\n\
             (assert (= v #b1))\n\
             (check-sat)\n",
            "a",
            ["#b00", "#b01", "#b10", "#b11"],
        ),
    ] {
        // One run for both commands.  `execute_script` parses the whole
        // script before it solves, so a `(get-value)` in the script puts its
        // terms into the graph the model is built over — asking the two
        // questions in two runs compares two different (both valid) models.
        let mut query = String::from(script);
        query.push_str("(get-model)\n(get-value (");
        for index in indices {
            query.push_str(&format!("(select {array} {index}) "));
        }
        query.push_str("))\n");
        let output = run(&query);
        assert_eq!(
            output.first().map(String::as_str),
            Some("sat"),
            "expected sat: {output:?}"
        );
        let model = output
            .iter()
            .find(|line| line.starts_with("(model"))
            .cloned()
            .unwrap_or_else(|| panic!("no model block: {output:?}"));
        let chain = binding_body(&model, array)
            .unwrap_or_else(|| panic!("no binding for {array} in {model}"));
        let response = output
            .iter()
            .find(|line| line.contains(&format!("(select {array} ")))
            .cloned()
            .unwrap_or_else(|| panic!("no (get-value) response for {script}"));
        for index in indices {
            let key = format!("(select {array} {index})");
            let answer = answered(&response, &key)
                .unwrap_or_else(|| panic!("no answer for {key} in {response}"));
            let printed = chain_value_at(&chain, index)
                .unwrap_or_else(|| panic!("cannot read {chain} at {index}"));
            assert_eq!(
                answer, printed,
                "(get-value) and (get-model) disagree about {key}: \
                 {answer} vs {printed} (chain {chain})"
            );
        }
        assert!(
            replay_is_sat(script, &model),
            "the model must satisfy its own script: {model}"
        );
    }
}

/// The read itself, at an index that is an uninterpreted application: the
/// answer `(get-value)` gives for `(select a (f k))` is the entry the printed
/// chain carries at the value `(get-value)` gives for `(f k)`.
///
/// Stronger than the fourth case of
/// [`every_array_read_agrees_with_the_printed_chain`], which only asks that
/// the chain cover the index sort.  The two sides resolve an `Apply` index by
/// different routes — the renderer publishes it through
/// `opaque_leaves::publish_index_leaves`, which gives an `Apply` leaf no
/// default because its value belongs to the printed interpretation of `f`,
/// while `get_value::array_read_value` resolves it through `model_value_in`
/// and its `euf_class_value` fallback.  Two routes to one index is exactly
/// how a read gets published at the wrong place.
#[test]
fn a_read_at_an_uninterpreted_index_agrees_with_the_printed_chain() {
    let script = "(set-logic QF_AUFBV)\n\
                  (declare-fun f ((_ BitVec 2)) (_ BitVec 2))\n\
                  (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
                  (declare-const k (_ BitVec 2))\n\
                  (declare-const v (_ BitVec 1))\n\
                  (assert (= v (select a (f k))))\n\
                  (assert (= v #b1))\n\
                  (check-sat)\n";
    // One run for both commands; see the array test above for why.
    let query = format!("{script}(get-model)\n(get-value ((f k) (select a (f k))))\n");
    let output = run(&query);
    assert_eq!(
        output.first().map(String::as_str),
        Some("sat"),
        "expected sat: {output:?}"
    );
    let model = output
        .iter()
        .find(|line| line.starts_with("(model"))
        .cloned()
        .unwrap_or_else(|| panic!("no model block: {output:?}"));
    let response = output
        .iter()
        .find(|line| line.contains("(select a (f k))"))
        .cloned()
        .unwrap_or_else(|| panic!("no (get-value) response: {output:?}"));
    let chain = binding_body(&model, "a").unwrap_or_else(|| panic!("no binding for a in {model}"));
    let index =
        answered(&response, "(f k)").unwrap_or_else(|| panic!("no answer for (f k) in {response}"));
    let answer = answered(&response, "(select a (f k))")
        .unwrap_or_else(|| panic!("no answer for the read in {response}"));
    let printed =
        chain_value_at(&chain, &index).unwrap_or_else(|| panic!("cannot read {chain} at {index}"));
    assert_eq!(
        answer, printed,
        "(get-value) reads a at {index} as {answer}, but the printed chain \
         carries {printed} there (chain {chain})"
    );
    assert!(
        replay_is_sat(script, &model),
        "the model must satisfy its own script: {model}"
    );
}

/// The same invariant for a datatype selector: `(get-value ((snd q)))` is the
/// field of the constructor value `(get-model)` prints for `q`.
#[test]
fn a_datatype_selector_agrees_with_the_printed_constructor() {
    let script = "(set-logic QF_AUFDTBV)\n\
                  (declare-datatypes ((Pair 0)) (((mk (fst (_ BitVec 2)) (snd (_ BitVec 2))))))\n\
                  (declare-const p Pair)\n\
                  (declare-const q Pair)\n\
                  (assert (distinct p q))\n\
                  (assert (= (fst p) #b01))\n\
                  (check-sat)\n";
    // One run for both commands; see the array test above for why.
    let query = format!("{script}(get-model)\n(get-value ((fst q) (snd q)))\n");
    let output = run(&query);
    assert_eq!(
        output.first().map(String::as_str),
        Some("sat"),
        "expected sat: {output:?}"
    );
    let model = output
        .iter()
        .find(|line| line.starts_with("(model"))
        .cloned()
        .unwrap_or_else(|| panic!("no model block: {output:?}"));
    let printed = binding_body(&model, "q").unwrap_or_else(|| panic!("no q: {model}"));
    let fields: Vec<&str> = printed
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split_whitespace()
        .collect();
    let response = output
        .iter()
        .find(|line| line.contains("(fst q)"))
        .cloned()
        .unwrap_or_else(|| panic!("no response"));
    assert_eq!(
        answered(&response, "(fst q)").as_deref(),
        fields.get(1).copied(),
        "fst disagrees with {printed}: {response}"
    );
    assert_eq!(
        answered(&response, "(snd q)").as_deref(),
        fields.get(2).copied(),
        "snd disagrees with {printed}: {response}"
    );
}
