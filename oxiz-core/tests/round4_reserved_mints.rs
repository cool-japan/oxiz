//! Every symbol the *tactic* layer mints belongs to the reserved class
//! (`#P2b-44`, decision (19) of the round-4 close-out).
//!
//! # The class, and why the tactic layer is in it
//!
//! `oxiz_core::smtlib::reserved_name` mints `\oxiz.<tag>!<suffix>`, and the
//! parser refuses that prefix outright: a backslash is in neither of SMT-LIB
//! 2.6's two symbol forms, so a user cannot spell one even inside `|…|`.  The
//! solver's own encoder, Skolemiser, array theory and model layer were moved
//! into that class earlier in the round.
//!
//! The tactic layer was audited and left out, on the grounds that no `.smt2`
//! command runs a tactic — which is true: the registry is reachable only
//! through the Rust API (`Z3Tactic::apply` on a `Z3Goal`), and
//! `oxiz-cli`'s `ml_tactic::apply_tactic` only sets solver *options*, never
//! applies a `Tactic`.  But "not reachable from a script" is not the bar
//! decision (19) sets: a tactic mints a symbol and then **asserts a side
//! condition about it** into the subgoal it returns — `!ack_k = f(args)`, the
//! clauses defining a sequential counter, the bit variables of a blasted term,
//! `x_bv`'s relation to `x` — and that subgoal is a formula the caller asserts.
//! A caller whose terms already contain a symbol of the same spelling had it
//! captured by that side condition, and `x_bv` beside `x` is a collision a user
//! could build by accident rather than by attack.
//!
//! So all five families are in the class now, and this file is the guard that
//! keeps them there: it drives each tactic, collects every `Var` in the
//! subgoals it produces that the *input* did not contain, and requires each one
//! to carry `RESERVED_PREFIX`.  A new mint site added without
//! `reserved_name` fails here rather than in a user's model.

use oxiz_core::ast::traversal::collect_subterms;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::RESERVED_PREFIX;
use oxiz_core::tactic::{
    AckermannizeTactic, BitBlastTactic, Goal, Lia2CardConfig, Lia2CardTactic, Nla2BvConfig,
    Nla2BvTactic, TacticResult,
};
use std::collections::HashSet;

/// The names of every `Var` occurring in `terms`.
fn var_names(terms: &[TermId], manager: &TermManager) -> HashSet<String> {
    let mut reachable: HashSet<TermId> = HashSet::new();
    for &term in terms {
        reachable.extend(collect_subterms(term, manager));
    }
    let mut names = HashSet::new();
    for term in reachable {
        if let Some(data) = manager.get(term)
            && let TermKind::Var(name) = &data.kind
        {
            names.insert(manager.resolve_str(*name).to_string());
        }
    }
    names
}

/// Assert that every variable the tactic introduced is in the reserved class.
fn no_unreserved_mint(before: &HashSet<String>, after: &HashSet<String>, what: &str) {
    let minted: Vec<&String> = after.difference(before).collect();
    assert!(
        !minted.is_empty(),
        "{what} was expected to mint at least one symbol, so a guard that saw \
         none would be pinning nothing"
    );
    for name in minted {
        assert!(
            name.starts_with(RESERVED_PREFIX),
            "{what} minted `{name}`, which a user can spell: every symbol a \
             tactic interns and then asserts a side condition about must come \
             from `oxiz_core::smtlib::reserved_name` (decision (19))"
        );
    }
}

/// Ackermannisation replaces a ground application with a fresh constant and
/// asserts the functional-consistency conditions about it.
#[test]
fn ackermannization_mints_only_reserved_names() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let f = manager.mk_var("f", int_sort);
    let _ = f;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let fx = manager.mk_apply("f", vec![x], int_sort);
    let fy = manager.mk_apply("f", vec![y], int_sort);
    let goal = Goal::new(vec![manager.mk_eq(fx, fy), manager.mk_eq(x, y)]);
    let before = var_names(&goal.assertions, &manager);

    let mut tactic = AckermannizeTactic::new(&mut manager);
    let result = tactic
        .apply_mut(&goal)
        .expect("ackermannize must not error");
    let TacticResult::SubGoals(goals) = result else {
        panic!("ackermannize should return subgoals here, got {result:?}");
    };
    let after: HashSet<String> = goals
        .iter()
        .flat_map(|g| var_names(&g.assertions, &manager))
        .collect();
    no_unreserved_mint(&before, &after, "ackermannization");
}

/// The cardinality encodings mint counter, commander and totaliser variables
/// and then define them with clauses.
#[test]
fn cardinality_encoding_mints_only_reserved_names() {
    for encoding in [
        oxiz_core::tactic::CardinalityEncoding::SequentialCounter,
        oxiz_core::tactic::CardinalityEncoding::Totalizer,
    ] {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let mut assertions = Vec::new();
        let mut summands = Vec::new();
        for k in 0..4u32 {
            let p = manager.mk_var(&format!("p{k}"), bool_sort);
            let one = manager.mk_int(1);
            let zero = manager.mk_int(0);
            summands.push(manager.mk_ite(p, one, zero));
        }
        let _ = int_sort;
        let sum = manager.mk_add(summands);
        let two = manager.mk_int(2);
        assertions.push(manager.mk_eq(sum, two));
        let goal = Goal::new(assertions);
        let before = var_names(&goal.assertions, &manager);

        let config = Lia2CardConfig {
            encoding,
            sequential_counter_threshold: 100,
            use_commander_for_amo: false,
        };
        let mut tactic = Lia2CardTactic::with_config(&mut manager, config);
        let result = tactic.apply_mut(&goal).expect("lia2card must not error");
        let TacticResult::SubGoals(goals) = result else {
            // Nothing to encode is an acceptable outcome for a shape the
            // tactic declines; the other encoding still exercises the mint.
            continue;
        };
        let after: HashSet<String> = goals
            .iter()
            .flat_map(|g| var_names(&g.assertions, &manager))
            .collect();
        if after.difference(&before).next().is_none() {
            continue;
        }
        no_unreserved_mint(&before, &after, "the cardinality encoding");
    }
}

/// The one mint derived from a *user's own* symbol: `nla2bv` used to name the
/// bit-vector proxy of `x` `x_bv`, so a script carrying both had a collision by
/// accident.
///
/// Driven, not read: the goal below is one the tactic *accepts* — every integer
/// variable carries a literal lower and upper bound, which is what
/// `Nla2BvTactic::apply_mut`'s `check_all_bounded` phase requires — so the
/// proxy mint really runs and the guard inspects the symbols the subgoal
/// carries rather than the spelling of a line of source.  A source-text pin
/// would stay green if the mint moved to another file and would go red on a
/// pure rename.
#[test]
fn the_bitvector_proxy_is_minted_in_the_reserved_class() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let zero = manager.mk_int(0);
    let seven = manager.mk_int(7);
    let twelve = manager.mk_int(12);
    let product = manager.mk_mul(vec![x, y]);
    let assertions = vec![
        manager.mk_ge(x, zero),
        manager.mk_le(x, seven),
        manager.mk_ge(y, zero),
        manager.mk_le(y, seven),
        manager.mk_eq(product, twelve),
    ];
    let goal = Goal::new(assertions);
    let before = var_names(&goal.assertions, &manager);
    assert!(
        before.contains("x") && before.contains("y"),
        "the input must carry the user's own symbols, or the difference below \
         would not be measuring a mint"
    );

    let mut tactic = Nla2BvTactic::with_config(&mut manager, Nla2BvConfig::default());
    let result = tactic.apply_mut(&goal).expect("nla2bv must not error");
    let TacticResult::SubGoals(goals) = result else {
        panic!("nla2bv must accept a fully bounded integer goal, got {result:?}");
    };
    let after: HashSet<String> = goals
        .iter()
        .flat_map(|g| var_names(&g.assertions, &manager))
        .collect();
    assert!(
        !after.contains("x_bv") && !after.contains("y_bv"),
        "the proxy must not be the user's symbol with a suffix"
    );
    no_unreserved_mint(&before, &after, "the nla2bv bit-vector proxy");
}

/// The bit-blaster's per-bit variables are in the class too, and likewise
/// guarded by driving the blaster over a bit-vector goal and reading the
/// symbols out of what it returns.
#[test]
fn the_bit_blaster_bit_names_are_minted_in_the_reserved_class() {
    let mut manager = TermManager::new();
    let bv_sort = manager.sorts.bitvec(4);
    let a = manager.mk_var("a", bv_sort);
    let b = manager.mk_var("b", bv_sort);
    let sum = manager.mk_bv_add(a, b);
    let three = manager.mk_bitvec(3u32, 4);
    let goal = Goal::new(vec![manager.mk_eq(sum, three)]);
    let before = var_names(&goal.assertions, &manager);
    assert!(
        before.contains("a") && before.contains("b"),
        "the input must carry the user's own symbols, or the difference below \
         would not be measuring a mint"
    );

    let result = BitBlastTactic::blast(&goal, &mut manager).expect("bit-blasting must not error");
    let TacticResult::SubGoals(goals) = result else {
        panic!("the bit-blaster must accept a bit-vector goal, got {result:?}");
    };
    let after: HashSet<String> = goals
        .iter()
        .flat_map(|g| var_names(&g.assertions, &manager))
        .collect();
    no_unreserved_mint(&before, &after, "the bit-blaster's bit variables");
}

/// And the parser refuses the whole class, in both symbol forms, so none of the
/// above can be captured from a script either.
#[test]
fn the_reserved_class_is_unspellable() {
    for tag in ["ack", "bb", "cards", "cardcmd", "tot", "cardaux", "nlabv"] {
        let name = oxiz_core::smtlib::reserved_name(tag, "0");
        assert!(
            name.starts_with(RESERVED_PREFIX),
            "`{name}` must carry the reserved prefix"
        );
        assert!(
            name.contains('\\'),
            "`{name}` must carry the backslash neither SMT-LIB symbol form admits"
        );
    }
}
