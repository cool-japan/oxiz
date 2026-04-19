use oxiz_core::ast::{EGraph, ENode, ENodeKind};
use oxiz_sat::{Lit, Solver, SolverResult};

#[test]
fn watchlist_in_place_unsat_formula() {
    let mut solver = Solver::new();
    let a = solver.new_var();
    let b = solver.new_var();

    solver.add_clause([Lit::pos(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);
    solver.add_clause([Lit::pos(a), Lit::neg(b)]);
    solver.add_clause([Lit::neg(a), Lit::neg(b)]);

    assert_eq!(solver.solve(), SolverResult::Unsat);
}

#[test]
fn eclass_smallvec_spills_past_inline_capacity() {
    let mut egraph = EGraph::new();
    let mut ids = Vec::new();

    for value in 0..8 {
        ids.push(egraph.add(ENode {
            kind: ENodeKind::IntConst(value),
            children: Vec::new(),
        }));
    }

    let root = ids[0];
    for &id in &ids[1..] {
        egraph.merge(root, id);
    }

    let class = egraph
        .get_class(root)
        .expect("merged root should still exist");
    assert_eq!(class.nodes.len(), 8);
}
