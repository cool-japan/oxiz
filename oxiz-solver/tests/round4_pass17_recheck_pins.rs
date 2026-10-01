//! Round-4 adversarial recheck 17 (decision (80): the commit criterion of
//! decision (78) measured) — pins for what the recheck found open on re-fix
//! pass 17's final tree.
//!
//! Every script below is a corpus script of the recheck's own fresh seed of
//! `scripts/round4/gen_dt.py` (seed 30100192, 600 scripts), quoted
//! **verbatim**, or a reduction of one.  z3 4.15.4 judged every verdict and
//! every model the doc of a test names.
//!
//! # How to read this file
//!
//! * A **HOLE** asserts what this tree answers today.  When the defect is
//!   fixed the assertion fails with **THE HOLE IS CLOSED** — invert the pin
//!   then (assert the right answer and, for a model, its exact evaluation),
//!   never delete it.  §1 pins a WRONG `sat`: today's answer *is* the wrong
//!   verdict, so the pin asserts it and closes on anything else (`unsat`, or
//!   an honest `unknown`).
//! * A printed datatype model is judged here by EXACT EVALUATION
//!   ([`DtEval`]), not by a replay through this solver: §2's falsifying model
//!   escapes the printed-model net precisely because a fresh solver of this
//!   build answers `sat` on the closed (ground) assertion it falsifies, so a
//!   replay through the same solver would confirm it.  The evaluator reads
//!   the printed `define-fun` lines (constants and the `ite` tables of
//!   functions) and evaluates every assertion in scope over integers,
//!   Booleans, constructor values and `@uc_` witnesses (distinct witnesses
//!   are distinct elements, as the printed model says); a selector applied to
//!   the wrong constructor is unspecified and makes a term undecided, never
//!   true or false.
//! * No test installs a wall clock (decision (16)); every script decides in
//!   well under a second in the release probe.

use oxiz_solver::Context;
use std::collections::HashMap;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// Every `sat` / `unsat` / `unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// An exact evaluator for the `gen_dt.py` fragment.
// ---------------------------------------------------------------------------

/// An S-expression.
#[derive(Clone, Debug, PartialEq)]
enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

fn parse_all(text: &str) -> Vec<Sx> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut in_comment = false;
    for ch in text.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            current.push(ch);
            if ch == '"' {
                in_string = false;
                tokens.push(std::mem::take(&mut current));
            }
            continue;
        }
        match ch {
            '"' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                in_string = true;
                current.push(ch);
            }
            ';' => in_comment = true,
            '(' | ')' => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    let mut stack: Vec<Vec<Sx>> = vec![Vec::new()];
    for token in tokens {
        match token.as_str() {
            "(" => stack.push(Vec::new()),
            ")" => {
                let Some(done) = stack.pop() else {
                    panic!("unbalanced S-expression")
                };
                match stack.last_mut() {
                    Some(parent) => parent.push(Sx::List(done)),
                    None => panic!("unbalanced S-expression"),
                }
            }
            _ => match stack.last_mut() {
                Some(top) => top.push(Sx::Atom(token)),
                None => panic!("unbalanced S-expression"),
            },
        }
    }
    stack.into_iter().next().unwrap_or_default()
}

/// A value of the fragment.
#[derive(Clone, Debug, PartialEq)]
enum Val {
    Int(i64),
    Bool(bool),
    Ctor(String, Vec<Val>),
    Witness(String),
}

/// The exact evaluator: the datatype declarations of a script and one
/// printed model.
#[derive(Default)]
struct DtEval {
    /// Constructor name → its selector names, in order.
    ctors: HashMap<String, Vec<String>>,
    /// Selector name → (its constructor, its position).
    selectors: HashMap<String, (String, usize)>,
    consts: HashMap<String, Val>,
    funcs: HashMap<String, (Vec<String>, Sx)>,
}

impl DtEval {
    fn declare_datatypes(&mut self, bodies: &Sx) {
        let Sx::List(bodies) = bodies else { return };
        for body in bodies {
            let Sx::List(ctors) = body else { continue };
            for ctor in ctors {
                let Sx::List(parts) = ctor else { continue };
                let Some(Sx::Atom(name)) = parts.first() else {
                    continue;
                };
                let mut fields = Vec::new();
                for (position, field) in parts.iter().skip(1).enumerate() {
                    if let Sx::List(pair) = field
                        && let Some(Sx::Atom(selector)) = pair.first()
                    {
                        fields.push(selector.clone());
                        self.selectors
                            .insert(selector.clone(), (name.clone(), position));
                    }
                }
                self.ctors.insert(name.clone(), fields);
            }
        }
    }

    /// Install one printed `(model …)`.
    fn load_model(&mut self, model: &Sx) {
        self.consts.clear();
        self.funcs.clear();
        let Sx::List(items) = model else { return };
        for item in items.iter().skip(1) {
            let Sx::List(def) = item else { continue };
            if def.len() != 5 || def.first() != Some(&Sx::Atom("define-fun".into())) {
                continue;
            }
            let (Some(Sx::Atom(name)), Some(Sx::List(params)), Some(body)) =
                (def.get(1), def.get(2), def.get(4))
            else {
                continue;
            };
            if params.is_empty() {
                if let Some(value) = self.eval(body, &HashMap::new()) {
                    self.consts.insert(name.clone(), value);
                }
            } else {
                let names = params
                    .iter()
                    .filter_map(|p| match p {
                        Sx::List(pair) => match pair.first() {
                            Some(Sx::Atom(param)) => Some(param.clone()),
                            _ => None,
                        },
                        Sx::Atom(_) => None,
                    })
                    .collect();
                self.funcs.insert(name.clone(), (names, body.clone()));
            }
        }
    }

    fn int(&self, term: &Sx, locals: &HashMap<String, Val>) -> Option<i64> {
        match self.eval(term, locals)? {
            Val::Int(n) => Some(n),
            _ => None,
        }
    }

    fn truth(&self, term: &Sx, locals: &HashMap<String, Val>) -> Option<bool> {
        match self.eval(term, locals)? {
            Val::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// The value of `term`; `None` where the printed model leaves it
    /// unspecified (a selector of the wrong constructor) or the fragment has
    /// no reading for it.
    fn eval(&self, term: &Sx, locals: &HashMap<String, Val>) -> Option<Val> {
        match term {
            Sx::Atom(atom) => {
                if let Ok(n) = atom.parse::<i64>() {
                    return Some(Val::Int(n));
                }
                match atom.as_str() {
                    "true" => return Some(Val::Bool(true)),
                    "false" => return Some(Val::Bool(false)),
                    _ => {}
                }
                if let Some(value) = locals.get(atom).or_else(|| self.consts.get(atom)) {
                    return Some(value.clone());
                }
                if atom.starts_with("@uc_") {
                    return Some(Val::Witness(atom.clone()));
                }
                match self.ctors.get(atom) {
                    Some(fields) if fields.is_empty() => Some(Val::Ctor(atom.clone(), Vec::new())),
                    _ => None,
                }
            }
            Sx::List(items) => {
                let (head, args) = items.split_first()?;
                if let Sx::List(index) = head {
                    // `((_ is C) t)`
                    if let (Some(Sx::Atom(us)), Some(Sx::Atom(is)), Some(Sx::Atom(ctor))) =
                        (index.first(), index.get(1), index.get(2))
                        && us == "_"
                        && is == "is"
                    {
                        let Val::Ctor(built, _) = self.eval(args.first()?, locals)? else {
                            return None;
                        };
                        return Some(Val::Bool(&built == ctor));
                    }
                    return None;
                }
                let Sx::Atom(op) = head else { return None };
                match op.as_str() {
                    "and" => {
                        let mut open = false;
                        for arg in args {
                            match self.truth(arg, locals) {
                                Some(false) => return Some(Val::Bool(false)),
                                Some(true) => {}
                                None => open = true,
                            }
                        }
                        (!open).then_some(Val::Bool(true))
                    }
                    "or" => {
                        let mut open = false;
                        for arg in args {
                            match self.truth(arg, locals) {
                                Some(true) => return Some(Val::Bool(true)),
                                Some(false) => {}
                                None => open = true,
                            }
                        }
                        (!open).then_some(Val::Bool(false))
                    }
                    "not" => Some(Val::Bool(!self.truth(args.first()?, locals)?)),
                    "=>" => {
                        let premise = self.truth(args.first()?, locals);
                        let conclusion = self.truth(args.get(1)?, locals);
                        match (premise, conclusion) {
                            (Some(false), _) | (_, Some(true)) => Some(Val::Bool(true)),
                            (Some(true), Some(false)) => Some(Val::Bool(false)),
                            _ => None,
                        }
                    }
                    "ite" => {
                        if self.truth(args.first()?, locals)? {
                            self.eval(args.get(1)?, locals)
                        } else {
                            self.eval(args.get(2)?, locals)
                        }
                    }
                    "=" => {
                        let values: Option<Vec<Val>> =
                            args.iter().map(|a| self.eval(a, locals)).collect();
                        let values = values?;
                        Some(Val::Bool(values.windows(2).all(|w| w[0] == w[1])))
                    }
                    "distinct" => {
                        let values: Option<Vec<Val>> =
                            args.iter().map(|a| self.eval(a, locals)).collect();
                        let values = values?;
                        let mut apart = true;
                        for (i, a) in values.iter().enumerate() {
                            if values.iter().skip(i + 1).any(|b| a == b) {
                                apart = false;
                            }
                        }
                        Some(Val::Bool(apart))
                    }
                    "+" => {
                        let mut sum: i64 = 0;
                        for arg in args {
                            sum = sum.checked_add(self.int(arg, locals)?)?;
                        }
                        Some(Val::Int(sum))
                    }
                    "*" => {
                        let mut product: i64 = 1;
                        for arg in args {
                            product = product.checked_mul(self.int(arg, locals)?)?;
                        }
                        Some(Val::Int(product))
                    }
                    "-" => {
                        let first = self.int(args.first()?, locals)?;
                        if args.len() == 1 {
                            return Some(Val::Int(first.checked_neg()?));
                        }
                        let mut out = first;
                        for arg in args.iter().skip(1) {
                            out = out.checked_sub(self.int(arg, locals)?)?;
                        }
                        Some(Val::Int(out))
                    }
                    "<=" | "<" | ">=" | ">" => {
                        let numbers: Option<Vec<i64>> =
                            args.iter().map(|a| self.int(a, locals)).collect();
                        let numbers = numbers?;
                        let holds = numbers.windows(2).all(|w| match op.as_str() {
                            "<=" => w[0] <= w[1],
                            "<" => w[0] < w[1],
                            ">=" => w[0] >= w[1],
                            _ => w[0] > w[1],
                        });
                        Some(Val::Bool(holds))
                    }
                    name => {
                        if self.ctors.contains_key(name) {
                            let fields: Option<Vec<Val>> =
                                args.iter().map(|a| self.eval(a, locals)).collect();
                            return Some(Val::Ctor(name.to_string(), fields?));
                        }
                        if let Some((ctor, position)) = self.selectors.get(name) {
                            let Val::Ctor(built, fields) = self.eval(args.first()?, locals)? else {
                                return None;
                            };
                            // A selector of another constructor is unspecified.
                            return if &built == ctor {
                                fields.get(*position).cloned()
                            } else {
                                None
                            };
                        }
                        let (params, body) = self.funcs.get(name)?;
                        let mut bound = HashMap::new();
                        for (param, arg) in params.iter().zip(args) {
                            bound.insert(param.clone(), self.eval(arg, locals)?);
                        }
                        self.eval(body, &bound)
                    }
                }
            }
        }
    }
}

/// How one check's printed model reads under exact evaluation.
#[derive(Debug, PartialEq)]
enum ModelReading {
    /// `(get-model)` answered `(error "model not certified: …")`.
    Withheld,
    /// Every assertion in scope evaluates to `true`.
    Holds,
    /// The assertion at this 1-based position among those in scope is
    /// `false`.
    Falsifies(usize),
    /// No assertion is false and at least one is undecided.
    Undecided,
    /// No `(get-model)` followed, or the check was not `sat`.
    NoModel,
}

/// Each check's verdict and how its printed model reads, the script's own
/// responses walked command by command (push / pop scopes kept).
fn judge(script: &str, lines: &[String]) -> Vec<(String, ModelReading)> {
    let commands = parse_all(script);
    let mut responses = lines
        .iter()
        .filter(|line| line.as_str() != "success")
        .cloned();
    let mut eval = DtEval::default();
    let mut frames: Vec<Vec<Sx>> = vec![Vec::new()];
    let mut out: Vec<(String, ModelReading)> = Vec::new();
    for command in &commands {
        let Sx::List(parts) = command else { continue };
        let Some(Sx::Atom(head)) = parts.first() else {
            continue;
        };
        match head.as_str() {
            "declare-datatypes" => {
                if let Some(bodies) = parts.get(2) {
                    eval.declare_datatypes(bodies);
                }
            }
            "assert" => {
                if let (Some(frame), Some(term)) = (frames.last_mut(), parts.get(1)) {
                    frame.push(term.clone());
                }
            }
            "push" => frames.push(Vec::new()),
            "pop" => {
                if frames.len() > 1 {
                    frames.pop();
                }
            }
            "check-sat" => {
                let verdict = responses.next().unwrap_or_default();
                out.push((verdict, ModelReading::NoModel));
            }
            "get-model" => {
                let response = responses.next().unwrap_or_default();
                let Some(last) = out.last_mut() else { continue };
                if last.0 != "sat" {
                    continue;
                }
                if response.contains("model not certified") {
                    last.1 = ModelReading::Withheld;
                    continue;
                }
                let Some(model) = parse_all(&response).into_iter().next() else {
                    continue;
                };
                eval.load_model(&model);
                let mut reading = ModelReading::Holds;
                for (position, assertion) in frames.iter().flatten().enumerate() {
                    match eval.truth(assertion, &HashMap::new()) {
                        Some(true) => {}
                        Some(false) => {
                            reading = ModelReading::Falsifies(position + 1);
                            break;
                        }
                        None => reading = ModelReading::Undecided,
                    }
                }
                last.1 = reading;
            }
            "get-value" | "get-info" | "get-option" | "echo" | "get-assignment"
            | "get-unsat-core" | "get-assertions" | "get-proof" => {
                let _ = responses.next();
            }
            _ => {}
        }
    }
    out
}

/// The datatypes every `gen_dt.py` script declares.
const DT_HEAD: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n";

/// The evaluator itself, on models whose reading is known: a list
/// equality, a table, an unspecified selector, a witness comparison.
#[test]
fn the_exact_evaluator_reads_the_gen_dt_fragment() {
    let script = format!(
        "{DT_HEAD}(declare-fun h (Int) L)\n(declare-const x Int)\n(declare-const l1 L)\n\
         (declare-const u1 U)\n(declare-const u2 U)\n\
         (assert (= (h x) (cons 1 nil)))\n(assert (distinct u1 u2))\n(check-sat)\n(get-model)\n\
         (assert (= l1 (cons x nil)))\n(check-sat)\n(get-model)\n\
         (assert (< 0 (hd (tl l1))))\n(check-sat)\n(get-model)\n"
    );
    let model = |l1: &str| {
        format!(
            "(model\n  (define-fun x () Int -2)\n  (define-fun l1 () L {l1})\n  \
             (define-fun u1 () U @uc_U_0)\n  (define-fun u2 () U @uc_U_1)\n  \
             (define-fun h ((x!0 Int)) L (ite (= x!0 -2) (cons 1 nil) nil))\n)"
        )
    };
    let lines: Vec<String> = vec![
        "sat".into(),
        model("nil"),
        "sat".into(),
        model("(cons (- 2) nil)"),
        "sat".into(),
        model("(cons -2 nil)"),
    ];
    let got = judge(&script, &lines);
    assert_eq!(
        got.iter().map(|(_, reading)| reading).collect::<Vec<_>>(),
        vec![
            &ModelReading::Holds,
            &ModelReading::Holds,
            &ModelReading::Undecided
        ],
        "{got:?}"
    );
    let wrong = vec![
        "sat".to_string(),
        model("nil").replace("(cons 1 nil) nil", "nil (cons 1 nil)"),
    ];
    assert_eq!(
        judge(&script, &wrong).first().map(|(_, r)| r),
        Some(&ModelReading::Falsifies(1))
    );
}

// ---------------------------------------------------------------------------
// §1. HOLE (BLOCKER: a WRONG `sat`, every build — 0.3.3, `c4b04b7`, `c702310`, re-fix pass 14, re-fix
//     pass 16 and this tree): a tester over a datatype-sorted `ite` is not tied to the `ite`'s value.
//
//     Found by this recheck's fresh `gen_dt.py` seed 30100192, `d00370` (verbatim below): `l2 = (cons
//     (px p) nil)`, the pushed `(= (ite ((_ is cons) l2) (tl l2) nil) l1)` makes `l1 = nil`, and then the
//     fourth assertion's `(ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) … 0)` is `0`, so `(< 0 …)`
//     is false.  z3: `unsat`; every build: `sat` (this tree withholds the model, "assertion 4 is false" —
//     its own net sees the model is false, but the verdict stands).  Reduced to four lines (`M10`): `l1 =
//     nil` beside `(< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) 1 0))` is `sat` on every
//     build; naming the inner `ite` with a constant (`(= p (ite …))`, then the tester over `p`) answers
//     `unsat`.  The GROUND formula `C318_B` — two testers / equalities over `ite` chains of list literals,
//     no declared symbol at all — is `sat` on every build too, and it is the closed assertion §2's net
//     confirms with a fresh solver.  The suspected mechanism is `TODO.md` `#P2b-88`'s: the datatype axioms
//     (`solver::dt_axioms`) read the terms as written while the encoder replaces every non-Bool `ite` by a
//     proxy (`encode::bool_euf_encoding`), so a tester applied to the written `ite` is never linked to the
//     proxy's class.
// ---------------------------------------------------------------------------

const D00370: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun g (Int) C)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
(declare-fun w (Int) U)\n\
(declare-fun v (L) L)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const l3 L)\n\
(declare-const c1 C)\n\
(declare-const c2 C)\n\
(declare-const p P)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
(assert (and (= l2 (cons (px p) (ite ((_ is cons) nil) (tl nil) nil))) (= (ite ((_ is cons) nil) (tl nil) nil) nil)))\n\
(assert (= (g (px p)) red))\n\
(assert (and (< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) (hd (ite ((_ is cons) l1) (tl l1) nil)) 0)) (or ((_ is nil) (v l2)) (= blue blue))))\n\
(assert (= l1 (h (ite ((_ is cons) (cons y l3)) (hd (cons y l3)) 0))))\n\
(push 1)\n\
(assert (= (ite ((_ is cons) l2) (tl l2) nil) l1))\n\
(check-sat)\n\
(get-model)\n";

const M10: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(assert (= l1 nil))\n\
(assert (< 0 (ite ((_ is cons) (ite ((_ is cons) l1) (tl l1) nil)) 1 0)))\n\
(check-sat)\n";

/// `M10` with the inner `ite` named by a constant: decided (`unsat`) today,
/// the in-test control.
const M10_NAMED: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-const l1 L)\n\
(declare-const p L)\n\
(assert (= l1 nil))\n\
(assert (= p (ite ((_ is cons) l1) (tl l1) nil)))\n\
(assert (< 0 (ite ((_ is cons) p) 1 0)))\n\
(check-sat)\n";

const C318_B: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(assert (or ((_ is nil) (ite (= (+ 2 (- 1)) 2) (cons 1 nil) (ite (= (+ 2 (- 1)) 0) (cons -1 (cons -1 (cons -1 (cons 1 nil)))) (ite (= (+ 2 (- 1)) 1) (cons -4 nil) (ite (= (+ 2 (- 1)) -1) (cons 1 (cons -1 (cons -1 (cons 1 nil)))) (cons 1 nil)))))) (= (ite (= (+ 0 (- 1)) 2) (cons 1 nil) (ite (= (+ 0 (- 1)) 0) (cons -1 (cons -1 (cons -1 (cons 1 nil)))) (ite (= (+ 0 (- 1)) 1) (cons -4 nil) (ite (= (+ 0 (- 1)) -1) (cons 1 (cons -1 (cons -1 (cons 1 nil)))) (cons 1 nil))))) nil)))\n\
(check-sat)\n";

#[test]
fn a_tester_over_a_datatype_ite_answers_sat_where_the_goal_is_unsat() {
    let control = verdicts(&run(M10_NAMED));
    assert_eq!(
        control,
        vec!["unsat"],
        "the named twin is refuted (z3: unsat)"
    );
    for (name, script) in [("d00370", D00370), ("m10", M10), ("c318_b", C318_B)] {
        let lines = run(script);
        let got = verdicts(&lines);
        assert_eq!(
            got,
            vec!["sat"],
            "THE HOLE IS CLOSED (`{name}`, a tester / equality over a datatype `ite`): answered \
             {got:?} (z3: unsat) — invert this pin to assert `unsat` (or at least never `sat`)\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §2. HOLE (BLOCKER: a published model that FALSIFIES its own script, quantifier-free, datatype): the
//     re-fix pass 17 net (decision (79)(a)) decides the falsified assertion by value and then asks a fresh
//     solver to confirm the refutation — and the fresh solver, hit by §1, answers `sat` on the closed
//     (ground) assertion, so the falsifying model is printed.
//
//     This recheck's fresh `gen_dt.py` seed 30100192, `d00318` (verbatim): the first check is `unknown`,
//     the second `sat` with a model whose sixth assertion in scope — `(or (or ((_ is nil) (h (+ 2 (- 1))))
//     (= (h (+ 0 x)) l2)) (= p (mk (ite ((_ is cons) l3) (hd l3) 0) (pc p))))` — is false: `h` prints
//     non-`nil` lists at 1 and at `x = -1`, `l2 = nil`, and `p = (mk -3 green)` against `(mk 1 green)`.
//     z3 judges the model falsifying.  The assertion closed over the printed model is exactly the
//     ground formula `C318_B`'s shape, which a fresh solver of this build answers `sat` (§1).
// ---------------------------------------------------------------------------

const D00318: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-sort U 0)\n\
(declare-fun h (Int) L)\n\
(declare-fun g (Int) C)\n\
(declare-fun k (L) Int)\n\
(declare-fun q (C) Int)\n\
(declare-fun w (Int) U)\n\
(declare-fun v (L) L)\n\
(declare-const x Int)\n\
(declare-const y Int)\n\
(declare-const z Int)\n\
(declare-const l1 L)\n\
(declare-const l2 L)\n\
(declare-const l3 L)\n\
(declare-const c1 C)\n\
(declare-const c2 C)\n\
(declare-const p P)\n\
(declare-const u1 U)\n\
(declare-const u2 U)\n\
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))\n\
(assert (not (or (distinct (pc p) (pc p)) (= (k (cons x nil)) y))))\n\
(assert (<= x (+ (ite ((_ is cons) l2) (hd l2) 0) (- 1))))\n\
(assert (distinct nil l3))\n\
(assert (and (or (distinct l2 (cons (px p) l1)) (= (cons (k l1) (cons z (h z))) nil)) (distinct (cons y (cons x (cons x l3))) (h (ite ((_ is cons) (cons z l1)) (hd (cons z l1)) 0)))))\n\
(push 1)\n\
(assert (or (or ((_ is nil) (h (+ 2 (- 1)))) (= (h (+ 0 x)) l2)) (= p (mk (ite ((_ is cons) l3) (hd l3) 0) (pc p)))))\n\
(assert (distinct (w (px p)) u1))\n\
(check-sat)\n\
(get-model)\n\
(push 1)\n\
(assert (and (= p (mk (k l1) green)) (= l3 (h (ite ((_ is cons) (cons 2 l2)) (hd (cons 2 l2)) 0)))))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_falsifying_datatype_model_escapes_the_net_when_its_confirming_solver_answers_sat() {
    let lines = run(D00318);
    let got = judge(D00318, &lines);
    assert_eq!(got.len(), 2, "{}", lines.join("\n"));
    // The soundness guard whatever the tree answers: z3 answers `sat` at both
    // checks, so neither may be `unsat`.
    assert!(
        got.iter().all(|(verdict, _)| verdict != "unsat"),
        "z3: sat at both checks\n{}",
        lines.join("\n")
    );
    // The first check's model, if one is printed, must read true.
    if let Some((verdict, reading)) = got.first()
        && verdict == "sat"
    {
        assert!(
            matches!(reading, ModelReading::Holds | ModelReading::Withheld),
            "check 0: {reading:?}\n{}",
            lines.join("\n")
        );
    }
    let second = got
        .get(1)
        .map(|(verdict, reading)| (verdict.as_str(), reading));
    assert_eq!(
        second,
        Some(("sat", &ModelReading::Falsifies(6))),
        "THE HOLE IS CLOSED (`d00318`, the net's confirming solver): the second check no longer \
         prints a model falsifying its sixth assertion — invert this pin to assert that every \
         printed model reads true (or is withheld)\n{}",
        lines.join("\n")
    );
}
