//! The exact evaluator of the round-4 datatype pins (adversarial recheck 17,
//! moved here by re-fix pass 18 so `round4_pass17_recheck_pins` and
//! `round4_pass18_fix_pins` judge printed models the same way): an
//! S-expression reader, the values of the `gen_dt.py` fragment (integers,
//! Booleans, constructors, `@uc_` witnesses, bit-vector literals, arrays —
//! nested ones included), the printed `define-fun` lines of one model, and
//! every assertion in scope evaluated per check with push / pop kept.  A
//! printed model is judged by EXACT EVALUATION, never by a replay through
//! the solver under test: recheck 17's `d00318` printed a falsifying model
//! precisely because a fresh solver of that build confirmed it.
//!
//! Included with `#[path = "support/dt_eval.rs"] mod dt_eval;`; not every
//! includer uses every item.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// An S-expression.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

pub(crate) fn parse_all(text: &str) -> Vec<Sx> {
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
pub(crate) enum Val {
    Int(i64),
    Bool(bool),
    Ctor(String, Vec<Val>),
    Witness(String),
    /// A bit-vector literal, compared by its spelling (`#b01`, `#x0f`).
    Bits(String),
    /// An array: its default and its stored entries, the latest first.
    Array(Box<Val>, Vec<(Val, Val)>),
}

impl Val {
    /// The value an array reads at `index`; `None` when an entry's index
    /// cannot be told equal or unequal to it.
    fn read(&self, index: &Val) -> Option<Val> {
        let Val::Array(default, entries) = self else {
            return None;
        };
        for (key, value) in entries {
            if key.same(index)? {
                return Some(value.clone());
            }
        }
        Some((**default).clone())
    }

    /// Value equality, `None` where the printed values do not decide it.
    /// Two arrays are equal when they read alike at every index either
    /// names and at every index neither names: their defaults are compared
    /// only while some index is unnamed.  Arrays with different defaults are
    /// unequal when neither names an index, or when an index is an integer
    /// (an infinite sort always has an unnamed index); over a finite or
    /// unknown index sort — `Bool`, a bit-vector, an enumeration, an
    /// uninterpreted sort — they can be extensionally equal, so the answer is
    /// open unless every index of the sort is named (adversarial recheck 18's
    /// minor 6: a false `distinct` must never read true).
    fn same(&self, other: &Val) -> Option<bool> {
        match (self, other) {
            (Val::Array(da, ea), Val::Array(db, eb)) => {
                let keys: Vec<&Val> = ea.iter().chain(eb.iter()).map(|(key, _)| key).collect();
                let mut pointwise = Some(true);
                for key in &keys {
                    match self.read(key)?.same(&other.read(key)?) {
                        Some(true) => {}
                        Some(false) => return Some(false),
                        None => pointwise = None,
                    }
                }
                if every_index_named(&keys) {
                    return pointwise;
                }
                match da.same(db) {
                    Some(true) => pointwise,
                    Some(false)
                        if keys.is_empty() || keys.iter().any(|key| matches!(key, Val::Int(_))) =>
                    {
                        Some(false)
                    }
                    _ => None,
                }
            }
            (Val::Ctor(na, fa), Val::Ctor(nb, fb)) => {
                if na != nb || fa.len() != fb.len() {
                    return Some(false);
                }
                let mut out = Some(true);
                for (x, y) in fa.iter().zip(fb) {
                    match x.same(y) {
                        Some(true) => {}
                        Some(false) => return Some(false),
                        None => out = None,
                    }
                }
                out
            }
            _ => Some(self == other),
        }
    }
}

/// Whether `keys` name every index of their sort: both Booleans, or all
/// `2^w` values of a bit-vector sort of width `w <= 16` (spelled `#b…` or
/// `#x…`).  Any other sort is infinite or of a size the values do not show.
fn every_index_named(keys: &[&Val]) -> bool {
    let first = keys.first();
    if keys.iter().all(|key| matches!(key, Val::Bool(_))) && first.is_some() {
        return keys.iter().any(|key| **key == Val::Bool(true))
            && keys.iter().any(|key| **key == Val::Bool(false));
    }
    let bits = |key: &Val| -> Option<(u32, u64)> {
        let Val::Bits(spelled) = key else { return None };
        if let Some(digits) = spelled.strip_prefix("#b") {
            Some((
                u32::try_from(digits.len()).ok()?,
                u64::from_str_radix(digits, 2).ok()?,
            ))
        } else {
            let digits = spelled.strip_prefix("#x")?;
            Some((
                u32::try_from(digits.len()).ok()?.checked_mul(4)?,
                u64::from_str_radix(digits, 16).ok()?,
            ))
        }
    };
    let Some(Some((width, _))) = first.map(|key| bits(key)) else {
        return false;
    };
    if width > 16 {
        return false;
    }
    let mut seen: HashSet<u64> = HashSet::new();
    for key in keys {
        match bits(key) {
            Some((w, value)) if w == width => {
                seen.insert(value);
            }
            _ => return false,
        }
    }
    seen.len() == 1usize << width
}

/// The exact evaluator: the datatype declarations of a script and one
/// printed model.
#[derive(Default)]
pub(crate) struct DtEval {
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
                if atom.starts_with("#b") || atom.starts_with("#x") {
                    return Some(Val::Bits(atom.clone()));
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
                    // `((as const (Array I E)) v)`
                    if let (Some(Sx::Atom(as_kw)), Some(Sx::Atom(const_kw))) =
                        (index.first(), index.get(1))
                        && as_kw == "as"
                        && const_kw == "const"
                    {
                        let default = self.eval(args.first()?, locals)?;
                        return Some(Val::Array(Box::new(default), Vec::new()));
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
                        let mut open = false;
                        for pair in values.windows(2) {
                            match pair[0].same(&pair[1]) {
                                Some(true) => {}
                                Some(false) => return Some(Val::Bool(false)),
                                None => open = true,
                            }
                        }
                        (!open).then_some(Val::Bool(true))
                    }
                    "select" => {
                        let array = self.eval(args.first()?, locals)?;
                        let index = self.eval(args.get(1)?, locals)?;
                        array.read(&index)
                    }
                    "store" => {
                        let Val::Array(default, mut entries) = self.eval(args.first()?, locals)?
                        else {
                            return None;
                        };
                        let index = self.eval(args.get(1)?, locals)?;
                        let value = self.eval(args.get(2)?, locals)?;
                        entries.retain(|(key, _)| key.same(&index) != Some(true));
                        entries.insert(0, (index, value));
                        Some(Val::Array(default, entries))
                    }
                    "distinct" => {
                        let values: Option<Vec<Val>> =
                            args.iter().map(|a| self.eval(a, locals)).collect();
                        let values = values?;
                        let mut open = false;
                        for (i, a) in values.iter().enumerate() {
                            for b in values.iter().skip(i + 1) {
                                match a.same(b) {
                                    Some(true) => return Some(Val::Bool(false)),
                                    Some(false) => {}
                                    None => open = true,
                                }
                            }
                        }
                        (!open).then_some(Val::Bool(true))
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
pub(crate) enum ModelReading {
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
pub(crate) fn judge(script: &str, lines: &[String]) -> Vec<(String, ModelReading)> {
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
            "pop" if frames.len() > 1 => {
                frames.pop();
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
