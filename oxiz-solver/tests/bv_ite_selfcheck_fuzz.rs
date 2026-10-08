//! Differential fuzz for bit-vector `ite` terms through the bit-blaster,
//! scored against an independent `u128` reference evaluator, with the
//! debug-only circuit self-check (`theory_bv_encode::debug_verify_bv_circuits`)
//! live.
//!
//! The scripts are shaped like the verification conditions cargo-formal emits:
//! a `(define-fun tN () <sort> <body>)` chain sharing sub-terms (which
//! `Context` turns into one `(= tN body)` atom per definition), `ite` terms
//! whose selectors are `and`/`or`/`not` trees over `bvult`/`bvule`/`=`/…
//! comparisons, nested `ite`s under `bvadd`/`bvmul`/`extract`/`concat`/
//! `zero_extend`, widths 8, 32, 63 and 64, and then one `(check-sat)` followed
//! by `(get-value …)`.  Every script runs plain, with `(set-option :timeout
//! …)`, and in cargo-formal's `explain --blame` form (`:produce-unsat-cores`,
//! every assertion `:named`, `(get-unsat-core)` after the check).
//!
//! Every `sat` answer's model is evaluated against every assertion; every
//! `unsat` at width 8 over two variables is checked exhaustively; a panic
//! (the self-check firing) is caught per script so one failure does not end
//! the run, and the offending script is printed and written under
//! `std::env::temp_dir()`.

use oxiz_solver::Context;
use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------------------
// PRNG (splitmix64, seed run through the finalizer first).
// ---------------------------------------------------------------------------

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        Rng {
            state: z ^ (z >> 33),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        let mut r = self.below(u64::from(total)) as u32;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= *w;
        }
        weights.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Term DAG.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Lshr,
    Ashr,
    Udiv,
    Urem,
}

impl BinOp {
    fn name(self) -> &'static str {
        match self {
            BinOp::Add => "bvadd",
            BinOp::Sub => "bvsub",
            BinOp::Mul => "bvmul",
            BinOp::And => "bvand",
            BinOp::Or => "bvor",
            BinOp::Xor => "bvxor",
            BinOp::Shl => "bvshl",
            BinOp::Lshr => "bvlshr",
            BinOp::Ashr => "bvashr",
            BinOp::Udiv => "bvudiv",
            BinOp::Urem => "bvurem",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CmpOp {
    Eq,
    Distinct,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

impl CmpOp {
    fn name(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Distinct => "distinct",
            CmpOp::Ult => "bvult",
            CmpOp::Ule => "bvule",
            CmpOp::Ugt => "bvugt",
            CmpOp::Uge => "bvuge",
            CmpOp::Slt => "bvslt",
            CmpOp::Sle => "bvsle",
            CmpOp::Sgt => "bvsgt",
            CmpOp::Sge => "bvsge",
        }
    }
}

type NodeId = usize;

#[derive(Clone, Debug)]
enum Node {
    /// `v<i>` at the base width.
    Var(usize),
    /// `p<i>`, a free Boolean.
    BoolVar(usize),
    Const {
        value: u128,
        width: u32,
    },
    Not(NodeId),
    Neg(NodeId),
    Bin {
        op: BinOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    Cmp {
        op: CmpOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    BNot(NodeId),
    BAnd(Vec<NodeId>),
    BOr(Vec<NodeId>),
    Ite {
        cond: NodeId,
        then: NodeId,
        els: NodeId,
    },
    Extract {
        hi: u32,
        lo: u32,
        arg: NodeId,
    },
    Concat {
        hi: NodeId,
        lo: NodeId,
    },
    ZeroExtend {
        by: u32,
        arg: NodeId,
    },
    SignExtend {
        by: u32,
        arg: NodeId,
    },
}

/// A DAG of nodes; `width[i] == 0` means `Bool`.
struct Dag {
    nodes: Vec<Node>,
    width: Vec<u32>,
    /// Whether node `i` is printed as its own `(define-fun t<i> …)` and
    /// referenced by name afterwards.
    named: Vec<bool>,
    num_vars: usize,
    num_bools: usize,
    base_width: u32,
}

impl Dag {
    fn push(&mut self, node: Node, width: u32, named: bool) -> NodeId {
        self.nodes.push(node);
        self.width.push(width);
        self.named.push(named);
        self.nodes.len() - 1
    }

    fn bv_nodes_of_width(&self, w: u32) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.width[i] == w)
            .collect()
    }

    fn bool_nodes(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.width[i] == 0)
            .collect()
    }
}

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn sort_name(width: u32) -> String {
    if width == 0 {
        "Bool".to_string()
    } else {
        format!("(_ BitVec {width})")
    }
}

fn print_const(value: u128, width: u32, out: &mut String) {
    if width.is_multiple_of(4) {
        let digits = (width / 4) as usize;
        let _ = write!(out, "#x{value:0digits$x}");
    } else {
        let digits = width as usize;
        let _ = write!(out, "#b{value:0digits$b}");
    }
}

/// Prints node `id`: by name if it is a `define-fun`, inline otherwise.
fn print_ref(dag: &Dag, id: NodeId, out: &mut String) {
    if dag.named[id] {
        let _ = write!(out, "t{id}");
    } else {
        print_body(dag, id, out);
    }
}

fn print_body(dag: &Dag, id: NodeId, out: &mut String) {
    match &dag.nodes[id] {
        Node::Var(i) => {
            let _ = write!(out, "v{i}");
        }
        Node::BoolVar(i) => {
            let _ = write!(out, "p{i}");
        }
        Node::Const { value, width } => print_const(*value, *width, out),
        Node::Not(a) => {
            out.push_str("(bvnot ");
            print_ref(dag, *a, out);
            out.push(')');
        }
        Node::Neg(a) => {
            out.push_str("(bvneg ");
            print_ref(dag, *a, out);
            out.push(')');
        }
        Node::Bin { op, lhs, rhs } => {
            let _ = write!(out, "({} ", op.name());
            print_ref(dag, *lhs, out);
            out.push(' ');
            print_ref(dag, *rhs, out);
            out.push(')');
        }
        Node::Cmp { op, lhs, rhs } => {
            let _ = write!(out, "({} ", op.name());
            print_ref(dag, *lhs, out);
            out.push(' ');
            print_ref(dag, *rhs, out);
            out.push(')');
        }
        Node::BNot(a) => {
            out.push_str("(not ");
            print_ref(dag, *a, out);
            out.push(')');
        }
        Node::BAnd(args) => {
            out.push_str("(and");
            for a in args {
                out.push(' ');
                print_ref(dag, *a, out);
            }
            out.push(')');
        }
        Node::BOr(args) => {
            out.push_str("(or");
            for a in args {
                out.push(' ');
                print_ref(dag, *a, out);
            }
            out.push(')');
        }
        Node::Ite { cond, then, els } => {
            out.push_str("(ite ");
            print_ref(dag, *cond, out);
            out.push(' ');
            print_ref(dag, *then, out);
            out.push(' ');
            print_ref(dag, *els, out);
            out.push(')');
        }
        Node::Extract { hi, lo, arg } => {
            let _ = write!(out, "((_ extract {hi} {lo}) ");
            print_ref(dag, *arg, out);
            out.push(')');
        }
        Node::Concat { hi, lo } => {
            out.push_str("(concat ");
            print_ref(dag, *hi, out);
            out.push(' ');
            print_ref(dag, *lo, out);
            out.push(')');
        }
        Node::ZeroExtend { by, arg } => {
            let _ = write!(out, "((_ zero_extend {by}) ");
            print_ref(dag, *arg, out);
            out.push(')');
        }
        Node::SignExtend { by, arg } => {
            let _ = write!(out, "((_ sign_extend {by}) ");
            print_ref(dag, *arg, out);
            out.push(')');
        }
    }
}

// ---------------------------------------------------------------------------
// Reference evaluator (SMT-LIB FixedSizeBitVectors semantics over u128).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Val {
    Bv(u128),
    B(bool),
}

/// Reads a `width`-bit value as a signed integer.
fn signed(value: u128, width: u32) -> i128 {
    let sign_bit = 1u128 << (width - 1);
    if value & sign_bit != 0 {
        (value as i128) - (1i128 << width)
    } else {
        value as i128
    }
}

/// Wraps a signed integer back into `width` bits (two's complement).
fn from_signed(value: i128, width: u32) -> u128 {
    (value as u128) & mask(width)
}

struct Assignment<'a> {
    vars: &'a [u128],
    bools: &'a [bool],
}

fn eval(dag: &Dag, id: NodeId, asg: &Assignment<'_>, memo: &mut Vec<Option<Val>>) -> Val {
    if let Some(v) = memo[id] {
        return v;
    }
    let w = dag.width[id];
    let bv = |v: Val| match v {
        Val::Bv(x) => x,
        Val::B(_) => 0,
    };
    let b = |v: Val| matches!(v, Val::B(true));
    let value = match &dag.nodes[id] {
        Node::Var(i) => Val::Bv(asg.vars[*i]),
        Node::BoolVar(i) => Val::B(asg.bools[*i]),
        Node::Const { value, .. } => Val::Bv(*value),
        Node::Not(a) => Val::Bv(!bv(eval(dag, *a, asg, memo)) & mask(w)),
        Node::Neg(a) => Val::Bv(bv(eval(dag, *a, asg, memo)).wrapping_neg() & mask(w)),
        Node::Bin { op, lhs, rhs } => {
            let x = bv(eval(dag, *lhs, asg, memo));
            let y = bv(eval(dag, *rhs, asg, memo));
            let m = mask(w);
            let r = match op {
                BinOp::Add => x.wrapping_add(y) & m,
                BinOp::Sub => x.wrapping_sub(y) & m,
                BinOp::Mul => x.wrapping_mul(y) & m,
                BinOp::And => x & y,
                BinOp::Or => x | y,
                BinOp::Xor => x ^ y,
                BinOp::Shl => {
                    if y >= u128::from(w) {
                        0
                    } else {
                        (x << y) & m
                    }
                }
                BinOp::Lshr => {
                    if y >= u128::from(w) {
                        0
                    } else {
                        x >> y
                    }
                }
                BinOp::Ashr => {
                    let s = signed(x, w);
                    if y >= u128::from(w) {
                        if s < 0 { m } else { 0 }
                    } else {
                        from_signed(s >> y, w)
                    }
                }
                // SMT-LIB totalises both: `x / 0` is all ones, `x % 0` is `x`.
                BinOp::Udiv => x.checked_div(y).unwrap_or(m),
                BinOp::Urem => x.checked_rem(y).unwrap_or(x),
            };
            Val::Bv(r)
        }
        Node::Cmp { op, lhs, rhs } => {
            let lw = dag.width[*lhs];
            let x = bv(eval(dag, *lhs, asg, memo));
            let y = bv(eval(dag, *rhs, asg, memo));
            let (sx, sy) = (signed(x, lw), signed(y, lw));
            Val::B(match op {
                CmpOp::Eq => x == y,
                CmpOp::Distinct => x != y,
                CmpOp::Ult => x < y,
                CmpOp::Ule => x <= y,
                CmpOp::Ugt => x > y,
                CmpOp::Uge => x >= y,
                CmpOp::Slt => sx < sy,
                CmpOp::Sle => sx <= sy,
                CmpOp::Sgt => sx > sy,
                CmpOp::Sge => sx >= sy,
            })
        }
        Node::BNot(a) => Val::B(!b(eval(dag, *a, asg, memo))),
        Node::BAnd(args) => {
            let mut all = true;
            for a in args {
                if !b(eval(dag, *a, asg, memo)) {
                    all = false;
                }
            }
            Val::B(all)
        }
        Node::BOr(args) => {
            let mut any = false;
            for a in args {
                if b(eval(dag, *a, asg, memo)) {
                    any = true;
                }
            }
            Val::B(any)
        }
        Node::Ite { cond, then, els } => {
            let c = b(eval(dag, *cond, asg, memo));
            let t = eval(dag, *then, asg, memo);
            let e = eval(dag, *els, asg, memo);
            if c { t } else { e }
        }
        Node::Extract { hi, lo, arg } => {
            let x = bv(eval(dag, *arg, asg, memo));
            Val::Bv((x >> lo) & mask(hi - lo + 1))
        }
        Node::Concat { hi, lo } => {
            let lw = dag.width[*lo];
            let h = bv(eval(dag, *hi, asg, memo));
            let l = bv(eval(dag, *lo, asg, memo));
            Val::Bv((h << lw) | l)
        }
        Node::ZeroExtend { arg, .. } => eval(dag, *arg, asg, memo),
        Node::SignExtend { arg, .. } => {
            let aw = dag.width[*arg];
            let x = bv(eval(dag, *arg, asg, memo));
            Val::Bv(from_signed(signed(x, aw), w))
        }
    };
    memo[id] = Some(value);
    value
}

fn all_assertions_hold(dag: &Dag, asserts: &[NodeId], asg: &Assignment<'_>) -> bool {
    let mut memo = vec![None; dag.nodes.len()];
    asserts
        .iter()
        .all(|&a| matches!(eval(dag, a, asg, &mut memo), Val::B(true)))
}

// ---------------------------------------------------------------------------
// Generator.
// ---------------------------------------------------------------------------

struct GenConfig {
    base_width: u32,
    num_vars: usize,
    num_bools: usize,
    /// Number of compound nodes to grow the DAG by.
    growth: usize,
    /// Number of top-level assertions.
    asserts: usize,
}

fn gen_const(rng: &mut Rng, width: u32) -> u128 {
    let m = mask(width);
    let raw = match rng.weighted(&[3, 2, 2, 2, 1]) {
        0 => u128::from(rng.below(16)),
        1 => u128::from(rng.next_u64()),
        2 => 1u128 << rng.below(u64::from(width)),
        3 => m ^ u128::from(rng.below(4)),
        _ => (m >> 1) + u128::from(rng.below(3)),
    };
    raw & m
}

/// Picks an existing node of width `w`, or creates a constant of that width.
fn pick_bv(rng: &mut Rng, dag: &mut Dag, w: u32) -> NodeId {
    let candidates = dag.bv_nodes_of_width(w);
    if candidates.is_empty() || rng.chance(1, 5) {
        let value = gen_const(rng, w);
        return dag.push(Node::Const { value, width: w }, w, false);
    }
    candidates[rng.below(candidates.len() as u64) as usize]
}

/// Picks an existing Boolean node, or creates a comparison at the base width.
fn pick_bool(rng: &mut Rng, dag: &mut Dag) -> NodeId {
    let candidates = dag.bool_nodes();
    if candidates.is_empty() || rng.chance(1, 3) {
        return gen_cmp(rng, dag, dag.base_width);
    }
    candidates[rng.below(candidates.len() as u64) as usize]
}

fn gen_cmp(rng: &mut Rng, dag: &mut Dag, w: u32) -> NodeId {
    const CMPS: [CmpOp; 10] = [
        CmpOp::Eq,
        CmpOp::Distinct,
        CmpOp::Ult,
        CmpOp::Ule,
        CmpOp::Ugt,
        CmpOp::Uge,
        CmpOp::Slt,
        CmpOp::Sle,
        CmpOp::Sgt,
        CmpOp::Sge,
    ];
    let op = CMPS[rng.weighted(&[4, 2, 4, 4, 3, 3, 2, 2, 1, 1])];
    let lhs = pick_bv(rng, dag, w);
    let rhs = pick_bv(rng, dag, w);
    dag.push(Node::Cmp { op, lhs, rhs }, 0, true)
}

/// A Boolean selector tree in the fragment the bit-blaster accepts for `ite`
/// conditions: comparisons, `and`, `or`, `not`, free Booleans.
fn gen_cond(rng: &mut Rng, dag: &mut Dag, depth: u32) -> NodeId {
    if depth == 0 {
        return match rng.weighted(&[6, 1, 2]) {
            0 => gen_cmp(rng, dag, dag.base_width),
            1 if dag.num_bools > 0 => {
                let i = rng.below(dag.num_bools as u64) as usize;
                // Bool vars are nodes num_vars..num_vars+num_bools.
                dag.num_vars + i
            }
            _ => pick_bool(rng, dag),
        };
    }
    match rng.weighted(&[3, 3, 2, 3]) {
        0 => {
            let n = 2 + rng.below(2) as usize;
            let args = (0..n).map(|_| gen_cond(rng, dag, depth - 1)).collect();
            dag.push(Node::BAnd(args), 0, true)
        }
        1 => {
            let n = 2 + rng.below(2) as usize;
            let args = (0..n).map(|_| gen_cond(rng, dag, depth - 1)).collect();
            dag.push(Node::BOr(args), 0, true)
        }
        2 => {
            let a = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BNot(a), 0, true)
        }
        _ => gen_cond(rng, dag, 0),
    }
}

/// Grows the DAG by one compound bit-vector node (possibly with nested
/// helpers) and returns it.
fn grow_bv(rng: &mut Rng, dag: &mut Dag) -> NodeId {
    let w = dag.base_width;
    match rng.weighted(&[10, 4, 6, 3, 3, 3, 2]) {
        // ite over the base width, selected by a comparison tree.
        0 => {
            let depth = rng.below(3) as u32;
            let cond = gen_cond(rng, dag, depth);
            let then = pick_bv(rng, dag, w);
            let els = pick_bv(rng, dag, w);
            dag.push(Node::Ite { cond, then, els }, w, true)
        }
        // unary
        1 => {
            let a = pick_bv(rng, dag, w);
            if rng.chance(1, 2) {
                dag.push(Node::Not(a), w, true)
            } else {
                dag.push(Node::Neg(a), w, true)
            }
        }
        // binary arithmetic / bitwise
        2 => {
            const OPS: [BinOp; 11] = [
                BinOp::Add,
                BinOp::Sub,
                BinOp::Mul,
                BinOp::And,
                BinOp::Or,
                BinOp::Xor,
                BinOp::Shl,
                BinOp::Lshr,
                BinOp::Ashr,
                BinOp::Udiv,
                BinOp::Urem,
            ];
            // Multiplication and division circuits at wide widths dominate a
            // debug-build trial; keep them rarer there.
            let weights: [u32; 11] = if w > 32 {
                [6, 6, 1, 3, 3, 3, 2, 2, 2, 1, 1]
            } else {
                [5, 5, 3, 3, 3, 3, 2, 2, 2, 2, 2]
            };
            let op = OPS[rng.weighted(&weights)];
            let lhs = pick_bv(rng, dag, w);
            let rhs = if op == BinOp::Mul && w > 16 {
                // A constant multiplier (often a power of two) at wide widths.
                let value = if rng.chance(1, 2) {
                    1u128 << rng.below(8)
                } else {
                    u128::from(rng.below(64))
                };
                dag.push(Node::Const { value, width: w }, w, false)
            } else {
                pick_bv(rng, dag, w)
            };
            dag.push(Node::Bin { op, lhs, rhs }, w, true)
        }
        // extract of a wider ite / term, then extended back to the base width
        3 if w > 2 => {
            let lo = rng.below(u64::from(w / 2)) as u32;
            let hi = lo + rng.below(u64::from(w - lo)) as u32;
            let arg = pick_bv(rng, dag, w);
            let narrow_w = hi - lo + 1;
            let narrow = dag.push(Node::Extract { hi, lo, arg }, narrow_w, true);
            if narrow_w == w {
                narrow
            } else if rng.chance(1, 2) {
                dag.push(
                    Node::ZeroExtend {
                        by: w - narrow_w,
                        arg: narrow,
                    },
                    w,
                    true,
                )
            } else {
                dag.push(
                    Node::SignExtend {
                        by: w - narrow_w,
                        arg: narrow,
                    },
                    w,
                    true,
                )
            }
        }
        // concat of two halves (each possibly an ite at the half width)
        4 if w >= 2 => {
            let hw = w / 2;
            let lw = w - hw;
            let hi_src = pick_bv(rng, dag, w);
            let lo_src = pick_bv(rng, dag, w);
            let hi = dag.push(
                Node::Extract {
                    hi: w - 1,
                    lo: w - hw,
                    arg: hi_src,
                },
                hw,
                true,
            );
            let lo_part = dag.push(
                Node::Extract {
                    hi: lw - 1,
                    lo: 0,
                    arg: lo_src,
                },
                lw,
                true,
            );
            let lo = if rng.chance(1, 2) {
                let cond = gen_cond(rng, dag, 1);
                let els = dag.push(
                    Node::Const {
                        value: gen_const(rng, lw),
                        width: lw,
                    },
                    lw,
                    false,
                );
                dag.push(
                    Node::Ite {
                        cond,
                        then: lo_part,
                        els,
                    },
                    lw,
                    true,
                )
            } else {
                lo_part
            };
            dag.push(Node::Concat { hi, lo }, w, true)
        }
        // an ite nested directly under an arithmetic op with an ite operand
        5 => {
            let cond = gen_cond(rng, dag, 1);
            let then = pick_bv(rng, dag, w);
            let els = pick_bv(rng, dag, w);
            let inner = dag.push(Node::Ite { cond, then, els }, w, rng.chance(1, 2));
            let other = pick_bv(rng, dag, w);
            let op = if rng.chance(1, 2) {
                BinOp::Add
            } else {
                BinOp::Sub
            };
            let (lhs, rhs) = if rng.chance(1, 2) {
                (inner, other)
            } else {
                (other, inner)
            };
            dag.push(Node::Bin { op, lhs, rhs }, w, true)
        }
        // zero_extend of a narrower slice, cargo-formal's widening idiom
        _ => {
            let narrow_w = 1 + rng.below(u64::from(w.saturating_sub(1).max(1))) as u32;
            let narrow_w = narrow_w.min(w);
            if narrow_w == w {
                let a = pick_bv(rng, dag, w);
                let b = pick_bv(rng, dag, w);
                return dag.push(
                    Node::Bin {
                        op: BinOp::Add,
                        lhs: a,
                        rhs: b,
                    },
                    w,
                    true,
                );
            }
            let src = pick_bv(rng, dag, w);
            let narrow = dag.push(
                Node::Extract {
                    hi: narrow_w - 1,
                    lo: 0,
                    arg: src,
                },
                narrow_w,
                true,
            );
            let cond = gen_cond(rng, dag, 1);
            let els = dag.push(
                Node::Const {
                    value: gen_const(rng, narrow_w),
                    width: narrow_w,
                },
                narrow_w,
                false,
            );
            let sel = dag.push(
                Node::Ite {
                    cond,
                    then: narrow,
                    els,
                },
                narrow_w,
                true,
            );
            dag.push(
                Node::ZeroExtend {
                    by: w - narrow_w,
                    arg: sel,
                },
                w,
                true,
            )
        }
    }
}

/// One generated problem: the DAG and the asserted Boolean nodes.
struct Problem {
    dag: Dag,
    asserts: Vec<NodeId>,
}

fn gen_problem(rng: &mut Rng, cfg: &GenConfig) -> Problem {
    let mut dag = Dag {
        nodes: Vec::new(),
        width: Vec::new(),
        named: Vec::new(),
        num_vars: cfg.num_vars,
        num_bools: cfg.num_bools,
        base_width: cfg.base_width,
    };
    for i in 0..cfg.num_vars {
        dag.push(Node::Var(i), cfg.base_width, false);
    }
    for i in 0..cfg.num_bools {
        dag.push(Node::BoolVar(i), 0, false);
    }
    for _ in 0..cfg.growth {
        grow_bv(rng, &mut dag);
    }
    // Top-level assertions: comparisons over the grown terms, sometimes
    // negated, sometimes a pin `v = const` so a useful share is unsat.
    let mut asserts = Vec::new();
    for _ in 0..cfg.asserts {
        let a = match rng.weighted(&[5, 3, 2, 2]) {
            0 => gen_cmp(rng, &mut dag, cfg.base_width),
            1 => {
                let c = gen_cmp(rng, &mut dag, cfg.base_width);
                dag.push(Node::BNot(c), 0, true)
            }
            2 => {
                let v = rng.below(cfg.num_vars as u64) as usize;
                let value = gen_const(rng, cfg.base_width);
                let k = dag.push(
                    Node::Const {
                        value,
                        width: cfg.base_width,
                    },
                    cfg.base_width,
                    false,
                );
                dag.push(
                    Node::Cmp {
                        op: CmpOp::Eq,
                        lhs: v,
                        rhs: k,
                    },
                    0,
                    true,
                )
            }
            _ => gen_cond(rng, &mut dag, 2),
        };
        asserts.push(a);
    }
    Problem { dag, asserts }
}

// ---------------------------------------------------------------------------
// Script rendering.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Plain,
    Timeout,
    Named,
    NamedTimeout,
}

fn render(problem: &Problem, seed_opt: Option<u64>, variant: Variant) -> String {
    let dag = &problem.dag;
    let mut out = String::new();
    out.push_str("(set-logic QF_BV)\n");
    if matches!(variant, Variant::Named | Variant::NamedTimeout) {
        out.push_str("(set-option :produce-unsat-cores true)\n");
    }
    if let Some(s) = seed_opt {
        let _ = writeln!(out, "(set-option :random-seed {s})");
    }
    out.push_str("(set-option :max-conflicts 200000)\n");
    if matches!(variant, Variant::Timeout | Variant::NamedTimeout) {
        out.push_str("(set-option :timeout 10000)\n");
    }
    out.push_str("(set-option :produce-models true)\n");
    for i in 0..dag.num_vars {
        let _ = writeln!(out, "(declare-const v{i} (_ BitVec {}))", dag.base_width);
    }
    for i in 0..dag.num_bools {
        let _ = writeln!(out, "(declare-const p{i} Bool)");
    }
    for id in 0..dag.nodes.len() {
        if dag.named[id] {
            let _ = write!(out, "(define-fun t{id} () {} ", sort_name(dag.width[id]));
            print_body(dag, id, &mut out);
            out.push_str(")\n");
        }
    }
    for (k, a) in problem.asserts.iter().enumerate() {
        if matches!(variant, Variant::Named | Variant::NamedTimeout) {
            out.push_str("(assert (! ");
            print_ref(dag, *a, &mut out);
            let _ = writeln!(out, " :named a{k}))");
        } else {
            out.push_str("(assert ");
            print_ref(dag, *a, &mut out);
            out.push_str(")\n");
        }
    }
    out.push_str("(check-sat)\n(get-value (");
    for i in 0..dag.num_vars {
        if i > 0 {
            out.push(' ');
        }
        let _ = write!(out, "v{i}");
    }
    for i in 0..dag.num_bools {
        let _ = write!(out, " p{i}");
    }
    out.push_str("))\n");
    if matches!(variant, Variant::Named | Variant::NamedTimeout) {
        out.push_str("(get-unsat-core)\n");
    }
    out
}

// ---------------------------------------------------------------------------
// Running and scoring.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Outcome {
    Sat(Vec<String>),
    Unsat(Vec<String>),
    Unknown,
    Error(String),
    Panic(String),
}

fn run(script: &str) -> Outcome {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = Context::new();
        ctx.execute_script(script)
    }));
    match result {
        Ok(Ok(lines)) => {
            let verdict = lines.iter().rev().find_map(|l| match l.trim() {
                "sat" => Some("sat"),
                "unsat" => Some("unsat"),
                "unknown" => Some("unknown"),
                _ => None,
            });
            match verdict {
                Some("sat") => Outcome::Sat(lines),
                Some("unsat") => Outcome::Unsat(lines),
                Some("unknown") => Outcome::Unknown,
                _ => Outcome::Error(lines.join(" | ")),
            }
        }
        Ok(Err(e)) => Outcome::Error(e.to_string()),
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            Outcome::Panic(message)
        }
    }
}

/// The value printed for `name` in a `(get-value …)` answer, radix-agnostic.
fn printed_bv(outputs: &[String], name: &str) -> Option<u128> {
    let joined = outputs.join("\n");
    let key = format!("({name} ");
    let after = joined.find(&key).map(|at| &joined[at + key.len()..])?;
    let after = after.trim_start();
    let rest = after.strip_prefix('#')?;
    let mut chars = rest.chars();
    let radix = match chars.next()? {
        'x' => 16,
        'b' => 2,
        _ => return None,
    };
    let digits: String = chars.take_while(char::is_ascii_alphanumeric).collect();
    u128::from_str_radix(&digits, radix).ok()
}

fn printed_bool(outputs: &[String], name: &str) -> Option<bool> {
    let joined = outputs.join("\n");
    let key = format!("({name} ");
    let after = joined.find(&key).map(|at| &joined[at + key.len()..])?;
    let after = after.trim_start();
    if after.starts_with("true") {
        Some(true)
    } else if after.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Exhaustive truth at width 8 over up to two variables (and any Booleans).
fn exhaustive_sat(problem: &Problem) -> Option<bool> {
    let dag = &problem.dag;
    if dag.base_width != 8 || dag.num_vars > 2 || dag.num_bools > 2 {
        return None;
    }
    let var_space = 1u64 << (8 * dag.num_vars as u64);
    let bool_space = 1u64 << dag.num_bools;
    for bits in 0..bool_space {
        let bools: Vec<bool> = (0..dag.num_bools).map(|i| (bits >> i) & 1 == 1).collect();
        for idx in 0..var_space {
            let vars: Vec<u128> = (0..dag.num_vars)
                .map(|i| u128::from((idx >> (8 * i)) & 0xff))
                .collect();
            let asg = Assignment {
                vars: &vars,
                bools: &bools,
            };
            if all_assertions_hold(dag, &problem.asserts, &asg) {
                return Some(true);
            }
        }
    }
    Some(false)
}

/// Random search for a witness (any width): the assignment if one was found.
fn sampled_witness(
    rng: &mut Rng,
    problem: &Problem,
    samples: usize,
) -> Option<(Vec<u128>, Vec<bool>)> {
    let dag = &problem.dag;
    let m = mask(dag.base_width);
    for _ in 0..samples {
        let vars: Vec<u128> = (0..dag.num_vars)
            .map(|_| match rng.below(4) {
                0 => u128::from(rng.below(8)),
                1 => m - u128::from(rng.below(8)),
                _ => u128::from(rng.next_u64()) & m,
            })
            .collect();
        let bools: Vec<bool> = (0..dag.num_bools).map(|_| rng.chance(1, 2)).collect();
        let asg = Assignment {
            vars: &vars,
            bools: &bools,
        };
        if all_assertions_hold(dag, &problem.asserts, &asg) {
            return Some((vars, bools));
        }
    }
    None
}

#[derive(Default, Debug)]
struct Tally {
    trials: usize,
    sat: usize,
    unsat: usize,
    unknown: usize,
    errors: usize,
    panics: usize,
    wrong_sat: usize,
    wrong_unsat: usize,
    core_not_subset: usize,
    named_disagrees: usize,
}

struct Failure {
    kind: &'static str,
    detail: String,
    script: String,
}

fn score(
    problem: &Problem,
    script: &str,
    outcome: &Outcome,
    truth: Option<bool>,
    tally: &mut Tally,
    failures: &mut Vec<Failure>,
) -> Option<&'static str> {
    let dag = &problem.dag;
    match outcome {
        Outcome::Panic(msg) => {
            tally.panics += 1;
            failures.push(Failure {
                kind: "panic",
                detail: msg.clone(),
                script: script.to_string(),
            });
            Some("panic")
        }
        Outcome::Error(msg) => {
            tally.errors += 1;
            failures.push(Failure {
                kind: "error",
                detail: msg.clone(),
                script: script.to_string(),
            });
            Some("error")
        }
        Outcome::Unknown => {
            tally.unknown += 1;
            // Not a failure, but worth keeping: on QF_BV an `unknown` is most
            // often the model gate refusing a model the circuit produced.
            failures.push(Failure {
                kind: "unknown",
                detail: String::new(),
                script: script.to_string(),
            });
            Some("unknown")
        }
        Outcome::Sat(lines) => {
            tally.sat += 1;
            let mut vars = Vec::new();
            for i in 0..dag.num_vars {
                match printed_bv(lines, &format!("v{i}")) {
                    Some(v) => vars.push(v),
                    None => {
                        failures.push(Failure {
                            kind: "sat-without-model-value",
                            detail: format!("no value for v{i}: {}", lines.join(" | ")),
                            script: script.to_string(),
                        });
                        return Some("sat-without-model-value");
                    }
                }
            }
            let mut bools = Vec::new();
            for i in 0..dag.num_bools {
                bools.push(printed_bool(lines, &format!("p{i}")).unwrap_or(false));
            }
            let asg = Assignment {
                vars: &vars,
                bools: &bools,
            };
            if !all_assertions_hold(dag, &problem.asserts, &asg) {
                tally.wrong_sat += 1;
                failures.push(Failure {
                    kind: "wrong-sat",
                    detail: format!("model {vars:x?} {bools:?} violates an assertion"),
                    script: script.to_string(),
                });
                return Some("wrong-sat");
            }
            if truth == Some(false) {
                tally.wrong_sat += 1;
                failures.push(Failure {
                    kind: "wrong-sat-vs-oracle",
                    detail: "oracle says unsat".to_string(),
                    script: script.to_string(),
                });
                return Some("wrong-sat-vs-oracle");
            }
            Some("sat")
        }
        Outcome::Unsat(_) => {
            tally.unsat += 1;
            if truth == Some(true) {
                tally.wrong_unsat += 1;
                failures.push(Failure {
                    kind: "wrong-unsat",
                    detail: "a witness exists".to_string(),
                    script: script.to_string(),
                });
                return Some("wrong-unsat");
            }
            Some("unsat")
        }
    }
}

/// The names in the `(get-unsat-core)` answer, if one was printed.
fn printed_core(lines: &[String]) -> Option<Vec<String>> {
    let line = lines.iter().rev().find(|l| {
        l.starts_with('(') && !l.starts_with("(error") && !l.starts_with("((") && {
            let inner = l.trim_start_matches('(').trim_end_matches(')');
            inner.is_empty() || inner.split_whitespace().all(|w| w.starts_with('a'))
        }
    })?;
    let inner = line.trim_start_matches('(').trim_end_matches(')');
    Some(inner.split_whitespace().map(str::to_string).collect())
}

fn write_failure(dir: &std::path::Path, tag: &str, index: usize, f: &Failure) {
    let path = dir.join(format!("{tag}-{index:04}-{}.smt2", f.kind));
    let mut text = String::new();
    let _ = writeln!(text, "; kind: {}", f.kind);
    for line in f.detail.lines() {
        let _ = writeln!(text, "; {line}");
    }
    text.push_str(&f.script);
    let _ = std::fs::write(&path, text);
}

struct Campaign {
    seeds: std::ops::Range<u64>,
    trials_per_seed: usize,
    widths: Vec<u32>,
    variants: Vec<Variant>,
    solver_seeds: Vec<Option<u64>>,
    tag: String,
}

fn run_campaign(c: &Campaign) -> (Tally, Vec<Failure>) {
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    let out_dir = std::env::temp_dir().join("oxiz-bv-ite-selfcheck-fuzz");
    let _ = std::fs::create_dir_all(&out_dir);
    let mut written = 0usize;
    // `OXIZ_ITE_FUZZ_DUMP_DIR`: also write every rendered script there, as
    // `<tag>-s<seed>-t<trial>-<variant>.smt2`; with `OXIZ_ITE_FUZZ_DUMP_ONLY`
    // set nothing is solved, which turns the generator into a corpus writer
    // for timing scripts one by one outside the harness.
    let dump_dir = std::env::var_os("OXIZ_ITE_FUZZ_DUMP_DIR").map(std::path::PathBuf::from);
    if let Some(dir) = &dump_dir {
        let _ = std::fs::create_dir_all(dir);
    }
    let dump_only = std::env::var_os("OXIZ_ITE_FUZZ_DUMP_ONLY").is_some();
    for seed in c.seeds.clone() {
        let mut rng = Rng::new(seed);
        for trial in 0..c.trials_per_seed {
            let width = c.widths[rng.below(c.widths.len() as u64) as usize];
            let cfg = GenConfig {
                base_width: width,
                num_vars: 2 + rng.below(3) as usize,
                num_bools: rng.below(2) as usize,
                growth: 3 + rng.below(6) as usize,
                asserts: 1 + rng.below(3) as usize,
            };
            let problem = gen_problem(&mut rng, &cfg);
            // Oracle: exhaustive at width 8 / 2 vars, sampled witness search
            // otherwise (only ever proves `sat`, never `unsat`).
            let mut witness: Option<(Vec<u128>, Vec<bool>)> = None;
            let truth = match exhaustive_sat(&problem) {
                Some(t) => Some(t),
                None => match sampled_witness(&mut rng, &problem, 4000) {
                    Some(w) => {
                        witness = Some(w);
                        Some(true)
                    }
                    None => None,
                },
            };
            let solver_seed = c.solver_seeds[rng.below(c.solver_seeds.len() as u64) as usize];
            let mut plain_verdict: Option<&'static str> = None;
            for &variant in &c.variants {
                let script = render(&problem, solver_seed, variant);
                if let Some(dir) = &dump_dir {
                    let name = format!("{}-s{seed}-t{trial}-{variant:?}.smt2", c.tag);
                    let _ = std::fs::write(dir.join(name), &script);
                }
                if dump_only {
                    continue;
                }
                tally.trials += 1;
                let outcome = run(&script);
                let before = failures.len();
                let verdict = score(
                    &problem,
                    &script,
                    &outcome,
                    truth,
                    &mut tally,
                    &mut failures,
                );
                if let Some(last) = failures.last_mut()
                    && last.kind == "wrong-unsat"
                    && let Some((vars, bools)) = &witness
                {
                    let _ = write!(last.detail, "; witness vars {vars:#x?} bools {bools:?}");
                }
                if let Some(v) = verdict
                    && matches!(v, "sat" | "unsat")
                {
                    match variant {
                        Variant::Plain => plain_verdict = Some(v),
                        Variant::Named | Variant::NamedTimeout => {
                            if let Some(p) = plain_verdict
                                && p != v
                            {
                                tally.named_disagrees += 1;
                                failures.push(Failure {
                                    kind: "named-disagrees",
                                    detail: format!("plain {p}, named {v}"),
                                    script: script.clone(),
                                });
                            }
                        }
                        Variant::Timeout => {}
                    }
                }
                if let (Outcome::Unsat(lines), Variant::Named | Variant::NamedTimeout) =
                    (&outcome, variant)
                {
                    let names: Vec<String> = (0..problem.asserts.len())
                        .map(|k| format!("a{k}"))
                        .collect();
                    match printed_core(lines) {
                        Some(core) if core.iter().all(|n| names.contains(n)) => {}
                        other => {
                            tally.core_not_subset += 1;
                            failures.push(Failure {
                                kind: "core-not-subset",
                                detail: format!("core {other:?} of names {names:?}"),
                                script: script.clone(),
                            });
                        }
                    }
                }
                for f in &failures[before..] {
                    eprintln!(
                        "[{}] seed {seed} trial {trial} width {width} {:?}: {} — {}",
                        c.tag, variant, f.kind, f.detail
                    );
                    if written < 200 {
                        write_failure(&out_dir, &c.tag, written, f);
                        written += 1;
                    }
                }
            }
        }
    }
    (tally, failures)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Long-running campaign, driven by `OXIZ_ITE_FUZZ_SEED_LO/HI`,
/// `OXIZ_ITE_FUZZ_TRIALS`, `OXIZ_ITE_FUZZ_WIDTHS` (comma list) and
/// `OXIZ_ITE_FUZZ_VARIANTS` (`p`,`t`,`n`,`nt`).
#[test]
#[ignore = "long-running differential campaign; run explicitly"]
fn ite_selfcheck_campaign() {
    let lo = env_u64("OXIZ_ITE_FUZZ_SEED_LO", 0);
    let hi = env_u64("OXIZ_ITE_FUZZ_SEED_HI", 8);
    let trials = env_u64("OXIZ_ITE_FUZZ_TRIALS", 200) as usize;
    let widths: Vec<u32> = std::env::var("OXIZ_ITE_FUZZ_WIDTHS")
        .ok()
        .map(|s| s.split(',').filter_map(|w| w.trim().parse().ok()).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![8, 32, 63, 64]);
    let variants: Vec<Variant> = std::env::var("OXIZ_ITE_FUZZ_VARIANTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|v| match v.trim() {
                    "p" => Some(Variant::Plain),
                    "t" => Some(Variant::Timeout),
                    "n" => Some(Variant::Named),
                    "nt" => Some(Variant::NamedTimeout),
                    _ => None,
                })
                .collect()
        })
        .filter(|v: &Vec<Variant>| !v.is_empty())
        .unwrap_or_else(|| vec![Variant::Plain, Variant::Timeout, Variant::Named]);
    let campaign = Campaign {
        seeds: lo..hi,
        trials_per_seed: trials,
        widths,
        variants,
        solver_seeds: vec![None, Some(1), Some(7), Some(42)],
        tag: format!("campaign-{lo}-{hi}"),
    };
    let (tally, failures) = run_campaign(&campaign);
    eprintln!("{tally:?}");
    let failures: Vec<Failure> = failures
        .into_iter()
        .filter(|f| f.kind != "unknown")
        .collect();
    assert!(
        failures.is_empty(),
        "{} failures ({} panics, {} wrong sat, {} wrong unsat); first: [{}] {}\n{}",
        failures.len(),
        tally.panics,
        tally.wrong_sat,
        tally.wrong_unsat,
        failures[0].kind,
        failures[0].detail,
        failures[0].script
    );
}

/// Bounded deterministic slice of the campaign, fast enough for every run.
#[test]
fn ite_selfcheck_bounded_differential() {
    let campaign = Campaign {
        seeds: 0..2,
        trials_per_seed: 25,
        widths: vec![8, 32, 64],
        variants: vec![Variant::Plain, Variant::Named],
        solver_seeds: vec![None, Some(1)],
        tag: "bounded".to_string(),
    };
    let (tally, failures) = run_campaign(&campaign);
    eprintln!("{tally:?}");
    let failures: Vec<Failure> = failures
        .into_iter()
        .filter(|f| f.kind != "unknown")
        .collect();
    assert!(
        failures.is_empty(),
        "{} failures; first: [{}] {}\n{}",
        failures.len(),
        failures[0].kind,
        failures[0].detail,
        failures[0].script
    );
}
