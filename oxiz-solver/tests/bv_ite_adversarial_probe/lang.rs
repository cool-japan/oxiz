//! The term language of the adversarial probe: a splitmix64 PRNG, the
//! bit-vector/Boolean term DAG, its SMT-LIB printer, the `u128` reference
//! evaluator and the random generator.  Split out of
//! `bv_ite_adversarial_probe.rs` to keep that file under the workspace
//! 2000-line limit.

use std::fmt::Write as _;

// ---------------------------------------------------------------------------
// PRNG (splitmix64).
// ---------------------------------------------------------------------------

pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Rng { state: z }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    pub fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------------------
// Term DAG.  `width == 0` means Bool.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
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
    Sdiv,
    Srem,
}

impl BinOp {
    pub fn name(self) -> &'static str {
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
            BinOp::Sdiv => "bvsdiv",
            BinOp::Srem => "bvsrem",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Eq,
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
    pub fn name(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
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

pub const ALL_CMPS: [CmpOp; 9] = [
    CmpOp::Eq,
    CmpOp::Ult,
    CmpOp::Ule,
    CmpOp::Ugt,
    CmpOp::Uge,
    CmpOp::Slt,
    CmpOp::Sle,
    CmpOp::Sgt,
    CmpOp::Sge,
];

pub const ALL_BINOPS: [BinOp; 13] = [
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
    BinOp::Sdiv,
    BinOp::Srem,
];

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum Node {
    /// `v<i>` at the base width.
    Var(usize),
    /// `p<i>`, a free Boolean.
    BoolVar(usize),
    /// `u0`: a declared constant asserted equal to `(f v0)`; free for the
    /// oracle, opaque for the bit-blaster.
    Uf,
    /// `w0`: a declared constant asserted equal to `(select arr v0)`.
    Sel,
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
    /// `distinct` over bit-vector operands (pairwise).
    Distinct(Vec<NodeId>),
    BNot(NodeId),
    BAnd(Vec<NodeId>),
    BOr(Vec<NodeId>),
    BXor(NodeId, NodeId),
    BImplies(NodeId, NodeId),
    /// Bool-sorted `ite`.
    BIte(NodeId, NodeId, NodeId),
    /// `=` over Bool operands (iff).
    BEq(NodeId, NodeId),
    /// `distinct` over Bool operands.
    BDistinct(Vec<NodeId>),
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

pub struct Dag {
    pub nodes: Vec<Node>,
    pub width: Vec<u32>,
    pub named: Vec<bool>,
    pub num_vars: usize,
    pub num_bools: usize,
    pub has_uf: bool,
    pub has_sel: bool,
    pub base_width: u32,
}

impl Dag {
    pub fn new(base_width: u32, num_vars: usize, num_bools: usize) -> Self {
        let mut dag = Dag {
            nodes: Vec::new(),
            width: Vec::new(),
            named: Vec::new(),
            num_vars,
            num_bools,
            has_uf: false,
            has_sel: false,
            base_width,
        };
        for i in 0..num_vars {
            dag.push(Node::Var(i), base_width, false);
        }
        for i in 0..num_bools {
            dag.push(Node::BoolVar(i), 0, false);
        }
        dag
    }

    pub fn push(&mut self, node: Node, width: u32, named: bool) -> NodeId {
        self.nodes.push(node);
        self.width.push(width);
        self.named.push(named);
        self.nodes.len() - 1
    }

    pub fn konst(&mut self, value: u128, width: u32) -> NodeId {
        self.push(
            Node::Const {
                value: value & mask(width),
                width,
            },
            width,
            false,
        )
    }

    pub fn bv_nodes_of_width(&self, w: u32) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.width[i] == w)
            .collect()
    }

    pub fn bool_nodes(&self) -> Vec<NodeId> {
        (0..self.nodes.len())
            .filter(|&i| self.width[i] == 0)
            .collect()
    }

    pub fn uf(&mut self) -> NodeId {
        self.has_uf = true;
        let w = self.base_width;
        self.push(Node::Uf, w, false)
    }

    pub fn sel(&mut self) -> NodeId {
        self.has_sel = true;
        let w = self.base_width;
        self.push(Node::Sel, w, false)
    }
}

pub fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

pub fn sort_name(width: u32) -> String {
    if width == 0 {
        "Bool".to_string()
    } else {
        format!("(_ BitVec {width})")
    }
}

pub fn print_const(value: u128, width: u32, out: &mut String) {
    if width.is_multiple_of(4) {
        let digits = (width / 4) as usize;
        let _ = write!(out, "#x{value:0digits$x}");
    } else {
        let digits = width as usize;
        let _ = write!(out, "#b{value:0digits$b}");
    }
}

pub fn print_ref(dag: &Dag, id: NodeId, out: &mut String) {
    if dag.named[id] {
        let _ = write!(out, "t{id}");
    } else {
        print_body(dag, id, out);
    }
}

pub fn print_list(dag: &Dag, head: &str, args: &[NodeId], out: &mut String) {
    out.push('(');
    out.push_str(head);
    for a in args {
        out.push(' ');
        print_ref(dag, *a, out);
    }
    out.push(')');
}

pub fn print_body(dag: &Dag, id: NodeId, out: &mut String) {
    match &dag.nodes[id] {
        Node::Var(i) => {
            let _ = write!(out, "v{i}");
        }
        Node::BoolVar(i) => {
            let _ = write!(out, "p{i}");
        }
        Node::Uf => out.push_str("(f v0)"),
        Node::Sel => out.push_str("(select arr v0)"),
        Node::Const { value, width } => print_const(*value, *width, out),
        Node::Not(a) => print_list(dag, "bvnot", &[*a], out),
        Node::Neg(a) => print_list(dag, "bvneg", &[*a], out),
        Node::Bin { op, lhs, rhs } => print_list(dag, op.name(), &[*lhs, *rhs], out),
        Node::Cmp { op, lhs, rhs } => print_list(dag, op.name(), &[*lhs, *rhs], out),
        Node::Distinct(args) | Node::BDistinct(args) => print_list(dag, "distinct", args, out),
        Node::BNot(a) => print_list(dag, "not", &[*a], out),
        Node::BAnd(args) => print_list(dag, "and", args, out),
        Node::BOr(args) => print_list(dag, "or", args, out),
        Node::BXor(a, b) => print_list(dag, "xor", &[*a, *b], out),
        Node::BImplies(a, b) => print_list(dag, "=>", &[*a, *b], out),
        Node::BIte(c, t, e) => print_list(dag, "ite", &[*c, *t, *e], out),
        Node::BEq(a, b) => print_list(dag, "=", &[*a, *b], out),
        Node::Ite { cond, then, els } => print_list(dag, "ite", &[*cond, *then, *els], out),
        Node::Extract { hi, lo, arg } => {
            let _ = write!(out, "((_ extract {hi} {lo}) ");
            print_ref(dag, *arg, out);
            out.push(')');
        }
        Node::Concat { hi, lo } => print_list(dag, "concat", &[*hi, *lo], out),
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
// Reference evaluator (SMT-LIB FixedSizeBitVectors over u128, widths ≤ 64).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Val {
    Bv(u128),
    B(bool),
}

pub fn signed(value: u128, width: u32) -> i128 {
    let sign_bit = 1u128 << (width - 1);
    if value & sign_bit != 0 {
        (value as i128) - (1i128 << width)
    } else {
        value as i128
    }
}

pub fn from_signed(value: i128, width: u32) -> u128 {
    (value as u128) & mask(width)
}

pub fn is_negative(value: u128, width: u32) -> bool {
    value & (1u128 << (width - 1)) != 0
}

pub fn magnitude(value: u128, width: u32) -> u128 {
    if is_negative(value, width) {
        value.wrapping_neg() & mask(width)
    } else {
        value
    }
}

pub struct Assignment<'a> {
    pub vars: &'a [u128],
    pub bools: &'a [bool],
    pub uf: u128,
    pub sel: u128,
}

pub fn bin_op(op: BinOp, x: u128, y: u128, w: u32) -> u128 {
    let m = mask(w);
    match op {
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
        BinOp::Udiv => x.checked_div(y).unwrap_or(m),
        BinOp::Urem => x.checked_rem(y).unwrap_or(x),
        // `(bvsdiv s 0)` is `-1` for a non-negative `s` and `1` for a
        // negative one; `(bvsrem s 0)` is `s`.
        BinOp::Sdiv => {
            if y == 0 {
                if is_negative(x, w) { 1 } else { m }
            } else {
                let q = magnitude(x, w) / magnitude(y, w);
                if is_negative(x, w) == is_negative(y, w) {
                    q & m
                } else {
                    q.wrapping_neg() & m
                }
            }
        }
        BinOp::Srem => {
            if y == 0 {
                x
            } else {
                let r = magnitude(x, w) % magnitude(y, w);
                if is_negative(x, w) {
                    r.wrapping_neg() & m
                } else {
                    r & m
                }
            }
        }
    }
}

pub fn cmp_op(op: CmpOp, x: u128, y: u128, w: u32) -> bool {
    let (sx, sy) = (signed(x, w), signed(y, w));
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ult => x < y,
        CmpOp::Ule => x <= y,
        CmpOp::Ugt => x > y,
        CmpOp::Uge => x >= y,
        CmpOp::Slt => sx < sy,
        CmpOp::Sle => sx <= sy,
        CmpOp::Sgt => sx > sy,
        CmpOp::Sge => sx >= sy,
    }
}

pub fn eval(dag: &Dag, id: NodeId, asg: &Assignment<'_>, memo: &mut Vec<Option<Val>>) -> Val {
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
        Node::Uf => Val::Bv(asg.uf),
        Node::Sel => Val::Bv(asg.sel),
        Node::Const { value, .. } => Val::Bv(*value),
        Node::Not(a) => Val::Bv(!bv(eval(dag, *a, asg, memo)) & mask(w)),
        Node::Neg(a) => Val::Bv(bv(eval(dag, *a, asg, memo)).wrapping_neg() & mask(w)),
        Node::Bin { op, lhs, rhs } => {
            let x = bv(eval(dag, *lhs, asg, memo));
            let y = bv(eval(dag, *rhs, asg, memo));
            Val::Bv(bin_op(*op, x, y, w))
        }
        Node::Cmp { op, lhs, rhs } => {
            let lw = dag.width[*lhs];
            let x = bv(eval(dag, *lhs, asg, memo));
            let y = bv(eval(dag, *rhs, asg, memo));
            Val::B(cmp_op(*op, x, y, lw))
        }
        Node::Distinct(args) => {
            let vals: Vec<u128> = args.iter().map(|&a| bv(eval(dag, a, asg, memo))).collect();
            let mut all_differ = true;
            for i in 0..vals.len() {
                for j in (i + 1)..vals.len() {
                    if vals[i] == vals[j] {
                        all_differ = false;
                    }
                }
            }
            Val::B(all_differ)
        }
        Node::BNot(a) => Val::B(!b(eval(dag, *a, asg, memo))),
        Node::BAnd(args) => {
            let vals: Vec<bool> = args.iter().map(|&a| b(eval(dag, a, asg, memo))).collect();
            Val::B(vals.iter().all(|&x| x))
        }
        Node::BOr(args) => {
            let vals: Vec<bool> = args.iter().map(|&a| b(eval(dag, a, asg, memo))).collect();
            Val::B(vals.iter().any(|&x| x))
        }
        Node::BXor(x, y) => Val::B(b(eval(dag, *x, asg, memo)) ^ b(eval(dag, *y, asg, memo))),
        Node::BImplies(x, y) => Val::B(!b(eval(dag, *x, asg, memo)) || b(eval(dag, *y, asg, memo))),
        Node::BIte(c, t, e) => {
            let cv = b(eval(dag, *c, asg, memo));
            let tv = b(eval(dag, *t, asg, memo));
            let ev = b(eval(dag, *e, asg, memo));
            Val::B(if cv { tv } else { ev })
        }
        Node::BEq(x, y) => Val::B(b(eval(dag, *x, asg, memo)) == b(eval(dag, *y, asg, memo))),
        Node::BDistinct(args) => {
            let vals: Vec<bool> = args.iter().map(|&a| b(eval(dag, a, asg, memo))).collect();
            let mut all_differ = true;
            for i in 0..vals.len() {
                for j in (i + 1)..vals.len() {
                    if vals[i] == vals[j] {
                        all_differ = false;
                    }
                }
            }
            Val::B(all_differ)
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

pub fn all_hold(dag: &Dag, asserts: &[NodeId], asg: &Assignment<'_>) -> bool {
    let mut memo = vec![None; dag.nodes.len()];
    asserts
        .iter()
        .all(|&a| matches!(eval(dag, a, asg, &mut memo), Val::B(true)))
}

// ---------------------------------------------------------------------------
// Generator.
// ---------------------------------------------------------------------------

pub const WIDTHS: [u32; 8] = [1, 7, 8, 31, 32, 33, 63, 64];

pub fn gen_const(rng: &mut Rng, width: u32) -> u128 {
    let m = mask(width);
    let raw = match rng.below(6) {
        0 => u128::from(rng.below(16)),
        1 => u128::from(rng.next_u64()),
        2 => 1u128 << rng.below(u64::from(width)),
        3 => m ^ u128::from(rng.below(4)),
        4 => (m >> 1) + u128::from(rng.below(3)),
        // A shift amount at or past the width.
        _ => u128::from(width) + u128::from(rng.below(4)),
    };
    raw & m
}

pub fn pick_bv(rng: &mut Rng, dag: &mut Dag, w: u32) -> NodeId {
    let candidates = dag.bv_nodes_of_width(w);
    if candidates.is_empty() || rng.chance(1, 5) {
        let value = gen_const(rng, w);
        return dag.konst(value, w);
    }
    *rng.pick(&candidates)
}

pub fn gen_cmp(rng: &mut Rng, dag: &mut Dag) -> NodeId {
    let w = dag.base_width;
    if rng.chance(1, 6) {
        let n = 2 + rng.below(2) as usize;
        let args = (0..n).map(|_| pick_bv(rng, dag, w)).collect();
        return dag.push(Node::Distinct(args), 0, true);
    }
    let op = *rng.pick(&ALL_CMPS);
    let lhs = pick_bv(rng, dag, w);
    let rhs = pick_bv(rng, dag, w);
    dag.push(Node::Cmp { op, lhs, rhs }, 0, true)
}

pub fn pick_bool(rng: &mut Rng, dag: &mut Dag) -> NodeId {
    let candidates = dag.bool_nodes();
    if candidates.is_empty() || rng.chance(1, 3) {
        return gen_cmp(rng, dag);
    }
    *rng.pick(&candidates)
}

/// A selector in the full fragment: comparisons, `distinct`, free Booleans,
/// `not`/`and`/`or`/`xor`/`=>`, a Bool `ite`, Bool `=` and Bool `distinct`.
pub fn gen_cond(rng: &mut Rng, dag: &mut Dag, depth: u32) -> NodeId {
    if depth == 0 {
        return match rng.below(9) {
            0..=5 => gen_cmp(rng, dag),
            6 if dag.num_bools > 0 => dag.num_vars + rng.below(dag.num_bools as u64) as usize,
            _ => pick_bool(rng, dag),
        };
    }
    match rng.below(9) {
        0 | 1 => {
            let n = 2 + rng.below(2) as usize;
            let args = (0..n).map(|_| gen_cond(rng, dag, depth - 1)).collect();
            dag.push(Node::BAnd(args), 0, true)
        }
        2 => {
            let n = 2 + rng.below(2) as usize;
            let args = (0..n).map(|_| gen_cond(rng, dag, depth - 1)).collect();
            dag.push(Node::BOr(args), 0, true)
        }
        3 => {
            let a = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BNot(a), 0, true)
        }
        4 => {
            let a = gen_cond(rng, dag, depth - 1);
            let b = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BXor(a, b), 0, true)
        }
        5 => {
            let a = gen_cond(rng, dag, depth - 1);
            let b = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BImplies(a, b), 0, true)
        }
        6 => {
            let c = gen_cond(rng, dag, depth - 1);
            let t = gen_cond(rng, dag, depth - 1);
            let e = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BIte(c, t, e), 0, true)
        }
        7 => {
            let a = gen_cond(rng, dag, depth - 1);
            let b = gen_cond(rng, dag, depth - 1);
            dag.push(Node::BEq(a, b), 0, true)
        }
        _ => {
            let n = 2 + rng.below(2) as usize;
            let args = (0..n).map(|_| gen_cond(rng, dag, depth - 1)).collect();
            dag.push(Node::BDistinct(args), 0, true)
        }
    }
}

/// One `ite` at the base width with a selector of random depth; nests up to
/// `nest` further `ite`s inside its branches.
pub fn gen_ite(rng: &mut Rng, dag: &mut Dag, nest: u32) -> NodeId {
    let w = dag.base_width;
    let depth = rng.below(3) as u32;
    let cond = gen_cond(rng, dag, depth);
    let then = if nest > 0 && rng.chance(1, 2) {
        gen_ite(rng, dag, nest - 1)
    } else {
        pick_bv(rng, dag, w)
    };
    let els = if nest > 0 && rng.chance(1, 2) {
        gen_ite(rng, dag, nest - 1)
    } else {
        pick_bv(rng, dag, w)
    };
    dag.push(Node::Ite { cond, then, els }, w, true)
}

/// Grows the DAG by one compound node.  Multiplication and division at wide
/// widths are kept rare so a debug-build trial stays affordable.
pub fn grow(rng: &mut Rng, dag: &mut Dag, opaque: bool) -> NodeId {
    let w = dag.base_width;
    let wide = w > 16;
    match rng.below(10) {
        0..=2 => gen_ite(rng, dag, 2),
        3 => {
            let a = pick_bv(rng, dag, w);
            if rng.chance(1, 2) {
                dag.push(Node::Not(a), w, true)
            } else {
                dag.push(Node::Neg(a), w, true)
            }
        }
        4 | 5 => {
            let op = loop {
                let op = *rng.pick(&ALL_BINOPS);
                let heavy = matches!(
                    op,
                    BinOp::Mul | BinOp::Udiv | BinOp::Urem | BinOp::Sdiv | BinOp::Srem
                );
                if !(wide && heavy) || rng.chance(1, 8) {
                    break op;
                }
            };
            // An `ite` feeds the operation directly half of the time.
            let lhs = if rng.chance(1, 2) {
                gen_ite(rng, dag, 1)
            } else {
                pick_bv(rng, dag, w)
            };
            let rhs = if matches!(op, BinOp::Mul) && wide {
                let value = 1u128 << rng.below(8);
                dag.konst(value, w)
            } else if matches!(op, BinOp::Shl | BinOp::Lshr | BinOp::Ashr) && rng.chance(1, 2) {
                // Shift amounts at or past the width.
                let value = u128::from(w) + u128::from(rng.below(3));
                dag.konst(value, w)
            } else {
                pick_bv(rng, dag, w)
            };
            dag.push(Node::Bin { op, lhs, rhs }, w, true)
        }
        6 if w > 2 => {
            let lo = rng.below(u64::from(w / 2)) as u32;
            let hi = lo + rng.below(u64::from(w - lo)) as u32;
            let arg = if rng.chance(1, 2) {
                gen_ite(rng, dag, 1)
            } else {
                pick_bv(rng, dag, w)
            };
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
        7 if w >= 2 => {
            let hw = w / 2;
            let lw = w - hw;
            let hi_src = if rng.chance(1, 2) {
                gen_ite(rng, dag, 1)
            } else {
                pick_bv(rng, dag, w)
            };
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
                let els = dag.konst(gen_const(rng, lw), lw);
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
        8 if opaque => {
            // An opaque leaf under an operation, or inside a selector.
            let leaf = if rng.chance(1, 2) {
                dag.uf()
            } else {
                dag.sel()
            };
            if rng.chance(1, 2) {
                let other = pick_bv(rng, dag, w);
                let op = *rng.pick(&[BinOp::Add, BinOp::Sub, BinOp::Xor, BinOp::And]);
                dag.push(
                    Node::Bin {
                        op,
                        lhs: leaf,
                        rhs: other,
                    },
                    w,
                    true,
                )
            } else {
                let other = pick_bv(rng, dag, w);
                let op = *rng.pick(&ALL_CMPS);
                let cond = dag.push(
                    Node::Cmp {
                        op,
                        lhs: leaf,
                        rhs: other,
                    },
                    0,
                    true,
                );
                let then = pick_bv(rng, dag, w);
                let els = pick_bv(rng, dag, w);
                dag.push(Node::Ite { cond, then, els }, w, true)
            }
        }
        _ => gen_ite(rng, dag, 2),
    }
}

pub struct Problem {
    pub dag: Dag,
    pub asserts: Vec<NodeId>,
}

pub struct GenConfig {
    pub base_width: u32,
    pub num_vars: usize,
    pub num_bools: usize,
    pub growth: usize,
    pub asserts: usize,
    pub opaque: bool,
}

pub fn gen_assert(rng: &mut Rng, dag: &mut Dag) -> NodeId {
    match rng.below(10) {
        0..=3 => gen_cmp(rng, dag),
        4 | 5 => {
            let c = gen_cmp(rng, dag);
            dag.push(Node::BNot(c), 0, true)
        }
        6 => {
            let v = rng.below(dag.num_vars as u64) as usize;
            let w = dag.base_width;
            let k = dag.konst(gen_const(rng, w), w);
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
        7 if dag.num_bools > 0 => {
            let p = dag.num_vars + rng.below(dag.num_bools as u64) as usize;
            if rng.chance(1, 2) {
                p
            } else {
                dag.push(Node::BNot(p), 0, true)
            }
        }
        _ => gen_cond(rng, dag, 2),
    }
}

pub fn gen_problem(rng: &mut Rng, cfg: &GenConfig) -> Problem {
    let mut dag = Dag::new(cfg.base_width, cfg.num_vars, cfg.num_bools);
    for _ in 0..cfg.growth {
        grow(rng, &mut dag, cfg.opaque);
    }
    let asserts = (0..cfg.asserts)
        .map(|_| gen_assert(rng, &mut dag))
        .collect();
    Problem { dag, asserts }
}
