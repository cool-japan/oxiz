#!/usr/bin/env python3
"""The shared shape generator behind gen_qmbqi.py and gen_q12.py.

One script = 2-3 arrays of sort (Array (_ BitVec w) (_ BitVec 1)), two Bool
selectors p and q, one symbolic bit d; 0-3 ground assertions built from
select equalities and bvxor atoms over ite-selected bases, stores at constant
indices and (as const ...) bases (some with the symbolic default d); and ONE
forall over the index sort whose body is one of the shapes below.  The ground
twin is the same script with the forall written out as an `and` over every
point of the index sort, in the same assertion position.

Everything is driven by one `random.Random(seed)`, so a (seed, index) pair
names exactly one pair of files.
"""
from __future__ import annotations

import random

BOUND = "i!q"


def bits(value: int, width: int) -> str:
    return "#b" + format(value, "0%db" % width)


class Gen:
    def __init__(self, rng: random.Random, width: int, n_arrays: int, use_ite: bool = True):
        self.r = rng
        self.w = width
        self.n = n_arrays
        self.use_ite = use_ite
        self.sort = "(Array (_ BitVec %d) (_ BitVec 1))" % width

    # -- leaves -----------------------------------------------------------
    def idx(self):
        return bits(self.r.randrange(1 << self.w), self.w)

    def bitc(self):
        return self.r.choice(["#b0", "#b1"])

    def var(self):
        return "a%d" % self.r.randrange(self.n)

    def const_array(self):
        v = self.r.choice(["#b0", "#b1", "#b0", "#b1", "d"])
        return "((as const %s) %s)" % (self.sort, v)

    # -- array terms ------------------------------------------------------
    def arr(self, depth: int):
        roll = self.r.random()
        if depth <= 0 or roll < 0.45:
            return self.var() if self.r.random() < 0.85 else self.const_array()
        if roll < 0.62:
            return self.const_array() if self.r.random() < 0.3 else self.var()
        if self.use_ite and roll < 0.85:
            c = self.r.choice(["p", "q"])
            return "(ite %s %s %s)" % (c, self.arr(depth - 1), self.arr(depth - 1))
        return "(store %s %s %s)" % (self.arr(depth - 1), self.idx(), self.bitc())

    # -- bit terms ---------------------------------------------------------
    def read(self, index=None, depth=2):
        return "(select %s %s)" % (self.arr(depth), index if index is not None else self.idx())

    def bit(self, depth=2, index=None):
        roll = self.r.random()
        if roll < 0.55:
            return self.read(index, depth)
        if roll < 0.85:
            return "(bvxor %s %s)" % (self.read(index, depth), self.bitc())
        if roll < 0.93:
            return "d"
        return self.bitc()

    # -- ground assertions -------------------------------------------------
    def atom(self):
        roll = self.r.random()
        if roll < 0.55:
            return "(= %s %s)" % (self.bit(), self.bit())
        if roll < 0.8:
            return "(distinct %s %s)" % (self.bit(), self.bit())
        return "(= %s %s)" % (self.bitc(), self.bit())

    def ground(self):
        roll = self.r.random()
        if roll < 0.5:
            return self.atom()
        if roll < 0.8:
            return "(or %s %s)" % (self.atom(), self.atom())
        if roll < 0.9:
            return "(and %s %s)" % (self.atom(), self.atom())
        return "(not %s)" % self.atom()

    # -- the quantifier body ----------------------------------------------
    def body(self):
        i = BOUND
        shapes = ["sel_eq", "sel_eq", "two_arr", "xor", "impl", "store_at", "min_const"]
        if self.use_ite:
            shapes += ["ite", "ite"]
        shape = self.r.choice(shapes)
        if shape == "min_const":
            # The #P2b-51 three-line family: one array read at the binder,
            # a constant on the other side.
            return shape, "(= (select %s %s) %s)" % (self.var(), i, self.bitc())
        if shape == "sel_eq":
            other = self.bit(depth=1) if self.r.random() < 0.6 else self.bitc()
            return shape, "(= (select %s %s) %s)" % (self.arr(1), i, other)
        if shape == "two_arr":
            op = self.r.choice(["=", "distinct"])
            rhs = "(select %s %s)" % (self.arr(1), i)
            if self.r.random() < 0.3:
                rhs = "(bvnot %s)" % rhs
            return shape, "(%s (select %s %s) %s)" % (op, self.arr(1), i, rhs)
        if shape == "xor":
            return shape, "(distinct (bvxor (select %s %s) %s) (select %s %s))" % (
                self.arr(1), i, self.bitc(), self.arr(1), i)
        if shape == "ite":
            return shape, "(= (select %s %s) (ite (= %s %s) %s %s))" % (
                self.arr(1), i, i, self.idx(), self.bitc(), self.bitc())
        if shape == "impl":
            guard = self.r.choice(["(= %s %s)" % (i, self.idx()), "(bvult %s %s)" % (i, self.idx())])
            return shape, "(=> %s (= (select %s %s) %s))" % (guard, self.arr(1), i, self.bit(depth=1))
        # store_at: read-over-write at the binder's own index
        return shape, "(= (select (store %s %s %s) %s) %s)" % (
            self.arr(1), i, self.bitc(), self.idx(), self.bit(depth=1))


def array_equality(g: "Gen") -> str:
    """`(= aK T)` with T an ite/store/(as const) term over the OTHER arrays,
    so the equality pins aK's value at every index (the shape of the
    wrong-`sat` family found 2026-09-28: an array fully determined by a
    ground equality, read under a binder)."""
    k = g.r.randrange(g.n)
    others = [j for j in range(g.n) if j != k]
    base = g.const_array() if g.r.random() < 0.6 or not others else "a%d" % g.r.choice(others)
    term = base
    for _ in range(g.r.choice([1, 1, 2])):
        term = "(store %s %s %s)" % (term, g.idx(), g.bitc())
    return "(= a%d %s)" % (k, term)


def make_pair(seed: int, index: int, widths, use_ite: bool, array_eq: bool = False):
    """(quantified script, ground twin, meta) for one corpus member."""
    r = random.Random("%d/%d" % (seed, index))
    width = r.choice(list(widths))
    n_arrays = r.choice([2, 3])
    g = Gen(r, width, n_arrays, use_ite)
    n_ground = r.choice([0, 1, 1, 2, 2, 3])
    grounds = [g.ground() for _ in range(n_ground)]
    if array_eq:
        grounds.insert(0, array_equality(g))
        n_ground += 1
    shape, body = g.body()
    position = r.randrange(n_ground + 1)
    decls = "".join("(declare-const a%d %s)\n" % (k, g.sort) for k in range(n_arrays))
    decls += "(declare-const p Bool)\n(declare-const q Bool)\n(declare-const d (_ BitVec 1))\n"
    quant = "(forall ((%s (_ BitVec %d))) %s)" % (BOUND, width, body)
    points = [body.replace(BOUND, bits(k, width)) for k in range(1 << width)]
    expanded = "(and %s)" % " ".join(points)
    q_asserts = list(grounds)
    g_asserts = list(grounds)
    q_asserts.insert(position, quant)
    g_asserts.insert(position, expanded)
    head = "(set-logic ALL)\n(set-option :produce-models true)\n"
    tail = "(check-sat)\n(get-model)\n"
    q_text = head + decls + "".join("(assert %s)\n" % a for a in q_asserts) + tail
    g_text = head + decls + "".join("(assert %s)\n" % a for a in g_asserts) + tail
    meta = {"width": width, "arrays": n_arrays, "ground": n_ground, "shape": shape}
    return q_text, g_text, meta
