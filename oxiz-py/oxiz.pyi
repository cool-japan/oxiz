"""
Type stubs for the OxiZ SMT Solver Python bindings.

OxiZ is a high-performance SMT (Satisfiability Modulo Theories) solver
implemented in Pure Rust, exposed to Python via PyO3/maturin.

Quick start (z3-python parity API)::

    import oxiz

    ctx = oxiz.Context()
    solver = oxiz.Solver()

    x = ctx.int_const("x")
    y = ctx.int_const("y")

    solver.add(x + y > ctx.int_val(0), ctx.tm)
    solver.add(x < ctx.int_val(10), ctx.tm)

    result = solver.check(ctx.tm)
    if result.is_sat:
        m = solver.model()
        print(m)  # e.g. {"x": 5, "y": -3}

    # --- Incremental solving with push/pop ---
    solver.push()
    solver.add(x == ctx.int_val(7), ctx.tm)
    result2 = solver.check(ctx.tm)
    solver.pop()

    # --- Unsat core ---
    solver.set_option("produce-unsat-cores", "true")
    solver.assert_and_track(x > ctx.int_val(0), "pos_x", ctx.tm)
    solver.assert_and_track(x < ctx.int_val(0), "neg_x", ctx.tm)
    if solver.check(ctx.tm).is_unsat:
        core = solver.unsat_core()   # ["pos_x", "neg_x"]

    # --- Timeout ---
    solver.set_timeout(milliseconds=5000)

    # --- Boolean combinators ---
    a = ctx.bool_const("a")
    b = ctx.bool_const("b")
    solver.add(oxiz.And(a, b), ctx.tm)
    solver.add(oxiz.Or(a, oxiz.Not(b)), ctx.tm)
    solver.add(oxiz.Implies(a, b), ctx.tm)
    solver.add(oxiz.If(a, x, y) > ctx.int_val(0), ctx.tm)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

__version__: str


# ====================================================================== #
# Term                                                                     #
# ====================================================================== #

class Term:
    """A reference to a term stored in a TermManager or Context.

    Terms are immutable handles into the term storage.  They are cheap to
    copy and compare.  When created via :class:`Context`, they carry a
    back-reference to the owning TermManager which enables Python operator
    overloads (``+``, ``-``, ``*``, ``<``, ``<=``, ``>``, ``>=``).
    """

    @property
    def id(self) -> int:
        """Raw numeric term ID."""
        ...

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

    # Arithmetic operators (return boolean or arithmetic Terms, not Python scalars)
    def __add__(self, other: Term) -> Term: ...
    def __sub__(self, other: Term) -> Term: ...
    def __mul__(self, other: Term) -> Term: ...
    def __neg__(self) -> Term: ...

    # Comparison operators (return a boolean-sort Term, not a Python bool)
    def __lt__(self, other: Term) -> Term: ...
    def __le__(self, other: Term) -> Term: ...
    def __gt__(self, other: Term) -> Term: ...
    def __ge__(self, other: Term) -> Term: ...

    def eq_term(self, other: Term) -> Term:
        """Return a boolean Term expressing structural equality (``self == other``).

        Note: Python's ``==`` operator is overloaded to test structural identity
        (returning ``bool``) for use as a dict key; use ``eq_term`` to obtain
        an SMT equality constraint.
        """
        ...


# ====================================================================== #
# SolverResult                                                             #
# ====================================================================== #

class SolverResult:
    """Result of a satisfiability check.

    One of: Sat, Unsat, Unknown.
    """

    Sat: SolverResult
    Unsat: SolverResult
    Unknown: SolverResult

    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        """Returns ``'sat'``, ``'unsat'``, or ``'unknown'``."""
        ...

    @property
    def is_sat(self) -> bool:
        """True if the result is satisfiable."""
        ...

    @property
    def is_unsat(self) -> bool:
        """True if the result is unsatisfiable."""
        ...

    @property
    def is_unknown(self) -> bool:
        """True if the result is unknown (timeout, incomplete, etc.)."""
        ...


# ====================================================================== #
# OptimizationResult                                                       #
# ====================================================================== #

class OptimizationResult:
    """Result of an optimization query.

    One of: Optimal, Unbounded, Unsat, Unknown.
    """

    Optimal: OptimizationResult
    Unbounded: OptimizationResult
    Unsat: OptimizationResult
    Unknown: OptimizationResult

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

    @property
    def is_optimal(self) -> bool: ...

    @property
    def is_unbounded(self) -> bool: ...

    @property
    def is_unsat(self) -> bool: ...


# ====================================================================== #
# TermManager                                                              #
# ====================================================================== #

class TermManager:
    """Low-level term factory.

    For most use cases prefer :class:`Context`, which wraps a TermManager and
    returns Terms with operator-overload support.

    All terms created by one TermManager must be used together with the same
    instance.  The TermManager is not thread-safe.

    Example::

        tm = oxiz.TermManager()
        x = tm.mk_var("x", "Int")
        zero = tm.mk_int(0)
        pos_x = tm.mk_gt(x, zero)
    """

    def __init__(self) -> None: ...

    # --- Boolean constants ---
    def mk_bool(self, value: bool) -> Term: ...

    # --- Integer constants ---
    def mk_int(self, value: int) -> Term: ...

    # --- Real constants ---
    def mk_real(self, numerator: int, denominator: int) -> Term:
        """Create a rational constant (numerator/denominator).

        Raises ``ValueError`` if denominator is zero.
        """
        ...

    # --- Variables ---
    def mk_var(self, name: str, sort_name: str) -> Term:
        """Declare a new variable.

        Args:
            name: Variable name (should be unique within the manager).
            sort_name: ``"Bool"``, ``"Int"``, ``"Real"``, or ``"BitVec[N]"``.

        Raises:
            ValueError: If the sort name is unrecognized.
        """
        ...

    # --- Boolean connectives ---
    def mk_not(self, term: Term) -> Term: ...
    def mk_and(self, terms: List[Term]) -> Term: ...
    def mk_or(self, terms: List[Term]) -> Term: ...
    def mk_implies(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_xor(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_eq(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_distinct(self, terms: List[Term]) -> Term: ...
    def mk_ite(self, cond: Term, then_branch: Term, else_branch: Term) -> Term: ...

    # --- Arithmetic ---
    def mk_add(self, terms: List[Term]) -> Term: ...
    def mk_sub(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_mul(self, terms: List[Term]) -> Term: ...
    def mk_neg(self, term: Term) -> Term: ...
    def mk_div(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_mod(self, lhs: Term, rhs: Term) -> Term: ...

    # --- Comparison ---
    def mk_lt(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_le(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_gt(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_ge(self, lhs: Term, rhs: Term) -> Term: ...

    # --- Bit-vectors ---
    def mk_bv(self, value: int, width: int) -> Term:
        """Create a bitvector constant with the given bit width."""
        ...

    def mk_bv_concat(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_extract(self, high: int, low: int, arg: Term) -> Term: ...
    def mk_bv_not(self, arg: Term) -> Term: ...
    def mk_bv_and(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_or(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_add(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_sub(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_mul(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_neg(self, arg: Term) -> Term: ...
    def mk_bv_ult(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_slt(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_ule(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_sle(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_udiv(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_sdiv(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_urem(self, lhs: Term, rhs: Term) -> Term: ...
    def mk_bv_srem(self, lhs: Term, rhs: Term) -> Term: ...

    # --- Arrays ---
    def mk_select(self, array: Term, index: Term) -> Term: ...
    def mk_store(self, array: Term, index: Term, value: Term) -> Term: ...

    # --- Utilities ---
    def term_to_string(self, term: Term) -> str:
        """Return a human-readable string representation of a term."""
        ...


# ====================================================================== #
# Context (z3-python parity)                                               #
# ====================================================================== #

class Context:
    """High-level context bundling a TermManager with named-constant factories.

    Terms returned by :meth:`int_const`, :meth:`real_const`,
    :meth:`bool_const`, and :meth:`bv_const` carry a reference to this
    context's TermManager, which enables Python operator overloads
    (``x + y``, ``x < y``, etc.) without explicit TermManager bookkeeping.

    Access the underlying :class:`TermManager` as ``ctx.tm``.

    Example::

        ctx = oxiz.Context()
        x = ctx.int_const("x")
        y = ctx.int_const("y")

        solver = oxiz.Solver()
        solver.add(x + y > ctx.int_val(0), ctx.tm)
        result = solver.check(ctx.tm)
        if result.is_sat:
            print(solver.model())
    """

    def __init__(self) -> None: ...

    @property
    def tm(self) -> TermManager:
        """The underlying TermManager owned by this Context."""
        ...

    # --- Named-constant factories ---

    def int_const(self, name: str) -> Term:
        """Declare and return a named integer constant."""
        ...

    def real_const(self, name: str) -> Term:
        """Declare and return a named real-valued constant."""
        ...

    def bool_const(self, name: str) -> Term:
        """Declare and return a named boolean constant."""
        ...

    def bv_const(self, name: str, width: int) -> Term:
        """Declare and return a named bitvector constant with ``width`` bits."""
        ...

    def const_of_sort(self, name: str, sort_name: str) -> Term:
        """Declare a constant with an explicit sort name.

        Args:
            name: Constant name.
            sort_name: ``"Bool"``, ``"Int"``, ``"Real"``, or ``"BitVec[N]"``.
        """
        ...

    # --- Literal factories ---

    def int_val(self, value: int) -> Term:
        """Create an integer literal."""
        ...

    def real_val(self, numerator: int, denominator: int) -> Term:
        """Create a rational literal (numerator/denominator).

        Raises ``ValueError`` if denominator is zero.
        """
        ...

    def bool_val(self, value: bool) -> Term:
        """Create a boolean literal."""
        ...

    def bv_val(self, value: int, width: int) -> Term:
        """Create a bitvector literal."""
        ...


# ====================================================================== #
# Solver                                                                   #
# ====================================================================== #

class Solver:
    """CDCL(T) SMT Solver with full z3-python parity.

    Supports:
    - Incremental solving via ``push()`` / ``pop()``.
    - Typed model retrieval via ``model()``.
    - Unsat-core computation via ``unsat_core()`` / ``get_unsat_core()``.
    - Timeouts via ``set_timeout()``.

    The Solver is not thread-safe; use separate instances per thread.

    Example::

        ctx = oxiz.Context()
        solver = oxiz.Solver()
        x = ctx.int_const("x")

        solver.push()
        solver.add(x > ctx.int_val(5), ctx.tm)
        if solver.check(ctx.tm).is_sat:
            print(solver.model())   # {"x": 6} (or similar)
        solver.pop()
    """

    def __init__(self) -> None: ...

    # ------------------------------------------------------------------ #
    # Assertion                                                            #
    # ------------------------------------------------------------------ #

    def assert_term(self, formula: Term, tm: TermManager) -> None:
        """Assert a boolean Term as a hard constraint."""
        ...

    def add(self, formula: Term, tm: TermManager) -> None:
        """Alias for :meth:`assert_term` matching z3-python's ``add()`` API."""
        ...

    def assert_formula(self, formula: str, tm: TermManager) -> None:
        """Assert a formula given as an SMT-LIB2 string expression.

        Raises:
            ValueError: If the formula string cannot be parsed.
        """
        ...

    def assert_expr(
        self, expr: str, tm: TermManager, name: Optional[str] = None
    ) -> None:
        """Assert an SMT-LIB2 expression string with an optional label.

        Named assertions participate in unsat-core reporting when
        ``produce-unsat-cores`` is enabled via :meth:`set_option`.

        Raises:
            ValueError: If the expression cannot be parsed.
        """
        ...

    def assert_and_track(
        self, term: Term, label: str, tm: TermManager
    ) -> None:
        """Assert a Term and associate it with a tracking label for unsat cores.

        Equivalent to z3-python's ``solver.assert_and_track(expr, label)``.
        Requires ``produce-unsat-cores`` to be enabled via :meth:`set_option`
        before calling :meth:`check_sat`.

        Args:
            term: A boolean Term.
            label: A string label used in the unsat core.
            tm: The TermManager that owns the term.
        """
        ...

    # ------------------------------------------------------------------ #
    # Check                                                                #
    # ------------------------------------------------------------------ #

    def check_sat(self, tm: TermManager) -> SolverResult:
        """Check satisfiability of the current assertion set.

        Returns :attr:`SolverResult.Sat`, :attr:`SolverResult.Unsat`,
        or :attr:`SolverResult.Unknown`.
        """
        ...

    def check(self, tm: TermManager) -> SolverResult:
        """Alias for :meth:`check_sat` matching z3-python's ``check()`` API."""
        ...

    # ------------------------------------------------------------------ #
    # Push / pop                                                           #
    # ------------------------------------------------------------------ #

    def push(self) -> None:
        """Push a new assertion scope.

        Subsequent assertions can be undone by a matching :meth:`pop` call.
        """
        ...

    def pop(self, n: int = 1) -> None:
        """Pop one or more assertion scopes.

        Args:
            n: Number of scopes to pop (default 1).
        """
        ...

    def reset(self) -> None:
        """Reset the solver, removing all assertions and learned clauses."""
        ...

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #

    def get_model(self, tm: TermManager) -> Dict[str, str]:
        """Get the satisfying model as a string-valued dictionary.

        Only meaningful after :meth:`check_sat` returns :attr:`SolverResult.Sat`.

        Returns:
            Dict mapping variable names to string-encoded values.
        """
        ...

    def model(self) -> Dict[str, Any]:
        """Get the satisfying model as a typed Python dictionary.

        Does not require passing a TermManager (uses cached state from the
        last :meth:`check_sat` call).  Only meaningful after
        :meth:`check_sat` returns :attr:`SolverResult.Sat`.

        Returns:
            Dict mapping variable names to typed Python values:

            - Booleans → ``bool``
            - Integers that fit in 64 bits → ``int``
            - Large integers → ``str`` (decimal)
            - Whole-number rationals → ``int``
            - Non-whole rationals → ``str`` (``"numer/denom"``)
            - Bitvectors → ``int`` (unsigned, ≤ 64 bits)
            - Other terms → ``str``

            Returns an empty dict if no model is available.

        Example::

            result = solver.check_sat(tm)
            if result.is_sat:
                m = solver.model()
                # e.g. {"x": 5, "y": True, "flag": False}
        """
        ...

    # ------------------------------------------------------------------ #
    # Unsat core                                                           #
    # ------------------------------------------------------------------ #

    def get_unsat_core(self) -> List[str]:
        """Get the unsat core as a list of assertion label strings.

        Only meaningful after :meth:`check_sat` returns
        :attr:`SolverResult.Unsat` AND ``produce-unsat-cores`` was enabled
        via :meth:`set_option`.

        Returns an empty list if no core is available.
        """
        ...

    def unsat_core(self) -> List[str]:
        """Alias for :meth:`get_unsat_core` matching z3-python's API."""
        ...

    # ------------------------------------------------------------------ #
    # Configuration                                                        #
    # ------------------------------------------------------------------ #

    def set_logic(self, logic: str) -> None:
        """Set the SMT-LIB2 logic (e.g., ``"QF_LIA"``, ``"QF_LRA"``)."""
        ...

    def set_option(self, key: str, value: str) -> None:
        """Set a solver option.

        Supported keys:

        ============================  ========================================
        Key                           Description
        ============================  ========================================
        ``produce-unsat-cores``       ``"true"``/``"false"`` — enable cores
        ``produce_unsat_cores``       Same, underscore variant
        ``logic``                     SMT-LIB2 logic name (same as set_logic)
        ``timeout``                   Integer milliseconds string
        ============================  ========================================

        Raises:
            ValueError: If the key is unknown or the value is invalid.
        """
        ...

    def set_timeout(self, milliseconds: int) -> None:
        """Set a solving timeout in milliseconds.

        When the timeout expires the solver returns :attr:`SolverResult.Unknown`.
        Pass ``0`` to disable the timeout.
        """
        ...

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def num_assertions(self) -> int:
        """Number of assertions currently in the solver."""
        ...

    @property
    def context_level(self) -> int:
        """Current push/pop context depth."""
        ...


# ====================================================================== #
# Optimizer                                                                #
# ====================================================================== #

class Optimizer:
    """MaxSMT / Optimization solver.

    Supports minimization and maximization of objective terms over
    a set of hard constraints.

    Example::

        tm = oxiz.TermManager()
        opt = oxiz.Optimizer()

        x = tm.mk_var("x", "Int")
        opt.assert_term(tm.mk_ge(x, tm.mk_int(0)))
        opt.minimize(x)
        result = opt.optimize(tm)
        # result == OptimizationResult.Optimal
    """

    def __init__(self) -> None: ...

    def assert_term(self, formula: Term) -> None:
        """Assert a hard constraint."""
        ...

    def minimize(self, objective: Term) -> None:
        """Add a minimization objective."""
        ...

    def maximize(self, objective: Term) -> None:
        """Add a maximization objective."""
        ...

    def set_logic(self, logic: str) -> None:
        """Set the SMT-LIB2 logic."""
        ...

    def push(self) -> None:
        """Push a new scope."""
        ...

    def pop(self) -> None:
        """Pop a scope."""
        ...

    def optimize(self, tm: TermManager) -> OptimizationResult:
        """Run optimization and return the result."""
        ...

    def get_model(self, tm: TermManager) -> Dict[str, str]:
        """Get the optimal model as a string-keyed dictionary."""
        ...


# ====================================================================== #
# Module-level combinators (z3-python parity)                             #
# ====================================================================== #

def And(*args: Term) -> Term:
    """Construct the conjunction of one or more boolean Terms.

    All terms must have been created by the same Context / TermManager.
    The TermManager is resolved from the first term with an owner reference.

    Example::

        result = oxiz.And(a, b, c)

    Raises:
        ValueError: If no arguments are provided or none have an owner.
    """
    ...


def Or(*args: Term) -> Term:
    """Construct the disjunction of one or more boolean Terms.

    Example::

        result = oxiz.Or(a, b, c)
    """
    ...


def Not(term: Term) -> Term:
    """Construct the logical negation of a boolean Term.

    Example::

        result = oxiz.Not(a)
    """
    ...


def Implies(lhs: Term, rhs: Term) -> Term:
    """Construct the implication ``lhs => rhs``.

    Example::

        result = oxiz.Implies(a, b)
    """
    ...


def If(cond: Term, then_: Term, else_: Term) -> Term:
    """Construct an if-then-else expression.

    Example::

        result = oxiz.If(cond, then_val, else_val)
    """
    ...


# ====================================================================== #
# Explicit-TM variants (for bare-TermManager workflows)                   #
# ====================================================================== #

def And_tm(tm: TermManager, *args: Term) -> Term:
    """Like :func:`And` but accepts an explicit TermManager as the first argument.

    Use when terms were created via ``TermManager.mk_var()`` (no Context).
    """
    ...


def Or_tm(tm: TermManager, *args: Term) -> Term:
    """Like :func:`Or` but with an explicit TermManager."""
    ...


def Not_tm(tm: TermManager, term: Term) -> Term:
    """Like :func:`Not` but with an explicit TermManager."""
    ...
