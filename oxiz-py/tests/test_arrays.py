"""Tests for array theory support in OxiZ Python bindings.

The TermManager exposes mk_select and mk_store (array read/write), but
there is no ArraySort constructor or dedicated array-sort string for
mk_var().  Tests reflect this partial support.
"""

import oxiz
import pytest


def test_tm_has_mk_select():
    """mk_select is exposed on TermManager."""
    tm = oxiz.TermManager()
    assert hasattr(tm, "mk_select")


def test_tm_has_mk_store():
    """mk_store is exposed on TermManager."""
    tm = oxiz.TermManager()
    assert hasattr(tm, "mk_store")


def test_array_sort_not_parseable_via_mk_var():
    """mk_var does not yet accept an 'Array[...]' sort string.

    # TODO: not yet wrapped — when an ArraySort is added to parse_sort_name,
    # update this test to create a real typed array variable.
    """
    tm = oxiz.TermManager()
    with pytest.raises((ValueError, Exception)):
        # This is expected to fail until array sorts are registered in
        # parse_sort_name().
        tm.mk_var("arr", "Array[Int,Int]")


def test_select_on_bare_term_does_not_panic():
    """mk_select and mk_store accept bare Term ids without crashing.

    We construct a synthetic scenario: use integer terms as stand-ins for
    array and index since there is no typed array variable yet.  The result
    is a term node that represents the select expression in the AST.
    """
    tm = oxiz.TermManager()
    # Use an integer variable as a stand-in for an array (untyped AST layer).
    arr_placeholder = tm.mk_var("arr_placeholder", "Int")
    idx = tm.mk_var("idx", "Int")
    val = tm.mk_int(42)

    # mk_select: builds AST node — should not raise
    sel = tm.mk_select(arr_placeholder, idx)
    assert sel is not None

    # mk_store: builds AST node — should not raise
    stored = tm.mk_store(arr_placeholder, idx, val)
    assert stored is not None

    # The resulting terms must be distinct term ids
    assert sel != stored


def test_select_store_produce_distinct_terms():
    """Repeated select/store calls on different indices produce distinct terms."""
    tm = oxiz.TermManager()
    arr = tm.mk_var("arr2", "Int")
    idx0 = tm.mk_int(0)
    idx1 = tm.mk_int(1)
    v10 = tm.mk_int(10)
    v20 = tm.mk_int(20)

    sel0 = tm.mk_select(arr, idx0)
    sel1 = tm.mk_select(arr, idx1)
    stored0 = tm.mk_store(arr, idx0, v10)
    stored1 = tm.mk_store(arr, idx1, v20)

    # All four should be distinct AST nodes
    assert sel0 != sel1
    assert stored0 != stored1
    assert sel0 != stored0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
