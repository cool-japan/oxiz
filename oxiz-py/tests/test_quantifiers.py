"""Tests for quantifier support in OxiZ Python bindings.

ForAll and Exists are not yet wrapped in the PyO3 layer.
These tests document the current gap and the intended z3-parity API.
"""

import oxiz


def test_forall_not_yet_available():
    """ForAll is not yet exposed in the oxiz Python module."""
    # TODO: not yet wrapped — oxiz-core has quantifier AST nodes but the
    # PyO3 wrapper does not export a ForAll() combinator yet.
    assert not hasattr(oxiz, "ForAll"), (
        "ForAll unexpectedly found — update this test to exercise it"
    )


def test_exists_not_yet_available():
    """Exists is not yet exposed in the oxiz Python module."""
    # TODO: not yet wrapped — same gap as ForAll above.
    assert not hasattr(oxiz, "Exists"), (
        "Exists unexpectedly found — update this test to exercise it"
    )


def test_module_has_expected_combinators():
    """Verify the boolean combinators that ARE available to confirm the
    module loads correctly and the gap is genuinely about quantifiers."""
    assert hasattr(oxiz, "And")
    assert hasattr(oxiz, "Or")
    assert hasattr(oxiz, "Not")
    assert hasattr(oxiz, "Implies")
    assert hasattr(oxiz, "If")


def test_term_manager_has_no_mk_forall():
    """TermManager does not expose mk_forall yet."""
    tm = oxiz.TermManager()
    # TODO: not yet wrapped — when implemented, remove this test and add
    # mk_forall / mk_exists tests that exercise quantified formulas.
    assert not hasattr(tm, "mk_forall")
    assert not hasattr(tm, "mk_exists")


def test_quantifier_gap_documented():
    """Regression sentinel: verifies that the exactly-missing names are still
    missing (not more, not fewer)."""
    missing = {"ForAll", "Exists"}
    for name in missing:
        assert not hasattr(oxiz, name), (
            f"{name!r} is now present — remove it from the missing set and add "
            "a real test exercising that combinator."
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
