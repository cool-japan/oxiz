"""Tests for floating-point theory support in OxiZ Python bindings.

FP sorts (FPSort, FPVal, fp_add, etc.) are not yet wrapped in the PyO3 layer.
These tests document the current gap and the intended z3-parity API surface.
"""

import oxiz


def test_fp_sort_not_yet_available():
    """FPSort constructor is not yet exposed at module level.

    # TODO: not yet wrapped — implement FPSort(eb, sb) in the PyO3 layer once
    # the floating-point theory is complete in oxiz-core.
    """
    assert not hasattr(oxiz, "FPSort"), (
        "FPSort unexpectedly found — update this test to exercise it"
    )


def test_fp_val_not_yet_available():
    """FPVal constructor is not yet exposed at module level.

    # TODO: not yet wrapped — add FPVal(sign, exp, sig, sort) once FP theory
    # is wrapped.
    """
    assert not hasattr(oxiz, "FPVal"), (
        "FPVal unexpectedly found — update this test to exercise it"
    )


def test_fp_combinators_not_yet_available():
    """FP arithmetic combinators are not yet exposed.

    # TODO: not yet wrapped — fp_add, fp_sub, fp_mul, fp_div, fp_abs, fp_neg,
    # fp_sqrt need to be added once the FP theory wrapper exists.
    """
    missing_fp_names = {"fp_add", "fp_sub", "fp_mul", "fp_div", "FPRoundingMode"}
    for name in missing_fp_names:
        assert not hasattr(oxiz, name), (
            f"{name!r} is now present — update this test to add a real FP test."
        )


def test_tm_has_no_mk_fp_methods():
    """TermManager does not yet expose mk_fp_* methods.

    # TODO: not yet wrapped — when FP theory is added to TermManager,
    # remove this test and add tests for mk_fp_val, mk_fp_add, etc.
    """
    tm = oxiz.TermManager()
    fp_methods = ["mk_fp_val", "mk_fp_add", "mk_fp_sub", "mk_fp_mul"]
    for method in fp_methods:
        assert not hasattr(tm, method), (
            f"TermManager.{method} is now present — add a real FP test for it."
        )


def test_module_version_present():
    """Sanity check: the module loads correctly and exposes __version__."""
    assert hasattr(oxiz, "__version__")
    assert isinstance(oxiz.__version__, str)
    assert len(oxiz.__version__) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
