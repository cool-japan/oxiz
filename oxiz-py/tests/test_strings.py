"""Tests for string theory support in OxiZ Python bindings.

String operations (StringVal, Length, Concat, Contains) are not yet wrapped
in the PyO3 layer.  These tests document the current gap.
"""

import oxiz


def test_string_val_not_yet_available():
    """StringVal is not yet exposed at module level.

    # TODO: not yet wrapped — add StringVal(s) once string theory is
    # implemented in the PyO3 layer.
    """
    assert not hasattr(oxiz, "StringVal"), (
        "StringVal unexpectedly found — update this test to exercise it"
    )


def test_string_combinators_not_yet_available():
    """String combinators (Length, Concat, Contains, PrefixOf, SuffixOf)
    are not yet exposed at module level.

    # TODO: not yet wrapped — add these once string theory is wrapped.
    """
    missing_names = {"Length", "Concat", "Contains", "PrefixOf", "SuffixOf"}
    for name in missing_names:
        assert not hasattr(oxiz, name), (
            f"{name!r} is now present — update this test to add a real string test."
        )


def test_tm_has_no_mk_string_methods():
    """TermManager does not yet expose mk_string_* methods.

    # TODO: not yet wrapped — when string theory is added to TermManager,
    # remove this test and add tests for mk_string_val, mk_str_concat, etc.
    """
    tm = oxiz.TermManager()
    string_methods = ["mk_string_val", "mk_str_concat", "mk_str_length", "mk_str_contains"]
    for method in string_methods:
        assert not hasattr(tm, method), (
            f"TermManager.{method} is now present — add a real string test."
        )


def test_string_sort_not_parseable_via_mk_var():
    """mk_var does not yet accept a 'String' sort string.

    # TODO: not yet wrapped — when string sort is registered in
    # parse_sort_name(), update this test.
    """
    import pytest
    tm = oxiz.TermManager()
    with pytest.raises((ValueError, Exception)):
        tm.mk_var("s", "String")


def test_context_has_no_string_const():
    """Context does not yet have a string_const() factory.

    # TODO: not yet wrapped — add ctx.string_const(name) when string sort
    # is integrated into Context.
    """
    ctx = oxiz.Context()
    assert not hasattr(ctx, "string_const"), (
        "Context.string_const is now present — add a real string test."
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
