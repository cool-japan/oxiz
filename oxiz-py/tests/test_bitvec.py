"""Bitvector tests for OxiZ Python bindings"""

import oxiz
import pytest

def test_bv_constant():
    """Test bitvector constant creation"""
    tm = oxiz.TermManager()
    bv = tm.mk_bv(42, 8)
    assert bv is not None

def test_bv_variable():
    """Test bitvector variable creation"""
    tm = oxiz.TermManager()
    x = tm.mk_var("x", "BitVec[8]")
    y = tm.mk_var("y", "BitVec[16]")
    assert x != y

def test_bv_arithmetic():
    """Test bitvector arithmetic operations"""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    x = tm.mk_var("x", "BitVec[8]")
    y = tm.mk_var("y", "BitVec[8]")

    ten = tm.mk_bv(10, 8)
    three = tm.mk_bv(3, 8)

    # x = 3, y = 10, x + y should equal 13
    solver.assert_term(tm.mk_eq(x, three), tm)
    solver.assert_term(tm.mk_eq(y, ten), tm)

    sum_xy = tm.mk_bv_add(x, y)
    thirteen = tm.mk_bv(13, 8)
    solver.assert_term(tm.mk_eq(sum_xy, thirteen), tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

def test_bv_bitwise():
    """Test bitvector bitwise operations"""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    x = tm.mk_var("x", "BitVec[4]")

    # x = 0b1010 (10)
    val = tm.mk_bv(0b1010, 4)
    solver.assert_term(tm.mk_eq(x, val), tm)

    # NOT x should be 0b0101 (5)
    not_x = tm.mk_bv_not(x)
    expected = tm.mk_bv(0b0101, 4)
    solver.assert_term(tm.mk_eq(not_x, expected), tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

def test_bv_comparison_unsigned():
    """Test unsigned bitvector comparisons"""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    x = tm.mk_var("x", "BitVec[8]")
    five = tm.mk_bv(5, 8)
    ten = tm.mk_bv(10, 8)

    # Assert: 5 < x < 10 (unsigned)
    solver.assert_term(tm.mk_bv_ult(five, x), tm)  # 5 < x
    solver.assert_term(tm.mk_bv_ult(x, ten), tm)   # x < 10

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

    model = solver.get_model(tm)
    # x should be between 6 and 9
    x_val = int(model["x"].split('x')[1], 16)  # Parse hex value
    assert 6 <= x_val <= 9

def test_bv_concat():
    """Test bitvector concatenation"""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    # Create 4-bit bitvectors
    high = tm.mk_bv(0xA, 4)  # 1010
    low = tm.mk_bv(0x5, 4)    # 0101

    # Concatenate to get 8-bit: 10100101 = 0xA5
    result = tm.mk_bv_concat(high, low)
    expected = tm.mk_bv(0xA5, 8)

    solver.assert_term(tm.mk_eq(result, expected), tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

def test_bv_extract():
    """Test bitvector extraction"""
    tm = oxiz.TermManager()
    solver = oxiz.Solver()
    solver.set_logic("QF_BV")

    # x = 0xA5 = 10100101
    x = tm.mk_var("x", "BitVec[8]")
    val = tm.mk_bv(0xA5, 8)
    solver.assert_term(tm.mk_eq(x, val), tm)

    # Extract high 4 bits: should be 0xA
    high_bits = tm.mk_bv_extract(7, 4, x)
    expected_high = tm.mk_bv(0xA, 4)
    solver.assert_term(tm.mk_eq(high_bits, expected_high), tm)

    # Extract low 4 bits: should be 0x5
    low_bits = tm.mk_bv_extract(3, 0, x)
    expected_low = tm.mk_bv(0x5, 4)
    solver.assert_term(tm.mk_eq(low_bits, expected_low), tm)

    result = solver.check_sat(tm)
    assert result == oxiz.SolverResult.Sat

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
