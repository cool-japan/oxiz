; Logical operations: Comparison operators (8-bit)
; Expected: UNSAT
; Tests bvult, bvugt, bvslt, bvsgt (unsigned and signed comparisons)

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))

; Contradictory unsigned comparisons
; x < y (unsigned)
; y < x (unsigned)
; These cannot both be true
(assert (bvult x y))
(assert (bvult y x))

; Should be UNSAT
(check-sat)
(exit)
