; Logical operations: NOT (8-bit)
; Expected: SAT
; Tests bvnot (bitwise NOT)

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))

; NOT(x) = y
; x XOR y = 0xFF (all bits different)
; NOT(NOT(x)) = x
(assert (= (bvnot x) y))
(assert (= (bvxor x y) #xFF))
(assert (= (bvnot (bvnot x)) x))

; Should be SAT: always satisfied for any x
(check-sat)
(exit)
