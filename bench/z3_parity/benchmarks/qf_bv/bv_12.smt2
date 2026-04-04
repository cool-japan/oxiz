; Arithmetic operations: Signed division and remainder (8-bit)
; Expected: unsat
; Tests bvsdiv and bvsrem (signed operations)
; Note: -13 = 6*(-2) + (-1), so bvsrem(-13,6) = -1 = 0xFF, not +1.
; The constraints are contradictory, hence unsat.

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))

; x / y = -2 (signed division)
; x % y = 1 (signed remainder)
; x = -13 (0xF3), y = 6
; -13 / 6 = -2, -13 % 6 = 1
(assert (= (bvsdiv x y) #xFE))  ; -2 in two's complement
(assert (= (bvsrem x y) #x01))
(assert (= y #x06))

; Should be SAT: x = 0xF3 (-13)
(check-sat)
(exit)
