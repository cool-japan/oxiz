; Arithmetic operations: Unsigned remainder (16-bit)
; Expected: UNSAT
; Tests bvurem with contradictory constraints

(set-logic QF_BV)
(declare-const x (_ BitVec 16))
(declare-const y (_ BitVec 16))

; x % y = 10
; y = 5
; These constraints are contradictory (remainder cannot be >= divisor)
(assert (= (bvurem x y) #x000A))
(assert (= y #x0005))

; Should be UNSAT: remainder 10 with divisor 5 is impossible
(check-sat)
(exit)
