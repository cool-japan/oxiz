; Arithmetic operations: Unsigned division (8-bit)
; Expected: SAT
; Tests bvudiv (unsigned division)

(set-logic QF_BV)
(declare-const dividend (_ BitVec 8))
(declare-const divisor (_ BitVec 8))
(declare-const quotient (_ BitVec 8))

; dividend / divisor = quotient
; dividend = 100
; quotient = 5
; divisor should be 20
(assert (= dividend #x64))
(assert (= quotient #x05))
(assert (= (bvudiv dividend divisor) quotient))

; Should be SAT: divisor = 20
(check-sat)
(exit)
