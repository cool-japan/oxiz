; Word-level reasoning: Multiplication (8-bit)
; Expected: SAT
; Tests bvmul with word-level solving

(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))

; a * b = 60
; a > 1 and b > 1
(assert (= (bvmul a b) #x3C))
(assert (bvugt a #x01))
(assert (bvugt b #x01))

; Should be SAT: e.g., a=12, b=5 or a=6, b=10, etc.
(check-sat)
(exit)
