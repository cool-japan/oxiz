; Bit-blasting: Basic AND operations (8-bit)
; Expected: SAT
; Tests basic bit-level reasoning with AND operations

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))

; x AND y = 0b10101010
; x = 0b11111111
(assert (= (bvand x y) #b10101010))
(assert (= x #b11111111))

; Should be SAT: y = 0b10101010
(check-sat)
(exit)
