; Shift operations: Left shift (8-bit)
; Expected: SAT
; Tests bvshl (shift left)

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const shift (_ BitVec 8))

; x << shift = 0b10000000
; shift = 3
(assert (= (bvshl x shift) #b10000000))
(assert (= shift #x03))

; Should be SAT: x = 0b00010000 (16)
(check-sat)
(exit)
