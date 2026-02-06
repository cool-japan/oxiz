; Shift operations: Arithmetic shift right (8-bit)
; Expected: SAT
; Tests bvashr (arithmetic shift right, sign-extend)

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const result (_ BitVec 8))

; x is negative (MSB = 1)
; x >> 2 (arithmetic) should preserve sign bit
; x = 0b11110000 = -16 in two's complement
; result = 0b11111100 = -4 (sign-extended)
(assert (= x #b11110000))
(assert (= (bvashr x #x02) result))
(assert (= result #b11111100))

; Should be SAT
(check-sat)
(exit)
