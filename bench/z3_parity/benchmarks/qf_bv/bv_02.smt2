; Bit-blasting: OR operations (8-bit)
; Expected: UNSAT
; Tests contradictory OR constraints

(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))

; a OR b = 0b11111111
; a = 0b10101010
; b = 0b01010100 (contradicts with a OR b)
(assert (= (bvor a b) #b11111111))
(assert (= a #b10101010))
(assert (= b #b01010100))

; Should be UNSAT: 0b10101010 OR 0b01010100 = 0b11111110, not 0b11111111
(check-sat)
(exit)
