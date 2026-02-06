; Logical operations: NAND (16-bit)
; Expected: SAT
; Tests bvnand (bitwise NAND)

(set-logic QF_BV)
(declare-const a (_ BitVec 16))
(declare-const b (_ BitVec 16))
(declare-const c (_ BitVec 16))

; NAND(a, b) = NOT(AND(a, b))
; a = 0xF0F0
; b = 0xFF00
; c = NAND(a, b) = NOT(0xF000) = 0x0FFF
(assert (= a #xF0F0))
(assert (= b #xFF00))
(assert (= (bvnand a b) c))
(assert (= c #x0FFF))

; Should be SAT
(check-sat)
(exit)
