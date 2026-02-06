; Shift operations: Logical shift right (16-bit)
; Expected: SAT
; Tests bvlshr (logical shift right, zero-fill)

(set-logic QF_BV)
(declare-const x (_ BitVec 16))
(declare-const y (_ BitVec 16))

; x >> 4 = y
; x = 0xABCD
; y should be 0x0ABC (logical shift, high bits filled with 0)
(assert (= x #xABCD))
(assert (= (bvlshr x #x0004) y))
(assert (= y #x0ABC))

; Should be SAT
(check-sat)
(exit)
