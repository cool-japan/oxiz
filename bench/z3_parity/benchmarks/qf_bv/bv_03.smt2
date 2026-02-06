; Bit-blasting: XOR operations (16-bit)
; Expected: SAT
; Tests XOR properties and bit-level solving

(set-logic QF_BV)
(declare-const x (_ BitVec 16))
(declare-const y (_ BitVec 16))
(declare-const z (_ BitVec 16))

; XOR associativity: (x XOR y) XOR z = x XOR (y XOR z)
; x = 0xABCD
; (x XOR y) XOR z = 0x1234
(assert (= x #xABCD))
(assert (= (bvxor (bvxor x y) z) #x1234))
(assert (= (bvxor x (bvxor y z)) #x1234))

; Should be SAT: multiple solutions for y and z
(check-sat)
(exit)
