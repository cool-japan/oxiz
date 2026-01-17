; Simple bitvector benchmark for BV solver performance
; Tests basic bitvector operations

(set-logic QF_BV)

(declare-const x (_ BitVec 32))
(declare-const y (_ BitVec 32))
(declare-const z (_ BitVec 32))

; Basic constraints
(assert (= (bvadd x y) #x00000064))  ; x + y = 100
(assert (bvult x #x00000050))         ; x < 80
(assert (bvugt y #x00000014))         ; y > 20

; Bitwise operations
(assert (= (bvand x #x0000000F) #x00000005))  ; x & 0xF = 5
(assert (= (bvor y #x000000F0) #x000000FF))   ; y | 0xF0 = 0xFF

; Result constraint
(assert (= z (bvxor x y)))

; This should be satisfiable
(check-sat)
(get-model)
(exit)
