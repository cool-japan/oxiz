; Word-level reasoning: Addition (8-bit)
; Expected: SAT
; Tests bvadd with overflow handling

(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 8))
(declare-const z (_ BitVec 8))

; x + y = z
; x + y + 10 = 0 (tests overflow in 8-bit)
(assert (= (bvadd x y) z))
(assert (= (bvadd (bvadd x y) #x0A) #x00))

; Should be SAT: e.g., x=0xF0, y=0x06, z=0xF6
(check-sat)
(exit)
