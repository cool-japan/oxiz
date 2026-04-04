; Test: Bitvector array bounds conflict
;; expected: unsat
; Pattern: BV comparison after store creates contradiction

(set-logic QF_ABV)
(declare-const mem (Array (_ BitVec 8) (_ BitVec 8)))

; Store 0x0A at address 0x00
(declare-const mem1 (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= mem1 (store mem #x00 #x0A)))

; Read value back
(declare-const val (_ BitVec 8))
(assert (= val (select mem1 #x00)))

; val must equal 0x0A (from store axiom)
; But require val to be bvugt 0x0A (strictly greater unsigned)
(assert (bvugt val #x0A))

; 0x0A is not > 0x0A => unsat
(check-sat)
(exit)
