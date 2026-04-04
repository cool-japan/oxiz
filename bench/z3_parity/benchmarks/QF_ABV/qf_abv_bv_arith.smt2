; Test: QF_ABV bitvector arithmetic contradiction - unsatisfiable
; Expected: unsat
; x = #x05, select(a, x) = bvadd(x, #x01) = #x06, but assert select(a, #x05) = #x10

(set-logic QF_ABV)
(declare-const x (_ BitVec 8))
(declare-const a (Array (_ BitVec 8) (_ BitVec 8)))
(assert (= (select a x) (bvadd x #x01)))
(assert (= x #x05))
(assert (= (select a #x05) #x10))
(check-sat)
; expected: unsat
(exit)
