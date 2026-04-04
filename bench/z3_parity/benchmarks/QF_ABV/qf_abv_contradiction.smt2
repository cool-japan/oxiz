; Test: QF_ABV contradiction at the same index - unsatisfiable
; Expected: unsat
; select(a, #x0) = #x5 and select(a, #x0) = #x6 is impossible

(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 4) (_ BitVec 4)))
(assert (= (select a #x0) #x5))
(assert (= (select a #x0) #x6))
(check-sat)
; expected: unsat
(exit)
