; Test: AUFLIRA unsatisfiable via arithmetic contradiction on function values
; Expected: unsat
; f(0) cannot simultaneously be 10 and 5

(set-logic AUFLIRA)
(declare-fun f (Int) Int)
(declare-fun g (Int) Real)
(assert (= (f 0) 10))
(assert (= (f 0) 5))
(check-sat)
; expected: unsat
(exit)
