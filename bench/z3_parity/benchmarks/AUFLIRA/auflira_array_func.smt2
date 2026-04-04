; Test: AUFLIRA array with function application - unsatisfiable
; Expected: unsat
; select(a, 0) = f(1) = 42, but select(a, 0) > 100 is impossible

(set-logic AUFLIRA)
(declare-fun f (Int) Int)
(declare-const a (Array Int Int))
(assert (= (select a 0) (f 1)))
(assert (= (f 1) 42))
(assert (> (select a 0) 100))
(check-sat)
; expected: unsat
(exit)
