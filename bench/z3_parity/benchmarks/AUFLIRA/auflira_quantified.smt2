; Test: AUFLIRA with quantifier - unsatisfiable
; Expected: unsat
; forall x. f(x) >= 0 contradicts f(5) < 0

(set-logic AUFLIRA)
(declare-fun f (Int) Int)
(assert (forall ((x Int)) (>= (f x) 0)))
(assert (< (f 5) 0))
(check-sat)
; expected: unsat
(exit)
