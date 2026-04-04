; Test: QF_NIRA real polynomial unsatisfiable
; Expected: unsat
; x^2 = -1 has no real solution

(set-logic QF_NIRA)
(declare-const x Real)
(assert (= (* x x) (- 1.0)))
(check-sat)
; expected: unsat
(exit)
