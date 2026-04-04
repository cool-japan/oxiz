; Test: QF_NIRA mixed integer/real unsatisfiable formula
; Expected: unsat
; x^2 > 0 and x^2 + y = 0 and y >= 0 is unsatisfiable
; because x^2 > 0 implies x^2 + y > 0 when y >= 0

(set-logic QF_NIRA)
(declare-const x Int)
(declare-const y Real)
(assert (> (* x x) 0))
(assert (= (+ (* x x) y) 0.0))
(assert (>= y 0.0))
(check-sat)
; expected: unsat
(exit)
