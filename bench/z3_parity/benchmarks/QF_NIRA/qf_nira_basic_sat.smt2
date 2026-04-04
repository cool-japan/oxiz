; Test: QF_NIRA basic satisfiable formula
; Expected: sat
; x^2 = 4 has solutions (x=2 or x=-2), and y in (1.0, 2.0) is feasible

(set-logic QF_NIRA)
(declare-const x Int)
(declare-const y Real)
(assert (= (* x x) 4))
(assert (> y 1.0))
(assert (< y 2.0))
(check-sat)
; expected: sat
(exit)
