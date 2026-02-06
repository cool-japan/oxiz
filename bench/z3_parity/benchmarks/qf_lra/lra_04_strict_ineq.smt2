; Test: Strict inequalities with reals
; Expected: sat
; Pattern: Mix of strict and non-strict

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (> x 5.0))
(assert (< y 10.0))
(assert (>= (+ x y) 10.0))
(assert (<= (+ x y) 20.0))

(check-sat)
