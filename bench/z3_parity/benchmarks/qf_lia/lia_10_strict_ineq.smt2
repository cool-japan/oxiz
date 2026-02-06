; Test: Strict inequalities
; Expected: sat
; Pattern: Mix of strict and non-strict inequalities

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (> x 5))
(assert (< y 10))
(assert (>= (+ x y) 10))
(assert (<= (+ x y) 20))

(check-sat)
