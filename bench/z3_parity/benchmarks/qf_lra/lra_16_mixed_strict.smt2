; Test: Mixed strict and non-strict inequalities
; Expected: sat
; Pattern: Combination of constraint types

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)
(declare-const z Real)

(assert (> (+ x y) 10.0))
(assert (<= (+ y z) 20.0))
(assert (>= (+ x z) 15.0))
(assert (< (+ x y z) 40.0))
(assert (>= x 0.0))
(assert (>= y 0.0))
(assert (>= z 0.0))

(check-sat)
