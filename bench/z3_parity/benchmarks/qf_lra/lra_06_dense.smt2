; Test: Dense constraint matrix
; Expected: sat
; Pattern: All variables in all constraints

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)
(declare-const z Real)

(assert (>= (+ x y z) 10.0))
(assert (<= (+ x y z) 30.0))
(assert (>= (+ (* 2.0 x) (* 1.5 y) (* -0.5 z)) 5.0))
(assert (<= (+ (* 0.5 x) (* -1.0 y) (* 2.0 z)) 20.0))
(assert (>= x 0.0))
(assert (>= y 0.0))
(assert (>= z 0.0))

(check-sat)
