; Test: Simplex feasibility
; Expected: sat
; Pattern: Standard simplex problem

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (>= (+ (* 2.0 x) (* 3.0 y)) 12.0))
(assert (<= (+ x y) 8.0))
(assert (>= x 0.0))
(assert (>= y 0.0))

(check-sat)
