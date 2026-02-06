; Test: Linear inequalities
; Expected: sat
; Pattern: Inequality constraints

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (>= (+ (* 2 x) (* 3 y)) 10))
(assert (<= (+ (* -1 x) (* 2 y)) 5))
(assert (>= x 0))
(assert (>= y 0))

(check-sat)
