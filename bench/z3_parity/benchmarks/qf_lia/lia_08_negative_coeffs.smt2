; Test: Negative coefficients
; Expected: sat
; Pattern: Mix of positive and negative multipliers

(set-logic QF_LIA)
(declare-const x Int)
(declare-const y Int)

(assert (= (+ (* -2 x) (* 3 y)) 6))
(assert (>= x 0))
(assert (<= x 10))
(assert (>= y 0))
(assert (<= y 10))

(check-sat)
