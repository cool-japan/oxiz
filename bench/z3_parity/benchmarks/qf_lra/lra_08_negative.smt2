; Test: Negative coefficients
; Expected: sat
; Pattern: Mix of positive and negative

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ (* -2.5 x) (* 3.7 y)) 10.0))
(assert (>= x 0.0))
(assert (<= x 20.0))
(assert (>= y 0.0))
(assert (<= y 20.0))

(check-sat)
