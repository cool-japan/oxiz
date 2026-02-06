; Test: Very small coefficients
; Expected: sat
; Pattern: Numerical stability test

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ (* 0.0001 x) (* 0.0002 y)) 0.005))
(assert (>= x 0.0))
(assert (<= x 100.0))
(assert (>= y 0.0))
(assert (<= y 100.0))

(check-sat)
