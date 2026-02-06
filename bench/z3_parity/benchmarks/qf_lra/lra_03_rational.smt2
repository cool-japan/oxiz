; Test: Rational coefficients
; Expected: sat
; Pattern: Fractional values

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ (* 0.5 x) (* 0.333 y)) 2.5))
(assert (>= x 0.0))
(assert (<= x 10.0))
(assert (>= y 0.0))

(check-sat)
