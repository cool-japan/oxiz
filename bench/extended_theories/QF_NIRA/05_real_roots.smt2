; Test: Polynomial root finding with real coefficients
;; expected: sat
; Pattern: x^2 - 3.0*x + 2.0 = 0 => x = 1.0 or x = 2.0

(set-logic QF_NIRA)

(declare-const x Real)

; x^2 - 3x + 2 = 0
(assert (= (+ (- (* x x) (* 3.0 x)) 2.0) 0.0))

; Require x > 0
(assert (> x 0.0))

; Bound the solution
(assert (<= x 3.0))

(check-sat)
(exit)
