; Test: Mixed integer/real nonlinear constraints
;; expected: sat
; Pattern: Integer n, real x, with n*x = 6.0 and n^2 <= 10

(set-logic QF_NIRA)

(declare-const n Int)
(declare-const x Real)

; n * x = 6.0
(assert (= (* (to_real n) x) 6.0))

; n^2 <= 10, so n in {-3, -2, -1, 0, 1, 2, 3} (but n != 0)
(assert (<= (* n n) 10))
(assert (not (= n 0)))

; x must be positive
(assert (> x 0.0))

; n must be positive (since x > 0 and product is positive)
(assert (> n 0))

(check-sat)
(exit)
