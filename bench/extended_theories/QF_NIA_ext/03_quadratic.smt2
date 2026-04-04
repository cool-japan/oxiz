; Test: Quadratic equation over integers
;; expected: sat
; Pattern: x^2 - 5*x + 6 = 0 => x = 2 or x = 3

(set-logic QF_NIA)

(declare-const x Int)

; x^2 - 5x + 6 = 0
(assert (= (+ (- (* x x) (* 5 x)) 6) 0))

; Bound the search
(assert (> x 0))
(assert (< x 10))

(check-sat)
(exit)
