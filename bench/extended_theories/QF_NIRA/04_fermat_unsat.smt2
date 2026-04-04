; Test: Simple impossible nonlinear equation
;; expected: unsat
; Pattern: x^2 + 1 = 0 has no real solution

(set-logic QF_NIRA)

(declare-const x Real)

; x^2 + 1 = 0 has no real solution since x^2 >= 0
(assert (= (+ (* x x) 1.0) 0.0))

(check-sat)
(exit)
