; Test: Quantified arithmetic contradiction
; Expected: unsat
; Pattern: f is bounded above by 10 for all inputs, but we need f(k) > 10

(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-const k Int)

; All function values are at most 10
(assert (forall ((x Int)) (<= (f x) 10)))

; But we want f(k) > 10
(assert (> (f k) 10))

(check-sat)
(exit)
