; Test: Strict inequality at boundary
; Expected: sat
; Pattern: Just above/below boundary

(set-logic QF_LRA)
(declare-const x Real)

(assert (> x 5.0))
(assert (< x 5.1))

(check-sat)
