; Test: Solution at zero
; Expected: sat
; Pattern: Zero is valid solution

(set-logic QF_LRA)
(declare-const x Real)
(declare-const y Real)

(assert (= (+ x y) 0.0))
(assert (>= x -10.0))
(assert (<= x 10.0))
(assert (>= y -10.0))
(assert (<= y 10.0))

(check-sat)
