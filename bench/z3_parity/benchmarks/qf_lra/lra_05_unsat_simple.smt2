; Test: Simple unsatisfiable system
; Expected: unsat
; Pattern: Contradictory real constraints

(set-logic QF_LRA)
(declare-const x Real)

(assert (< x 0.0))
(assert (> x 10.0))

(check-sat)
