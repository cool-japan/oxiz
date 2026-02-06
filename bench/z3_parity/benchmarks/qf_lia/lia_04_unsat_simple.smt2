; Test: Simple unsatisfiable system
; Expected: unsat
; Pattern: Contradictory constraints

(set-logic QF_LIA)
(declare-const x Int)

(assert (< x 0))
(assert (> x 10))

(check-sat)
