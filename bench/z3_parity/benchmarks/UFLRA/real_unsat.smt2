; Test: Contradictory real quantified constraints
; Expected: unsat
; Pattern: f bounded above by 1.0 for all reals, but f(c) required to be > 1.0

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-const c Real)

; f is bounded above by 1
(assert (forall ((x Real)) (<= (f x) 1.0)))

; But f(c) must exceed 1
(assert (> (f c) 1.0))

(check-sat)
(exit)
