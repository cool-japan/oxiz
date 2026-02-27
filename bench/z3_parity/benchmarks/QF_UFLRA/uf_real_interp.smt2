; Interpolation-style problem with UF + LRA
; Expected: unsat
; Tests interaction between two groups of constraints sharing a UF

(set-logic QF_UFLRA)
(declare-fun f (Real) Real)
(declare-fun a () Real)
(declare-fun b () Real)
(declare-fun c () Real)

; Group A constraints: f(a) + f(b) > 10.0
(assert (> (+ (f a) (f b)) 10.0))

; Group B constraints: f(a) < 3.0 and f(b) < 3.0
(assert (< (f a) 3.0))
(assert (< (f b) 3.0))

; These are contradictory: f(a) < 3 and f(b) < 3 imply f(a) + f(b) < 6 < 10

(check-sat)
; expected: unsat
(exit)
