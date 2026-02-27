; Unsatisfiable UF + LRA problem
; Expected: unsat
; Tests congruence with real arithmetic contradiction

(set-logic QF_UFLRA)
(declare-fun f (Real) Real)
(declare-fun x () Real)
(declare-fun y () Real)

; x = y
(assert (= x y))

; f(x) > 5.0
(assert (> (f x) 5.0))

; f(y) < 3.0  -- contradiction since x = y implies f(x) = f(y)
(assert (< (f y) 3.0))

(check-sat)
; expected: unsat
(exit)
