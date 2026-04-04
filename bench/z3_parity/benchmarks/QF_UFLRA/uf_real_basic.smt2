; Basic UF + real arithmetic
; Expected: sat
; Tests uninterpreted function over reals

(set-logic QF_UFLRA)
(declare-fun f (Real) Real)
(declare-fun x () Real)
(declare-fun y () Real)

; f(x) > f(y)
(assert (> (f x) (f y)))

; x > y (consistent with a monotone interpretation of f)
(assert (> x y))

; Bounds
(assert (>= x 0.0))
(assert (<= x 10.0))
(assert (>= y 0.0))

(check-sat)
; expected: sat
(exit)
