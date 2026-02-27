; Absolute value via UF + LRA
; Expected: sat
; Models absolute value as UF with appropriate constraints

(set-logic QF_UFLRA)
(declare-fun abs_f (Real) Real)
(declare-fun x () Real)
(declare-fun y () Real)

; abs_f behaves like absolute value for x and y
(assert (=> (>= x 0.0) (= (abs_f x) x)))
(assert (=> (< x 0.0) (= (abs_f x) (- 0.0 x))))
(assert (=> (>= y 0.0) (= (abs_f y) y)))
(assert (=> (< y 0.0) (= (abs_f y) (- 0.0 y))))

; abs_f(x) = abs_f(y) but x != y
(assert (= (abs_f x) (abs_f y)))
(assert (not (= x y)))

; x is negative, y is positive (so |x| = |y| means y = -x)
(assert (< x 0.0))
(assert (> y 0.0))

(check-sat)
; expected: sat
(exit)
