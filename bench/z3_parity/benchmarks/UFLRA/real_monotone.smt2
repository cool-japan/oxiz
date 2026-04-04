; Test: Monotonicity with real arithmetic
; Expected: sat
; Pattern: forall x y. x <= y => f(x) <= f(y) over reals

(set-logic UFLRA)
(declare-fun f (Real) Real)

; Monotonicity
(assert (forall ((x Real) (y Real))
  (=> (<= x y) (<= (f x) (f y)))))

; Concrete values
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 2.5))
(assert (= (f 3.0) 7.0))

; Consistent query
(assert (<= (f 0.5) (f 2.0)))

(check-sat)
(exit)
