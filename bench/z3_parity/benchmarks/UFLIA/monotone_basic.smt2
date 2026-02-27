; Test: Monotonicity of an uninterpreted function with concrete bounds
; Expected: sat
; Pattern: forall x y. x <= y => f(x) <= f(y) with concrete instantiations

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Monotonicity axiom
(assert (forall ((x Int) (y Int))
  (=> (<= x y) (<= (f x) (f y)))))

; Concrete bounds
(assert (= (f 0) 0))
(assert (= (f 10) 100))

; Additional constraints consistent with monotonicity
(assert (>= (f 5) 30))
(assert (<= (f 5) 70))

(check-sat)
(exit)
