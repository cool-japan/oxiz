; Test: Monotonicity that forces contradiction on function values
; Expected: unsat
; Pattern: Monotone f with f(3) > f(7), which contradicts 3 <= 7

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Monotonicity axiom
(assert (forall ((x Int) (y Int))
  (=> (<= x y) (<= (f x) (f y)))))

; f(3) = 50 and f(7) = 20, but 3 <= 7 so we need f(3) <= f(7)
(assert (= (f 3) 50))
(assert (= (f 7) 20))

(check-sat)
(exit)
