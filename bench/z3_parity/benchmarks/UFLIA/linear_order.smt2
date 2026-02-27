; Test: Subadditive function with bounds
; Expected: sat
; Pattern: f(x) + f(y) >= f(x + y) for bounded inputs

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Subadditivity for small inputs only
(assert (forall ((x Int) (y Int))
  (=> (and (>= x 0) (<= x 5) (>= y 0) (<= y 5))
      (>= (+ (f x) (f y)) (f (+ x y))))))

; Specific values consistent with subadditivity
(assert (= (f 0) 0))
(assert (= (f 1) 3))
(assert (= (f 2) 5))
; f(1) + f(1) = 6 >= f(2) = 5, OK
(assert (= (f 3) 7))
; f(1) + f(2) = 8 >= f(3) = 7, OK

(check-sat)
(exit)
