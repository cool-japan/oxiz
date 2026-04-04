; Test: Injectivity of an uninterpreted function
; Expected: sat
; Pattern: forall x y. f(x) = f(y) => x = y with consistent assignments

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Injectivity axiom (bounded to avoid timeout)
(assert (forall ((x Int) (y Int))
  (=> (and (>= x 0) (<= x 10) (>= y 0) (<= y 10) (= (f x) (f y)))
      (= x y))))

; Specific distinct function values (consistent with injectivity)
(assert (= (f 1) 10))
(assert (= (f 2) 20))
(assert (= (f 3) 30))

(check-sat)
(exit)
