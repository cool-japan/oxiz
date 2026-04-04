; Test: Injectivity axiom for uninterpreted function
;; expected: sat
; Pattern: f(x) = f(y) => x = y, with consistent ground instances

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Injectivity axiom (bounded domain)
(assert (forall ((x Int) (y Int))
  (=> (and (>= x 0) (<= x 5) (>= y 0) (<= y 5)
           (= (f x) (f y)))
      (= x y))))

; Ground instances consistent with injectivity
(assert (= (f 0) 100))
(assert (= (f 1) 200))
(assert (= (f 2) 300))
(assert (= (f 3) 400))

; Query: distinct images
(assert (not (= (f 0) (f 1))))

(check-sat)
(exit)
