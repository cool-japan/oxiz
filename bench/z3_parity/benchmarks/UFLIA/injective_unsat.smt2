; Test: Injectivity with pigeonhole-style contradiction
; Expected: unsat
; Pattern: Injective f maps 3 distinct inputs to only 2 possible outputs

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Injectivity axiom
(assert (forall ((x Int) (y Int))
  (=> (= (f x) (f y)) (= x y))))

; Three distinct inputs
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)
(assert (not (= a b)))
(assert (not (= b c)))
(assert (not (= a c)))

; Map all three into range {0, 1} -- pigeonhole forces collision
(assert (or (= (f a) 0) (= (f a) 1)))
(assert (or (= (f b) 0) (= (f b) 1)))
(assert (or (= (f c) 0) (= (f c) 1)))

(check-sat)
(exit)
