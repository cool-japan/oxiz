; Test: Forall-exists alternation
; Expected: sat
; Pattern: forall x. exists y. f(x, y) > 0

(set-logic UFLIA)
(declare-fun f (Int Int) Int)

; For every x, there exists a y such that f(x, y) > 0
(assert (forall ((x Int))
  (exists ((y Int)) (> (f x y) 0))))

; Some specific function values
(assert (= (f 0 0) 1))
(assert (= (f 1 1) 2))
(assert (= (f 2 0) 3))

(check-sat)
(exit)
