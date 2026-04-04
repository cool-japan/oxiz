; Test: Triple-nested quantifiers
; Expected: sat
; Pattern: forall x. exists y. forall z. (z >= y => f(x,z) >= 0)

(set-logic UFLIA)
(declare-fun f (Int Int) Int)

; For every x, there exists a threshold y such that for all z >= y, f(x,z) >= 0
(assert (forall ((x Int))
  (exists ((y Int))
    (forall ((z Int))
      (=> (>= z y) (>= (f x z) 0))))))

; Some specific values consistent with the property
(assert (= (f 0 0) (- 1)))
(assert (= (f 0 5) 10))
(assert (= (f 0 6) 12))
(assert (= (f 1 3) 7))

(check-sat)
(exit)
