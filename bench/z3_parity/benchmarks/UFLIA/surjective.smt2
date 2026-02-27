; Test: Surjectivity -- for specific target values, a preimage exists
; Expected: sat
; Pattern: exists y. f(y) = x for specific x values

(set-logic UFLIA)
(declare-fun f (Int) Int)

; For each target value, there exists a preimage
(assert (exists ((y1 Int)) (= (f y1) 0)))
(assert (exists ((y2 Int)) (= (f y2) 1)))
(assert (exists ((y3 Int)) (= (f y3) 2)))

; Additional constraint: f is bounded
(assert (forall ((x Int))
  (=> (and (>= x 0) (<= x 10))
      (and (>= (f x) 0) (<= (f x) 5)))))

(check-sat)
(exit)
