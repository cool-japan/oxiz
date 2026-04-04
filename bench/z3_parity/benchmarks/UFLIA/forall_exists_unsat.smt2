; Test: Contradictory forall-exists alternation
; Expected: unsat
; Pattern: All outputs are non-positive, but we need a positive witness for every input

(set-logic UFLIA)
(declare-fun f (Int Int) Int)

; For every x, there exists y such that f(x, y) > 0
(assert (forall ((x Int))
  (exists ((y Int)) (> (f x y) 0))))

; But all function values are <= 0
(assert (forall ((x Int) (y Int)) (<= (f x y) 0)))

(check-sat)
(exit)
