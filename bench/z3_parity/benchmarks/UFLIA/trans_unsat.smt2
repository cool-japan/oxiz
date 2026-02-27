; Test: Transitivity with contradictory constraints
; Expected: unsat
; Pattern: Transitivity forces f(a) <= f(c) but we assert f(a) > f(c)

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Transitivity axiom
(assert (forall ((x Int) (y Int) (z Int))
  (=> (and (<= (f x) (f y)) (<= (f y) (f z)))
      (<= (f x) (f z)))))

; Chain: f(1) <= f(2) <= f(3)
(assert (<= (f 1) (f 2)))
(assert (<= (f 2) (f 3)))

; Contradiction: f(1) > f(3), which violates transitivity
(assert (> (f 1) (f 3)))

(check-sat)
(exit)
