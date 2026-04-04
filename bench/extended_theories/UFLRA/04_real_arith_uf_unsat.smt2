; Test: Real arithmetic contradiction with UF
;; expected: unsat
; Pattern: f is monotone but we require f(1) > f(2) with 1 < 2

(set-logic UFLRA)
(declare-fun f (Real) Real)

; Monotonicity on bounded domain
(assert (forall ((x Real) (y Real))
  (=> (and (>= x 0.0) (<= x 10.0) (>= y 0.0) (<= y 10.0) (<= x y))
      (<= (f x) (f y)))))

; Contradictory requirement: f(1) > f(2) where 1 <= 2
(assert (> (f 1.0) (f 2.0)))

(check-sat)
(exit)
