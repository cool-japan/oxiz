; Test: Contradictory UF axioms
;; expected: unsat
; Pattern: f is both injective and constant => contradiction with distinct inputs

(set-logic UFLIA)
(declare-fun f (Int) Int)

; f is constant (maps everything to 0) on bounded domain
(assert (forall ((x Int))
  (=> (and (>= x 0) (<= x 10))
      (= (f x) 0))))

; But we also require distinct outputs for distinct inputs
(assert (not (= (f 1) (f 2))))

; Domain constraints
(assert (>= 1 0))
(assert (<= 1 10))
(assert (>= 2 0))
(assert (<= 2 10))

(check-sat)
(exit)
