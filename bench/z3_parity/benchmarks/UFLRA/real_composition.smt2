; Test: Commuting functions under composition
; Expected: sat
; Pattern: forall x. f(g(x)) = g(f(x)) (bounded domain)

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-fun g (Real) Real)

; f and g commute on bounded domain
(assert (forall ((x Real))
  (=> (and (>= x 0.0) (<= x 5.0))
      (= (f (g x)) (g (f x))))))

; Specific function values consistent with commutativity
; Use f = identity, g = identity (trivially commuting)
(assert (= (f 0.0) 0.0))
(assert (= (g 0.0) 0.0))
(assert (= (f 1.0) 1.0))
(assert (= (g 1.0) 1.0))
(assert (= (f 2.0) 2.0))
(assert (= (g 2.0) 2.0))

(check-sat)
(exit)
