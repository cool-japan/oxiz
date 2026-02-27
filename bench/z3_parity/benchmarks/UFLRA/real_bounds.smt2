; Test: Bounded real function
; Expected: sat
; Pattern: forall x. 0 <= f(x) <= 1 (f maps to unit interval)

(set-logic UFLRA)
(declare-fun f (Real) Real)

; f is bounded in [0, 1]
(assert (forall ((x Real))
  (and (>= (f x) 0.0) (<= (f x) 1.0))))

; Specific values within bounds
(assert (= (f 0.0) 0.5))
(assert (= (f 1.0) 0.75))
(assert (= (f (- 1.0)) 0.25))

; Sum of three outputs is at most 3
(assert (<= (+ (f 0.0) (+ (f 1.0) (f (- 1.0)))) 3.0))

(check-sat)
(exit)
