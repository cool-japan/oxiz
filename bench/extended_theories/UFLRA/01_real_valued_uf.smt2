; Test: UF with real-valued functions
;; expected: sat
; Pattern: f: Real -> Real with ground constraints

(set-logic UFLRA)
(declare-fun f (Real) Real)
(declare-const x Real)
(declare-const y Real)

; f preserves sign on bounded domain
(assert (forall ((z Real))
  (=> (and (>= z 0.0) (<= z 10.0))
      (>= (f z) 0.0))))

; Ground values
(assert (= x 2.5))
(assert (= y 7.3))

; f(x) + f(y) > 0
(assert (> (+ (f x) (f y)) 0.0))

; Specific value
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 1.5))

(check-sat)
(exit)
