; Test: Monotonicity axiom
;; expected: sat
; Pattern: x <= y => f(x) <= f(y)

(set-logic UFLRA)
(declare-fun f (Real) Real)

; Monotonicity on bounded domain
(assert (forall ((x Real) (y Real))
  (=> (and (>= x 0.0) (<= x 100.0) (>= y 0.0) (<= y 100.0) (<= x y))
      (<= (f x) (f y)))))

; Ground instances demonstrating monotonicity
(assert (= (f 0.0) 0.0))
(assert (= (f 1.0) 2.0))
(assert (= (f 2.0) 3.5))
(assert (= (f 10.0) 50.0))

; Derived: f(1) <= f(2) and f(0) <= f(10)
(declare-const r1 Real)
(declare-const r2 Real)
(assert (= r1 (f 1.0)))
(assert (= r2 (f 2.0)))
(assert (<= r1 r2))

(check-sat)
(exit)
