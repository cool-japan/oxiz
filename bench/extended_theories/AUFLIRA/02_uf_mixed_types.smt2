; Test: Uninterpreted functions over mixed types with quantified axioms
;; expected: sat
; Pattern: f: Int -> Real with quantified axioms about monotonicity

(set-logic AUFLIRA)

; UF from Int to Real
(declare-fun f (Int) Real)

; f is monotone: forall x y, x < y => f(x) < f(y)
(assert (forall ((x Int) (y Int))
  (=> (< x y) (< (f x) (f y)))))

; Specific values
(assert (= (f 0) 0.0))
(assert (= (f 10) 5.5))

; Since 0 < 10, we should have f(0) < f(10), which is 0.0 < 5.5: consistent
; Also check an intermediate value
(declare-const v Real)
(assert (= v (f 5)))

; f(0) < f(5) < f(10)
(assert (> v 0.0))
(assert (< v 5.5))

(check-sat)
(exit)
