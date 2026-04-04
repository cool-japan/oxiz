; Test: Quantified linear arithmetic bounds
;; expected: sat
; Pattern: forall x in [0,n]. f(x) >= 0 and f(x) <= x*x

(set-logic UFLIA)
(declare-fun f (Int) Int)

; f is non-negative for non-negative inputs (bounded domain)
(assert (forall ((x Int))
  (=> (and (>= x 0) (<= x 10))
      (>= (f x) 0))))

; f is bounded above by x*x
(assert (forall ((x Int))
  (=> (and (>= x 0) (<= x 10))
      (<= (f x) (* x x)))))

; Specific ground instances
(assert (= (f 0) 0))
(assert (= (f 1) 1))
(assert (= (f 2) 3))
(assert (= (f 3) 5))

; f(2) < f(3)
(assert (< (f 2) (f 3)))

(check-sat)
(exit)
