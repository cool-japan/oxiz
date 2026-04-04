; Test: Transitivity of a monotone function
; Expected: sat
; Pattern: forall x y z. f(x) <= f(y) /\ f(y) <= f(z) => f(x) <= f(z) with specific values

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Transitivity axiom for f
(assert (forall ((x Int) (y Int) (z Int))
  (=> (and (<= (f x) (f y)) (<= (f y) (f z)))
      (<= (f x) (f z)))))

; Specific function values
(assert (= (f 0) 1))
(assert (= (f 5) 10))
(assert (= (f 10) 20))

; This should be satisfiable: f(0) <= f(5) <= f(10) and transitivity holds
(assert (<= (f 0) (f 5)))
(assert (<= (f 5) (f 10)))

(check-sat)
(exit)
