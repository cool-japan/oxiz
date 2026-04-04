; Test: Skolemization test -- existential inside universal
; Expected: sat
; Pattern: forall x. exists y. g(x) = f(y) (Skolem function maps x to witness y)

(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

; For every x, there exists y such that g(x) = f(y)
; After Skolemization: there exists a Skolem function sk such that g(x) = f(sk(x))
(assert (forall ((x Int))
  (exists ((y Int)) (= (g x) (f y)))))

; Specific values
(assert (= (g 0) 10))
(assert (= (g 1) 20))
(assert (= (f 5) 10))
(assert (= (f 7) 20))

; This is satisfiable: sk(0) = 5, sk(1) = 7 witnesses the existentials
(check-sat)
(exit)
