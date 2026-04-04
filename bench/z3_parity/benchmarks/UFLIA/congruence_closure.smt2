; Test: Quantified congruence closure
; Expected: sat
; Pattern: Congruence axiom plus equality chains

(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-fun g (Int) Int)

; Congruence: equal inputs produce equal outputs
; (This is built-in for UF, but we test with explicit quantified form)
(assert (forall ((x Int) (y Int))
  (=> (= x y) (= (f x) (f y)))))

(assert (forall ((x Int) (y Int))
  (=> (= x y) (= (g x) (g y)))))

; Equality chain
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)

(assert (= a b))
(assert (= b c))

; By congruence: f(a) = f(b) = f(c) and g(a) = g(b) = g(c)
(assert (= (f a) 42))
(assert (= (g c) 100))

; These should follow from congruence
(assert (= (f c) 42))
(assert (= (g a) 100))

(check-sat)
(exit)
