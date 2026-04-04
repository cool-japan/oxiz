; Unsatisfiable combined UF + Array + LIA problem
; Expected: unsat
; Tests interaction of UF congruence with array axioms and arithmetic

(set-logic QF_AUFLIA)
(declare-fun a () (Array Int Int))
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)

; f is injective on relevant domain
(assert (=> (= (f x) (f y)) (= x y)))

; a[f(x)] = 10 and a[f(y)] = 20
(assert (= (select a (f x)) 10))
(assert (= (select a (f y)) 20))

; But x = y, so f(x) = f(y), so a[f(x)] = a[f(y)]
; This means 10 = 20, which is impossible
(assert (= x y))

(check-sat)
; expected: unsat
(exit)
