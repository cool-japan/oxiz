; Unsatisfiable UF + LIA problem
; Expected: unsat
; Tests arithmetic bounds combined with UF injectivity

(set-logic QF_UFLIA)
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)
(declare-fun z () Int)

; f is injective (contrapositive: distinct inputs give distinct outputs)
(assert (=> (= (f x) (f y)) (= x y)))
(assert (=> (= (f x) (f z)) (= x z)))
(assert (=> (= (f y) (f z)) (= y z)))

; All three have the same f-value
(assert (= (f x) (f y)))
(assert (= (f y) (f z)))

; But they're pairwise distinct -- contradiction with injectivity
(assert (not (= x y)))
(assert (not (= y z)))

(check-sat)
; expected: unsat
(exit)
