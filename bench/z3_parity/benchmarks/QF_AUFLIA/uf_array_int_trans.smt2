; Transitivity with UF + arrays
; Expected: unsat
; Tests congruence closure with array reads

(set-logic QF_AUFLIA)
(declare-fun a () (Array Int Int))
(declare-fun f (Int) Int)
(declare-fun x () Int)
(declare-fun y () Int)

; x = y (so a[x] = a[y] by congruence)
(assert (= x y))

; f(a[x]) != f(a[y]) -- impossible since x = y implies a[x] = a[y]
; and congruence gives f(a[x]) = f(a[y])
(assert (not (= (f (select a x)) (f (select a y)))))

(check-sat)
; expected: unsat
(exit)
