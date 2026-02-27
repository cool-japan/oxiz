; Congruence closure with arithmetic
; Expected: unsat
; Tests that f(a) = f(b) when a = b, combined with arithmetic

(set-logic QF_UFLIA)
(declare-fun f (Int) Int)
(declare-fun a () Int)
(declare-fun b () Int)

; a = b
(assert (= a b))

; f(a) != f(b) -- impossible by congruence
(assert (not (= (f a) (f b))))

(check-sat)
; expected: unsat
(exit)
