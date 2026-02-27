; Uninterpreted functions with integer arrays
; Expected: sat
; Tests UF applied to array elements

(set-logic QF_AUFLIA)
(declare-fun a () (Array Int Int))
(declare-fun f (Int) Int)

; f applied to array elements
(assert (= (f (select a 0)) 10))
(assert (= (f (select a 1)) 20))

; Array elements are different
(assert (not (= (select a 0) (select a 1))))

; f is well-defined (consistent with the above)
(assert (>= (select a 0) 0))
(assert (>= (select a 1) 0))

(check-sat)
; expected: sat
(exit)
