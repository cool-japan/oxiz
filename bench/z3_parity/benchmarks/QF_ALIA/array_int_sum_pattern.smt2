; Array with arithmetic constraints on elements
; Expected: sat
; Tests combining array reads with arithmetic

(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))

; Elements at positions 0, 1, 2 sum to 60
(assert (= (+ (select a 0) (select a 1) (select a 2)) 60))

; Each element is between 10 and 30
(assert (>= (select a 0) 10))
(assert (<= (select a 0) 30))
(assert (>= (select a 1) 10))
(assert (<= (select a 1) 30))
(assert (>= (select a 2) 10))
(assert (<= (select a 2) 30))

(check-sat)
; expected: sat
(exit)
