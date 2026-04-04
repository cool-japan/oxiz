; Test: No integer satisfies x^2 + 1 = 0
;; expected: unsat
; Pattern: x^2 >= 0 for all integers, so x^2 + 1 >= 1 > 0

(set-logic QF_NIA)

(declare-const x Int)

; x^2 + 1 = 0 is impossible over integers
(assert (= (+ (* x x) 1) 0))

(check-sat)
(exit)
