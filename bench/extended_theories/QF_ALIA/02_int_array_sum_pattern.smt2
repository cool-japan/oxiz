; Test: Integer array with arithmetic on elements
;; expected: sat
; Pattern: Array elements satisfy arithmetic constraints

(set-logic QF_ALIA)
(declare-const a (Array Int Int))

; a[0] + a[1] + a[2] = 100
(assert (= (+ (select a 0) (+ (select a 1) (select a 2))) 100))

; All elements non-negative
(assert (>= (select a 0) 0))
(assert (>= (select a 1) 0))
(assert (>= (select a 2) 0))

; a[0] > a[1] > a[2]
(assert (> (select a 0) (select a 1)))
(assert (> (select a 1) (select a 2)))

; a[2] >= 10
(assert (>= (select a 2) 10))

(check-sat)
(exit)
