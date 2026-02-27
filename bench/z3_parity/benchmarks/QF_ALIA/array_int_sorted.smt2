; Sorted array constraint
; Expected: sat
; Tests array elements in sorted order with arithmetic

(set-logic QF_ALIA)
(declare-fun a () (Array Int Int))

; Elements are sorted: a[0] <= a[1] <= a[2] <= a[3]
(assert (<= (select a 0) (select a 1)))
(assert (<= (select a 1) (select a 2)))
(assert (<= (select a 2) (select a 3)))

; First element is 1, last is 10
(assert (= (select a 0) 1))
(assert (= (select a 3) 10))

; Middle elements are distinct
(assert (not (= (select a 1) (select a 2))))

(check-sat)
; expected: sat
(exit)
