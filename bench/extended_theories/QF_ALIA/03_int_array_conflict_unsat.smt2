; Test: Integer array conflicting constraints
;; expected: unsat
; Pattern: Array element must be both > 10 and < 5

(set-logic QF_ALIA)
(declare-const a (Array Int Int))

; a[0] = 42
(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 42)))

; a1[0] > 10 (true since 42 > 10)
(assert (> (select a1 0) 10))

; But also require a1[0] < 5 => contradiction with store
; select(store(a, 0, 42), 0) = 42, and 42 < 5 is false
(assert (< (select a1 0) 5))

(check-sat)
(exit)
