; Test: Array initialization with quantified constraint
; Expected: sat
; Pattern: forall i. 0 <= i < n => a[i] = 0

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const n Int)

; n = 5
(assert (= n 5))

; All elements in [0, n) are initialized to 0
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i n))
      (= (select a i) 0))))

; Verify specific elements
(assert (= (select a 0) 0))
(assert (= (select a 4) 0))

(check-sat)
(exit)
