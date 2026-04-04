; Test: Array-like function with non-negative elements and sum constraint
; Expected: sat
; Pattern: forall i. 0 <= i /\ i < n => a(i) >= 0 with sum constraint

(set-logic UFLIA)
(declare-fun a (Int) Int)
(declare-const n Int)

; All elements in range [0, n) are non-negative
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i n))
      (>= (a i) 0))))

; n = 3
(assert (= n 3))

; Sum constraint: a(0) + a(1) + a(2) = 10
(assert (= (+ (a 0) (+ (a 1) (a 2))) 10))

; Each element is at most 5
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i n))
      (<= (a i) 5))))

(check-sat)
(exit)
