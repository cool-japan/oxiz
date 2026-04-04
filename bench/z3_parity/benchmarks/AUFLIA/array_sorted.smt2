; Test: Sorted array
; Expected: sat
; Pattern: forall i j. 0 <= i <= j < n => a[i] <= a[j]

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const n Int)

(assert (= n 4))

; Array is sorted in non-decreasing order
(assert (forall ((i Int) (j Int))
  (=> (and (>= i 0) (<= i j) (< j n))
      (<= (select a i) (select a j)))))

; Specific sorted values
(assert (= (select a 0) 1))
(assert (= (select a 1) 3))
(assert (= (select a 2) 5))
(assert (= (select a 3) 7))

(check-sat)
(exit)
