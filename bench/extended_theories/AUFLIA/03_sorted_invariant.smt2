; Test: Sorted array invariant with quantifiers
;; expected: sat
; Pattern: forall i j. 0<=i<j<=n => a[i] <= a[j]

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const n Int)

(assert (= n 4))

; Array is sorted from index 0 to n
(assert (forall ((i Int) (j Int))
  (=> (and (>= i 0) (>= j 0) (<= i n) (<= j n) (< i j))
      (<= (select a i) (select a j)))))

; Concrete sorted values
(assert (= (select a 0) 1))
(assert (= (select a 1) 3))
(assert (= (select a 2) 5))
(assert (= (select a 3) 7))
(assert (= (select a 4) 9))

(check-sat)
(exit)
