; Test: Array partitioned by value around a pivot
; Expected: sat
; Pattern: Elements below pivot index are negative, elements at or above are non-negative

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const pivot Int)
(declare-const n Int)

(assert (= n 6))
(assert (= pivot 3))

; Elements below pivot are negative
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i pivot))
      (< (select a i) 0))))

; Elements at pivot or above are non-negative
(assert (forall ((i Int))
  (=> (and (>= i pivot) (< i n))
      (>= (select a i) 0))))

; Concrete values
(assert (= (select a 0) (- 5)))
(assert (= (select a 1) (- 3)))
(assert (= (select a 2) (- 1)))
(assert (= (select a 3) 0))
(assert (= (select a 4) 2))
(assert (= (select a 5) 7))

(check-sat)
(exit)
