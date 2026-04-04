; Test: Contradictory quantified array constraints
; Expected: unsat
; Pattern: All elements are 0, but we need an element that is 1

(set-logic AUFLIA)
(declare-const a (Array Int Int))

; All elements in [0, 10) are 0
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i 10))
      (= (select a i) 0))))

; But element at index 5 must be 1
(assert (= (select a 5) 1))

(check-sat)
(exit)
