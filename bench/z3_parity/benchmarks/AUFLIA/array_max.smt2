; Test: Array maximum exists
; Expected: sat
; Pattern: exists m. forall i. 0 <= i < n => a[i] <= m

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const n Int)
(declare-const m Int)

(assert (= n 5))

; m is an upper bound on all array elements in [0, n)
(assert (forall ((i Int))
  (=> (and (>= i 0) (< i n))
      (<= (select a i) m))))

; m is achieved by some element (m is the actual maximum)
(assert (exists ((j Int))
  (and (>= j 0) (< j n) (= (select a j) m))))

; Concrete values
(assert (= (select a 0) 3))
(assert (= (select a 1) 7))
(assert (= (select a 2) 1))
(assert (= (select a 3) 9))
(assert (= (select a 4) 5))

; Maximum should be 9
(assert (= m 9))

(check-sat)
(exit)
