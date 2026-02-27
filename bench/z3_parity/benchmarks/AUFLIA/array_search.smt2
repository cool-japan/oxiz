; Test: Array search -- existential quantifier over array index
; Expected: sat
; Pattern: exists i. a[i] = v

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const v Int)

; Search value
(assert (= v 42))

; There exists an index where the array contains v
(assert (exists ((i Int))
  (and (>= i 0) (<= i 9) (= (select a i) v))))

; Some concrete array values
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select a 5) 42))
(assert (= (select a 9) 90))

(check-sat)
(exit)
