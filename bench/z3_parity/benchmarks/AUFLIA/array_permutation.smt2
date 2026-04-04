; Test: Array is a permutation of {1, 2, 3}
; Expected: sat
; Pattern: Array contains exactly the elements {1, 2, 3} in some order

(set-logic AUFLIA)
(declare-const a (Array Int Int))

; Array of size 3 is a permutation of {1, 2, 3}
; Each position has a value in {1, 2, 3}
(assert (and (>= (select a 0) 1) (<= (select a 0) 3)))
(assert (and (>= (select a 1) 1) (<= (select a 1) 3)))
(assert (and (>= (select a 2) 1) (<= (select a 2) 3)))

; All distinct (injectivity ensures permutation for same-size sets)
(assert (not (= (select a 0) (select a 1))))
(assert (not (= (select a 1) (select a 2))))
(assert (not (= (select a 0) (select a 2))))

; Each value appears: surjectivity for bounded domain
(assert (exists ((i Int))
  (and (>= i 0) (<= i 2) (= (select a i) 1))))
(assert (exists ((i Int))
  (and (>= i 0) (<= i 2) (= (select a i) 2))))
(assert (exists ((i Int))
  (and (>= i 0) (<= i 2) (= (select a i) 3))))

(check-sat)
(exit)
