; Test: Array store preserves other elements
; Expected: sat
; Pattern: forall i. i != k => store(a, k, v)[i] = a[i]

(set-logic AUFLIA)
(declare-const a (Array Int Int))
(declare-const b (Array Int Int))
(declare-const k Int)
(declare-const v Int)

; b = store(a, k, v)
(assert (= b (store a k v)))

; Update at position k = 3 with value v = 99
(assert (= k 3))
(assert (= v 99))

; The store axiom guarantees: for all i != k, b[i] = a[i]
(assert (forall ((i Int))
  (=> (not (= i k))
      (= (select b i) (select a i)))))

; And b[k] = v
(assert (= (select b k) v))

; Some concrete values for the original array
(assert (= (select a 0) 10))
(assert (= (select a 1) 20))
(assert (= (select a 3) 30))

; After store, a[3] was 30 but b[3] = 99
(assert (= (select b 3) 99))
(assert (= (select b 0) 10))
(assert (= (select b 1) 20))

(check-sat)
(exit)
