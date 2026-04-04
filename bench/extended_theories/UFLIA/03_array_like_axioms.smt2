; Test: Array-like axioms using select/store patterns with UF
;; expected: sat
; Pattern: Read-over-write axiom encoded with UF functions

(set-logic UFLIA)

; Encode arrays as functions: sel(a, i) and sto(a, i, v)
(declare-fun sel (Int Int) Int)
(declare-fun sto (Int Int Int) Int)

; Read-after-write same index: sel(sto(a, i, v), i) = v
; We model this with ground instances
(declare-const arr Int)

; Store value 42 at index 0 in arr, get new array id
(declare-const arr1 Int)
(assert (= (sto arr 0 42) arr1))
(assert (= (sel arr1 0) 42))

; Store value 99 at index 1 in arr1, get arr2
(declare-const arr2 Int)
(assert (= (sto arr1 1 99) arr2))
(assert (= (sel arr2 1) 99))

; Read-after-write different index: sel(sto(a, i, v), j) = sel(a, j) when i != j
(assert (= (sel arr2 0) 42))

; Original array has some default
(assert (= (sel arr 5) 0))

; Non-stored index in arr1 should still be 0
(assert (= (sel arr1 5) 0))

(check-sat)
(exit)
