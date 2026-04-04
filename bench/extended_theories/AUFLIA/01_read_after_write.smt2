; Test: Array read-after-write (store then select)
;; expected: sat
; Pattern: select(store(a, i, v), i) = v

(set-logic AUFLIA)
(declare-const a (Array Int Int))

; Store value 42 at index 0
(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 42)))

; Read back: should be 42
(assert (= (select a1 0) 42))

; Store value 99 at index 1
(declare-const a2 (Array Int Int))
(assert (= a2 (store a1 1 99)))

; Read back both
(assert (= (select a2 0) 42))
(assert (= (select a2 1) 99))

; Store at index 2, verify index 0 and 1 unchanged
(declare-const a3 (Array Int Int))
(assert (= a3 (store a2 2 7)))
(assert (= (select a3 0) 42))
(assert (= (select a3 1) 99))
(assert (= (select a3 2) 7))

(check-sat)
(exit)
