; Test: Conflicting array state
;; expected: unsat
; Pattern: store at index i then require different value at same index

(set-logic AUFLIA)
(declare-const a (Array Int Int))

; Store 42 at index 0
(declare-const a1 (Array Int Int))
(assert (= a1 (store a 0 42)))

; select(store(a, 0, 42), 0) must be 42
; But we require it to be 99
(assert (= (select a1 0) 99))

(check-sat)
(exit)
