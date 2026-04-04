; Test: Mixed integer/real array with store/select and arithmetic
;; expected: sat
; Pattern: Array<Int, Real> with store, select, and real arithmetic

(set-logic AUFLIRA)

; Declare an array from Int to Real
(declare-const a (Array Int Real))

; Store 3.14 at index 0
(declare-const a1 (Array Int Real))
(assert (= a1 (store a 0 3.14)))

; Store 2.71 at index 1
(declare-const a2 (Array Int Real))
(assert (= a2 (store a1 1 2.71)))

; Verify stored values
(assert (= (select a2 0) 3.14))
(assert (= (select a2 1) 2.71))

; Arithmetic on selected values: sum should be 5.85
(declare-const sum Real)
(assert (= sum (+ (select a2 0) (select a2 1))))
(assert (= sum 5.85))

; Store sum at index 2
(declare-const a3 (Array Int Real))
(assert (= a3 (store a2 2 sum)))
(assert (= (select a3 2) 5.85))

(check-sat)
(exit)
