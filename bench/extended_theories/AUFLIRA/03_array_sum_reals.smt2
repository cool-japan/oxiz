; Test: Array sum pattern with real-valued elements
;; expected: sat
; Pattern: Sum of real-valued array elements over integer indices

(set-logic AUFLIRA)

; Array of reals indexed by integers
(declare-const arr (Array Int Real))

; Define specific elements
(declare-const arr1 (Array Int Real))
(assert (= arr1 (store arr 0 1.5)))
(declare-const arr2 (Array Int Real))
(assert (= arr2 (store arr1 1 2.5)))
(declare-const arr3 (Array Int Real))
(assert (= arr3 (store arr2 2 3.0)))

; Compute partial sums
(declare-const s0 Real)
(declare-const s1 Real)
(declare-const s2 Real)
(assert (= s0 (select arr3 0)))
(assert (= s1 (+ s0 (select arr3 1))))
(assert (= s2 (+ s1 (select arr3 2))))

; Total sum should be 7.0
(assert (= s2 7.0))

; Average should be 7.0/3
(declare-const avg Real)
(assert (= (* avg 3.0) s2))

(check-sat)
(exit)
