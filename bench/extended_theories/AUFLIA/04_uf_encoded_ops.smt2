; Test: Array with UF-encoded operations
;; expected: sat
; Pattern: UF function transforms array values, result stored back

(set-logic AUFLIA)
(declare-fun transform (Int) Int)
(declare-const src (Array Int Int))
(declare-const dst (Array Int Int))

; Transform doubles each value (ground instances)
(assert (= (transform 10) 20))
(assert (= (transform 20) 40))
(assert (= (transform 30) 60))

; Source array values
(assert (= (select src 0) 10))
(assert (= (select src 1) 20))
(assert (= (select src 2) 30))

; Destination array is transform applied to source (ground instances)
(assert (= (select dst 0) (transform (select src 0))))
(assert (= (select dst 1) (transform (select src 1))))
(assert (= (select dst 2) (transform (select src 2))))

; Verify destination values
(assert (= (select dst 0) 20))
(assert (= (select dst 1) 40))
(assert (= (select dst 2) 60))

(check-sat)
(exit)
