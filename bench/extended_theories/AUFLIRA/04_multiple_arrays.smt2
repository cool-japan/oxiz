; Test: Operations across integer and real arrays
;; expected: sat
; Pattern: Integer array and real array with cross-array constraints

(set-logic AUFLIRA)

; Integer array: counts
(declare-const counts (Array Int Int))
(declare-const counts1 (Array Int Int))
(assert (= counts1 (store counts 0 3)))
(declare-const counts2 (Array Int Int))
(assert (= counts2 (store counts1 1 5)))
(declare-const counts3 (Array Int Int))
(assert (= counts3 (store counts2 2 2)))

; Real array: weights
(declare-const weights (Array Int Real))
(declare-const weights1 (Array Int Real))
(assert (= weights1 (store weights 0 1.5)))
(declare-const weights2 (Array Int Real))
(assert (= weights2 (store weights1 1 2.0)))
(declare-const weights3 (Array Int Real))
(assert (= weights3 (store weights2 2 0.5)))

; Weighted sum: counts[i] * weights[i] for i = 0, 1, 2
; 3*1.5 + 5*2.0 + 2*0.5 = 4.5 + 10.0 + 1.0 = 15.5
(declare-const ws Real)
(assert (= ws (+ (+ (* (to_real (select counts3 0)) (select weights3 0))
                    (* (to_real (select counts3 1)) (select weights3 1)))
                 (* (to_real (select counts3 2)) (select weights3 2)))))
(assert (= ws 15.5))

(check-sat)
(exit)
