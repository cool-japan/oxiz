; Test: Quantified non-negative array property
;; expected: sat
; Pattern: forall i: select(a, i) >= 0.0, with concrete checks

(set-logic AUFLIRA)

(declare-const a (Array Int Real))

; All elements are non-negative
(assert (forall ((i Int)) (>= (select a i) 0.0)))

; Check specific indices
(declare-const v0 Real)
(declare-const v1 Real)
(declare-const v42 Real)
(assert (= v0 (select a 0)))
(assert (= v1 (select a 1)))
(assert (= v42 (select a 42)))

; These should all be satisfiable given the universal property
(assert (>= v0 0.0))
(assert (>= v1 0.0))
(assert (>= v42 0.0))

; Sum of non-negative values is non-negative
(declare-const total Real)
(assert (= total (+ v0 (+ v1 v42))))
(assert (>= total 0.0))

(check-sat)
(exit)
