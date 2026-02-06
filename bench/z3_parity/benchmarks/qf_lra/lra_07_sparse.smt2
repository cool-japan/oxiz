; Test: Sparse constraint matrix
; Expected: sat
; Pattern: Few constraints, many variables

(set-logic QF_LRA)
(declare-const a Real)
(declare-const b Real)
(declare-const c Real)
(declare-const d Real)
(declare-const e Real)

(assert (= (+ a b) 10.0))
(assert (= (+ c d) 20.0))
(assert (>= e 5.0))
(assert (>= a 0.0))
(assert (>= b 0.0))
(assert (>= c 0.0))
(assert (>= d 0.0))

(check-sat)
