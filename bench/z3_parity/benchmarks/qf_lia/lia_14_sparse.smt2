; Test: Sparse constraint matrix
; Expected: sat
; Pattern: Many variables, few constraints

(set-logic QF_LIA)
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)
(declare-const d Int)
(declare-const e Int)
(declare-const f Int)

(assert (= (+ a b) 10))
(assert (= (+ c d) 20))
(assert (= (+ e f) 30))
(assert (>= a 0))
(assert (>= b 0))
(assert (>= c 0))
(assert (>= d 0))
(assert (>= e 0))
(assert (>= f 0))

(check-sat)
