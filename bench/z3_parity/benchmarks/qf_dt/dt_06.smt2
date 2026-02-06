; QF_DT Benchmark: Tree Constraints with Contradictions
; Expected: UNSAT
; Description: Test tree with contradictory selector constraints

(set-logic ALL)

; Define a binary tree of integers
(declare-datatypes ((IntTree 0)) (
  ((leaf (value Int))
   (node (left IntTree) (right IntTree)))
))

; Declare tree variable
(declare-const t IntTree)

; t is a leaf
(assert ((_ is leaf) t))

; The value of the leaf is 100
(assert (= (value t) 100))

; But we also try to access it as a node - contradiction!
; This should be UNSAT because a leaf has no left/right children
(assert ((_ is node) t))

; Try to assert something about its left child
(assert ((_ is leaf) (left t)))

(check-sat)
(exit)
