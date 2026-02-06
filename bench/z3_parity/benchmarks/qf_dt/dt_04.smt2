; QF_DT Benchmark: Binary Tree Construction with node/leaf
; Expected: SAT
; Description: Test binary tree construction and basic properties

(set-logic ALL)

; Define a binary tree of integers
(declare-datatypes ((IntTree 0)) (
  ((leaf (value Int))
   (node (left IntTree) (right IntTree)))
))

; Declare tree variables
(declare-const t1 IntTree)
(declare-const t2 IntTree)
(declare-const t3 IntTree)

; t1 is a leaf with value 5
(assert (= t1 (leaf 5)))

; t2 is a leaf with value 10
(assert (= t2 (leaf 10)))

; t3 is a node with t1 as left child and t2 as right child
(assert (= t3 (node t1 t2)))

; Check that t3 is indeed a node
(assert ((_ is node) t3))

; Check that left child of t3 is a leaf
(assert ((_ is leaf) (left t3)))

; Check that the value of the left leaf is 5
(assert (= (value (left t3)) 5))

; Check that the value of the right leaf is 10
(assert (= (value (right t3)) 10))

(check-sat)
(exit)
