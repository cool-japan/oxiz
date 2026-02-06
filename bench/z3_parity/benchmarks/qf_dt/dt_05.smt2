; QF_DT Benchmark: Tree Selectors left/right
; Expected: SAT
; Description: Test tree left/right selectors with complex structure

(set-logic ALL)

; Define a binary tree of integers
(declare-datatypes ((IntTree 0)) (
  ((leaf (value Int))
   (node (left IntTree) (right IntTree)))
))

; Declare tree variables
(declare-const t IntTree)
(declare-const left_subtree IntTree)
(declare-const right_subtree IntTree)

; t is a node
(assert ((_ is node) t))

; Extract left and right subtrees
(assert (= left_subtree (left t)))
(assert (= right_subtree (right t)))

; Left subtree is a node
(assert ((_ is node) left_subtree))

; Right subtree is a leaf with value 20
(assert ((_ is leaf) right_subtree))
(assert (= (value right_subtree) 20))

; Left's left is a leaf with value 7
(assert ((_ is leaf) (left left_subtree)))
(assert (= (value (left left_subtree)) 7))

; Left's right is a leaf with value 13
(assert ((_ is leaf) (right left_subtree)))
(assert (= (value (right left_subtree)) 13))

; Reconstruct t to verify structure
(declare-const t_reconstructed IntTree)
(assert (= t_reconstructed
           (node (node (leaf 7) (leaf 13))
                 (leaf 20))))

(assert (= t t_reconstructed))

(check-sat)
(exit)
