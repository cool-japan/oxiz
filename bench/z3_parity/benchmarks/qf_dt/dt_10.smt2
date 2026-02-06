; QF_DT Benchmark: Multiple Datatypes with Pattern Matching
; Expected: SAT
; Description: Test interaction between multiple datatypes with complex pattern matching

(set-logic ALL)

; Define a binary tree of integers
(declare-datatypes ((IntTree 0)) (
  ((Leaf (leaf_value Int))
   (Node (node_left IntTree) (node_right IntTree)))
))

; Define an option type for tree results
(declare-datatypes ((TreeOption 0)) (
  ((None)
   (Some (some_value IntTree)))
))

; Define a result type that combines trees and options
(declare-datatypes ((ProcessResult 0)) (
  ((Success (result_tree IntTree) (result_depth Int))
   (Failure (error_code Int)))
))

; Declare variables
(declare-const tree1 IntTree)
(declare-const tree2 IntTree)
(declare-const opt1 TreeOption)
(declare-const opt2 TreeOption)
(declare-const result1 ProcessResult)

; tree1 is a Node with two leaves
(assert (= tree1 (Node (Leaf 10) (Leaf 20))))

; tree2 is extracted from tree1's left child
(assert (= tree2 (node_left tree1)))

; tree2 should be a Leaf
(assert ((_ is Leaf) tree2))
(assert (= (leaf_value tree2) 10))

; opt1 contains tree1
(assert (= opt1 (Some tree1)))

; opt1 is Some (not None)
(assert ((_ is Some) opt1))

; opt2 is None
(assert (= opt2 None))
(assert ((_ is None) opt2))

; result1 is a Success with tree1 and depth 2
(assert (= result1 (Success tree1 2)))

; Verify result1 is Success
(assert ((_ is Success) result1))

; Verify the tree in result1 has a left leaf with value 10
(assert (= (leaf_value (node_left (result_tree result1))) 10))

; Verify the depth is 2
(assert (= (result_depth result1) 2))

(check-sat)
(exit)
