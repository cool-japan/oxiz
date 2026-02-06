; QF_DT Benchmark: List Equality with Contradictions
; Expected: UNSAT
; Description: Test list equality with contradictory constraints

(set-logic ALL)

; Define a list of integers
(declare-datatypes ((IntList 0)) (
  ((nil)
   (cons (head Int) (tail IntList)))
))

; Declare list variables
(declare-const l1 IntList)
(declare-const l2 IntList)

; l1 is a cons cell
(assert (not ((_ is nil) l1)))

; l2 is nil
(assert ((_ is nil) l2))

; But we assert they are equal - contradiction!
(assert (= l1 l2))

; Additional contradictory constraint
(assert (= (head l1) 10))

(check-sat)
(exit)
