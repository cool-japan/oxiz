; QF_DT Benchmark: Basic List Operations with nil/cons
; Expected: SAT
; Description: Test basic list construction with nil and cons constructors

(set-logic ALL)

; Define a list of integers
(declare-datatypes ((IntList 0)) (
  ((nil)
   (cons (head Int) (tail IntList)))
))

; Declare list variables
(declare-const l1 IntList)
(declare-const l2 IntList)
(declare-const l3 IntList)

; Build a list: cons(1, cons(2, nil))
(assert (= l1 (cons 1 (cons 2 nil))))

; Build another list: cons(3, nil)
(assert (= l2 (cons 3 nil)))

; l3 is the concatenation concept: cons(1, cons(2, cons(3, nil)))
(assert (= l3 (cons 1 (cons 2 l2))))

; Check that l3's head is 1
(assert (= (head l3) 1))

; Check that the head of tail of l3 is 2
(assert (= (head (tail l3)) 2))

; Check that the head of tail of tail of l3 is 3
(assert (= (head (tail (tail l3))) 3))

(check-sat)
(exit)
