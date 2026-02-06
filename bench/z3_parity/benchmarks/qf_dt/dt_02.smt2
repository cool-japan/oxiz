; QF_DT Benchmark: Head/Tail Selectors and is-nil Tester
; Expected: SAT
; Description: Test head/tail selectors and is-nil predicate

(set-logic ALL)

; Define a list of integers
(declare-datatypes ((IntList 0)) (
  ((nil)
   (cons (head Int) (tail IntList)))
))

; Declare list variables
(declare-const l1 IntList)
(declare-const l2 IntList)

; l1 is not nil (it's a cons)
(assert (not ((_ is nil) l1)))

; l1 has head 42
(assert (= (head l1) 42))

; The tail of l1 is nil
(assert ((_ is nil) (tail l1)))

; l2 is constructed as cons(42, nil)
(assert (= l2 (cons 42 nil)))

; l1 and l2 should be equal
(assert (= l1 l2))

; Additional constraint: the tail of l2 must be nil
(assert (= (tail l2) nil))

(check-sat)
(exit)
