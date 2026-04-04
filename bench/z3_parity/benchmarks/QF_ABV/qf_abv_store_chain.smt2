; Test: QF_ABV chained store operations - satisfiable
; Expected: sat
; Chain of stores with distinct indices: c[0]=1, c[1]=2

(set-logic QF_ABV)
(declare-const a (Array (_ BitVec 4) (_ BitVec 4)))
(define-fun b () (Array (_ BitVec 4) (_ BitVec 4)) (store a #x0 #x1))
(define-fun c () (Array (_ BitVec 4) (_ BitVec 4)) (store b #x1 #x2))
(assert (= (select c #x0) #x1))
(assert (= (select c #x1) #x2))
(check-sat)
; expected: sat
(exit)
