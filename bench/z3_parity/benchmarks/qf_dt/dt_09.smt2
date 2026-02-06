; QF_DT Benchmark: Complex Pattern Matching with Multiple Constructors
; Expected: SAT
; Description: Test complex datatype with multiple constructors and pattern matching

(set-logic ALL)

; Define an expression datatype with multiple constructors
(declare-datatypes ((Expr 0)) (
  ((Const (const_val Int))
   (Add (add_left Expr) (add_right Expr))
   (Mul (mul_left Expr) (mul_right Expr))
   (Neg (neg_arg Expr)))
))

; Declare expression variables
(declare-const e1 Expr)
(declare-const e2 Expr)
(declare-const e3 Expr)
(declare-const e4 Expr)

; e1 is a constant 5
(assert (= e1 (Const 5)))

; e2 is a constant 3
(assert (= e2 (Const 3)))

; e3 is Add(e1, e2) = Add(Const(5), Const(3))
(assert (= e3 (Add e1 e2)))

; e4 is Mul(e3, Neg(e1)) = Mul(Add(Const(5), Const(3)), Neg(Const(5)))
(assert (= e4 (Mul e3 (Neg e1))))

; Check pattern matching: e3 is an Add
(assert ((_ is Add) e3))

; Check that left side of e3 is a Const
(assert ((_ is Const) (add_left e3)))

; Check that the constant value is 5
(assert (= (const_val (add_left e3)) 5))

; Check that e4 is a Mul
(assert ((_ is Mul) e4))

; Check that right side of e4 is a Neg
(assert ((_ is Neg) (mul_right e4)))

; Check that the argument of Neg is a Const with value 5
(assert ((_ is Const) (neg_arg (mul_right e4))))
(assert (= (const_val (neg_arg (mul_right e4))) 5))

(check-sat)
(exit)
