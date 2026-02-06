; Benchmark: Arithmetic Operations - fp.div with Constraints
; Expected: UNSAT
; Description: Tests division with impossible constraints

(set-logic QF_FP)
(set-info :status unsat)

; Declare Float32 variables
(declare-fun x () (_ FloatingPoint 8 24))
(declare-fun y () (_ FloatingPoint 8 24))
(declare-fun z1 () (_ FloatingPoint 8 24))
(declare-fun z2 () (_ FloatingPoint 8 24))

; x = 10.0
(assert (= x ((_ to_fp 8 24) RNE 10.0)))

; y = 3.0
(assert (= y ((_ to_fp 8 24) RNE 3.0)))

; z1 = x / y
(assert (= z1 (fp.div RNE x y)))

; z2 = y (exactly 3.0)
(assert (= z2 ((_ to_fp 8 24) RNE 3.0)))

; Impossible constraint: require z1 * z2 = x exactly
; But floating point arithmetic is not exact: (10/3)*3 != 10 in floating point
(declare-fun product () (_ FloatingPoint 8 24))
(assert (= product (fp.mul RNE z1 z2)))
(assert (= product x))

; Additional impossible constraints
; Require z1 to be both > 3.333 and < 3.333
(assert (fp.gt z1 ((_ to_fp 8 24) RNE 3.333)))
(assert (fp.lt z1 ((_ to_fp 8 24) RNE 3.333)))

(check-sat)
; (exit)
