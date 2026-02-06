; Benchmark: Arithmetic Operations - fp.add and fp.mul
; Expected: SAT
; Description: Tests addition and multiplication with various constraints

(set-logic QF_FP)
(set-info :status sat)

; Declare Float64 variables
(declare-fun a () (_ FloatingPoint 11 53))
(declare-fun b () (_ FloatingPoint 11 53))
(declare-fun c () (_ FloatingPoint 11 53))
(declare-fun sum () (_ FloatingPoint 11 53))
(declare-fun product () (_ FloatingPoint 11 53))

; a = 5.0
(assert (= a ((_ to_fp 11 53) RNE 5.0)))

; b = 7.0
(assert (= b ((_ to_fp 11 53) RNE 7.0)))

; c = 3.0
(assert (= c ((_ to_fp 11 53) RNE 3.0)))

; sum = a + b
(assert (= sum (fp.add RNE a b)))

; product = a * c
(assert (= product (fp.mul RNE a c)))

; Check sum = 12.0
(assert (= sum ((_ to_fp 11 53) RNE 12.0)))

; Check product = 15.0
(assert (= product ((_ to_fp 11 53) RNE 15.0)))

; Additional complex constraint: (a + b) * c
(declare-fun result () (_ FloatingPoint 11 53))
(assert (= result (fp.mul RNE sum c)))
(assert (= result ((_ to_fp 11 53) RNE 36.0)))

; Test commutativity
(declare-fun product2 () (_ FloatingPoint 11 53))
(assert (= product2 (fp.mul RNE c a)))
(assert (= product product2))

; Test associativity of addition (should hold for these simple values)
(declare-fun sum_alt () (_ FloatingPoint 11 53))
(declare-fun temp () (_ FloatingPoint 11 53))
(assert (= temp (fp.add RNE b c)))
(assert (= sum_alt (fp.add RNE a temp)))
(assert (= sum_alt ((_ to_fp 11 53) RNE 15.0)))

(check-sat)
; (exit)
