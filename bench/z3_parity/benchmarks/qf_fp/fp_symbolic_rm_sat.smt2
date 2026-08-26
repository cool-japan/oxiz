; Benchmark: Symbolic rounding mode - RoundingMode as a first-class sort
; Expected: SAT
; Description: The rounding mode of an fp.add is a declared RoundingMode
; constant rather than a literal RNE/RTZ/... symbol. The two operands are
; pinned to concrete Float32 values, so the sum is determined once a mode is
; chosen; every one of the five modes yields the exact value 4.0 here, so the
; formula is satisfiable whichever mode the solver picks for m.

(set-logic QF_FP)
(set-info :status sat)

(declare-const m RoundingMode)
(declare-const x (_ FloatingPoint 8 24))
(declare-const y (_ FloatingPoint 8 24))
(declare-const z (_ FloatingPoint 8 24))

(assert (= x ((_ to_fp 8 24) RNE 1.5)))
(assert (= y ((_ to_fp 8 24) RNE 2.5)))

; z = x + y under the *symbolic* mode m
(assert (= z (fp.add m x y)))

; 1.5 + 2.5 is exactly representable, so the result is 4.0 in every mode
(assert (fp.eq z ((_ to_fp 8 24) RNE 4.0)))

(check-sat)
; (exit)
