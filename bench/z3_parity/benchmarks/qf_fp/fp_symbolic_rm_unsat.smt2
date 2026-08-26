; Benchmark: RoundingMode cardinality - the sort has exactly five elements
; Expected: UNSAT
; Description: Six pairwise-distinct RoundingMode constants cannot exist: the
; SMT-LIB FloatingPoint theory fixes the RoundingMode domain at exactly five
; elements (RNE, RNA, RTP, RTN, RTZ). A solver that models RoundingMode as an
; ordinary uninterpreted sort answers `sat` here.

(set-logic QF_FP)
(set-info :status unsat)

(declare-const m0 RoundingMode)
(declare-const m1 RoundingMode)
(declare-const m2 RoundingMode)
(declare-const m3 RoundingMode)
(declare-const m4 RoundingMode)
(declare-const m5 RoundingMode)

(assert (distinct m0 m1 m2 m3 m4 m5))

(check-sat)
; (exit)
