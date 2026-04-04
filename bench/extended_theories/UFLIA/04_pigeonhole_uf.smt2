; Test: Pigeonhole principle with uninterpreted functions
;; expected: unsat
; Pattern: 4 pigeons into 3 holes (injective function cannot exist)

(set-logic UFLIA)

; f maps pigeons (0..3) to holes (0..2)
(declare-fun f (Int) Int)

; Range constraint: each pigeon goes to a valid hole
(assert (and (>= (f 0) 0) (<= (f 0) 2)))
(assert (and (>= (f 1) 0) (<= (f 1) 2)))
(assert (and (>= (f 2) 0) (<= (f 2) 2)))
(assert (and (>= (f 3) 0) (<= (f 3) 2)))

; Injectivity: no two pigeons share a hole
(assert (not (= (f 0) (f 1))))
(assert (not (= (f 0) (f 2))))
(assert (not (= (f 0) (f 3))))
(assert (not (= (f 1) (f 2))))
(assert (not (= (f 1) (f 3))))
(assert (not (= (f 2) (f 3))))

(check-sat)
(exit)
