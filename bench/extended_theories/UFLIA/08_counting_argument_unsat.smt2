; Test: Counting argument with UF
;; expected: unsat
; Pattern: f maps {0,1,2} to {0,1} injectively => impossible

(set-logic UFLIA)
(declare-fun f (Int) Int)

; Range: f maps to {0, 1}
(assert (or (= (f 0) 0) (= (f 0) 1)))
(assert (or (= (f 1) 0) (= (f 1) 1)))
(assert (or (= (f 2) 0) (= (f 2) 1)))

; Injectivity on {0, 1, 2}
(assert (not (= (f 0) (f 1))))
(assert (not (= (f 0) (f 2))))
(assert (not (= (f 1) (f 2))))

; 3 values mapped injectively to 2 targets => unsat
(check-sat)
(exit)
