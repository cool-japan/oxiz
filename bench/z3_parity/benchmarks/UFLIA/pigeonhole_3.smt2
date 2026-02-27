; Test: 3-pigeonhole principle via quantifiers
; Expected: unsat
; Pattern: 4 pigeons mapped injectively to 3 holes is impossible

(set-logic UFLIA)
(declare-fun hole (Int) Int)

; 4 pigeons (0..3) must map to 3 holes (1..3)
; Each pigeon goes to a valid hole
(assert (and (>= (hole 0) 1) (<= (hole 0) 3)))
(assert (and (>= (hole 1) 1) (<= (hole 1) 3)))
(assert (and (>= (hole 2) 1) (<= (hole 2) 3)))
(assert (and (>= (hole 3) 1) (<= (hole 3) 3)))

; Injectivity: no two pigeons share a hole
(assert (forall ((i Int) (j Int))
  (=> (and (>= i 0) (<= i 3) (>= j 0) (<= j 3) (not (= i j)))
      (not (= (hole i) (hole j))))))

(check-sat)
(exit)
