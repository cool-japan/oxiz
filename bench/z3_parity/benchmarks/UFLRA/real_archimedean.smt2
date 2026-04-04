; Test: Archimedean property
; Expected: sat
; Pattern: forall x. exists n. n > x (for bounded domain)

(set-logic UFLRA)
(declare-fun ceil (Real) Real)

; For any real in a bounded range, there's a larger value
; We encode: for specific reals, we find witnesses
(declare-const x1 Real)
(declare-const x2 Real)
(declare-const x3 Real)
(declare-const n1 Real)
(declare-const n2 Real)
(declare-const n3 Real)

(assert (= x1 3.7))
(assert (= x2 (- 2.1)))
(assert (= x3 100.5))

; Each n_i > x_i
(assert (> n1 x1))
(assert (> n2 x2))
(assert (> n3 x3))

; ceil function rounds up
(assert (forall ((r Real))
  (=> (and (>= r 0.0) (<= r 10.0))
      (>= (ceil r) r))))

(assert (= (ceil 3.7) 4.0))
(assert (= (ceil 0.0) 0.0))

(check-sat)
(exit)
