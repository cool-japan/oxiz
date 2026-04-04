; Test: Mixed UF and LIA constraints
;; expected: sat
; Pattern: UF congruence combined with arithmetic reasoning

(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-fun g (Int Int) Int)
(declare-const a Int)
(declare-const b Int)
(declare-const c Int)

; Arithmetic constraints
(assert (= a (+ b 1)))
(assert (>= b 0))
(assert (<= b 100))
(assert (= c (* 2 b)))

; UF axiom: g is commutative on bounded domain
(assert (forall ((x Int) (y Int))
  (=> (and (>= x 0) (<= x 50) (>= y 0) (<= y 50))
      (= (g x y) (g y x)))))

; f preserves order on bounded domain
(assert (forall ((x Int) (y Int))
  (=> (and (>= x 0) (<= x 50) (>= y 0) (<= y 50) (< x y))
      (<= (f x) (f y)))))

; Ground: f(b) + f(a) = g(a, b) + g(b, a)
; Since g is commutative, g(a,b) = g(b,a), so this equals 2*g(a,b)
(assert (= (+ (f b) (f a)) (+ (g a b) (g b a))))

; b = 5 for concreteness
(assert (= b 5))

(check-sat)
(exit)
