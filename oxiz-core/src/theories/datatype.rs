//! Algebraic Datatype theory implementation
//!
//! Implements a lightweight theory of algebraic datatypes: it tracks
//! constructor, selector and tester applications, instantiates the datatype
//! axioms as real terms, propagates the equalities those axioms justify
//! (injectivity, selector application, tester application), and reports the
//! conflicts it can see (one class holding applications of two different
//! constructors).
//!
//! This is `oxiz-core`'s self-contained layer, not the datatype solver that
//! `oxiz-solver` runs — that is `oxiz_theories::datatype`. It is incomplete:
//! finding no conflict here says nothing about satisfiability.
//!
//! Reference: Z3's `src/smt/theory_datatype.cpp`

use super::combination::{Theory, TheoryResult};
use super::eq_classes::EqClasses;
use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind, SortManager};

/// Datatype theory axioms
///
/// Constructor, selector and tester names are stored as strings rather than
/// interned keys: the axioms travel between the term manager's interner and
/// the sort manager's (which are separate), and a key from one resolves to
/// nonsense in the other.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DatatypeAxiom {
    /// Constructor distinctness: C_i(...) ≠ C_j(...) for i ≠ j
    ConstructorDistinctness {
        /// Name of the first constructor
        cons1: String,
        /// Name of the second constructor
        cons2: String,
        /// Datatype sort
        datatype: SortId,
    },
    /// Selector axiom: sel_i(C(x_1, ..., x_n)) = x_i
    SelectorAxiom {
        /// Constructor application the selector is applied to
        constructor: TermId,
        /// Name of the selector
        selector: String,
        /// Field index
        field_index: usize,
        /// Field value
        field_value: TermId,
    },
    /// Tester axiom: is_C(C(x_1, ..., x_n)) = true
    TesterAxiom {
        /// Constructor application the tester is applied to
        constructor: TermId,
        /// Name of the constructor the tester tests for
        tester: String,
    },
    /// Negative tester: is_C_i(C_j(...)) = false for i ≠ j
    NegativeTesterAxiom {
        /// Constructor application with constructor C_j
        constructor: TermId,
        /// Name of the different constructor C_i the tester tests for
        tester: String,
    },
    /// Acyclicity: prevents cyclic structures in strictly positive positions
    Acyclicity {
        /// Term that would create a cycle
        term: TermId,
    },
    /// Injectivity: C(x_1, ..., x_n) = C(y_1, ..., y_n) ⟹ x_i = y_i
    Injectivity {
        /// First constructor application
        cons1: TermId,
        /// Second constructor application
        cons2: TermId,
        /// Name of the shared constructor
        constructor: String,
    },
}

/// Datatype theory reasoning engine
#[derive(Debug, Clone)]
pub struct DatatypeTheory {
    /// Tracked datatype terms (maps term to its sort)
    datatypes: FxHashMap<TermId, SortId>,
    /// Constructor applications: maps constructor term to its arguments
    constructors: FxHashMap<TermId, ConstructorInfo>,
    /// Selector applications: maps selector term to (datatype, selector_name)
    selectors: FxHashMap<TermId, (TermId, crate::interner::Spur)>,
    /// Tester applications: maps tester term to (datatype, tester_name)
    testers: FxHashMap<TermId, (TermId, crate::interner::Spur)>,
    /// Equality classes over every term this theory tracks
    classes: EqClasses,
    /// Equalities already reported by `propagate`, so each is reported once
    reported: FxHashSet<(TermId, TermId)>,
    /// Pending axiom instantiations
    pending_axioms: Vec<DatatypeAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<DatatypeAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

/// Information about a constructor application
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstructorInfo {
    /// Constructor name, interned in the `TermManager`'s interner
    name: crate::interner::Spur,
    /// Arguments to the constructor
    args: Vec<TermId>,
    /// Datatype sort
    sort: SortId,
}

impl Default for DatatypeTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl DatatypeTheory {
    /// Create a new datatype theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            datatypes: FxHashMap::default(),
            constructors: FxHashMap::default(),
            selectors: FxHashMap::default(),
            testers: FxHashMap::default(),
            classes: EqClasses::new(),
            reported: FxHashSet::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a datatype term
    pub fn register_datatype(&mut self, term: TermId, sort: SortId) {
        self.datatypes.insert(term, sort);
        self.classes.add(term);
    }

    /// Register a constructor application
    ///
    /// This is registration only. The axioms a constructor application
    /// justifies are instantiated by [`DatatypeTheory::add_term`], which has
    /// the managers needed to name the datatype's selectors and its other
    /// constructors.
    pub fn register_constructor(
        &mut self,
        term: TermId,
        name: crate::interner::Spur,
        args: Vec<TermId>,
        sort: SortId,
    ) {
        for &arg in &args {
            self.classes.add(arg);
        }
        self.constructors
            .insert(term, ConstructorInfo { name, args, sort });
        self.register_datatype(term, sort);
    }

    /// Register a selector application
    pub fn register_selector(
        &mut self,
        term: TermId,
        datatype: TermId,
        selector: crate::interner::Spur,
    ) {
        self.selectors.insert(term, (datatype, selector));
        self.classes.add(term);
        self.classes.add(datatype);
    }

    /// Register a tester application
    pub fn register_tester(
        &mut self,
        term: TermId,
        datatype: TermId,
        tester: crate::interner::Spur,
    ) {
        self.testers.insert(term, (datatype, tester));
        self.classes.add(term);
        self.classes.add(datatype);
    }

    /// Add a term to the theory and extract datatype operations
    ///
    /// Returns `true` when the term belongs to this theory: a datatype-sorted
    /// term, or an application of a constructor, selector or tester (a tester
    /// is Boolean and a selector has its field's sort, but both are datatype
    /// operations).
    pub fn add_term(
        &mut self,
        term: TermId,
        manager: &TermManager,
        sort_manager: &SortManager,
    ) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        match t.kind.clone() {
            TermKind::DtConstructor { constructor, args } => {
                let sort = t.sort;
                self.register_constructor(term, constructor, args.to_vec(), sort);
                self.generate_selector_axioms(term, manager, sort_manager);
                self.generate_distinctness_axioms(constructor, sort, manager, sort_manager);
                true
            }
            TermKind::DtTester { arg, constructor } => {
                self.register_tester(term, arg, constructor);
                self.generate_tester_axiom(arg, constructor, manager);
                true
            }
            TermKind::DtSelector { arg, selector } => {
                self.register_selector(term, arg, selector);
                self.generate_selector_axioms(arg, manager, sort_manager);
                true
            }
            _ => {
                if let Some(sort) = sort_manager.get(t.sort)
                    && matches!(sort.kind, SortKind::Datatype(_))
                {
                    let sort_id = t.sort;
                    self.register_datatype(term, sort_id);
                    return true;
                }
                false
            }
        }
    }

    /// Record that two terms known to this theory are equal
    ///
    /// Returns `true` when this was new information. Equalities about terms
    /// the theory has never seen are ignored, and so are equalities between
    /// datatype terms of two different datatype sorts, which are not well
    /// sorted.
    pub fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        if !self.classes.contains(a) || !self.classes.contains(b) {
            return false;
        }

        if let (Some(&sort_a), Some(&sort_b)) = (self.datatypes.get(&a), self.datatypes.get(&b))
            && sort_a != sort_b
        {
            return false;
        }

        self.classes.union(a, b)
    }

    /// Whether two terms are currently known to be equal by this theory
    ///
    /// Takes `&mut self` because the lookup compresses the union-find paths.
    pub fn known_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.classes.are_equal(a, b)
    }

    /// Instantiate the selector axioms of a constructor application
    ///
    /// One axiom per field of the constructor, naming the real selector from
    /// the datatype declaration. Nothing is generated when the term is not a
    /// registered constructor application, or when its datatype has not been
    /// declared on the sort manager (the selector names live there).
    fn generate_selector_axioms(
        &mut self,
        constructor_term: TermId,
        manager: &TermManager,
        sort_manager: &SortManager,
    ) {
        let Some(info) = self.constructors.get(&constructor_term) else {
            return;
        };
        let args = info.args.clone();
        let Some(selectors) = constructor_selectors(info.name, info.sort, manager, sort_manager)
        else {
            return;
        };

        for (field_index, &field_value) in args.iter().enumerate() {
            let Some(selector) = selectors.get(field_index) else {
                break;
            };
            self.add_axiom(DatatypeAxiom::SelectorAxiom {
                constructor: constructor_term,
                selector: selector.clone(),
                field_index,
                field_value,
            });
        }
    }

    /// Instantiate the tester axiom for a tester applied to a constructor
    ///
    /// Positive when the tester's constructor is the one that built the term,
    /// negative otherwise. Nothing is generated when the argument is not a
    /// registered constructor application.
    fn generate_tester_axiom(
        &mut self,
        argument: TermId,
        tester: crate::interner::Spur,
        manager: &TermManager,
    ) {
        let Some(info) = self.constructors.get(&argument) else {
            return;
        };

        let tester_name = manager.resolve_str(tester).to_string();
        let built_by = manager.resolve_str(info.name).to_string();

        if tester_name == built_by {
            self.add_axiom(DatatypeAxiom::TesterAxiom {
                constructor: argument,
                tester: tester_name,
            });
        } else {
            self.add_axiom(DatatypeAxiom::NegativeTesterAxiom {
                constructor: argument,
                tester: tester_name,
            });
        }
    }

    /// Generate distinctness axioms between a constructor and every other
    /// constructor of its datatype
    ///
    /// Nothing is generated when the datatype has not been declared on the
    /// sort manager, or when it has only this one constructor.
    fn generate_distinctness_axioms(
        &mut self,
        cons_name: crate::interner::Spur,
        sort: SortId,
        manager: &TermManager,
        sort_manager: &SortManager,
    ) {
        let name = manager.resolve_str(cons_name).to_string();
        let Some(others) = sibling_constructors(&name, sort, sort_manager) else {
            return;
        };

        for other in others {
            self.add_axiom(DatatypeAxiom::ConstructorDistinctness {
                cons1: name.clone(),
                cons2: other,
                datatype: sort,
            });
        }
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: DatatypeAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<DatatypeAxiom> {
        core::mem::take(&mut self.pending_axioms)
    }

    /// Build the formula stated by an axiom
    ///
    /// Every axiom is emitted as a real term over `manager`:
    ///
    /// * `ConstructorDistinctness` — a ground instance
    ///   `not (C(v_1, …, v_n) = D(w_1, …, w_m))`, where the `v_i` and `w_j`
    ///   are fresh variables named after the constructor and field position.
    ///   It is one instance, not the universally quantified axiom: assert it
    ///   for the variables you care about, or instantiate it yourself.
    /// * `SelectorAxiom` — `sel(C(x_1, …, x_n)) = x_i`.
    /// * `TesterAxiom` — `is_C(C(…)) = true`.
    /// * `NegativeTesterAxiom` — `not is_C(D(…))`.
    /// * `Acyclicity` — the conjunction of `t ≠ s` over the proper
    ///   sub-terms `s` of `t` that have the same datatype sort and are
    ///   reachable through registered constructor applications. This is the
    ///   ground part of acyclicity, not the full axiom.
    /// * `Injectivity` — `C(x⃗) = C(y⃗) ⟹ x_1 = y_1 ∧ … ∧ x_n = y_n`.
    ///
    /// Returns `None` when the axiom cannot be built, or when it does not fit
    /// the terms it names — the emitted formula has to be true, so an axiom
    /// that does not describe its own terms produces nothing rather than a
    /// false one. That covers: a term that is not in `manager` or is not the
    /// constructor application the axiom claims; a datatype that has not been
    /// declared on the sort manager; a `ConstructorDistinctness` between one
    /// constructor and itself; a `SelectorAxiom` whose selector is not the
    /// named constructor's field at that index, or whose field value is not
    /// the argument there; a `TesterAxiom` whose term was built by a different
    /// constructor (and a `NegativeTesterAxiom` whose term was built by that
    /// very one); an `Injectivity` over two different constructors, different
    /// arities, or a nullary one; and an `Acyclicity` term with no sub-term of
    /// its own sort.
    pub fn axiom_to_term(
        &self,
        axiom: &DatatypeAxiom,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        match axiom {
            DatatypeAxiom::ConstructorDistinctness {
                cons1,
                cons2,
                datatype,
            } => {
                // C(v⃗) ≠ C(v⃗) is false, so two different names are required.
                if cons1 == cons2 {
                    return None;
                }
                let first = self.ground_instance(cons1, *datatype, manager)?;
                let second = self.ground_instance(cons2, *datatype, manager)?;
                let equality = manager.mk_eq(first, second);
                Some(manager.mk_not(equality))
            }
            DatatypeAxiom::SelectorAxiom {
                constructor,
                selector,
                field_index,
                field_value,
            } => {
                let (built_by, args) = constructor_application(*constructor, manager)?;
                // The equation only holds if the selector really is this
                // constructor's field at this index, holding this value.
                if args.get(*field_index) != Some(field_value) {
                    return None;
                }
                let sort = manager.get(*constructor)?.sort;
                let selectors = constructor_selectors(built_by, sort, manager, &manager.sorts)?;
                if selectors.get(*field_index).map(String::as_str) != Some(selector.as_str()) {
                    return None;
                }

                let result_sort = manager.get(*field_value)?.sort;
                let application = manager.mk_dt_selector(selector, *constructor, result_sort);
                Some(manager.mk_eq(application, *field_value))
            }
            DatatypeAxiom::TesterAxiom {
                constructor,
                tester,
            } => {
                let (built_by, _) = constructor_application(*constructor, manager)?;
                // is_C(t) is true only when t was built by C.
                if manager.resolve_str(built_by) != tester {
                    return None;
                }

                let application = manager.mk_dt_tester(tester, *constructor);
                let truth = manager.mk_true();
                Some(manager.mk_eq(application, truth))
            }
            DatatypeAxiom::NegativeTesterAxiom {
                constructor,
                tester,
            } => {
                let (built_by, _) = constructor_application(*constructor, manager)?;
                // not is_C(t) holds only when t was built by something else.
                if manager.resolve_str(built_by) == tester {
                    return None;
                }

                let application = manager.mk_dt_tester(tester, *constructor);
                Some(manager.mk_not(application))
            }
            DatatypeAxiom::Acyclicity { term } => {
                let descendants = self.proper_subterms_of_same_sort(*term);
                if descendants.is_empty() {
                    return None;
                }

                let mut conjuncts = Vec::with_capacity(descendants.len());
                for descendant in descendants {
                    let equality = manager.mk_eq(*term, descendant);
                    conjuncts.push(manager.mk_not(equality));
                }
                Some(manager.mk_and(conjuncts))
            }
            DatatypeAxiom::Injectivity {
                cons1,
                cons2,
                constructor,
            } => {
                // Injectivity is per constructor: `C(x⃗) = D(y⃗)` says nothing
                // about the arguments when C and D differ.
                let (first_name, first) = constructor_application(*cons1, manager)?;
                let (second_name, second) = constructor_application(*cons2, manager)?;
                if first_name != second_name
                    || manager.resolve_str(first_name) != constructor
                    || first.len() != second.len()
                    || first.is_empty()
                {
                    return None;
                }

                let premise = manager.mk_eq(*cons1, *cons2);
                let mut conclusions = Vec::with_capacity(first.len());
                for (left, right) in first.iter().zip(second.iter()) {
                    conclusions.push(manager.mk_eq(*left, *right));
                }
                let conclusion = manager.mk_and(conclusions);
                Some(manager.mk_implies(premise, conclusion))
            }
        }
    }

    /// Build `C(v_1, …, v_n)` over fresh variables named after the constructor
    fn ground_instance(
        &self,
        constructor: &str,
        datatype: SortId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let field_sorts = constructor_field_sorts(constructor, datatype, &manager.sorts)?;

        let mut arguments = Vec::with_capacity(field_sorts.len());
        for (index, sort) in field_sorts.into_iter().enumerate() {
            let name = format!("dt!{constructor}!{index}");
            arguments.push(manager.mk_var(&name, sort));
        }

        Some(manager.mk_dt_constructor(constructor, arguments, datatype))
    }

    /// Proper sub-terms of `term` with the same datatype sort, reachable
    /// through registered constructor applications
    fn proper_subterms_of_same_sort(&self, term: TermId) -> Vec<TermId> {
        let Some(sort) = self.datatypes.get(&term).copied() else {
            return Vec::new();
        };
        let Some(info) = self.constructors.get(&term) else {
            return Vec::new();
        };

        let mut found = Vec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = info.args.clone();

        while let Some(current) = stack.pop() {
            if current == term || !seen.insert(current) {
                continue;
            }
            if self.datatypes.get(&current) == Some(&sort) {
                found.push(current);
            }
            if let Some(info) = self.constructors.get(&current) {
                stack.extend(info.args.iter().copied());
            }
        }

        found.sort_unstable_by_key(|term| term.0);
        found
    }

    /// Deduce new equalities from the datatype axioms
    ///
    /// Three rules are applied to a fixpoint over the current classes:
    ///
    /// * *injectivity* — two applications of the same constructor that are in
    ///   one class have pairwise equal arguments;
    /// * *selector application* — `sel(t)` where `t` is known equal to
    ///   `C(x_1, …, x_n)` and `sel` is `C`'s i-th selector equals `x_i`;
    /// * *tester application* — `is_C(t)` where `t` is known equal to a
    ///   constructor application is `true` if that constructor is `C`, and
    ///   `false` otherwise.
    ///
    /// Each deduced equality is merged into this theory's classes and returned
    /// once; later calls return only what is new since the previous one.
    /// A selector applied to a constructor it does not belong to deduces
    /// nothing: SMT-LIB leaves that value unspecified.
    pub fn propagate(&mut self, manager: &TermManager) -> Vec<(TermId, TermId)> {
        let mut deduced = Vec::new();

        loop {
            let mut round = self.injectivity_pairs();
            round.extend(self.selector_values(manager));
            round.extend(self.tester_values(manager));

            let mut progressed = false;
            for (a, b) in round {
                let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                let merged = self.classes.union(a, b);
                if self.reported.insert(key) {
                    deduced.push(key);
                    progressed = true;
                } else if merged {
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }

        self.propagations += deduced.len();
        deduced
    }

    /// One constructor application per class, keyed by representative
    fn constructor_representatives(&mut self) -> FxHashMap<TermId, TermId> {
        let mut representatives = FxHashMap::default();

        for class in self.classes.classes() {
            for &term in &class {
                if self.constructors.contains_key(&term) {
                    let representative = self.classes.find(term);
                    representatives.entry(representative).or_insert(term);
                    break;
                }
            }
        }

        representatives
    }

    /// Argument equalities forced by two same-constructor applications in one class
    fn injectivity_pairs(&mut self) -> Vec<(TermId, TermId)> {
        let mut forced = Vec::new();

        for class in self.classes.classes() {
            let applications: Vec<TermId> = class
                .iter()
                .copied()
                .filter(|term| self.constructors.contains_key(term))
                .collect();

            for (position, &first) in applications.iter().enumerate() {
                for &second in &applications[position + 1..] {
                    let (Some(left), Some(right)) = (
                        self.constructors.get(&first),
                        self.constructors.get(&second),
                    ) else {
                        continue;
                    };
                    if left.name != right.name || left.args.len() != right.args.len() {
                        continue;
                    }

                    let pairs: Vec<(TermId, TermId)> = left
                        .args
                        .iter()
                        .copied()
                        .zip(right.args.iter().copied())
                        .collect();
                    for (a, b) in pairs {
                        if a != b {
                            forced.push((a, b));
                        }
                    }
                }
            }
        }

        forced
    }

    /// Equalities from selectors applied to known constructor applications
    fn selector_values(&mut self, manager: &TermManager) -> Vec<(TermId, TermId)> {
        let representatives = self.constructor_representatives();
        let applications: Vec<(TermId, TermId, crate::interner::Spur)> = self
            .selectors
            .iter()
            .map(|(&term, &(argument, selector))| (term, argument, selector))
            .collect();

        let mut values = Vec::new();
        for (term, argument, selector) in applications {
            let representative = self.classes.find(argument);
            let Some(&constructor_term) = representatives.get(&representative) else {
                continue;
            };
            let Some(info) = self.constructors.get(&constructor_term) else {
                continue;
            };
            let (name, sort, args) = (info.name, info.sort, info.args.clone());

            let Some(selectors) = constructor_selectors(name, sort, manager, &manager.sorts) else {
                continue;
            };
            let wanted = manager.resolve_str(selector);
            let Some(index) = selectors.iter().position(|candidate| candidate == wanted) else {
                continue;
            };
            let Some(&value) = args.get(index) else {
                continue;
            };

            self.classes.add(term);
            self.classes.add(value);
            if term != value {
                values.push((term, value));
            }
        }

        values
    }

    /// Equalities from testers applied to known constructor applications
    fn tester_values(&mut self, manager: &TermManager) -> Vec<(TermId, TermId)> {
        let representatives = self.constructor_representatives();
        let applications: Vec<(TermId, TermId, crate::interner::Spur)> = self
            .testers
            .iter()
            .map(|(&term, &(argument, tester))| (term, argument, tester))
            .collect();

        let mut values = Vec::new();
        for (term, argument, tester) in applications {
            let representative = self.classes.find(argument);
            let Some(&constructor_term) = representatives.get(&representative) else {
                continue;
            };
            let Some(info) = self.constructors.get(&constructor_term) else {
                continue;
            };

            let built_by = manager.resolve_str(info.name);
            let tested_for = manager.resolve_str(tester);
            let answer = if built_by == tested_for {
                manager.mk_true()
            } else {
                manager.mk_false()
            };

            self.classes.add(term);
            self.classes.add(answer);
            values.push((term, answer));
        }

        values
    }

    /// Check for conflicts in the current state
    ///
    /// Two contradictions are detected:
    ///
    /// * a class holding applications of two different constructors, which
    ///   constructor distinctness forbids;
    /// * a class holding both `true` and `false`, which is how two testers
    ///   that disagree about one term show up.
    ///
    /// The returned chain starts at one offending term and ends at the other,
    /// and every consecutive pair in it was asserted or deduced equal.
    ///
    /// `None` means "no conflict found by this theory", never "satisfiable".
    pub fn check_for_conflicts(&mut self, manager: &TermManager) -> Option<Vec<TermId>> {
        let (truth, falsity) = (manager.mk_true(), manager.mk_false());

        for class in self.classes.classes() {
            let mut witness: Option<(TermId, crate::interner::Spur)> = None;

            for &term in &class {
                let Some(info) = self.constructors.get(&term) else {
                    continue;
                };
                match witness {
                    Some((first, name)) if name != info.name => {
                        self.conflicts += 1;
                        return Some(self.explanation_between(first, term));
                    }
                    Some(_) => {}
                    None => witness = Some((term, info.name)),
                }
            }

            if class.contains(&truth) && class.contains(&falsity) {
                self.conflicts += 1;
                return Some(self.explanation_between(truth, falsity));
            }
        }

        None
    }

    /// Chain of terms linking two conflicting terms, or just the pair
    fn explanation_between(&self, from: TermId, to: TermId) -> Vec<TermId> {
        let explanation = self.classes.explain(from, to);
        if explanation.is_empty() {
            vec![from, to]
        } else {
            explanation
        }
    }

    /// Reset the theory state (for backtracking)
    pub fn reset(&mut self) {
        self.datatypes.clear();
        self.constructors.clear();
        self.selectors.clear();
        self.testers.clear();
        self.classes.reset();
        self.reported.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> DatatypeStatistics {
        DatatypeStatistics {
            num_datatypes: self.datatypes.len(),
            num_constructors: self.constructors.len(),
            num_selectors: self.selectors.len(),
            num_testers: self.testers.len(),
            num_axioms: self.instantiated.len(),
            num_equality_nodes: self.classes.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// The constructor name and arguments of a constructor application
///
/// Reads them from the term itself rather than from the theory's registry, so
/// that an axiom about a term this theory never saw is still checked against
/// what the term actually is. Returns `None` for anything else.
fn constructor_application(
    term: TermId,
    manager: &TermManager,
) -> Option<(crate::interner::Spur, Vec<TermId>)> {
    match &manager.get(term)?.kind {
        TermKind::DtConstructor { constructor, args } => Some((*constructor, args.to_vec())),
        _ => None,
    }
}

/// Selector names of a constructor, in field order
///
/// `cons_name` is interned in the term manager, the datatype declaration in
/// the sort manager, so the two are matched by their resolved strings.
fn constructor_selectors(
    cons_name: crate::interner::Spur,
    sort: SortId,
    manager: &TermManager,
    sort_manager: &SortManager,
) -> Option<Vec<String>> {
    let wanted = manager.resolve_str(cons_name);
    let datatype_name = sort_manager.datatype_name(sort)?;
    let definition = sort_manager.get_datatype(datatype_name)?;

    let constructor = definition
        .constructors
        .iter()
        .find(|candidate| sort_manager.resolve_spur(candidate.name) == wanted)?;

    Some(
        constructor
            .selectors
            .iter()
            .map(|(name, _)| sort_manager.resolve_spur(*name).to_string())
            .collect(),
    )
}

/// Field sorts of a constructor, in field order
fn constructor_field_sorts(
    cons_name: &str,
    sort: SortId,
    sort_manager: &SortManager,
) -> Option<Vec<SortId>> {
    let datatype_name = sort_manager.datatype_name(sort)?;
    let definition = sort_manager.get_datatype(datatype_name)?;

    let constructor = definition
        .constructors
        .iter()
        .find(|candidate| sort_manager.resolve_spur(candidate.name) == cons_name)?;

    Some(
        constructor
            .selectors
            .iter()
            .map(|(_, field_sort)| *field_sort)
            .collect(),
    )
}

/// Names of the datatype's other constructors
fn sibling_constructors(
    cons_name: &str,
    sort: SortId,
    sort_manager: &SortManager,
) -> Option<Vec<String>> {
    let datatype_name = sort_manager.datatype_name(sort)?;
    let definition = sort_manager.get_datatype(datatype_name)?;

    Some(
        definition
            .constructors
            .iter()
            .map(|constructor| sort_manager.resolve_spur(constructor.name).to_string())
            .filter(|name| name != cons_name)
            .collect(),
    )
}

impl Theory for DatatypeTheory {
    fn add_term(&mut self, term: TermId, manager: &TermManager) -> bool {
        DatatypeTheory::add_term(self, term, manager, &manager.sorts)
    }

    fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        DatatypeTheory::assert_equality(self, a, b)
    }

    /// Propagates, then reports a conflict, then states the axioms it can
    ///
    /// An axiom that [`DatatypeTheory::axiom_to_term`] cannot build stays
    /// queued rather than being dropped, so
    /// [`DatatypeTheory::take_pending_axioms`] still hands it to the caller.
    /// It produces no lemma, so leaving it queued does not keep the combiner's
    /// loop running.
    fn check(&mut self, manager: &mut TermManager) -> TheoryResult {
        let deduced = self.propagate(manager);

        if let Some(explanation) = self.check_for_conflicts(manager) {
            return TheoryResult::Unsat { explanation };
        }

        if !deduced.is_empty() {
            return TheoryResult::Propagate(deduced);
        }

        let queued = self.take_pending_axioms();
        let mut lemmas = Vec::with_capacity(queued.len());
        for axiom in queued {
            match self.axiom_to_term(&axiom, manager) {
                Some(term) => lemmas.push(term),
                None => self.pending_axioms.push(axiom),
            }
        }

        if lemmas.is_empty() {
            TheoryResult::Sat
        } else {
            TheoryResult::Lemmas(lemmas)
        }
    }

    fn name(&self) -> &str {
        "datatype"
    }

    fn reset(&mut self) {
        DatatypeTheory::reset(self);
    }
}

/// Statistics for datatype theory
#[derive(Debug, Clone, Copy)]
pub struct DatatypeStatistics {
    /// Number of datatype terms
    pub num_datatypes: usize,
    /// Number of constructor applications
    pub num_constructors: usize,
    /// Number of selector applications
    pub num_selectors: usize,
    /// Number of tester applications
    pub num_testers: usize,
    /// Number of axioms instantiated
    pub num_axioms: usize,
    /// Number of terms held in the equality classes
    pub num_equality_nodes: usize,
    /// Number of equalities deduced by `propagate`
    pub num_propagations: usize,
    /// Number of conflicts detected
    pub num_conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interner::Key;
    use crate::sort::DataTypeConstructor;

    /// Declare `Pair` with one constructor `mk(first: BV8, second: BV8)` and a
    /// nullary constructor `none`.
    fn declare_pair(manager: &mut TermManager) -> (SortId, SortId) {
        let bv_sort = manager.sorts.bitvec(8);
        let pair_sort = manager.sorts.mk_datatype_sort("Pair");

        let first = manager.sorts.intern_str("first");
        let second = manager.sorts.intern_str("second");
        let mk = manager.sorts.intern_str("mk");
        let none = manager.sorts.intern_str("none");

        manager.sorts.declare_datatype(
            "Pair",
            vec![
                DataTypeConstructor {
                    name: mk,
                    selectors: smallvec::smallvec![(first, bv_sort), (second, bv_sort)],
                },
                DataTypeConstructor {
                    name: none,
                    selectors: smallvec::smallvec![],
                },
            ],
        );

        (pair_sort, bv_sort)
    }

    #[test]
    fn test_empty_theory() {
        let theory = DatatypeTheory::new();
        assert_eq!(theory.datatypes.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_datatype() {
        let mut theory = DatatypeTheory::new();
        let term = TermId(42);
        let sort = SortId(1);
        theory.register_datatype(term, sort);
        assert_eq!(theory.datatypes.get(&term), Some(&sort));
    }

    #[test]
    fn test_register_constructor() {
        let mut theory = DatatypeTheory::new();
        let cons_name =
            crate::interner::Spur::try_from_usize(1).expect("test operation should succeed");
        let term = TermId(42);
        let sort = SortId(1);
        let args = vec![TermId(1), TermId(2)];

        theory.register_constructor(term, cons_name, args.clone(), sort);

        assert_eq!(
            theory
                .constructors
                .get(&term)
                .expect("key should exist in map")
                .name,
            cons_name
        );
        assert_eq!(
            theory
                .constructors
                .get(&term)
                .expect("key should exist in map")
                .args,
            args
        );
    }

    #[test]
    fn test_register_selector() {
        let mut theory = DatatypeTheory::new();
        let selector =
            crate::interner::Spur::try_from_usize(2).expect("test operation should succeed");
        let term = TermId(42);
        let datatype = TermId(10);

        theory.register_selector(term, datatype, selector);
        assert_eq!(theory.selectors.get(&term), Some(&(datatype, selector)));
    }

    #[test]
    fn test_register_tester() {
        let mut theory = DatatypeTheory::new();
        let tester =
            crate::interner::Spur::try_from_usize(3).expect("test operation should succeed");
        let term = TermId(42);
        let datatype = TermId(10);

        theory.register_tester(term, datatype, tester);
        assert_eq!(theory.testers.get(&term), Some(&(datatype, tester)));
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = DatatypeTheory::new();
        let axiom = DatatypeAxiom::TesterAxiom {
            constructor: TermId(1),
            tester: "mk".to_string(),
        };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = DatatypeTheory::new();
        let cons_name =
            crate::interner::Spur::try_from_usize(1).expect("test operation should succeed");

        theory.register_datatype(TermId(1), SortId(1));
        theory.register_constructor(TermId(2), cons_name, vec![], SortId(1));

        theory.reset();

        assert_eq!(theory.datatypes.len(), 0);
        assert_eq!(theory.constructors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_add_term_generates_the_real_selector_axioms() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);

        assert!(theory.add_term(pair, &manager, &manager.sorts));

        let axioms = theory.take_pending_axioms();
        let selector_axioms: Vec<&DatatypeAxiom> = axioms
            .iter()
            .filter(|axiom| matches!(axiom, DatatypeAxiom::SelectorAxiom { .. }))
            .collect();
        assert_eq!(selector_axioms.len(), 2);

        assert!(axioms.contains(&DatatypeAxiom::SelectorAxiom {
            constructor: pair,
            selector: "first".to_string(),
            field_index: 0,
            field_value: x,
        }));
        assert!(axioms.contains(&DatatypeAxiom::SelectorAxiom {
            constructor: pair,
            selector: "second".to_string(),
            field_index: 1,
            field_value: y,
        }));
    }

    #[test]
    fn test_selector_axiom_term_has_the_expected_shape() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        theory.add_term(pair, &manager, &manager.sorts);

        let axiom = DatatypeAxiom::SelectorAxiom {
            constructor: pair,
            selector: "first".to_string(),
            field_index: 0,
            field_value: x,
        };
        let term = theory
            .axiom_to_term(&axiom, &mut manager)
            .expect("first(mk(x, y)) = x should be buildable");

        let TermKind::Eq(lhs, rhs) =
            manager
                .get(term)
                .map(|t| t.kind.clone())
                .unwrap_or_else(|| {
                    panic!("the axiom term should exist");
                })
        else {
            panic!("expected an equality");
        };
        let (selector_side, value_side) = if lhs == x { (rhs, lhs) } else { (lhs, rhs) };
        assert_eq!(value_side, x);

        match manager.get(selector_side).map(|t| t.kind.clone()) {
            Some(TermKind::DtSelector { selector, arg }) => {
                assert_eq!(manager.resolve_str(selector), "first");
                assert_eq!(arg, pair);
            }
            other => panic!("expected a selector application, got {other:?}"),
        }
    }

    #[test]
    fn test_tester_axioms_have_the_expected_shapes() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        theory.add_term(pair, &manager, &manager.sorts);

        let positive = theory
            .axiom_to_term(
                &DatatypeAxiom::TesterAxiom {
                    constructor: pair,
                    tester: "mk".to_string(),
                },
                &mut manager,
            )
            .expect("is_mk(mk(x, y)) = true should be buildable");
        let truth = manager.mk_true();
        match manager.get(positive).map(|t| t.kind.clone()) {
            Some(TermKind::Eq(lhs, rhs)) => {
                let tester_side = if lhs == truth { rhs } else { lhs };
                assert!(lhs == truth || rhs == truth, "one side should be true");
                assert!(matches!(
                    manager.get(tester_side).map(|t| t.kind.clone()),
                    Some(TermKind::DtTester { .. })
                ));
            }
            other => panic!("expected an equality, got {other:?}"),
        }

        let negative = theory
            .axiom_to_term(
                &DatatypeAxiom::NegativeTesterAxiom {
                    constructor: pair,
                    tester: "none".to_string(),
                },
                &mut manager,
            )
            .expect("not is_none(mk(x, y)) should be buildable");
        match manager.get(negative).map(|t| t.kind.clone()) {
            Some(TermKind::Not(inner)) => {
                assert!(matches!(
                    manager.get(inner).map(|t| t.kind.clone()),
                    Some(TermKind::DtTester { .. })
                ));
            }
            other => panic!("expected a negation, got {other:?}"),
        }
    }

    #[test]
    fn test_distinctness_axiom_is_a_ground_disequality() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        theory.add_term(pair, &manager, &manager.sorts);

        let axioms = theory.take_pending_axioms();
        let distinctness = axioms
            .iter()
            .find(|axiom| matches!(axiom, DatatypeAxiom::ConstructorDistinctness { .. }))
            .expect("mk should be distinct from none");

        let term = theory
            .axiom_to_term(distinctness, &mut manager)
            .expect("the ground instance should be buildable");

        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Not(inner)) => match manager.get(inner).map(|t| t.kind.clone()) {
                Some(TermKind::Eq(lhs, rhs)) => {
                    for side in [lhs, rhs] {
                        assert!(matches!(
                            manager.get(side).map(|t| t.kind.clone()),
                            Some(TermKind::DtConstructor { .. })
                        ));
                    }
                }
                other => panic!("expected an equality under the negation, got {other:?}"),
            },
            other => panic!("expected a negation, got {other:?}"),
        }
    }

    #[test]
    fn test_injectivity_axiom_is_an_implication() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let z = manager.mk_var("z", bv_sort);
        let left = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let right = manager.mk_dt_constructor("mk", [z, y], pair_sort);
        theory.add_term(left, &manager, &manager.sorts);
        theory.add_term(right, &manager, &manager.sorts);

        let term = theory
            .axiom_to_term(
                &DatatypeAxiom::Injectivity {
                    cons1: left,
                    cons2: right,
                    constructor: "mk".to_string(),
                },
                &mut manager,
            )
            .expect("the injectivity instance should be buildable");

        assert!(matches!(
            manager.get(term).map(|t| t.kind.clone()),
            Some(TermKind::Implies(_, _))
        ));
    }

    #[test]
    fn test_acyclicity_axiom_covers_the_nested_subterm() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();

        let list_sort = manager.sorts.mk_datatype_sort("List");
        let int_sort = manager.sorts.int_sort;
        let head = manager.sorts.intern_str("head");
        let tail = manager.sorts.intern_str("tail");
        let cons = manager.sorts.intern_str("cons");
        let nil = manager.sorts.intern_str("nil");
        manager.sorts.declare_datatype(
            "List",
            vec![
                DataTypeConstructor {
                    name: cons,
                    selectors: smallvec::smallvec![(head, int_sort), (tail, list_sort)],
                },
                DataTypeConstructor {
                    name: nil,
                    selectors: smallvec::smallvec![],
                },
            ],
        );

        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let empty = manager.mk_dt_constructor("nil", [], list_sort);
        let inner = manager.mk_dt_constructor("cons", [one, empty], list_sort);
        let outer = manager.mk_dt_constructor("cons", [two, inner], list_sort);

        for term in [empty, inner, outer] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        let term = theory
            .axiom_to_term(&DatatypeAxiom::Acyclicity { term: outer }, &mut manager)
            .expect("outer has proper sub-terms of its own sort");

        // Two disequalities: outer != inner and outer != nil.
        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::And(conjuncts)) => assert_eq!(conjuncts.len(), 2),
            other => panic!("expected a conjunction, got {other:?}"),
        }
    }

    #[test]
    fn test_axiom_to_term_rejects_axioms_that_do_not_fit_their_terms() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let empty = manager.mk_dt_constructor("none", [], pair_sort);
        theory.add_term(pair, &manager, &manager.sorts);
        theory.add_term(empty, &manager, &manager.sorts);
        let _ = theory.take_pending_axioms();

        // is_none(mk(x, y)) is false, so it is not a positive tester axiom.
        assert!(
            theory
                .axiom_to_term(
                    &DatatypeAxiom::TesterAxiom {
                        constructor: pair,
                        tester: "none".to_string(),
                    },
                    &mut manager
                )
                .is_none()
        );

        // ...and not is_mk(mk(x, y)) is false too.
        assert!(
            theory
                .axiom_to_term(
                    &DatatypeAxiom::NegativeTesterAxiom {
                        constructor: pair,
                        tester: "mk".to_string(),
                    },
                    &mut manager
                )
                .is_none()
        );

        // second(mk(x, y)) is y, not x.
        assert!(
            theory
                .axiom_to_term(
                    &DatatypeAxiom::SelectorAxiom {
                        constructor: pair,
                        selector: "second".to_string(),
                        field_index: 0,
                        field_value: x,
                    },
                    &mut manager
                )
                .is_none()
        );

        // mk(x, y) = none says nothing about arguments: the constructors differ.
        assert!(
            theory
                .axiom_to_term(
                    &DatatypeAxiom::Injectivity {
                        cons1: pair,
                        cons2: empty,
                        constructor: "mk".to_string(),
                    },
                    &mut manager
                )
                .is_none()
        );

        // A constructor is not distinct from itself.
        assert!(
            theory
                .axiom_to_term(
                    &DatatypeAxiom::ConstructorDistinctness {
                        cons1: "mk".to_string(),
                        cons2: "mk".to_string(),
                        datatype: pair_sort,
                    },
                    &mut manager
                )
                .is_none()
        );
    }

    #[test]
    fn test_propagate_applies_selectors_and_testers() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let first = manager.mk_dt_selector("first", pair, bv_sort);
        let is_mk = manager.mk_dt_tester("mk", pair);
        let is_none = manager.mk_dt_tester("none", pair);

        for term in [pair, first, is_mk, is_none] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        theory.propagate(&manager);

        assert!(theory.known_equal(first, x));
        let truth = manager.mk_true();
        let falsity = manager.mk_false();
        assert!(theory.known_equal(is_mk, truth));
        assert!(theory.known_equal(is_none, falsity));
    }

    #[test]
    fn test_propagate_applies_injectivity() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let z = manager.mk_var("z", bv_sort);
        let left = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let right = manager.mk_dt_constructor("mk", [z, y], pair_sort);

        theory.add_term(left, &manager, &manager.sorts);
        theory.add_term(right, &manager, &manager.sorts);
        assert!(theory.assert_equality(left, right));

        let deduced = theory.propagate(&manager);
        let expected = if x.0 < z.0 { (x, z) } else { (z, x) };
        assert!(
            deduced.contains(&expected),
            "injectivity should force x = z, got {deduced:?}"
        );
    }

    #[test]
    fn test_conflict_on_two_constructors_in_one_class() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let empty = manager.mk_dt_constructor("none", [], pair_sort);

        theory.add_term(pair, &manager, &manager.sorts);
        theory.add_term(empty, &manager, &manager.sorts);
        assert!(theory.assert_equality(pair, empty));

        let explanation = theory
            .check_for_conflicts(&manager)
            .expect("mk(...) = none is a conflict");
        assert!(explanation.contains(&pair));
        assert!(explanation.contains(&empty));
        assert_eq!(theory.statistics().num_conflicts, 1);
    }

    #[test]
    fn test_equalities_between_unknown_terms_are_ignored() {
        let mut theory = DatatypeTheory::new();
        assert!(!theory.assert_equality(TermId(1), TermId(2)));
    }

    #[test]
    fn test_statistics_counts_deduced_equalities() {
        let mut theory = DatatypeTheory::new();
        let mut manager = TermManager::new();
        let (pair_sort, bv_sort) = declare_pair(&mut manager);

        let x = manager.mk_var("x", bv_sort);
        let y = manager.mk_var("y", bv_sort);
        let pair = manager.mk_dt_constructor("mk", [x, y], pair_sort);
        let second = manager.mk_dt_selector("second", pair, bv_sort);

        theory.add_term(pair, &manager, &manager.sorts);
        theory.add_term(second, &manager, &manager.sorts);
        let deduced = theory.propagate(&manager);

        let stats = theory.statistics();
        assert_eq!(stats.num_constructors, 1);
        assert_eq!(stats.num_selectors, 1);
        assert_eq!(stats.num_propagations, deduced.len());
        assert!(stats.num_equality_nodes >= 3);
    }
}
