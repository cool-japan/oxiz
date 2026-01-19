//! String Automata for Constraint Solving
//!
//! Implements automata-based operations for string constraints:
//! - NFA construction from regex
//! - DFA conversion and minimization
//! - Product automaton for intersection
//! - Automata-based constraint solving

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// State ID
pub type StateId = u32;

/// Transition label
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Label {
    /// Single character
    Char(char),
    /// Character range [lo, hi]
    Range(char, char),
    /// Any character
    Any,
    /// Epsilon (empty) transition
    Epsilon,
}

impl Label {
    /// Check if this label matches a character
    pub fn matches(&self, c: char) -> bool {
        match self {
            Label::Char(ch) => *ch == c,
            Label::Range(lo, hi) => c >= *lo && c <= *hi,
            Label::Any => true,
            Label::Epsilon => false,
        }
    }

    /// Check if this label overlaps with another
    pub fn overlaps(&self, other: &Label) -> bool {
        match (self, other) {
            (Label::Any, _) | (_, Label::Any) => true,
            (Label::Epsilon, _) | (_, Label::Epsilon) => false,
            (Label::Char(a), Label::Char(b)) => a == b,
            (Label::Char(c), Label::Range(lo, hi)) | (Label::Range(lo, hi), Label::Char(c)) => {
                *c >= *lo && *c <= *hi
            }
            (Label::Range(lo1, hi1), Label::Range(lo2, hi2)) => !(*hi1 < *lo2 || *hi2 < *lo1),
        }
    }
}

/// NFA transition
#[derive(Debug, Clone)]
pub struct Transition {
    /// Source state
    pub from: StateId,
    /// Target state
    pub to: StateId,
    /// Label
    pub label: Label,
}

/// Non-deterministic Finite Automaton
#[derive(Debug, Clone)]
pub struct Nfa {
    /// Initial state
    pub initial: StateId,
    /// Accepting states
    pub accepting: FxHashSet<StateId>,
    /// Transitions grouped by source state
    transitions: FxHashMap<StateId, Vec<Transition>>,
    /// Number of states
    num_states: StateId,
}

impl Nfa {
    /// Create a new NFA with a single state
    pub fn new() -> Self {
        Self {
            initial: 0,
            accepting: FxHashSet::default(),
            transitions: FxHashMap::default(),
            num_states: 1,
        }
    }

    /// Create NFA accepting empty string
    pub fn epsilon() -> Self {
        let mut nfa = Self::new();
        nfa.accepting.insert(0);
        nfa
    }

    /// Create NFA accepting no strings
    pub fn empty() -> Self {
        Self::new()
    }

    /// Create NFA accepting single character
    pub fn char(c: char) -> Self {
        let mut nfa = Self::new();
        nfa.num_states = 2;
        nfa.add_transition(0, 1, Label::Char(c));
        nfa.accepting.insert(1);
        nfa
    }

    /// Create NFA accepting character range
    pub fn range(lo: char, hi: char) -> Self {
        let mut nfa = Self::new();
        nfa.num_states = 2;
        nfa.add_transition(0, 1, Label::Range(lo, hi));
        nfa.accepting.insert(1);
        nfa
    }

    /// Create NFA accepting any single character
    pub fn any_char() -> Self {
        let mut nfa = Self::new();
        nfa.num_states = 2;
        nfa.add_transition(0, 1, Label::Any);
        nfa.accepting.insert(1);
        nfa
    }

    /// Add a state and return its ID
    pub fn add_state(&mut self) -> StateId {
        let id = self.num_states;
        self.num_states += 1;
        id
    }

    /// Add a transition
    pub fn add_transition(&mut self, from: StateId, to: StateId, label: Label) {
        self.transitions
            .entry(from)
            .or_default()
            .push(Transition { from, to, label });
    }

    /// Get transitions from a state
    pub fn transitions_from(&self, state: StateId) -> &[Transition] {
        self.transitions
            .get(&state)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Concatenation: self followed by other
    pub fn concat(mut self, mut other: Nfa) -> Nfa {
        let offset = self.num_states;

        // Remap other's states
        other.initial += offset;
        let new_accepting: FxHashSet<_> = other.accepting.iter().map(|s| s + offset).collect();

        // Add epsilon transitions from self's accepting to other's initial
        let accepting_copy: Vec<_> = self.accepting.iter().copied().collect();
        for acc in accepting_copy {
            self.add_transition(acc, other.initial, Label::Epsilon);
        }

        // Move other's transitions
        for (from, trans) in other.transitions {
            for t in trans {
                self.add_transition(from + offset, t.to + offset, t.label);
            }
        }

        self.accepting = new_accepting;
        self.num_states += other.num_states;
        self
    }

    /// Union: accept either self or other
    pub fn union(mut self, mut other: Nfa) -> Nfa {
        let new_initial = self.num_states + other.num_states;

        // Offset other's states
        let offset = self.num_states;
        other.initial += offset;

        // Add epsilon transitions from new initial
        let old_initial = self.initial;
        self.add_transition(new_initial, old_initial, Label::Epsilon);
        self.add_transition(new_initial, other.initial, Label::Epsilon);

        // Move other's transitions
        for (from, trans) in other.transitions {
            for t in trans {
                self.add_transition(from + offset, t.to + offset, t.label);
            }
        }

        // Union accepting states
        for &acc in &other.accepting {
            self.accepting.insert(acc + offset);
        }

        self.initial = new_initial;
        self.num_states = new_initial + 1;
        self
    }

    /// Kleene star: zero or more repetitions
    pub fn star(mut self) -> Nfa {
        let new_initial = self.num_states;
        let new_final = self.num_states + 1;

        // Epsilon from new initial to old initial and new final
        let old_initial = self.initial;
        self.add_transition(new_initial, old_initial, Label::Epsilon);
        self.add_transition(new_initial, new_final, Label::Epsilon);

        // Epsilon from old accepting to old initial and new final
        let accepting_copy: Vec<_> = self.accepting.iter().copied().collect();
        for acc in accepting_copy {
            self.add_transition(acc, old_initial, Label::Epsilon);
            self.add_transition(acc, new_final, Label::Epsilon);
        }

        self.initial = new_initial;
        self.accepting.clear();
        self.accepting.insert(new_final);
        self.num_states += 2;
        self
    }

    /// Kleene plus: one or more repetitions
    pub fn plus(self) -> Nfa {
        let copy = self.clone();
        self.concat(copy.star())
    }

    /// Optional: zero or one
    pub fn optional(mut self) -> Nfa {
        // Add epsilon from initial to all accepting states
        let initial = self.initial;
        self.accepting.insert(initial);
        self
    }

    /// Compute epsilon closure of a set of states
    pub fn epsilon_closure(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut closure = states.clone();
        let mut worklist: Vec<_> = states.iter().copied().collect();

        while let Some(state) = worklist.pop() {
            for trans in self.transitions_from(state) {
                if trans.label == Label::Epsilon && !closure.contains(&trans.to) {
                    closure.insert(trans.to);
                    worklist.push(trans.to);
                }
            }
        }

        closure
    }

    /// Check if NFA accepts a string
    pub fn accepts(&self, s: &str) -> bool {
        let mut current = FxHashSet::default();
        current.insert(self.initial);
        let mut current = self.epsilon_closure(&current);

        for c in s.chars() {
            let mut next = FxHashSet::default();
            for &state in &current {
                for trans in self.transitions_from(state) {
                    if trans.label.matches(c) {
                        next.insert(trans.to);
                    }
                }
            }
            current = self.epsilon_closure(&next);
        }

        current.iter().any(|s| self.accepting.contains(s))
    }

    /// Number of states
    pub fn num_states(&self) -> StateId {
        self.num_states
    }

    /// Check if accepting a state
    pub fn is_accepting(&self, state: StateId) -> bool {
        self.accepting.contains(&state)
    }
}

impl Default for Nfa {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic Finite Automaton
#[derive(Debug, Clone)]
pub struct Dfa {
    /// Initial state
    pub initial: StateId,
    /// Accepting states
    pub accepting: FxHashSet<StateId>,
    /// Transitions: state -> char -> state
    transitions: FxHashMap<StateId, FxHashMap<char, StateId>>,
    /// Default transitions (for any char)
    default_trans: FxHashMap<StateId, StateId>,
    /// Number of states
    num_states: StateId,
    /// Dead state (sink)
    #[allow(dead_code)]
    dead_state: Option<StateId>,
}

impl Dfa {
    /// Create a new DFA
    pub fn new() -> Self {
        Self {
            initial: 0,
            accepting: FxHashSet::default(),
            transitions: FxHashMap::default(),
            default_trans: FxHashMap::default(),
            num_states: 1,
            dead_state: None,
        }
    }

    /// Add a state
    pub fn add_state(&mut self) -> StateId {
        let id = self.num_states;
        self.num_states += 1;
        id
    }

    /// Add a transition
    pub fn add_transition(&mut self, from: StateId, c: char, to: StateId) {
        self.transitions.entry(from).or_default().insert(c, to);
    }

    /// Add a default transition (any char)
    pub fn add_default_transition(&mut self, from: StateId, to: StateId) {
        self.default_trans.insert(from, to);
    }

    /// Get next state for a character
    pub fn next_state(&self, state: StateId, c: char) -> Option<StateId> {
        self.transitions
            .get(&state)
            .and_then(|m| m.get(&c).copied())
            .or_else(|| self.default_trans.get(&state).copied())
    }

    /// Check if DFA accepts a string
    pub fn accepts(&self, s: &str) -> bool {
        let mut state = self.initial;
        for c in s.chars() {
            match self.next_state(state, c) {
                Some(next) => state = next,
                None => return false,
            }
        }
        self.accepting.contains(&state)
    }

    /// Convert from NFA using subset construction
    pub fn from_nfa(nfa: &Nfa) -> Self {
        let mut dfa = Dfa::new();
        let mut state_map: FxHashMap<Vec<StateId>, StateId> = FxHashMap::default();
        let mut worklist: VecDeque<Vec<StateId>> = VecDeque::new();

        // Initial state is epsilon closure of NFA initial
        let mut initial_set = FxHashSet::default();
        initial_set.insert(nfa.initial);
        let initial_closure = nfa.epsilon_closure(&initial_set);
        let initial_vec: Vec<_> = {
            let mut v: Vec<_> = initial_closure.iter().copied().collect();
            v.sort_unstable();
            v
        };

        state_map.insert(initial_vec.clone(), 0);
        worklist.push_back(initial_vec.clone());

        if initial_closure.iter().any(|s| nfa.is_accepting(*s)) {
            dfa.accepting.insert(0);
        }

        // Collect all possible characters from transitions
        let mut alphabet: FxHashSet<char> = FxHashSet::default();
        for trans_list in nfa.transitions.values() {
            for t in trans_list {
                match &t.label {
                    Label::Char(c) => {
                        alphabet.insert(*c);
                    }
                    Label::Range(lo, hi) => {
                        for c in *lo..=*hi {
                            alphabet.insert(c);
                            if alphabet.len() > 256 {
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Limit alphabet size for practical purposes
        if alphabet.is_empty() {
            alphabet.insert('a');
        }

        while let Some(current) = worklist.pop_front() {
            let current_dfa_state = *state_map.get(&current).unwrap_or(&0);
            let current_set: FxHashSet<_> = current.iter().copied().collect();

            for &c in &alphabet {
                let mut next_set = FxHashSet::default();

                for &nfa_state in &current_set {
                    for trans in nfa.transitions_from(nfa_state) {
                        if trans.label.matches(c) {
                            next_set.insert(trans.to);
                        }
                    }
                }

                let next_closure = nfa.epsilon_closure(&next_set);
                if next_closure.is_empty() {
                    continue;
                }

                let next_vec: Vec<_> = {
                    let mut v: Vec<_> = next_closure.iter().copied().collect();
                    v.sort_unstable();
                    v
                };

                let next_dfa_state = if let Some(&s) = state_map.get(&next_vec) {
                    s
                } else {
                    let s = dfa.add_state();
                    state_map.insert(next_vec.clone(), s);
                    worklist.push_back(next_vec.clone());

                    if next_closure.iter().any(|s| nfa.is_accepting(*s)) {
                        dfa.accepting.insert(s);
                    }
                    s
                };

                dfa.add_transition(current_dfa_state, c, next_dfa_state);
            }
        }

        dfa
    }

    /// Minimize DFA using Hopcroft's algorithm
    pub fn minimize(&self) -> Self {
        if self.num_states <= 1 {
            return self.clone();
        }

        // Partition refinement
        let mut partitions: Vec<FxHashSet<StateId>> = vec![
            self.accepting.clone(),
            (0..self.num_states)
                .filter(|s| !self.accepting.contains(s))
                .collect(),
        ];

        partitions.retain(|p| !p.is_empty());

        let mut changed = true;
        while changed {
            changed = false;
            let mut new_partitions = Vec::new();

            for partition in &partitions {
                if partition.len() <= 1 {
                    new_partitions.push(partition.clone());
                    continue;
                }

                // Try to split partition
                let first = *partition.iter().next().expect("partition non-empty");
                let mut same = FxHashSet::default();
                let mut different = FxHashSet::default();
                same.insert(first);

                for &state in partition {
                    if state == first {
                        continue;
                    }

                    // Check if state behaves same as first
                    let same_behavior = self.same_behavior(first, state, &partitions);
                    if same_behavior {
                        same.insert(state);
                    } else {
                        different.insert(state);
                    }
                }

                if different.is_empty() {
                    new_partitions.push(partition.clone());
                } else {
                    new_partitions.push(same);
                    new_partitions.push(different);
                    changed = true;
                }
            }

            partitions = new_partitions;
        }

        // Build minimized DFA
        let mut min_dfa = Dfa::new();
        let mut state_to_partition: FxHashMap<StateId, usize> = FxHashMap::default();

        for (i, partition) in partitions.iter().enumerate() {
            for &state in partition {
                state_to_partition.insert(state, i);
            }
        }

        // Create states
        min_dfa.num_states = partitions.len() as StateId;

        // Set initial and accepting
        min_dfa.initial = state_to_partition[&self.initial] as StateId;
        for &acc in &self.accepting {
            min_dfa
                .accepting
                .insert(state_to_partition[&acc] as StateId);
        }

        // Copy transitions (taking representative from each partition)
        for (i, partition) in partitions.iter().enumerate() {
            if let Some(&rep) = partition.iter().next() {
                if let Some(trans) = self.transitions.get(&rep) {
                    for (&c, &to) in trans {
                        let min_to = state_to_partition[&to] as StateId;
                        min_dfa.add_transition(i as StateId, c, min_to);
                    }
                }
                if let Some(&default_to) = self.default_trans.get(&rep) {
                    min_dfa.add_default_transition(
                        i as StateId,
                        state_to_partition[&default_to] as StateId,
                    );
                }
            }
        }

        min_dfa
    }

    /// Check if two states have the same behavior
    fn same_behavior(&self, s1: StateId, s2: StateId, partitions: &[FxHashSet<StateId>]) -> bool {
        // Get all characters that have transitions
        let chars1: FxHashSet<_> = self
            .transitions
            .get(&s1)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default();
        let chars2: FxHashSet<_> = self
            .transitions
            .get(&s2)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default();

        let all_chars: FxHashSet<_> = chars1.union(&chars2).copied().collect();

        for c in all_chars {
            let next1 = self.next_state(s1, c);
            let next2 = self.next_state(s2, c);

            match (next1, next2) {
                (None, None) => {}
                (Some(n1), Some(n2)) => {
                    // Check if n1 and n2 are in the same partition
                    let mut same = false;
                    for partition in partitions {
                        if partition.contains(&n1) && partition.contains(&n2) {
                            same = true;
                            break;
                        }
                    }
                    if !same {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }

    /// Number of states
    pub fn num_states(&self) -> StateId {
        self.num_states
    }
}

impl Default for Dfa {
    fn default() -> Self {
        Self::new()
    }
}

/// Product automaton for intersection
#[derive(Debug)]
#[allow(dead_code)]
pub struct ProductAutomaton {
    /// DFA 1
    dfa1: Dfa,
    /// DFA 2
    dfa2: Dfa,
    /// Product DFA
    product: Dfa,
    /// State mapping: product state -> (dfa1 state, dfa2 state)
    state_map: FxHashMap<StateId, (StateId, StateId)>,
}

impl ProductAutomaton {
    /// Build product automaton for intersection
    pub fn intersection(dfa1: Dfa, dfa2: Dfa) -> Self {
        let mut product = Dfa::new();
        let mut state_map: FxHashMap<StateId, (StateId, StateId)> = FxHashMap::default();
        let mut reverse_map: FxHashMap<(StateId, StateId), StateId> = FxHashMap::default();
        let mut worklist: VecDeque<(StateId, StateId)> = VecDeque::new();

        // Initial state
        let initial = (dfa1.initial, dfa2.initial);
        state_map.insert(0, initial);
        reverse_map.insert(initial, 0);
        worklist.push_back(initial);

        if dfa1.accepting.contains(&dfa1.initial) && dfa2.accepting.contains(&dfa2.initial) {
            product.accepting.insert(0);
        }

        // Collect alphabet
        let mut alphabet: FxHashSet<char> = FxHashSet::default();
        for trans in dfa1.transitions.values() {
            alphabet.extend(trans.keys());
        }
        for trans in dfa2.transitions.values() {
            alphabet.extend(trans.keys());
        }

        if alphabet.is_empty() {
            alphabet.insert('a');
        }

        while let Some((s1, s2)) = worklist.pop_front() {
            let current = reverse_map[&(s1, s2)];

            for &c in &alphabet {
                let next1 = dfa1.next_state(s1, c);
                let next2 = dfa2.next_state(s2, c);

                if let (Some(n1), Some(n2)) = (next1, next2) {
                    let next = (n1, n2);
                    let next_state = if let Some(&s) = reverse_map.get(&next) {
                        s
                    } else {
                        let s = product.add_state();
                        state_map.insert(s, next);
                        reverse_map.insert(next, s);
                        worklist.push_back(next);

                        if dfa1.accepting.contains(&n1) && dfa2.accepting.contains(&n2) {
                            product.accepting.insert(s);
                        }
                        s
                    };

                    product.add_transition(current, c, next_state);
                }
            }
        }

        Self {
            dfa1,
            dfa2,
            product,
            state_map,
        }
    }

    /// Check if intersection is empty
    pub fn is_empty(&self) -> bool {
        self.product.accepting.is_empty()
    }

    /// Get the product DFA
    pub fn product(&self) -> &Dfa {
        &self.product
    }

    /// Check if a string is accepted
    pub fn accepts(&self, s: &str) -> bool {
        self.product.accepts(s)
    }
}

/// String constraint automaton
#[derive(Debug)]
pub struct ConstraintAutomaton {
    /// The underlying DFA
    dfa: Dfa,
    /// Length bounds (min, max)
    length_bounds: Option<(usize, Option<usize>)>,
    /// Prefix constraints
    prefixes: Vec<String>,
    /// Suffix constraints
    suffixes: Vec<String>,
}

impl ConstraintAutomaton {
    /// Create from DFA
    pub fn from_dfa(dfa: Dfa) -> Self {
        Self {
            dfa,
            length_bounds: None,
            prefixes: Vec::new(),
            suffixes: Vec::new(),
        }
    }

    /// Add length bounds
    pub fn with_length_bounds(mut self, min: usize, max: Option<usize>) -> Self {
        self.length_bounds = Some((min, max));
        self
    }

    /// Add prefix constraint
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefixes.push(prefix);
        self
    }

    /// Add suffix constraint
    pub fn with_suffix(mut self, suffix: String) -> Self {
        self.suffixes.push(suffix);
        self
    }

    /// Check if a string satisfies all constraints
    pub fn accepts(&self, s: &str) -> bool {
        // Check length
        if let Some((min, max)) = &self.length_bounds {
            if s.len() < *min {
                return false;
            }
            if let Some(max) = max
                && s.len() > *max
            {
                return false;
            }
        }

        // Check prefixes
        for prefix in &self.prefixes {
            if !s.starts_with(prefix) {
                return false;
            }
        }

        // Check suffixes
        for suffix in &self.suffixes {
            if !s.ends_with(suffix) {
                return false;
            }
        }

        // Check DFA
        self.dfa.accepts(s)
    }

    /// Generate a sample string (if possible)
    pub fn sample(&self) -> Option<String> {
        // BFS to find shortest accepting path
        let mut visited: FxHashSet<StateId> = FxHashSet::default();
        let mut queue: VecDeque<(StateId, String)> = VecDeque::new();

        queue.push_back((self.dfa.initial, String::new()));
        visited.insert(self.dfa.initial);

        while let Some((state, path)) = queue.pop_front() {
            // Check length constraints
            if let Some((_, Some(max))) = &self.length_bounds
                && path.len() > *max
            {
                continue;
            }

            if self.accepts(&path) {
                return Some(path);
            }

            // Try all transitions
            if let Some(trans) = self.dfa.transitions.get(&state) {
                for (&c, &next) in trans {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        let mut new_path = path.clone();
                        new_path.push(c);
                        queue.push_back((next, new_path));
                    }
                }
            }
        }

        None
    }

    /// Check if constraint is satisfiable
    pub fn is_satisfiable(&self) -> bool {
        self.sample().is_some()
    }

    /// Get the underlying DFA
    pub fn dfa(&self) -> &Dfa {
        &self.dfa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfa_epsilon() {
        let nfa = Nfa::epsilon();
        assert!(nfa.accepts(""));
        assert!(!nfa.accepts("a"));
    }

    #[test]
    fn test_nfa_char() {
        let nfa = Nfa::char('a');
        assert!(nfa.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts("aa"));
    }

    #[test]
    fn test_nfa_concat() {
        let a = Nfa::char('a');
        let b = Nfa::char('b');
        let ab = a.concat(b);
        assert!(ab.accepts("ab"));
        assert!(!ab.accepts("a"));
        assert!(!ab.accepts("b"));
        assert!(!ab.accepts("ba"));
    }

    #[test]
    fn test_nfa_union() {
        let a = Nfa::char('a');
        let b = Nfa::char('b');
        let ab = a.union(b);
        assert!(ab.accepts("a"));
        assert!(ab.accepts("b"));
        assert!(!ab.accepts("ab"));
    }

    #[test]
    fn test_nfa_star() {
        let a = Nfa::char('a');
        let a_star = a.star();
        assert!(a_star.accepts(""));
        assert!(a_star.accepts("a"));
        assert!(a_star.accepts("aa"));
        assert!(a_star.accepts("aaa"));
        assert!(!a_star.accepts("b"));
    }

    #[test]
    fn test_nfa_plus() {
        let a = Nfa::char('a');
        let a_plus = a.plus();
        assert!(!a_plus.accepts(""));
        assert!(a_plus.accepts("a"));
        assert!(a_plus.accepts("aa"));
    }

    #[test]
    fn test_nfa_optional() {
        let a = Nfa::char('a');
        let a_opt = a.optional();
        assert!(a_opt.accepts(""));
        assert!(a_opt.accepts("a"));
        assert!(!a_opt.accepts("aa"));
    }

    #[test]
    fn test_dfa_from_nfa() {
        let a = Nfa::char('a');
        let b = Nfa::char('b');
        let ab = a.union(b);
        let dfa = Dfa::from_nfa(&ab);
        assert!(dfa.accepts("a"));
        assert!(dfa.accepts("b"));
        assert!(!dfa.accepts("c"));
    }

    #[test]
    fn test_dfa_accepts() {
        let mut dfa = Dfa::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 'a', s1);
        dfa.accepting.insert(s1);

        assert!(dfa.accepts("a"));
        assert!(!dfa.accepts(""));
        assert!(!dfa.accepts("aa"));
    }

    #[test]
    fn test_product_automaton() {
        // DFA accepting strings starting with 'a'
        let mut dfa1 = Dfa::new();
        let s1 = dfa1.add_state();
        dfa1.add_transition(0, 'a', s1);
        dfa1.add_default_transition(s1, s1);
        dfa1.accepting.insert(s1);

        // DFA accepting strings ending with 'b'
        let mut dfa2 = Dfa::new();
        let s1 = dfa2.add_state();
        dfa2.add_transition(0, 'b', s1);
        dfa2.add_transition(0, 'a', 0);
        dfa2.add_transition(s1, 'b', s1);
        dfa2.add_transition(s1, 'a', 0);
        dfa2.accepting.insert(s1);

        let product = ProductAutomaton::intersection(dfa1, dfa2);
        assert!(product.accepts("ab"));
        assert!(!product.accepts("a"));
        assert!(!product.accepts("b"));
    }

    #[test]
    fn test_constraint_automaton() {
        let mut dfa = Dfa::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 'a', s1);
        dfa.add_default_transition(s1, s1);
        dfa.accepting.insert(s1);

        let ca = ConstraintAutomaton::from_dfa(dfa)
            .with_length_bounds(2, Some(5))
            .with_prefix("a".to_string());

        assert!(!ca.accepts("a"));
        assert!(ca.accepts("ab"));
        assert!(!ca.accepts("ba"));
    }

    #[test]
    fn test_label_matches() {
        assert!(Label::Char('a').matches('a'));
        assert!(!Label::Char('a').matches('b'));
        assert!(Label::Range('a', 'z').matches('m'));
        assert!(!Label::Range('a', 'z').matches('A'));
        assert!(Label::Any.matches('x'));
    }

    #[test]
    fn test_label_overlaps() {
        assert!(Label::Char('a').overlaps(&Label::Char('a')));
        assert!(!Label::Char('a').overlaps(&Label::Char('b')));
        assert!(Label::Char('m').overlaps(&Label::Range('a', 'z')));
        assert!(Label::Range('a', 'e').overlaps(&Label::Range('c', 'h')));
        assert!(!Label::Range('a', 'e').overlaps(&Label::Range('f', 'z')));
    }

    #[test]
    fn test_dfa_minimize() {
        // Create a DFA with redundant states
        let a = Nfa::char('a');
        let ab = a.star();
        let dfa = Dfa::from_nfa(&ab);
        let min = dfa.minimize();

        // Should accept same language
        assert!(min.accepts(""));
        assert!(min.accepts("a"));
        assert!(min.accepts("aaa"));
    }

    #[test]
    fn test_nfa_range() {
        let nfa = Nfa::range('a', 'c');
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("c"));
        assert!(!nfa.accepts("d"));
    }

    #[test]
    fn test_constraint_automaton_sample() {
        let mut dfa = Dfa::new();
        let s1 = dfa.add_state();
        dfa.add_transition(0, 'h', s1);
        dfa.accepting.insert(s1);

        let ca = ConstraintAutomaton::from_dfa(dfa);
        let sample = ca.sample();
        assert!(sample.is_some());
        assert_eq!(sample.unwrap(), "h");
    }
}
