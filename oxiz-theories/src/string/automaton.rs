//! Finite Automaton for String Theory.
//!
//! Implements:
//! - NFA (Nondeterministic Finite Automaton)
//! - DFA (Deterministic Finite Automaton)
//! - NFA to DFA conversion (subset construction)
//! - Automaton operations (union, concatenation, Kleene star)
//! - Language membership testing

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Finite automaton for string constraints.
#[derive(Debug, Clone)]
pub struct Automaton {
    /// States
    pub states: FxHashSet<StateId>,
    /// Initial state
    pub initial: StateId,
    /// Accepting states
    pub accepting: FxHashSet<StateId>,
    /// Transitions: (from, symbol) → to_states
    pub transitions: FxHashMap<(StateId, Symbol), FxHashSet<StateId>>,
    /// Whether this is a DFA (deterministic)
    pub is_deterministic: bool,
}

/// State identifier.
pub type StateId = usize;

/// Symbol in the alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// Character
    Char(char),
    /// Epsilon transition (empty string)
    Epsilon,
    /// Any character (wildcard)
    Any,
}

/// Automaton builder for constructing automata from regex patterns.
pub struct AutomatonBuilder {
    /// Next available state ID
    next_state: StateId,
}

impl Automaton {
    /// Create a new empty automaton.
    pub fn new(initial: StateId) -> Self {
        let mut states = FxHashSet::default();
        states.insert(initial);

        Self {
            states,
            initial,
            accepting: FxHashSet::default(),
            transitions: FxHashMap::default(),
            is_deterministic: true,
        }
    }

    /// Add a state.
    pub fn add_state(&mut self, state: StateId) {
        self.states.insert(state);
    }

    /// Mark a state as accepting.
    pub fn mark_accepting(&mut self, state: StateId) {
        self.accepting.insert(state);
    }

    /// Add a transition.
    pub fn add_transition(&mut self, from: StateId, symbol: Symbol, to: StateId) {
        if symbol == Symbol::Epsilon {
            self.is_deterministic = false;
        }

        self.states.insert(from);
        self.states.insert(to);

        self.transitions
            .entry((from, symbol))
            .or_insert_with(FxHashSet::default)
            .insert(to);

        // Check determinism
        if self.transitions.get(&(from, symbol)).map_or(0, |s| s.len()) > 1 {
            self.is_deterministic = false;
        }
    }

    /// Check if a string is accepted by the automaton.
    pub fn accepts(&self, input: &str) -> bool {
        if self.is_deterministic {
            self.accepts_dfa(input)
        } else {
            self.accepts_nfa(input)
        }
    }

    /// DFA acceptance check.
    fn accepts_dfa(&self, input: &str) -> bool {
        let mut current = self.initial;

        for ch in input.chars() {
            // Try character transition
            if let Some(next_states) = self.transitions.get(&(current, Symbol::Char(ch))) {
                if let Some(&next) = next_states.iter().next() {
                    current = next;
                    continue;
                }
            }

            // Try wildcard transition
            if let Some(next_states) = self.transitions.get(&(current, Symbol::Any)) {
                if let Some(&next) = next_states.iter().next() {
                    current = next;
                    continue;
                }
            }

            // No transition found
            return false;
        }

        self.accepting.contains(&current)
    }

    /// NFA acceptance check (with epsilon closure).
    fn accepts_nfa(&self, input: &str) -> bool {
        let mut current_states = FxHashSet::default();
        current_states.insert(self.initial);

        // Epsilon closure of initial state
        current_states = self.epsilon_closure(&current_states);

        for ch in input.chars() {
            let mut next_states = FxHashSet::default();

            for &state in &current_states {
                // Character transitions
                if let Some(dests) = self.transitions.get(&(state, Symbol::Char(ch))) {
                    for &dest in dests {
                        next_states.insert(dest);
                    }
                }

                // Wildcard transitions
                if let Some(dests) = self.transitions.get(&(state, Symbol::Any)) {
                    for &dest in dests {
                        next_states.insert(dest);
                    }
                }
            }

            // Epsilon closure
            next_states = self.epsilon_closure(&next_states);
            current_states = next_states;

            if current_states.is_empty() {
                return false;
            }
        }

        // Check if any current state is accepting
        current_states.iter().any(|s| self.accepting.contains(s))
    }

    /// Compute epsilon closure of a set of states.
    fn epsilon_closure(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<StateId> = states.iter().copied().collect();

        while let Some(state) = queue.pop_front() {
            if let Some(eps_dests) = self.transitions.get(&(state, Symbol::Epsilon)) {
                for &dest in eps_dests {
                    if closure.insert(dest) {
                        queue.push_back(dest);
                    }
                }
            }
        }

        closure
    }

    /// Convert NFA to DFA using subset construction.
    pub fn to_dfa(&self) -> Automaton {
        if self.is_deterministic {
            return self.clone();
        }

        let mut dfa = Automaton::new(0);
        dfa.is_deterministic = true;

        // Map from NFA state sets to DFA states
        let mut state_map: FxHashMap<Vec<StateId>, StateId> = FxHashMap::default();
        let mut next_state_id = 0;

        // Initial DFA state
        let initial_closure = self.epsilon_closure(&{
            let mut s = FxHashSet::default();
            s.insert(self.initial);
            s
        });

        let mut initial_sorted: Vec<StateId> = initial_closure.iter().copied().collect();
        initial_sorted.sort_unstable();

        state_map.insert(initial_sorted.clone(), next_state_id);
        dfa.initial = next_state_id;
        next_state_id += 1;

        // Check if initial is accepting
        if initial_sorted.iter().any(|s| self.accepting.contains(s)) {
            dfa.mark_accepting(dfa.initial);
        }

        // Queue of DFA states to process
        let mut queue: VecDeque<Vec<StateId>> = VecDeque::new();
        queue.push_back(initial_sorted);

        // Collect alphabet (all non-epsilon symbols)
        let alphabet = self.get_alphabet();

        while let Some(nfa_states) = queue.pop_front() {
            let from_state = state_map[&nfa_states];

            for &symbol in &alphabet {
                // Compute next NFA states
                let mut next_nfa_states = FxHashSet::default();

                for &nfa_state in &nfa_states {
                    if let Some(dests) = self.transitions.get(&(nfa_state, symbol)) {
                        for &dest in dests {
                            next_nfa_states.insert(dest);
                        }
                    }
                }

                // Epsilon closure
                let next_closure = self.epsilon_closure(&next_nfa_states);

                if next_closure.is_empty() {
                    continue;
                }

                let mut next_sorted: Vec<StateId> = next_closure.iter().copied().collect();
                next_sorted.sort_unstable();

                // Get or create DFA state for this NFA state set
                let to_state = if let Some(&existing) = state_map.get(&next_sorted) {
                    existing
                } else {
                    let new_state = next_state_id;
                    next_state_id += 1;
                    state_map.insert(next_sorted.clone(), new_state);
                    dfa.add_state(new_state);

                    // Check if accepting
                    if next_sorted.iter().any(|s| self.accepting.contains(s)) {
                        dfa.mark_accepting(new_state);
                    }

                    queue.push_back(next_sorted);
                    new_state
                };

                // Add transition in DFA
                dfa.add_transition(from_state, symbol, to_state);
            }
        }

        dfa
    }

    /// Get alphabet (all non-epsilon symbols used).
    fn get_alphabet(&self) -> FxHashSet<Symbol> {
        let mut alphabet = FxHashSet::default();

        for ((_, symbol), _) in &self.transitions {
            if *symbol != Symbol::Epsilon {
                alphabet.insert(*symbol);
            }
        }

        alphabet
    }

    /// Compute union of two automata: L(a1) ∪ L(a2).
    pub fn union(&self, other: &Automaton) -> Automaton {
        let mut result = Automaton::new(0);
        result.is_deterministic = false;

        let offset1 = 1; // Offset for first automaton states
        let offset2 = 1 + self.states.len(); // Offset for second automaton states

        // Add epsilon transitions from new initial to both automata
        result.add_transition(0, Symbol::Epsilon, self.initial + offset1);
        result.add_transition(0, Symbol::Epsilon, other.initial + offset2);

        // Copy first automaton
        for &state in &self.states {
            result.add_state(state + offset1);
        }
        for &acc in &self.accepting {
            result.mark_accepting(acc + offset1);
        }
        for ((from, sym), dests) in &self.transitions {
            for &to in dests {
                result.add_transition(*from + offset1, *sym, to + offset1);
            }
        }

        // Copy second automaton
        for &state in &other.states {
            result.add_state(state + offset2);
        }
        for &acc in &other.accepting {
            result.mark_accepting(acc + offset2);
        }
        for ((from, sym), dests) in &other.transitions {
            for &to in dests {
                result.add_transition(*from + offset2, *sym, to + offset2);
            }
        }

        result
    }

    /// Compute concatenation: L(a1) · L(a2).
    pub fn concatenation(&self, other: &Automaton) -> Automaton {
        let mut result = self.clone();
        result.is_deterministic = false;

        let offset = self.states.len();

        // Copy second automaton
        for &state in &other.states {
            result.add_state(state + offset);
        }
        for ((from, sym), dests) in &other.transitions {
            for &to in dests {
                result.add_transition(*from + offset, *sym, to + offset);
            }
        }

        // Connect accepting states of first to initial of second
        for &acc in &self.accepting {
            result.add_transition(acc, Symbol::Epsilon, other.initial + offset);
        }

        // Update accepting states
        result.accepting.clear();
        for &acc in &other.accepting {
            result.mark_accepting(acc + offset);
        }

        result
    }

    /// Compute Kleene star: L(a)*.
    pub fn kleene_star(&self) -> Automaton {
        let mut result = Automaton::new(0);
        result.is_deterministic = false;
        result.mark_accepting(0); // Empty string is accepted

        let offset = 1;

        // Copy automaton
        for &state in &self.states {
            result.add_state(state + offset);
        }
        for ((from, sym), dests) in &self.transitions {
            for &to in dests {
                result.add_transition(*from + offset, *sym, to + offset);
            }
        }

        // Add epsilon from new initial to old initial
        result.add_transition(0, Symbol::Epsilon, self.initial + offset);

        // Add epsilon from accepting states back to initial
        for &acc in &self.accepting {
            result.add_transition(acc + offset, Symbol::Epsilon, self.initial + offset);
            result.mark_accepting(acc + offset);
        }

        result
    }
}

impl AutomatonBuilder {
    /// Create a new automaton builder.
    pub fn new() -> Self {
        Self { next_state: 0 }
    }

    /// Get next state ID.
    fn next_state(&mut self) -> StateId {
        let id = self.next_state;
        self.next_state += 1;
        id
    }

    /// Build automaton for single character.
    pub fn char_automaton(&mut self, ch: char) -> Automaton {
        let start = self.next_state();
        let end = self.next_state();

        let mut nfa = Automaton::new(start);
        nfa.add_transition(start, Symbol::Char(ch), end);
        nfa.mark_accepting(end);

        nfa
    }

    /// Build automaton for empty string.
    pub fn empty_automaton(&mut self) -> Automaton {
        let state = self.next_state();
        let mut nfa = Automaton::new(state);
        nfa.mark_accepting(state);
        nfa
    }
}

impl Default for AutomatonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automaton_creation() {
        let nfa = Automaton::new(0);
        assert_eq!(nfa.initial, 0);
        assert!(nfa.accepting.is_empty());
    }

    #[test]
    fn test_single_char() {
        let mut builder = AutomatonBuilder::new();
        let nfa = builder.char_automaton('a');

        assert!(nfa.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts("aa"));
    }

    #[test]
    fn test_union() {
        let mut builder = AutomatonBuilder::new();
        let a = builder.char_automaton('a');
        let b = builder.char_automaton('b');

        let union = a.union(&b);

        assert!(union.accepts("a"));
        assert!(union.accepts("b"));
        assert!(!union.accepts("c"));
    }

    #[test]
    fn test_concatenation() {
        let mut builder = AutomatonBuilder::new();
        let a = builder.char_automaton('a');
        let b = builder.char_automaton('b');

        let concat = a.concatenation(&b);

        assert!(concat.accepts("ab"));
        assert!(!concat.accepts("a"));
        assert!(!concat.accepts("b"));
        assert!(!concat.accepts("ba"));
    }

    #[test]
    fn test_kleene_star() {
        let mut builder = AutomatonBuilder::new();
        let a = builder.char_automaton('a');

        let star = a.kleene_star();

        assert!(star.accepts(""));
        assert!(star.accepts("a"));
        assert!(star.accepts("aa"));
        assert!(star.accepts("aaa"));
        assert!(!star.accepts("b"));
    }

    #[test]
    fn test_nfa_to_dfa() {
        let mut nfa = Automaton::new(0);
        nfa.add_transition(0, Symbol::Epsilon, 1);
        nfa.add_transition(1, Symbol::Char('a'), 2);
        nfa.mark_accepting(2);

        let dfa = nfa.to_dfa();

        assert!(dfa.is_deterministic);
        assert!(dfa.accepts("a"));
        assert!(!dfa.accepts("b"));
    }
}
