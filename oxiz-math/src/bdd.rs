//! Binary Decision Diagrams (BDDs) for efficient boolean function representation.
//!
//! This module implements Reduced Ordered Binary Decision Diagrams (ROBDDs),
//! which provide a canonical representation for boolean functions.
//!
//! Reference: Bryant, "Graph-Based Algorithms for Boolean Function Manipulation" (1986)

use rustc_hash::FxHashMap;
use std::fmt;

/// Variable identifier in a BDD.
pub type VarId = u32;

/// Node identifier in the BDD.
/// Special values: 0 = FALSE, 1 = TRUE
pub type NodeId = usize;

/// The constant FALSE BDD node (node id 0).
pub const BDD_FALSE: NodeId = 0;
/// The constant TRUE BDD node (node id 1).
pub const BDD_TRUE: NodeId = 1;

/// A BDD node representing: if var then high else low
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BddNode {
    var: VarId,
    low: NodeId,
    high: NodeId,
}

/// A Binary Decision Diagram manager.
/// Maintains a unique table to ensure node sharing and canonical representation.
#[derive(Debug, Clone)]
pub struct BddManager {
    /// All nodes in the BDD. Index 0 is FALSE, index 1 is TRUE.
    nodes: Vec<BddNode>,
    /// Unique table mapping (var, low, high) to node ID.
    unique_table: FxHashMap<(VarId, NodeId, NodeId), NodeId>,
    /// Cache for binary operations: (op, left, right) -> result
    op_cache: FxHashMap<(BddOp, NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BddOp {
    And,
    Or,
    Xor,
}

impl BddManager {
    /// Creates a new BDD manager.
    pub fn new() -> Self {
        let mut manager = Self {
            nodes: Vec::new(),
            unique_table: FxHashMap::default(),
            op_cache: FxHashMap::default(),
        };

        // Initialize constant nodes
        // FALSE node (id = 0): dummy node, never actually used for var/low/high
        manager.nodes.push(BddNode {
            var: u32::MAX,
            low: 0,
            high: 0,
        });

        // TRUE node (id = 1): dummy node, never actually used for var/low/high
        manager.nodes.push(BddNode {
            var: u32::MAX,
            low: 1,
            high: 1,
        });

        manager
    }

    /// Returns the FALSE constant.
    pub fn constant_false(&self) -> NodeId {
        BDD_FALSE
    }

    /// Returns the TRUE constant.
    pub fn constant_true(&self) -> NodeId {
        BDD_TRUE
    }

    /// Creates a BDD for a single variable.
    pub fn variable(&mut self, var: VarId) -> NodeId {
        self.make_node(var, BDD_FALSE, BDD_TRUE)
    }

    /// Creates or retrieves a BDD node.
    /// Implements the reduction rules:
    /// - If low == high, return low (redundancy elimination)
    /// - Otherwise, check unique table for existing node
    fn make_node(&mut self, var: VarId, low: NodeId, high: NodeId) -> NodeId {
        // Reduction rule: if both branches are the same, return that branch
        if low == high {
            return low;
        }

        // Check if this node already exists
        let key = (var, low, high);
        if let Some(&node_id) = self.unique_table.get(&key) {
            return node_id;
        }

        // Create new node
        let node_id = self.nodes.len();
        self.nodes.push(BddNode { var, low, high });
        self.unique_table.insert(key, node_id);
        node_id
    }

    /// Negates a BDD node.
    pub fn not(&mut self, node: NodeId) -> NodeId {
        self.apply_not_rec(node)
    }

    fn apply_not_rec(&mut self, node: NodeId) -> NodeId {
        if node == BDD_FALSE {
            return BDD_TRUE;
        }
        if node == BDD_TRUE {
            return BDD_FALSE;
        }

        let n = self.nodes[node];
        let low = self.apply_not_rec(n.low);
        let high = self.apply_not_rec(n.high);
        self.make_node(n.var, low, high)
    }

    /// Computes the AND of two BDD nodes.
    pub fn and(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(BddOp::And, left, right)
    }

    /// Computes the OR of two BDD nodes.
    pub fn or(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(BddOp::Or, left, right)
    }

    /// Computes the XOR of two BDD nodes.
    pub fn xor(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(BddOp::Xor, left, right)
    }

    /// Computes the implication: left => right (equivalent to !left | right)
    pub fn implies(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let not_left = self.not(left);
        self.or(not_left, right)
    }

    /// Computes the equivalence: left <=> right
    pub fn iff(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let not_xor = self.xor(left, right);
        self.not(not_xor)
    }

    /// Apply a binary operation with memoization.
    fn apply(&mut self, op: BddOp, left: NodeId, right: NodeId) -> NodeId {
        // Terminal cases
        match op {
            BddOp::And => {
                if left == BDD_FALSE || right == BDD_FALSE {
                    return BDD_FALSE;
                }
                if left == BDD_TRUE {
                    return right;
                }
                if right == BDD_TRUE {
                    return left;
                }
                if left == right {
                    return left;
                }
            }
            BddOp::Or => {
                if left == BDD_TRUE || right == BDD_TRUE {
                    return BDD_TRUE;
                }
                if left == BDD_FALSE {
                    return right;
                }
                if right == BDD_FALSE {
                    return left;
                }
                if left == right {
                    return left;
                }
            }
            BddOp::Xor => {
                if left == BDD_FALSE {
                    return right;
                }
                if right == BDD_FALSE {
                    return left;
                }
                if left == BDD_TRUE {
                    return self.apply_not_rec(right);
                }
                if right == BDD_TRUE {
                    return self.apply_not_rec(left);
                }
                if left == right {
                    return BDD_FALSE;
                }
            }
        }

        // Check cache
        let cache_key = (op, left, right);
        if let Some(&result) = self.op_cache.get(&cache_key) {
            return result;
        }

        // Recursive case: Shannon expansion
        let left_node = self.nodes[left];
        let right_node = self.nodes[right];

        let result = if left_node.var == right_node.var {
            // Same variable
            let low = self.apply(op, left_node.low, right_node.low);
            let high = self.apply(op, left_node.high, right_node.high);
            self.make_node(left_node.var, low, high)
        } else if left_node.var < right_node.var {
            // Left has smaller variable (higher in ordering)
            let low = self.apply(op, left_node.low, right);
            let high = self.apply(op, left_node.high, right);
            self.make_node(left_node.var, low, high)
        } else {
            // Right has smaller variable
            let low = self.apply(op, left, right_node.low);
            let high = self.apply(op, left, right_node.high);
            self.make_node(right_node.var, low, high)
        };

        self.op_cache.insert(cache_key, result);
        result
    }

    /// If-then-else operation: if cond then then_node else else_node
    pub fn ite(&mut self, cond: NodeId, then_node: NodeId, else_node: NodeId) -> NodeId {
        // Terminal cases
        if cond == BDD_TRUE {
            return then_node;
        }
        if cond == BDD_FALSE {
            return else_node;
        }
        if then_node == else_node {
            return then_node;
        }
        if then_node == BDD_TRUE && else_node == BDD_FALSE {
            return cond;
        }

        let cond_node = self.nodes[cond];
        let then_n = if then_node <= BDD_TRUE {
            None
        } else {
            Some(self.nodes[then_node])
        };
        let else_n = if else_node <= BDD_TRUE {
            None
        } else {
            Some(self.nodes[else_node])
        };

        // Find the top variable
        let mut top_var = cond_node.var;
        if let Some(n) = then_n
            && n.var < top_var
        {
            top_var = n.var;
        }
        if let Some(n) = else_n
            && n.var < top_var
        {
            top_var = n.var;
        }

        // Cofactors
        let cond_low = if cond_node.var == top_var {
            cond_node.low
        } else {
            cond
        };
        let cond_high = if cond_node.var == top_var {
            cond_node.high
        } else {
            cond
        };

        let then_low = if let Some(n) = then_n {
            if n.var == top_var { n.low } else { then_node }
        } else {
            then_node
        };
        let then_high = if let Some(n) = then_n {
            if n.var == top_var { n.high } else { then_node }
        } else {
            then_node
        };

        let else_low = if let Some(n) = else_n {
            if n.var == top_var { n.low } else { else_node }
        } else {
            else_node
        };
        let else_high = if let Some(n) = else_n {
            if n.var == top_var { n.high } else { else_node }
        } else {
            else_node
        };

        let low = self.ite(cond_low, then_low, else_low);
        let high = self.ite(cond_high, then_high, else_high);
        self.make_node(top_var, low, high)
    }

    /// Evaluates a BDD with a given variable assignment.
    pub fn eval(&self, node: NodeId, assignment: &FxHashMap<VarId, bool>) -> bool {
        if node == BDD_FALSE {
            return false;
        }
        if node == BDD_TRUE {
            return true;
        }

        let n = self.nodes[node];
        let var_value = assignment.get(&n.var).copied().unwrap_or(false);

        if var_value {
            self.eval(n.high, assignment)
        } else {
            self.eval(n.low, assignment)
        }
    }

    /// Counts the number of satisfying assignments for a BDD.
    pub fn sat_count(&self, node: NodeId, num_vars: u32) -> u64 {
        let mut cache = FxHashMap::default();
        self.sat_count_rec(node, &mut cache)
            * (1u64 << num_vars.saturating_sub(self.max_var(node).map_or(0, |v| v + 1)))
    }

    fn sat_count_rec(&self, node: NodeId, cache: &mut FxHashMap<NodeId, u64>) -> u64 {
        if node == BDD_FALSE {
            return 0;
        }
        if node == BDD_TRUE {
            return 1;
        }

        if let Some(&count) = cache.get(&node) {
            return count;
        }

        let n = self.nodes[node];
        let low_count = self.sat_count_rec(n.low, cache);
        let high_count = self.sat_count_rec(n.high, cache);

        // Account for the difference in variable levels
        let low_mult = if n.low <= BDD_TRUE {
            1
        } else {
            let low_var = self.nodes[n.low].var;
            1u64 << (low_var - n.var - 1)
        };
        let high_mult = if n.high <= BDD_TRUE {
            1
        } else {
            let high_var = self.nodes[n.high].var;
            1u64 << (high_var - n.var - 1)
        };

        let count = low_count * low_mult + high_count * high_mult;
        cache.insert(node, count);
        count
    }

    /// Returns the maximum variable ID in a BDD.
    fn max_var(&self, node: NodeId) -> Option<VarId> {
        if node <= BDD_TRUE {
            return None;
        }
        Some(self.nodes[node].var)
    }

    /// Returns the number of nodes in the manager.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clears the operation cache (useful for memory management).
    pub fn clear_cache(&mut self) {
        self.op_cache.clear();
    }
}

impl Default for BddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BddManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BDD Manager: {} nodes", self.nodes.len())?;
        writeln!(f, "  Cache size: {}", self.op_cache.len())?;
        Ok(())
    }
}

/// Zero-suppressed Binary Decision Diagram (ZDD) manager.
/// ZDDs use a different reduction rule optimized for sparse sets:
/// - Eliminate node if high == 0 (instead of low == high for BDDs)
///
/// ZDDs are particularly efficient for representing:
/// - Sparse sets and families of sets
/// - Combinatorial structures
/// - Paths in graphs
#[derive(Debug, Clone)]
pub struct ZddManager {
    nodes: Vec<BddNode>,
    unique_table: FxHashMap<(VarId, NodeId, NodeId), NodeId>,
    op_cache: FxHashMap<(ZddOp, NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ZddOp {
    Union,
    Intersection,
    Difference,
}

/// The empty set in ZDD.
pub const ZDD_EMPTY: NodeId = 0;
/// The set containing only the empty set in ZDD.
pub const ZDD_BASE: NodeId = 1;

impl ZddManager {
    /// Creates a new ZDD manager.
    pub fn new() -> Self {
        let mut manager = Self {
            nodes: Vec::new(),
            unique_table: FxHashMap::default(),
            op_cache: FxHashMap::default(),
        };

        // EMPTY set (0)
        manager.nodes.push(BddNode {
            var: u32::MAX,
            low: 0,
            high: 0,
        });

        // BASE set containing empty set (1)
        manager.nodes.push(BddNode {
            var: u32::MAX,
            low: 1,
            high: 1,
        });

        manager
    }

    /// Returns the empty set.
    pub fn empty(&self) -> NodeId {
        ZDD_EMPTY
    }

    /// Returns the base set (set containing only the empty set).
    pub fn base(&self) -> NodeId {
        ZDD_BASE
    }

    /// Creates a ZDD for a singleton set containing just one variable.
    pub fn singleton(&mut self, var: VarId) -> NodeId {
        self.make_node(var, ZDD_EMPTY, ZDD_BASE)
    }

    /// Creates or retrieves a ZDD node with ZDD reduction rule.
    /// ZDD reduction: If high == 0 (empty), return low
    fn make_node(&mut self, var: VarId, low: NodeId, high: NodeId) -> NodeId {
        // ZDD reduction rule: if high branch is empty, eliminate node
        if high == ZDD_EMPTY {
            return low;
        }

        // Check if node exists
        let key = (var, low, high);
        if let Some(&node_id) = self.unique_table.get(&key) {
            return node_id;
        }

        // Create new node
        let node_id = self.nodes.len();
        self.nodes.push(BddNode { var, low, high });
        self.unique_table.insert(key, node_id);
        node_id
    }

    /// Computes the union of two ZDD sets.
    pub fn union(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(ZddOp::Union, left, right)
    }

    /// Computes the intersection of two ZDD sets.
    pub fn intersection(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(ZddOp::Intersection, left, right)
    }

    /// Computes the set difference: left \ right
    pub fn difference(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(ZddOp::Difference, left, right)
    }

    fn apply(&mut self, op: ZddOp, left: NodeId, right: NodeId) -> NodeId {
        // Terminal cases
        match op {
            ZddOp::Union => {
                if left == ZDD_EMPTY {
                    return right;
                }
                if right == ZDD_EMPTY {
                    return left;
                }
                if left == right {
                    return left;
                }
            }
            ZddOp::Intersection => {
                if left == ZDD_EMPTY || right == ZDD_EMPTY {
                    return ZDD_EMPTY;
                }
                if left == right {
                    return left;
                }
                // BASE only intersects with itself
                if left == ZDD_BASE || right == ZDD_BASE {
                    return ZDD_EMPTY;
                }
            }
            ZddOp::Difference => {
                if left == ZDD_EMPTY {
                    return ZDD_EMPTY;
                }
                if right == ZDD_EMPTY {
                    return left;
                }
                if left == right {
                    return ZDD_EMPTY;
                }
            }
        }

        // Check cache
        let cache_key = (op, left, right);
        if let Some(&result) = self.op_cache.get(&cache_key) {
            return result;
        }

        // Get top variable
        let left_node = self.nodes[left];
        let right_node = self.nodes[right];

        let result = if left_node.var == right_node.var {
            let low = self.apply(op, left_node.low, right_node.low);
            let high = self.apply(op, left_node.high, right_node.high);
            self.make_node(left_node.var, low, high)
        } else if left_node.var < right_node.var {
            let low = self.apply(op, left_node.low, right);
            let high = match op {
                ZddOp::Union => left_node.high,
                ZddOp::Intersection => self.apply(op, left_node.high, right),
                ZddOp::Difference => left_node.high,
            };
            self.make_node(left_node.var, low, high)
        } else {
            let low = self.apply(op, left, right_node.low);
            let high = match op {
                ZddOp::Union => right_node.high,
                ZddOp::Intersection => self.apply(op, left, right_node.high),
                ZddOp::Difference => ZDD_EMPTY,
            };
            self.make_node(right_node.var, low, high)
        };

        self.op_cache.insert(cache_key, result);
        result
    }

    /// Returns the number of sets in the family represented by this ZDD.
    pub fn count(&self, node: NodeId) -> u64 {
        let mut cache = FxHashMap::default();
        self.count_rec(node, &mut cache)
    }

    fn count_rec(&self, node: NodeId, cache: &mut FxHashMap<NodeId, u64>) -> u64 {
        if node == ZDD_EMPTY {
            return 0;
        }
        if node == ZDD_BASE {
            return 1;
        }

        if let Some(&count) = cache.get(&node) {
            return count;
        }

        let n = self.nodes[node];
        let count = self.count_rec(n.low, cache) + self.count_rec(n.high, cache);
        cache.insert(node, count);
        count
    }

    /// Returns the number of nodes in the manager.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clears the operation cache.
    pub fn clear_cache(&mut self) {
        self.op_cache.clear();
    }
}

impl Default for ZddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ZddManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ZDD Manager: {} nodes", self.nodes.len())?;
        writeln!(f, "  Cache size: {}", self.op_cache.len())?;
        Ok(())
    }
}

/// Algebraic Decision Diagram (ADD) for arithmetic functions.
/// ADDs generalize BDDs by allowing arbitrary terminal values (not just 0/1).
/// They can represent functions from boolean vectors to any algebraic domain.
///
/// Reference: Bahar et al., "Algebraic Decision Diagrams and Their Applications" (1993)
use num_rational::BigRational;
use num_traits::Zero;

/// An ADD node with rational number terminals.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AddNode {
    /// Terminal node with a rational value.
    Terminal(BigRational),
    /// Internal node: if var then high else low
    Internal {
        var: VarId,
        low: NodeId,
        high: NodeId,
    },
}

/// ADD manager for rational-valued functions.
#[derive(Debug, Clone)]
pub struct AddManager {
    nodes: Vec<AddNode>,
    unique_table: FxHashMap<AddNode, NodeId>,
    op_cache: FxHashMap<(AddOp, NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AddOp {
    Add,
    Mul,
    Max,
    Min,
}

impl AddManager {
    /// Creates a new ADD manager.
    pub fn new() -> Self {
        let mut manager = Self {
            nodes: Vec::new(),
            unique_table: FxHashMap::default(),
            op_cache: FxHashMap::default(),
        };

        // Create zero terminal
        let zero = AddNode::Terminal(BigRational::zero());
        manager.nodes.push(zero.clone());
        manager.unique_table.insert(zero, 0);

        manager
    }

    /// Creates a constant ADD.
    pub fn constant(&mut self, value: BigRational) -> NodeId {
        let node = AddNode::Terminal(value);

        // Check if this terminal already exists
        if let Some(&id) = self.unique_table.get(&node) {
            return id;
        }

        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.unique_table.insert(node, id);
        id
    }

    /// Creates an ADD for a single variable (returns 1 if true, 0 if false).
    pub fn variable(&mut self, var: VarId) -> NodeId {
        let zero = self.constant(BigRational::zero());
        let one = self.constant(BigRational::from_integer(1.into()));
        self.make_node(var, zero, one)
    }

    /// Creates or retrieves an ADD node.
    fn make_node(&mut self, var: VarId, low: NodeId, high: NodeId) -> NodeId {
        // Reduction rule: if both branches are the same, return that branch
        if low == high {
            return low;
        }

        let node = AddNode::Internal { var, low, high };

        // Check if this node already exists
        if let Some(&id) = self.unique_table.get(&node) {
            return id;
        }

        // Create new node
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.unique_table.insert(node, id);
        id
    }

    /// Evaluates the ADD given a variable assignment.
    pub fn eval(&self, node: NodeId, assignment: &FxHashMap<VarId, bool>) -> BigRational {
        match &self.nodes[node] {
            AddNode::Terminal(value) => value.clone(),
            AddNode::Internal { var, low, high } => {
                if *assignment.get(var).unwrap_or(&false) {
                    self.eval(*high, assignment)
                } else {
                    self.eval(*low, assignment)
                }
            }
        }
    }

    /// Computes the sum of two ADDs.
    pub fn add(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(AddOp::Add, left, right)
    }

    /// Computes the product of two ADDs.
    pub fn mul(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(AddOp::Mul, left, right)
    }

    /// Computes the maximum of two ADDs.
    pub fn max(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(AddOp::Max, left, right)
    }

    /// Computes the minimum of two ADDs.
    pub fn min(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.apply(AddOp::Min, left, right)
    }

    /// Multiplies an ADD by a scalar.
    pub fn scale(&mut self, node: NodeId, scalar: &BigRational) -> NodeId {
        let scalar_node = self.constant(scalar.clone());
        self.mul(node, scalar_node)
    }

    /// Apply a binary operation with memoization.
    fn apply(&mut self, op: AddOp, left: NodeId, right: NodeId) -> NodeId {
        // Terminal cases
        if let (AddNode::Terminal(l), AddNode::Terminal(r)) =
            (&self.nodes[left], &self.nodes[right])
        {
            let result_value = match op {
                AddOp::Add => l + r,
                AddOp::Mul => l * r,
                AddOp::Max => {
                    if l > r {
                        l.clone()
                    } else {
                        r.clone()
                    }
                }
                AddOp::Min => {
                    if l < r {
                        l.clone()
                    } else {
                        r.clone()
                    }
                }
            };
            return self.constant(result_value);
        }

        // Check cache
        let cache_key = (op, left, right);
        if let Some(&result) = self.op_cache.get(&cache_key) {
            return result;
        }

        // Recursive case
        let (var, left_low, left_high, right_low, right_high) = self.split_nodes(left, right);

        let low = self.apply(op, left_low, right_low);
        let high = self.apply(op, left_high, right_high);

        let result = self.make_node(var, low, high);
        self.op_cache.insert(cache_key, result);
        result
    }

    /// Helper function to split nodes for apply operation.
    fn split_nodes(&self, left: NodeId, right: NodeId) -> (VarId, NodeId, NodeId, NodeId, NodeId) {
        let (left_var, left_low, left_high) = match &self.nodes[left] {
            AddNode::Terminal(_) => (u32::MAX, left, left),
            AddNode::Internal { var, low, high } => (*var, *low, *high),
        };

        let (right_var, right_low, right_high) = match &self.nodes[right] {
            AddNode::Terminal(_) => (u32::MAX, right, right),
            AddNode::Internal { var, low, high } => (*var, *low, *high),
        };

        if left_var == u32::MAX && right_var == u32::MAX {
            panic!("Both nodes are terminals - should have been handled earlier");
        }

        if left_var == right_var {
            (left_var, left_low, left_high, right_low, right_high)
        } else if left_var < right_var {
            (left_var, left_low, left_high, right, right)
        } else {
            (right_var, left, left, right_low, right_high)
        }
    }

    /// If-then-else operation for ADDs.
    pub fn ite(&mut self, cond: NodeId, then_add: NodeId, else_add: NodeId) -> NodeId {
        // Convert boolean condition to ADD if needed
        // ite(c, t, e) = c * t + (1 - c) * e
        let one = self.constant(BigRational::from_integer(1.into()));
        let neg_one = BigRational::from_integer((-1).into());
        let scaled_cond = self.scale(cond, &neg_one);
        let not_cond = self.apply(AddOp::Add, one, scaled_cond);

        let then_part = self.mul(cond, then_add);
        let else_part = self.mul(not_cond, else_add);
        self.add(then_part, else_part)
    }

    /// Returns the number of nodes in the manager.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Clears the operation cache.
    pub fn clear_cache(&mut self) {
        self.op_cache.clear();
    }

    /// Gets the variable of a node, if it's internal.
    pub fn get_var(&self, node: NodeId) -> Option<VarId> {
        match &self.nodes[node] {
            AddNode::Terminal(_) => None,
            AddNode::Internal { var, .. } => Some(*var),
        }
    }

    /// Gets the terminal value of a node, if it's a terminal.
    pub fn get_value(&self, node: NodeId) -> Option<&BigRational> {
        match &self.nodes[node] {
            AddNode::Terminal(value) => Some(value),
            AddNode::Internal { .. } => None,
        }
    }
}

impl Default for AddManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AddManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ADD Manager: {} nodes", self.nodes.len())?;
        writeln!(f, "  Cache size: {}", self.op_cache.len())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdd_constants() {
        let manager = BddManager::new();
        assert_eq!(manager.constant_false(), BDD_FALSE);
        assert_eq!(manager.constant_true(), BDD_TRUE);
    }

    #[test]
    fn test_bdd_variable() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assert!(manager.eval(x, &assignment));

        assignment.insert(0, false);
        assert!(!manager.eval(x, &assignment));
    }

    #[test]
    fn test_bdd_not() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let not_x = manager.not(x);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assert!(!manager.eval(not_x, &assignment));

        assignment.insert(0, false);
        assert!(manager.eval(not_x, &assignment));
    }

    #[test]
    fn test_bdd_and() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(1);
        let x_and_y = manager.and(x, y);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assert!(manager.eval(x_and_y, &assignment));

        assignment.insert(0, false);
        assert!(!manager.eval(x_and_y, &assignment));
    }

    #[test]
    fn test_bdd_or() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(1);
        let x_or_y = manager.or(x, y);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, false);
        assignment.insert(1, false);
        assert!(!manager.eval(x_or_y, &assignment));

        assignment.insert(0, true);
        assert!(manager.eval(x_or_y, &assignment));
    }

    #[test]
    fn test_bdd_xor() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(1);
        let x_xor_y = manager.xor(x, y);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assert!(!manager.eval(x_xor_y, &assignment));

        assignment.insert(0, true);
        assignment.insert(1, false);
        assert!(manager.eval(x_xor_y, &assignment));
    }

    #[test]
    fn test_bdd_ite() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(1);
        let z = manager.variable(2);

        // if x then y else z
        let result = manager.ite(x, y, z);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assignment.insert(2, false);
        assert!(manager.eval(result, &assignment)); // x=T, y=T => T

        assignment.insert(0, false);
        assignment.insert(1, true);
        assignment.insert(2, false);
        assert!(!manager.eval(result, &assignment)); // x=F, z=F => F
    }

    #[test]
    fn test_bdd_implies() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(1);
        let x_implies_y = manager.implies(x, y);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        assignment.insert(1, false);
        assert!(!manager.eval(x_implies_y, &assignment));

        assignment.insert(0, false);
        assignment.insert(1, false);
        assert!(manager.eval(x_implies_y, &assignment));
    }

    #[test]
    fn test_bdd_sharing() {
        let mut manager = BddManager::new();
        let x = manager.variable(0);
        let y = manager.variable(0); // Same variable
        assert_eq!(x, y); // Should return the same node
    }

    // ZDD tests
    #[test]
    fn test_zdd_empty_and_base() {
        let manager = ZddManager::new();
        assert_eq!(manager.empty(), ZDD_EMPTY);
        assert_eq!(manager.base(), ZDD_BASE);
        assert_eq!(manager.count(ZDD_EMPTY), 0);
        assert_eq!(manager.count(ZDD_BASE), 1);
    }

    #[test]
    fn test_zdd_singleton() {
        let mut manager = ZddManager::new();
        let x = manager.singleton(0);
        assert_eq!(manager.count(x), 1); // One set: {0}
        assert_ne!(x, ZDD_EMPTY);
        assert_ne!(x, ZDD_BASE);
    }

    #[test]
    fn test_zdd_union() {
        let mut manager = ZddManager::new();
        let x = manager.singleton(0);
        let y = manager.singleton(1);
        let union = manager.union(x, y);

        // Union of {0} and {1} should give us 2 sets: {0} and {1}
        assert_eq!(manager.count(union), 2);
    }

    #[test]
    fn test_zdd_intersection() {
        let mut manager = ZddManager::new();
        let x = manager.singleton(0);
        let y = manager.singleton(1);
        let intersection = manager.intersection(x, y);

        // Intersection of {0} and {1} should be empty
        assert_eq!(intersection, ZDD_EMPTY);
        assert_eq!(manager.count(intersection), 0);
    }

    #[test]
    fn test_zdd_difference() {
        let mut manager = ZddManager::new();
        let x = manager.singleton(0);
        let y = manager.singleton(1);

        let diff1 = manager.difference(x, y);
        assert_eq!(manager.count(diff1), 1); // {0} - {1} = {0}

        let diff2 = manager.difference(x, x);
        assert_eq!(diff2, ZDD_EMPTY); // {0} - {0} = empty
    }

    #[test]
    fn test_zdd_union_associative() {
        let mut manager = ZddManager::new();
        let x = manager.singleton(0);
        let y = manager.singleton(1);
        let z = manager.singleton(2);

        let xy = manager.union(x, y);
        let xyz1 = manager.union(xy, z);

        let yz = manager.union(y, z);
        let xyz2 = manager.union(x, yz);

        assert_eq!(manager.count(xyz1), manager.count(xyz2));
        assert_eq!(manager.count(xyz1), 3); // Three sets
    }

    // ADD tests
    #[test]
    fn test_add_constant() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let five = manager.constant(BigRational::from_integer(BigInt::from(5)));

        let assignment = FxHashMap::default();
        let result = manager.eval(five, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(5)));
    }

    #[test]
    fn test_add_variable() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let x = manager.variable(0);

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        let result = manager.eval(x, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(1)));

        assignment.insert(0, false);
        let result = manager.eval(x, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(0)));
    }

    #[test]
    fn test_add_addition() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let three = manager.constant(BigRational::from_integer(BigInt::from(3)));
        let five = manager.constant(BigRational::from_integer(BigInt::from(5)));
        let sum = manager.add(three, five);

        let assignment = FxHashMap::default();
        let result = manager.eval(sum, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(8)));
    }

    #[test]
    fn test_add_multiplication() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let three = manager.constant(BigRational::from_integer(BigInt::from(3)));
        let five = manager.constant(BigRational::from_integer(BigInt::from(5)));
        let product = manager.mul(three, five);

        let assignment = FxHashMap::default();
        let result = manager.eval(product, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(15)));
    }

    #[test]
    fn test_add_max_min() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let three = manager.constant(BigRational::from_integer(BigInt::from(3)));
        let five = manager.constant(BigRational::from_integer(BigInt::from(5)));

        let max_val = manager.max(three, five);
        let min_val = manager.min(three, five);

        let assignment = FxHashMap::default();
        assert_eq!(
            manager.eval(max_val, &assignment),
            BigRational::from_integer(BigInt::from(5))
        );
        assert_eq!(
            manager.eval(min_val, &assignment),
            BigRational::from_integer(BigInt::from(3))
        );
    }

    #[test]
    fn test_add_scale() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let x = manager.variable(0);
        let scaled = manager.scale(x, &BigRational::from_integer(BigInt::from(5)));

        let mut assignment = FxHashMap::default();
        assignment.insert(0, true);
        let result = manager.eval(scaled, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(5)));

        assignment.insert(0, false);
        let result = manager.eval(scaled, &assignment);
        assert_eq!(result, BigRational::from_integer(BigInt::from(0)));
    }

    #[test]
    fn test_add_sharing() {
        use num_bigint::BigInt;
        let mut manager = AddManager::new();
        let x = manager.constant(BigRational::from_integer(BigInt::from(5)));
        let y = manager.constant(BigRational::from_integer(BigInt::from(5)));
        assert_eq!(x, y); // Should return the same node
    }
}
