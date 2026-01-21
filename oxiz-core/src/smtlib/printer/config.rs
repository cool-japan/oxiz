//\! Pretty printing configuration


/// Configuration for pretty printing
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PrettyConfig {
    /// Number of spaces per indentation level
    pub indent_width: usize,
    /// Maximum line width before breaking
    pub max_width: usize,
    /// Whether to use tabs instead of spaces
    pub use_tabs: bool,
    /// Whether to print sorts for terms
    pub print_sorts: bool,
    /// Minimum depth before breaking
    pub break_depth: usize,
}

impl Default for PrettyConfig {
    fn default() -> Self {
        Self {
            indent_width: 2,
            max_width: 80,
            use_tabs: false,
            print_sorts: false,
            break_depth: 2,
        }
    }
}

#[allow(dead_code)]
impl PrettyConfig {
    /// Create a compact configuration (minimal whitespace)
    #[must_use]
    pub fn compact() -> Self {
        Self {
            indent_width: 0,
            max_width: usize::MAX,
            use_tabs: false,
            print_sorts: false,
            break_depth: usize::MAX,
        }
    }

    /// Create an expanded configuration (one term per line)
    #[must_use]
    pub fn expanded() -> Self {
        Self {
            indent_width: 2,
            max_width: 40,
            use_tabs: false,
            print_sorts: false,
            break_depth: 1,
        }
    }

    /// Set the indentation width
    #[must_use]
    pub fn with_indent_width(mut self, width: usize) -> Self {
        self.indent_width = width;
        self
    }

    /// Set the maximum line width
    #[must_use]
    pub fn with_max_width(mut self, width: usize) -> Self {
        self.max_width = width;
        self
    }

    /// Set whether to use tabs
    #[must_use]
    pub fn with_tabs(mut self, use_tabs: bool) -> Self {
        self.use_tabs = use_tabs;
        self
    }

    /// Set whether to print sorts
    #[must_use]
    pub fn with_print_sorts(mut self, print_sorts: bool) -> Self {
        self.print_sorts = print_sorts;
        self
    }

    /// Set the minimum depth before breaking
    #[must_use]
    pub fn with_break_depth(mut self, depth: usize) -> Self {
        self.break_depth = depth;
        self
    }
}
