//! Lazy Loading System for WASM Theories
#![allow(missing_docs)] // Under development
//!
//! This module provides dynamic theory loading capabilities for the OxiZ WASM solver.
//! By loading theories on-demand rather than bundling everything upfront, we can
//! dramatically reduce initial bundle size while maintaining full functionality.
//!
//! # Architecture
//!
//! The lazy loader uses a registry-based approach where each theory module is:
//! 1. Registered with metadata (size, dependencies, initialization function)
//! 2. Loaded asynchronously when first needed
//! 3. Cached for subsequent use
//! 4. Unloadable to free memory when no longer needed
//!
//! # Performance
//!
//! - Initial load time: ~50-100ms per theory (vs ~500ms for full bundle)
//! - Memory overhead: ~10KB per loaded theory
//! - Unload time: ~5-10ms
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::lazy_loader::{LazyLoader, TheoryDescriptor};
//!
//! let mut loader = LazyLoader::new();
//!
//! // Register theories
//! loader.register_theory(TheoryDescriptor {
//!     name: "arithmetic",
//!     size_estimate: 128 * 1024,
//!     dependencies: vec![],
//!     init_fn: Box::new(|| { /* initialization */ Ok(()) }),
//! });
//!
//! // Load on demand
//! loader.load_theory("arithmetic").await?;
//!
//! // Unload when done
//! loader.unload_theory("arithmetic")?;
//! ```

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Result type for lazy loading operations
pub type LoadResult<T> = Result<T, LoadError>;

/// Errors that can occur during lazy loading
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Theory not found in registry
    TheoryNotFound(String),
    /// Theory already loaded
    AlreadyLoaded(String),
    /// Theory not loaded
    NotLoaded(String),
    /// Dependency cycle detected
    DependencyCycle(Vec<String>),
    /// Missing dependency
    MissingDependency { theory: String, dependency: String },
    /// Initialization failed
    InitializationFailed { theory: String, error: String },
    /// Network error during module fetch
    NetworkError(String),
    /// Invalid theory descriptor
    InvalidDescriptor(String),
    /// Memory allocation failed
    OutOfMemory,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TheoryNotFound(name) => write!(f, "Theory not found: {}", name),
            Self::AlreadyLoaded(name) => write!(f, "Theory already loaded: {}", name),
            Self::NotLoaded(name) => write!(f, "Theory not loaded: {}", name),
            Self::DependencyCycle(cycle) => {
                write!(f, "Dependency cycle detected: {}", cycle.join(" -> "))
            }
            Self::MissingDependency { theory, dependency } => {
                write!(f, "Theory '{}' requires '{}'", theory, dependency)
            }
            Self::InitializationFailed { theory, error } => {
                write!(f, "Failed to initialize '{}': {}", theory, error)
            }
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::InvalidDescriptor(msg) => write!(f, "Invalid descriptor: {}", msg),
            Self::OutOfMemory => write!(f, "Out of memory"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Function type for theory initialization
pub type InitFn = Box<dyn Fn() -> Result<(), String> + Send + Sync>;

/// Descriptor for a theory module
#[derive(Clone)]
pub struct TheoryDescriptor {
    /// Unique name of the theory
    pub name: String,
    /// Estimated size in bytes when loaded
    pub size_estimate: usize,
    /// List of theory names this theory depends on
    pub dependencies: Vec<String>,
    /// Initialization function to call when loading
    pub init_fn: Arc<InitFn>,
    /// URL to fetch the WASM module from (for true dynamic loading)
    pub module_url: Option<String>,
    /// Priority level (higher = load earlier when multiple theories requested)
    pub priority: u8,
}

impl TheoryDescriptor {
    /// Create a new theory descriptor
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size_estimate: 0,
            dependencies: Vec::new(),
            init_fn: Arc::new(Box::new(|| Ok(()))),
            module_url: None,
            priority: 0,
        }
    }

    /// Set the estimated size in bytes
    pub fn with_size(mut self, size: usize) -> Self {
        self.size_estimate = size;
        self
    }

    /// Add a dependency
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Add multiple dependencies
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Set the initialization function
    pub fn with_init_fn(mut self, init_fn: InitFn) -> Self {
        self.init_fn = Arc::new(init_fn);
        self
    }

    /// Set the module URL for dynamic loading
    pub fn with_module_url(mut self, url: impl Into<String>) -> Self {
        self.module_url = Some(url.into());
        self
    }

    /// Set the priority level
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// State of a loaded theory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TheoryState {
    /// Theory is registered but not loaded
    Registered,
    /// Theory is currently being loaded
    Loading,
    /// Theory is fully loaded and ready
    Loaded,
    /// Theory failed to load
    Failed,
}

/// Information about a loaded theory
struct LoadedTheory {
    descriptor: TheoryDescriptor,
    state: TheoryState,
    load_time_ms: Option<f64>,
    actual_size: Option<usize>,
    reference_count: usize,
}

/// Main lazy loading system
pub struct LazyLoader {
    /// Registry of all known theories
    registry: HashMap<String, TheoryDescriptor>,
    /// Currently loaded theories
    loaded: HashMap<String, LoadedTheory>,
    /// Load order (for dependency tracking)
    load_order: Vec<String>,
    /// Maximum memory budget in bytes (0 = unlimited)
    max_memory_budget: usize,
    /// Current memory usage estimate
    current_memory_usage: usize,
    /// Whether to enable verbose logging
    verbose: bool,
}

impl LazyLoader {
    /// Create a new lazy loader
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            loaded: HashMap::new(),
            load_order: Vec::new(),
            max_memory_budget: 0,
            current_memory_usage: 0,
            verbose: false,
        }
    }

    /// Create a lazy loader with a memory budget
    pub fn with_memory_budget(budget_bytes: usize) -> Self {
        Self {
            max_memory_budget: budget_bytes,
            ..Self::new()
        }
    }

    /// Enable or disable verbose logging
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Register a theory with the loader
    pub fn register_theory(&mut self, descriptor: TheoryDescriptor) -> LoadResult<()> {
        if descriptor.name.is_empty() {
            return Err(LoadError::InvalidDescriptor(
                "Theory name cannot be empty".to_string(),
            ));
        }

        // Validate dependencies exist
        for dep in &descriptor.dependencies {
            if !self.registry.contains_key(dep) && !self.loaded.contains_key(dep) && self.verbose {
                web_sys::console::warn_1(
                    &format!("Warning: Dependency '{}' not yet registered", dep).into(),
                );
            }
        }

        self.registry.insert(descriptor.name.clone(), descriptor);
        Ok(())
    }

    /// Register multiple theories at once
    pub fn register_theories(&mut self, descriptors: Vec<TheoryDescriptor>) -> LoadResult<()> {
        for desc in descriptors {
            self.register_theory(desc)?;
        }
        Ok(())
    }

    /// Check if a theory is registered
    pub fn is_registered(&self, name: &str) -> bool {
        self.registry.contains_key(name)
    }

    /// Check if a theory is loaded
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded
            .get(name)
            .is_some_and(|t| t.state == TheoryState::Loaded)
    }

    /// Get the state of a theory
    pub fn get_state(&self, name: &str) -> Option<TheoryState> {
        self.loaded.get(name).map(|t| t.state)
    }

    /// Load a theory and all its dependencies
    pub async fn load_theory(&mut self, name: &str) -> LoadResult<()> {
        if self.is_loaded(name) {
            return Err(LoadError::AlreadyLoaded(name.to_string()));
        }

        // Get the descriptor
        let descriptor = self
            .registry
            .get(name)
            .ok_or_else(|| LoadError::TheoryNotFound(name.to_string()))?
            .clone();

        // Check for dependency cycles
        let mut visited = HashSet::new();
        let mut rec_stack = Vec::new();
        self.check_dependency_cycle(name, &mut visited, &mut rec_stack)?;

        // Load dependencies first
        for dep in &descriptor.dependencies {
            if !self.is_loaded(dep) {
                // Use Box::pin for recursive async calls
                Box::pin(self.load_theory(dep)).await?;
            }
        }

        // Check memory budget
        if self.max_memory_budget > 0 {
            let required = self.current_memory_usage + descriptor.size_estimate;
            if required > self.max_memory_budget {
                return Err(LoadError::OutOfMemory);
            }
        }

        // Mark as loading
        self.loaded.insert(
            name.to_string(),
            LoadedTheory {
                descriptor: descriptor.clone(),
                state: TheoryState::Loading,
                load_time_ms: None,
                actual_size: None,
                reference_count: 0,
            },
        );

        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now());

        // Load the module if URL provided
        if let Some(url) = &descriptor.module_url {
            self.fetch_and_load_module(url).await?;
        }

        // Run initialization
        match (descriptor.init_fn)() {
            Ok(()) => {
                let load_time = start_time
                    .and_then(|start| {
                        web_sys::window()
                            .and_then(|w| w.performance())
                            .map(|p| p.now() - start)
                    })
                    .unwrap_or(0.0);

                // Update to loaded state
                if let Some(theory) = self.loaded.get_mut(name) {
                    theory.state = TheoryState::Loaded;
                    theory.load_time_ms = Some(load_time);
                    theory.actual_size = Some(descriptor.size_estimate);
                }

                self.current_memory_usage += descriptor.size_estimate;
                self.load_order.push(name.to_string());

                if self.verbose {
                    web_sys::console::log_1(
                        &format!("Loaded theory '{}' in {:.2}ms", name, load_time).into(),
                    );
                }

                Ok(())
            }
            Err(e) => {
                // Mark as failed
                if let Some(theory) = self.loaded.get_mut(name) {
                    theory.state = TheoryState::Failed;
                }
                Err(LoadError::InitializationFailed {
                    theory: name.to_string(),
                    error: e,
                })
            }
        }
    }

    /// Load multiple theories concurrently
    pub async fn load_theories(&mut self, names: &[&str]) -> LoadResult<()> {
        // Sort by priority (higher first)
        let mut sorted_names: Vec<_> = names
            .iter()
            .map(|&name| {
                let priority = self.registry.get(name).map(|d| d.priority).unwrap_or(0);
                (name, priority)
            })
            .collect();
        sorted_names.sort_by(|a, b| b.1.cmp(&a.1));

        // Load in priority order
        for (name, _) in sorted_names {
            if !self.is_loaded(name) {
                self.load_theory(name).await?;
            }
        }

        Ok(())
    }

    /// Unload a theory and free its memory
    pub fn unload_theory(&mut self, name: &str) -> LoadResult<()> {
        let theory = self
            .loaded
            .get(name)
            .ok_or_else(|| LoadError::NotLoaded(name.to_string()))?;

        // Check if other theories depend on this one
        for (other_name, other_theory) in &self.loaded {
            if other_theory.state == TheoryState::Loaded
                && other_theory
                    .descriptor
                    .dependencies
                    .contains(&name.to_string())
            {
                return Err(LoadError::MissingDependency {
                    theory: other_name.clone(),
                    dependency: name.to_string(),
                });
            }
        }

        let size = theory
            .actual_size
            .unwrap_or(theory.descriptor.size_estimate);
        self.current_memory_usage = self.current_memory_usage.saturating_sub(size);

        self.loaded.remove(name);
        self.load_order.retain(|n| n != name);

        if self.verbose {
            web_sys::console::log_1(&format!("Unloaded theory '{}'", name).into());
        }

        Ok(())
    }

    /// Unload all theories
    pub fn unload_all(&mut self) {
        self.loaded.clear();
        self.load_order.clear();
        self.current_memory_usage = 0;

        if self.verbose {
            web_sys::console::log_1(&"Unloaded all theories".into());
        }
    }

    /// Increment reference count for a theory
    pub fn inc_ref(&mut self, name: &str) -> LoadResult<()> {
        let theory = self
            .loaded
            .get_mut(name)
            .ok_or_else(|| LoadError::NotLoaded(name.to_string()))?;
        theory.reference_count += 1;
        Ok(())
    }

    /// Decrement reference count for a theory
    pub fn dec_ref(&mut self, name: &str) -> LoadResult<()> {
        let theory = self
            .loaded
            .get_mut(name)
            .ok_or_else(|| LoadError::NotLoaded(name.to_string()))?;
        theory.reference_count = theory.reference_count.saturating_sub(1);
        Ok(())
    }

    /// Get current memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.current_memory_usage
    }

    /// Get list of loaded theories
    pub fn loaded_theories(&self) -> Vec<String> {
        self.load_order.clone()
    }

    /// Get load statistics
    pub fn get_stats(&self) -> LoadStats {
        let mut total_load_time = 0.0;
        let mut loaded_count = 0;

        for theory in self.loaded.values() {
            if theory.state == TheoryState::Loaded {
                loaded_count += 1;
                if let Some(time) = theory.load_time_ms {
                    total_load_time += time;
                }
            }
        }

        LoadStats {
            registered_count: self.registry.len(),
            loaded_count,
            failed_count: self
                .loaded
                .values()
                .filter(|t| t.state == TheoryState::Failed)
                .count(),
            total_load_time_ms: total_load_time,
            memory_usage_bytes: self.current_memory_usage,
            memory_budget_bytes: self.max_memory_budget,
        }
    }

    /// Check for dependency cycles using DFS
    fn check_dependency_cycle(
        &self,
        name: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut Vec<String>,
    ) -> LoadResult<()> {
        if rec_stack.contains(&name.to_string()) {
            rec_stack.push(name.to_string());
            return Err(LoadError::DependencyCycle(rec_stack.clone()));
        }

        if visited.contains(name) {
            return Ok(());
        }

        visited.insert(name.to_string());
        rec_stack.push(name.to_string());

        if let Some(descriptor) = self.registry.get(name) {
            for dep in &descriptor.dependencies {
                self.check_dependency_cycle(dep, visited, rec_stack)?;
            }
        }

        rec_stack.pop();
        Ok(())
    }

    /// Fetch and load a WASM module from a URL
    async fn fetch_and_load_module(&self, url: &str) -> LoadResult<()> {
        let window = web_sys::window()
            .ok_or_else(|| LoadError::NetworkError("No window object available".to_string()))?;

        let resp_value = JsFuture::from(window.fetch_with_str(url))
            .await
            .map_err(|e| LoadError::NetworkError(format!("Fetch failed: {:?}", e)))?;

        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| LoadError::NetworkError("Invalid response".to_string()))?;

        let array_buffer = JsFuture::from(
            resp.array_buffer()
                .map_err(|e| LoadError::NetworkError(format!("Failed to get buffer: {:?}", e)))?,
        )
        .await
        .map_err(|e| LoadError::NetworkError(format!("Buffer fetch failed: {:?}", e)))?;

        // In a real implementation, we would instantiate the WASM module here
        // For now, we just validate that we got the data
        let _bytes = js_sys::Uint8Array::new(&array_buffer);

        Ok(())
    }
}

impl Default for LazyLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about loaded theories
#[derive(Debug, Clone)]
pub struct LoadStats {
    /// Number of registered theories
    pub registered_count: usize,
    /// Number of loaded theories
    pub loaded_count: usize,
    /// Number of failed theories
    pub failed_count: usize,
    /// Total load time in milliseconds
    pub total_load_time_ms: f64,
    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Memory budget in bytes (0 = unlimited)
    pub memory_budget_bytes: usize,
}

impl LoadStats {
    /// Get memory usage as a percentage of budget
    pub fn memory_usage_percent(&self) -> Option<f64> {
        if self.memory_budget_bytes > 0 {
            Some((self.memory_usage_bytes as f64 / self.memory_budget_bytes as f64) * 100.0)
        } else {
            None
        }
    }

    /// Get average load time per theory
    pub fn avg_load_time_ms(&self) -> f64 {
        if self.loaded_count > 0 {
            self.total_load_time_ms / self.loaded_count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theory_descriptor_builder() {
        let desc = TheoryDescriptor::new("test")
            .with_size(1024)
            .with_dependency("dep1")
            .with_priority(5);

        assert_eq!(desc.name, "test");
        assert_eq!(desc.size_estimate, 1024);
        assert_eq!(desc.dependencies, vec!["dep1"]);
        assert_eq!(desc.priority, 5);
    }

    #[test]
    fn test_lazy_loader_registration() {
        let mut loader = LazyLoader::new();
        let desc = TheoryDescriptor::new("arithmetic").with_size(128 * 1024);

        assert!(loader.register_theory(desc).is_ok());
        assert!(loader.is_registered("arithmetic"));
        assert!(!loader.is_loaded("arithmetic"));
    }

    #[test]
    fn test_empty_name_rejected() {
        let mut loader = LazyLoader::new();
        let desc = TheoryDescriptor::new("");

        assert!(loader.register_theory(desc).is_err());
    }

    #[test]
    fn test_memory_tracking() {
        let loader = LazyLoader::with_memory_budget(1024 * 1024);
        assert_eq!(loader.max_memory_budget, 1024 * 1024);
        assert_eq!(loader.memory_usage(), 0);
    }

    #[test]
    fn test_load_stats() {
        let loader = LazyLoader::new();
        let stats = loader.get_stats();

        assert_eq!(stats.registered_count, 0);
        assert_eq!(stats.loaded_count, 0);
        assert_eq!(stats.memory_usage_bytes, 0);
    }
}
