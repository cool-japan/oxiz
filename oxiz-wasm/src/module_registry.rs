//! Module Registry for Theory Management
#![allow(missing_docs)] // Under development
//!
//! This module provides a centralized registry for tracking and managing
//! loaded theory modules in the WASM environment. It handles module lifecycle,
//! versioning, and capability detection.
//!
//! # Features
//!
//! - **Module Tracking**: Keep track of all loaded modules and their state
//! - **Version Management**: Handle different versions of theories
//! - **Capability Detection**: Query which features are available
//! - **Hot Reloading**: Support for reloading modules without full restart
//! - **Dependency Resolution**: Automatic handling of inter-module dependencies
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::module_registry::ModuleRegistry;
//!
//! let mut registry = ModuleRegistry::new();
//!
//! // Register a module
//! registry.register_module(ModuleInfo {
//!     id: "arithmetic_v1".to_string(),
//!     name: "arithmetic".to_string(),
//!     version: Version::new(1, 0, 0),
//!     capabilities: vec!["linear", "nonlinear"],
//!     size_bytes: 128 * 1024,
//! });
//!
//! // Query capabilities
//! if registry.has_capability("arithmetic", "linear") {
//!     println!("Linear arithmetic is available");
//! }
//!
//! // Unregister when done
//! registry.unregister_module("arithmetic_v1");
//! ```

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Semantic version for modules
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string (e.g., "1.2.3")
    pub fn parse(s: &str) -> Result<Self, VersionParseError> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(VersionParseError::InvalidFormat(s.to_string()));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber(parts[0].to_string()))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber(parts[1].to_string()))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| VersionParseError::InvalidNumber(parts[2].to_string()))?;

        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        self.major == other.major && self >= other
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Error type for version parsing
#[derive(Debug, Clone)]
pub enum VersionParseError {
    /// Invalid version format
    InvalidFormat(String),
    /// Invalid number in version
    InvalidNumber(String),
}

impl fmt::Display for VersionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat(s) => write!(f, "Invalid version format: {}", s),
            Self::InvalidNumber(s) => write!(f, "Invalid version number: {}", s),
        }
    }
}

impl std::error::Error for VersionParseError {}

/// Information about a registered module
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Unique identifier for this module instance
    pub id: String,
    /// Module name (can have multiple versions)
    pub name: String,
    /// Version of the module
    pub version: Version,
    /// List of capabilities this module provides
    pub capabilities: Vec<String>,
    /// Size in bytes when loaded
    pub size_bytes: usize,
    /// Whether this module is currently active
    pub active: bool,
    /// Load timestamp (milliseconds since epoch)
    pub load_timestamp: Option<f64>,
    /// Module metadata
    pub metadata: HashMap<String, String>,
}

impl ModuleInfo {
    /// Create a new module info
    pub fn new(name: impl Into<String>, version: Version) -> Self {
        let name = name.into();
        let id = format!("{}_v{}", name, version);
        Self {
            id,
            name,
            version,
            capabilities: Vec::new(),
            size_bytes: 0,
            active: false,
            load_timestamp: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a capability
    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        self.capabilities.push(cap.into());
        self
    }

    /// Add multiple capabilities
    pub fn with_capabilities(mut self, caps: Vec<String>) -> Self {
        self.capabilities = caps;
        self
    }

    /// Set the size in bytes
    pub fn with_size(mut self, size: usize) -> Self {
        self.size_bytes = size;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if this module has a specific capability
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

/// Module registry error types
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// Module already registered
    AlreadyRegistered(String),
    /// Module not found
    NotFound(String),
    /// Version conflict
    VersionConflict {
        module: String,
        existing: Version,
        requested: Version,
    },
    /// Invalid module ID
    InvalidId(String),
    /// Module is currently active and cannot be unregistered
    ModuleActive(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRegistered(id) => write!(f, "Module already registered: {}", id),
            Self::NotFound(id) => write!(f, "Module not found: {}", id),
            Self::VersionConflict {
                module,
                existing,
                requested,
            } => write!(
                f,
                "Version conflict for '{}': existing={}, requested={}",
                module, existing, requested
            ),
            Self::InvalidId(id) => write!(f, "Invalid module ID: {}", id),
            Self::ModuleActive(id) => write!(f, "Module is active: {}", id),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Result type for registry operations
pub type RegistryResult<T> = Result<T, RegistryError>;

/// Module registry for tracking loaded modules
pub struct ModuleRegistry {
    /// All registered modules by ID
    modules: HashMap<String, ModuleInfo>,
    /// Modules indexed by name (for version lookup)
    by_name: HashMap<String, Vec<String>>,
    /// Capability index (capability -> module IDs)
    capabilities: HashMap<String, HashSet<String>>,
    /// Active module IDs
    active: HashSet<String>,
    /// Total size of all loaded modules
    total_size: usize,
}

impl ModuleRegistry {
    /// Create a new module registry
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            by_name: HashMap::new(),
            capabilities: HashMap::new(),
            active: HashSet::new(),
            total_size: 0,
        }
    }

    /// Register a new module
    pub fn register_module(&mut self, mut info: ModuleInfo) -> RegistryResult<()> {
        if info.id.is_empty() {
            return Err(RegistryError::InvalidId(info.id.clone()));
        }

        if self.modules.contains_key(&info.id) {
            return Err(RegistryError::AlreadyRegistered(info.id.clone()));
        }

        // Check for version conflicts if same name exists
        if let Some(existing_ids) = self.by_name.get(&info.name) {
            for existing_id in existing_ids {
                if let Some(existing) = self.modules.get(existing_id)
                    && existing.version == info.version
                {
                    return Err(RegistryError::VersionConflict {
                        module: info.name.clone(),
                        existing: existing.version,
                        requested: info.version,
                    });
                }
            }
        }

        // Set load timestamp
        #[cfg(target_arch = "wasm32")]
        {
            info.load_timestamp = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            info.load_timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64() * 1000.0)
                .ok();
        }

        // Index capabilities
        for cap in &info.capabilities {
            self.capabilities
                .entry(cap.clone())
                .or_default()
                .insert(info.id.clone());
        }

        // Index by name
        self.by_name
            .entry(info.name.clone())
            .or_default()
            .push(info.id.clone());

        self.total_size += info.size_bytes;
        self.modules.insert(info.id.clone(), info);

        Ok(())
    }

    /// Unregister a module
    pub fn unregister_module(&mut self, id: &str) -> RegistryResult<()> {
        if !self.modules.contains_key(id) {
            return Err(RegistryError::NotFound(id.to_string()));
        }

        if self.active.contains(id) {
            return Err(RegistryError::ModuleActive(id.to_string()));
        }

        let info = self.modules.remove(id).expect("module exists");

        // Remove from capability index
        for cap in &info.capabilities {
            if let Some(module_set) = self.capabilities.get_mut(cap) {
                module_set.remove(id);
                if module_set.is_empty() {
                    self.capabilities.remove(cap);
                }
            }
        }

        // Remove from name index
        if let Some(ids) = self.by_name.get_mut(&info.name) {
            ids.retain(|module_id| module_id != id);
            if ids.is_empty() {
                self.by_name.remove(&info.name);
            }
        }

        self.total_size = self.total_size.saturating_sub(info.size_bytes);

        Ok(())
    }

    /// Activate a module
    pub fn activate_module(&mut self, id: &str) -> RegistryResult<()> {
        if !self.modules.contains_key(id) {
            return Err(RegistryError::NotFound(id.to_string()));
        }

        self.active.insert(id.to_string());

        if let Some(info) = self.modules.get_mut(id) {
            info.active = true;
        }

        Ok(())
    }

    /// Deactivate a module
    pub fn deactivate_module(&mut self, id: &str) -> RegistryResult<()> {
        self.active.remove(id);

        if let Some(info) = self.modules.get_mut(id) {
            info.active = false;
        }

        Ok(())
    }

    /// Get module info by ID
    pub fn get_module(&self, id: &str) -> Option<&ModuleInfo> {
        self.modules.get(id)
    }

    /// Get module info by name and version
    pub fn get_module_by_version(&self, name: &str, version: &Version) -> Option<&ModuleInfo> {
        self.by_name
            .get(name)?
            .iter()
            .find_map(|id| self.modules.get(id))
            .filter(|info| info.version == *version)
    }

    /// Get the latest version of a module by name
    pub fn get_latest_module(&self, name: &str) -> Option<&ModuleInfo> {
        self.by_name
            .get(name)?
            .iter()
            .filter_map(|id| self.modules.get(id))
            .max_by(|a, b| a.version.cmp(&b.version))
    }

    /// Get all versions of a module
    pub fn get_all_versions(&self, name: &str) -> Vec<&ModuleInfo> {
        self.by_name
            .get(name)
            .map(|ids| {
                let mut versions: Vec<_> =
                    ids.iter().filter_map(|id| self.modules.get(id)).collect();
                versions.sort_by(|a, b| a.version.cmp(&b.version));
                versions
            })
            .unwrap_or_default()
    }

    /// Check if a module is registered
    pub fn is_registered(&self, id: &str) -> bool {
        self.modules.contains_key(id)
    }

    /// Check if a module is active
    pub fn is_active(&self, id: &str) -> bool {
        self.active.contains(id)
    }

    /// Check if any module provides a capability
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities
            .get(cap)
            .is_some_and(|modules| !modules.is_empty())
    }

    /// Get all modules that provide a capability
    pub fn modules_with_capability(&self, cap: &str) -> Vec<&ModuleInfo> {
        self.capabilities
            .get(cap)
            .map(|ids| ids.iter().filter_map(|id| self.modules.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all registered module IDs
    pub fn module_ids(&self) -> Vec<String> {
        self.modules.keys().cloned().collect()
    }

    /// Get all active module IDs
    pub fn active_modules(&self) -> Vec<String> {
        self.active.iter().cloned().collect()
    }

    /// Get total size of all registered modules
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_modules: self.modules.len(),
            active_modules: self.active.len(),
            unique_names: self.by_name.len(),
            total_capabilities: self.capabilities.len(),
            total_size_bytes: self.total_size,
        }
    }

    /// Clear all modules (only if none are active)
    pub fn clear(&mut self) -> RegistryResult<()> {
        if !self.active.is_empty() {
            let active_id = self.active.iter().next().expect("not empty").clone();
            return Err(RegistryError::ModuleActive(active_id));
        }

        self.modules.clear();
        self.by_name.clear();
        self.capabilities.clear();
        self.total_size = 0;

        Ok(())
    }

    /// Export module list as JSON-compatible structure
    pub fn export_module_list(&self) -> Vec<ModuleExport> {
        self.modules
            .values()
            .map(|info| ModuleExport {
                id: info.id.clone(),
                name: info.name.clone(),
                version: info.version.to_string(),
                capabilities: info.capabilities.clone(),
                size_bytes: info.size_bytes,
                active: info.active,
                load_timestamp: info.load_timestamp,
            })
            .collect()
    }

    /// Find modules matching a predicate
    pub fn find_modules<F>(&self, predicate: F) -> Vec<&ModuleInfo>
    where
        F: Fn(&ModuleInfo) -> bool,
    {
        self.modules
            .values()
            .filter(|info| predicate(info))
            .collect()
    }
}

impl Default for ModuleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of registered modules
    pub total_modules: usize,
    /// Number of active modules
    pub active_modules: usize,
    /// Number of unique module names
    pub unique_names: usize,
    /// Total number of capabilities
    pub total_capabilities: usize,
    /// Total size in bytes
    pub total_size_bytes: usize,
}

/// Exported module information (JSON-serializable)
#[derive(Debug, Clone)]
pub struct ModuleExport {
    /// Module ID
    pub id: String,
    /// Module name
    pub name: String,
    /// Version string
    pub version: String,
    /// Capabilities
    pub capabilities: Vec<String>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether active
    pub active: bool,
    /// Load timestamp
    pub load_timestamp: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_invalid_format() {
        assert!(Version::parse("1.2").is_err());
        assert!(Version::parse("1.2.3.4").is_err());
        assert!(Version::parse("a.b.c").is_err());
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 3, 0);
        let v3 = Version::new(2, 0, 0);

        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
        assert!(!v3.is_compatible_with(&v1));
    }

    #[test]
    fn test_version_display() {
        let v = Version::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_module_info_builder() {
        let info = ModuleInfo::new("test", Version::new(1, 0, 0))
            .with_capability("linear")
            .with_capability("nonlinear")
            .with_size(1024);

        assert_eq!(info.name, "test");
        assert_eq!(info.capabilities.len(), 2);
        assert_eq!(info.size_bytes, 1024);
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = ModuleRegistry::new();
        let info = ModuleInfo::new("arithmetic", Version::new(1, 0, 0));

        assert!(registry.register_module(info).is_ok());
        assert!(registry.is_registered("arithmetic_v1.0.0"));
    }

    #[test]
    fn test_registry_duplicate_registration() {
        let mut registry = ModuleRegistry::new();
        let info = ModuleInfo::new("arithmetic", Version::new(1, 0, 0));

        assert!(registry.register_module(info.clone()).is_ok());
        assert!(registry.register_module(info).is_err());
    }

    #[test]
    fn test_capability_index() {
        let mut registry = ModuleRegistry::new();
        let info = ModuleInfo::new("arithmetic", Version::new(1, 0, 0))
            .with_capability("linear")
            .with_capability("integer");

        registry.register_module(info).unwrap();

        assert!(registry.has_capability("linear"));
        assert!(registry.has_capability("integer"));
        assert!(!registry.has_capability("nonlinear"));
    }

    #[test]
    fn test_module_activation() {
        let mut registry = ModuleRegistry::new();
        let info = ModuleInfo::new("test", Version::new(1, 0, 0));
        let id = info.id.clone();

        registry.register_module(info).unwrap();
        assert!(!registry.is_active(&id));

        registry.activate_module(&id).unwrap();
        assert!(registry.is_active(&id));

        registry.deactivate_module(&id).unwrap();
        assert!(!registry.is_active(&id));
    }

    #[test]
    fn test_cannot_unregister_active_module() {
        let mut registry = ModuleRegistry::new();
        let info = ModuleInfo::new("test", Version::new(1, 0, 0));
        let id = info.id.clone();

        registry.register_module(info).unwrap();
        registry.activate_module(&id).unwrap();

        assert!(registry.unregister_module(&id).is_err());
    }

    #[test]
    fn test_version_lookup() {
        let mut registry = ModuleRegistry::new();

        let v1 = ModuleInfo::new("test", Version::new(1, 0, 0));
        let v2 = ModuleInfo::new("test", Version::new(2, 0, 0));

        registry.register_module(v1).unwrap();
        registry.register_module(v2).unwrap();

        let versions = registry.get_all_versions("test");
        assert_eq!(versions.len(), 2);

        let latest = registry.get_latest_module("test").unwrap();
        assert_eq!(latest.version, Version::new(2, 0, 0));
    }

    #[test]
    fn test_registry_stats() {
        let mut registry = ModuleRegistry::new();

        let info1 = ModuleInfo::new("arith", Version::new(1, 0, 0)).with_size(1024);
        let info2 = ModuleInfo::new("logic", Version::new(1, 0, 0)).with_size(2048);

        registry.register_module(info1).unwrap();
        registry.register_module(info2).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_modules, 2);
        assert_eq!(stats.total_size_bytes, 3072);
    }
}
