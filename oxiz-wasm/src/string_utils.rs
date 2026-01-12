//! Optimized string handling for WASM
//!
//! This module provides efficient string conversion between JavaScript and Rust
//! using TextEncoder/TextDecoder patterns and zero-copy techniques where possible.

use wasm_bindgen::prelude::*;

/// A helper for efficiently building strings that will be returned to JavaScript
///
/// This uses a StringBuilder pattern to minimize allocations when constructing
/// strings from multiple parts.
#[derive(Default)]
#[allow(dead_code)]
pub struct JsStringBuilder {
    buffer: String,
}

#[allow(dead_code)]
impl JsStringBuilder {
    /// Create a new JsStringBuilder with default capacity
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new JsStringBuilder with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
        }
    }

    /// Append a string slice to the builder
    pub fn push_str(&mut self, s: &str) -> &mut Self {
        self.buffer.push_str(s);
        self
    }

    /// Append a character to the builder
    pub fn push(&mut self, ch: char) -> &mut Self {
        self.buffer.push(ch);
        self
    }

    /// Append a formatted string to the builder
    pub fn push_fmt(&mut self, args: std::fmt::Arguments<'_>) -> &mut Self {
        use std::fmt::Write;
        let _ = write!(&mut self.buffer, "{}", args);
        self
    }

    /// Get the current length of the string
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the builder is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the builder
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Build the final JsValue
    pub fn build(self) -> JsValue {
        JsValue::from_str(&self.buffer)
    }

    /// Build a String (consuming the builder)
    pub fn into_string(self) -> String {
        self.buffer
    }

    /// Get a reference to the current buffer
    pub fn as_str(&self) -> &str {
        &self.buffer
    }
}

impl From<String> for JsStringBuilder {
    fn from(s: String) -> Self {
        Self { buffer: s }
    }
}

impl From<&str> for JsStringBuilder {
    fn from(s: &str) -> Self {
        Self {
            buffer: s.to_string(),
        }
    }
}

/// Join a vector of strings efficiently
///
/// This is optimized for the common case of joining SMT-LIB2 output lines.
pub fn join_strings(strings: &[String], separator: &str) -> String {
    if strings.is_empty() {
        return String::new();
    }

    if strings.len() == 1 {
        return strings[0].clone();
    }

    // Pre-calculate capacity to avoid reallocations
    let total_len: usize = strings.iter().map(|s| s.len()).sum();
    let sep_len = separator.len() * (strings.len() - 1);
    let capacity = total_len + sep_len;

    let mut result = String::with_capacity(capacity);

    for (i, s) in strings.iter().enumerate() {
        if i > 0 {
            result.push_str(separator);
        }
        result.push_str(s);
    }

    result
}

/// Join a vector of strings with newlines
///
/// This is a common operation when building SMT-LIB2 output.
#[inline]
pub fn join_lines(strings: &[String]) -> String {
    join_strings(strings, "\n")
}

/// Efficiently convert a Vec<String> to a JavaScript array
///
/// This avoids intermediate string concatenation.
#[allow(dead_code)]
pub fn vec_to_js_array(strings: Vec<String>) -> js_sys::Array {
    let arr = js_sys::Array::new();
    for s in strings {
        arr.push(&JsValue::from_str(&s));
    }
    arr
}

/// Efficiently convert a JavaScript array to Vec<String>
///
/// This is optimized for the common case of receiving string arrays from JavaScript.
#[allow(dead_code)]
pub fn js_array_to_vec(arr: &js_sys::Array) -> Result<Vec<String>, JsValue> {
    let len = arr.length() as usize;
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let val = arr.get(i as u32);
        if let Some(s) = val.as_string() {
            result.push(s);
        } else {
            return Err(JsValue::from_str(&format!(
                "Array element at index {} is not a string",
                i
            )));
        }
    }

    Ok(result)
}

/// Trim whitespace efficiently
///
/// This is a simple wrapper but helps document intent.
#[inline]
pub fn trim(s: &str) -> &str {
    s.trim()
}

/// Check if a string is effectively empty (empty or all whitespace)
#[inline]
pub fn is_effectively_empty(s: &str) -> bool {
    trim(s).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_builder() {
        let mut builder = JsStringBuilder::new();
        builder.push_str("Hello").push(' ').push_str("World");
        assert_eq!(builder.as_str(), "Hello World");
        assert_eq!(builder.len(), 11);
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_string_builder_with_capacity() {
        let builder = JsStringBuilder::with_capacity(100);
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());
    }

    #[test]
    fn test_string_builder_clear() {
        let mut builder = JsStringBuilder::new();
        builder.push_str("test");
        assert!(!builder.is_empty());
        builder.clear();
        assert!(builder.is_empty());
    }

    #[test]
    fn test_string_builder_from_string() {
        let builder = JsStringBuilder::from("test".to_string());
        assert_eq!(builder.as_str(), "test");
    }

    #[test]
    fn test_string_builder_from_str() {
        let builder = JsStringBuilder::from("test");
        assert_eq!(builder.as_str(), "test");
    }

    #[test]
    fn test_join_strings_empty() {
        let strings: Vec<String> = vec![];
        let result = join_strings(&strings, ", ");
        assert_eq!(result, "");
    }

    #[test]
    fn test_join_strings_single() {
        let strings = vec!["hello".to_string()];
        let result = join_strings(&strings, ", ");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_join_strings_multiple() {
        let strings = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = join_strings(&strings, ", ");
        assert_eq!(result, "a, b, c");
    }

    #[test]
    fn test_join_lines() {
        let strings = vec![
            "line1".to_string(),
            "line2".to_string(),
            "line3".to_string(),
        ];
        let result = join_lines(&strings);
        assert_eq!(result, "line1\nline2\nline3");
    }

    #[test]
    fn test_is_effectively_empty() {
        assert!(is_effectively_empty(""));
        assert!(is_effectively_empty("   "));
        assert!(is_effectively_empty("\t\n"));
        assert!(!is_effectively_empty("a"));
        assert!(!is_effectively_empty("  a  "));
    }

    #[test]
    fn test_trim() {
        assert_eq!(trim("  hello  "), "hello");
        assert_eq!(trim("\thello\n"), "hello");
        assert_eq!(trim("hello"), "hello");
    }
}
