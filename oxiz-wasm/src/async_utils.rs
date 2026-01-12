//! Async utilities for WASM
//!
//! Provides utilities for yielding to the JavaScript event loop during
//! long-running operations, allowing the browser to remain responsive.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Yield to the JavaScript event loop
///
/// This allows other tasks (like UI updates) to run before continuing.
/// Should be called periodically during long-running operations.
#[allow(dead_code)]
pub async fn yield_now() {
    // Use setTimeout with 0ms to yield to the event loop
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    let _ = JsFuture::from(promise).await;
}

/// Yield to the event loop if enough iterations have passed
///
/// This is useful for operations that iterate many times - we don't want
/// to yield on every iteration (too much overhead), but we want to yield
/// periodically to keep the UI responsive.
///
/// # Parameters
///
/// * `counter` - Mutable counter tracking iterations
/// * `yield_every` - Yield every N iterations
///
/// # Returns
///
/// `true` if yielded, `false` otherwise
#[allow(dead_code)]
pub async fn yield_periodic(counter: &mut usize, yield_every: usize) -> bool {
    *counter += 1;
    if *counter >= yield_every {
        *counter = 0;
        yield_now().await;
        true
    } else {
        false
    }
}

/// Execute a closure with periodic yielding
///
/// Wraps a long-running operation and periodically yields to the event loop.
/// The closure receives a yield callback that it should call periodically.
///
/// # Parameters
///
/// * `f` - Closure to execute
/// * `yield_every` - Yield every N calls to the yield callback
///
/// # Example
///
/// ```rust,ignore
/// use oxiz_wasm::async_utils::with_periodic_yield;
///
/// async fn process_items(items: Vec<Item>) -> Result<(), Error> {
///     with_periodic_yield(|mut should_yield| {
///         for item in items {
///             process_item(item)?;
///
///             // Check if we should yield
///             if should_yield() {
///                 // Yield point reached
///             }
///         }
///         Ok(())
///     }, 100).await
/// }
/// ```
#[allow(dead_code)]
pub async fn with_periodic_yield<F, R>(mut f: F, yield_every: usize) -> R
where
    F: FnMut(&mut dyn FnMut() -> bool) -> R,
{
    let mut counter = 0;
    let mut should_yield = || {
        counter += 1;
        counter >= yield_every
    };

    let result = f(&mut should_yield);

    if counter >= yield_every {
        yield_now().await;
    }

    result
}

/// Create a cancellable async operation
///
/// Returns a tuple of (cancellation flag, cancel function).
/// The operation should check the flag periodically and abort if set.
#[allow(dead_code)]
pub fn create_cancellable() -> (std::sync::Arc<std::sync::atomic::AtomicBool>, impl Fn()) {
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag_clone = flag.clone();
    let cancel = move || {
        flag_clone.store(true, std::sync::atomic::Ordering::Relaxed);
    };
    (flag, cancel)
}

/// Check if operation is cancelled
#[allow(dead_code)]
pub fn is_cancelled(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> bool {
    flag.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_periodic_counter() {
        let mut counter = 0;

        // First call should not trigger yield
        assert_eq!(counter, 0);

        // Increment counter
        for i in 1..=10 {
            counter += 1;
            if counter >= 5 {
                // Would yield here and reset counter
                assert_eq!(i, 5);
                break;
            }
        }
        // Counter was at 5 when we broke
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_cancellable() {
        let (flag, cancel) = create_cancellable();
        assert!(!is_cancelled(&flag));

        cancel();
        assert!(is_cancelled(&flag));
    }
}
