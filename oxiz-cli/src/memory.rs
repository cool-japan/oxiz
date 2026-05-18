//! Memory sampling utilities for the OxiZ CLI
//!
//! Provides (current_rss, peak_rss) sampling via the OS.
//! On Linux the kernel high-water-mark is read from `/proc/self/status` (`VmHWM`).
//! On other platforms peak falls back to the current RSS value.

/// Returns `(current_rss_bytes, peak_rss_bytes)`.
///
/// On Linux `peak_rss_bytes` is read from `/proc/self/status` (`VmHWM:`), which
/// is the true kernel-tracked high-water-mark and is therefore always ≥ the
/// current RSS. On other platforms it falls back to the current RSS obtained
/// from `sysinfo`.
pub fn rss_and_peak() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        rss_and_peak_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        rss_and_peak_fallback()
    }
}

/// Linux implementation: parse `/proc/self/status` for `VmRSS:` and `VmHWM:`.
#[cfg(target_os = "linux")]
fn rss_and_peak_linux() -> (u64, u64) {
    let mut current_rss: u64 = 0;
    let mut peak_rss: u64 = 0;

    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                // Values are in kB; convert to bytes
                let kb = parts[1].parse::<u64>().unwrap_or(0);
                match parts[0] {
                    "VmRSS:" => current_rss = kb * 1024,
                    "VmHWM:" => peak_rss = kb * 1024,
                    _ => {}
                }
            }
        }
    }

    // If VmHWM is somehow missing, fall back to current.
    if peak_rss == 0 {
        peak_rss = current_rss;
    }

    (current_rss, peak_rss)
}

/// Non-Linux fallback: use `sysinfo` for the current RSS; report it as peak as
/// well since we have no OS-level high-water-mark.
#[cfg(not(target_os = "linux"))]
fn rss_and_peak_fallback() -> (u64, u64) {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

    let pid = Pid::from_u32(std::process::id());
    let mut sys = System::new();
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let rss = sys.process(pid).map(|p| p.memory()).unwrap_or(0);
    (rss, rss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rss_and_peak_returns_nonzero() {
        let (current, peak) = rss_and_peak();
        // The process must use at least some memory.
        assert!(current > 0, "current RSS should be > 0");
        assert!(peak > 0, "peak RSS should be > 0");
    }

    #[test]
    fn test_peak_geq_current() {
        let (current, peak) = rss_and_peak();
        assert!(
            peak >= current,
            "peak ({}) must be >= current ({})",
            peak,
            current
        );
    }
}
