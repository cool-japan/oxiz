//! End-to-end tests for SMT-COMP submission generation.

#[cfg(test)]
mod tests {
    use oxiz_smtcomp::submission::{generate_submission_package, SubmissionConfig, Track};
    use std::fs;

    fn temp_dir(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oxiz_e2e_{}_{}", suffix, std::process::id()))
    }

    #[test]
    fn test_submission_module_accessible() {
        let _ = SubmissionConfig::default_oxiz_2026();
    }

    #[test]
    fn test_all_tracks_have_unique_suffixes() {
        let mut suffixes: Vec<&str> =
            Track::all().iter().map(|t| t.as_starexec_suffix()).collect();
        suffixes.sort_unstable();
        suffixes.dedup();
        assert_eq!(suffixes.len(), Track::all().len());
    }

    #[test]
    fn test_generate_package_has_all_track_scripts() {
        let dir = temp_dir("track_scripts");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        assert!(pkg.root_dir.exists());

        for track in Track::all() {
            let name = format!("starexec_run_{}", track.as_starexec_suffix());
            let script = pkg.root_dir.join("bin").join(&name);
            assert!(script.exists(), "missing: {name}");
            let content = fs::read_to_string(&script).expect("read script");
            assert!(content.contains("smtcomp2026"), "script missing binary ref");
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_generate_package_has_description_and_conf() {
        let dir = temp_dir("conf");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        assert!(pkg.root_dir.join("description.txt").exists());
        assert!(pkg.root_dir.join("starexec_conf.xml").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_generate_package_conf_contains_solver_name() {
        let dir = temp_dir("solver_name");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        let conf =
            fs::read_to_string(pkg.root_dir.join("starexec_conf.xml")).expect("read conf");
        assert!(
            conf.contains("OxiZ") || conf.contains("oxiz"),
            "conf missing solver name"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_single_query_script_has_no_track_flag() {
        let dir = temp_dir("single_query");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        let default_script = fs::read_to_string(
            pkg.root_dir.join("bin").join("starexec_run_default"),
        )
        .expect("read default script");
        // Single-query track should not pass --track to preserve backward compat.
        assert!(
            default_script.contains("smtcomp2026"),
            "default script must invoke binary"
        );
        // The default script must NOT carry a --track flag (backward compat).
        assert!(
            !default_script.contains("--track"),
            "default script must not pass --track flag"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unsat_core_script_passes_track_flag() {
        let dir = temp_dir("unsat_core");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        let script = fs::read_to_string(
            pkg.root_dir.join("bin").join("starexec_run_unsat_core"),
        )
        .expect("read unsat_core script");
        assert!(
            script.contains("unsat"),
            "unsat-core script should reference unsat-core track"
        );
        assert!(
            script.contains("--track"),
            "unsat-core script should pass --track flag"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_track_scripts_list_in_package() {
        let dir = temp_dir("track_list");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        // Package should list all five tracks
        assert_eq!(pkg.track_scripts.len(), Track::all().len());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_summary_contains_track_info() {
        let dir = temp_dir("summary");
        let cfg = SubmissionConfig::default_oxiz_2026();
        let pkg = generate_submission_package(&cfg, &dir).expect("generate failed");
        let summary = pkg.summary();
        // Summary must mention at least one track's display name
        assert!(
            summary.contains("Single Query") || summary.contains("Incremental"),
            "summary should contain track names"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
