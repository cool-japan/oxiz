use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use oxiz_smtcomp::regression::RegressionAnalysis;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug)]
struct HistoryEntry {
    timestamp: String,
    analysis: RegressionAnalysis,
}

#[derive(Debug, Deserialize)]
struct WrappedRegressionAnalysis {
    analysis: RegressionAnalysis,
}

pub fn render_perf_dashboard(history_dir: &Path, output_dir: &Path) -> Result<()> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory for perf dashboard: {}",
            output_dir.display()
        )
    })?;

    let mut entries = load_history(history_dir)?;
    entries.sort_by(|lhs, rhs| lhs.timestamp.cmp(&rhs.timestamp));

    let rows = entries
        .iter()
        .rev()
        .take(30)
        .map(render_row)
        .collect::<Vec<_>>()
        .join("\n");

    let body = if rows.is_empty() {
        "<tr><td colspan=\"5\">No regression history found.</td></tr>".to_string()
    } else {
        rows
    };

    let html = format!(
        "<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>OxiZ Performance Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5efe6;
      --panel: #fffaf3;
      --ink: #1e1a17;
      --muted: #6e6156;
      --ok: #d9f2d9;
      --bad: #f7d6d0;
      --border: #d2c2b3;
      --accent: #8f4e2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, \"Iowan Old Style\", serif;
      background:
        radial-gradient(circle at top left, rgba(143, 78, 44, 0.10), transparent 35%),
        linear-gradient(180deg, #f9f3eb 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 48px 20px 64px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 3.4rem);
      letter-spacing: -0.04em;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 24px;
      font-size: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 40px rgba(61, 44, 31, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      background: rgba(143, 78, 44, 0.08);
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    tr.ok {{ background: var(--ok); }}
    tr.regression {{ background: var(--bad); }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
      background: rgba(30, 26, 23, 0.08);
    }}
    .small {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    @media (max-width: 760px) {{
      th:nth-child(3), td:nth-child(3),
      th:nth-child(4), td:nth-child(4) {{
        display: none;
      }}
      main {{ padding: 28px 12px 40px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Performance Dashboard</h1>
    <p>Latest 30 perf regression runs rendered from <code>{}</code>.</p>
    <section class=\"panel\">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Findings</th>
            <th>Improvement Notes</th>
            <th>Solves</th>
          </tr>
        </thead>
        <tbody>
          {}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>",
        history_dir.display(),
        body
    );

    let output_path = output_dir.join("index.html");
    fs::write(&output_path, html).with_context(|| {
        format!(
            "Failed to write perf dashboard HTML: {}",
            output_path.display()
        )
    })?;
    Ok(())
}

fn load_history(history_dir: &Path) -> Result<Vec<HistoryEntry>> {
    if !history_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for entry in fs::read_dir(history_dir)
        .with_context(|| format!("Failed to read perf history dir: {}", history_dir.display()))?
    {
        let entry = entry.with_context(|| "Failed to read perf history entry")?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        if let Some(loaded) = load_history_entry(&path)? {
            entries.push(loaded);
        }
    }

    Ok(entries)
}

fn load_history_entry(path: &Path) -> Result<Option<HistoryEntry>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read perf history file: {}", path.display()))?;

    let analysis = if let Ok(analysis) = serde_json::from_str::<RegressionAnalysis>(&content) {
        analysis
    } else if let Ok(wrapped) = serde_json::from_str::<WrappedRegressionAnalysis>(&content) {
        wrapped.analysis
    } else {
        return Ok(None);
    };

    Ok(Some(HistoryEntry {
        timestamp: history_timestamp(path)?,
        analysis,
    }))
}

fn history_timestamp(path: &Path) -> Result<String> {
    let modified = fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for {}", path.display()))?
        .modified()
        .unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(format_timestamp(modified))
}

fn format_timestamp(timestamp: SystemTime) -> String {
    let datetime: DateTime<Utc> = DateTime::<Utc>::from(timestamp);
    datetime.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

fn render_row(entry: &HistoryEntry) -> String {
    let status_class = if entry.analysis.has_regressions {
        "regression"
    } else {
        "ok"
    };
    let status_text = if entry.analysis.has_regressions {
        "regression"
    } else {
        "ok"
    };
    let findings = if entry.analysis.findings.is_empty() {
        "No regressions detected.".to_string()
    } else {
        entry
            .analysis
            .findings
            .iter()
            .map(|finding| finding.description.clone())
            .collect::<Vec<_>>()
            .join("<br>")
    };
    let improvements = if entry.analysis.improvements.is_empty() {
        "<span class=\"small\">None</span>".to_string()
    } else {
        entry.analysis.improvements.join("<br>")
    };
    format!(
        "<tr class=\"{status_class}\">
  <td><strong>{}</strong><div class=\"small\">{}</div></td>
  <td><span class=\"badge\">{}</span></td>
  <td>{}</td>
  <td>{}</td>
  <td>{} / {}</td>
</tr>",
        entry.timestamp,
        summary_label(&entry.analysis),
        status_text,
        findings,
        improvements,
        entry.analysis.current_summary.solved(),
        entry.analysis.current_summary.total
    )
}

fn summary_label(analysis: &RegressionAnalysis) -> &'static str {
    if analysis.has_critical_regressions {
        "critical regressions"
    } else if analysis.has_regressions {
        "regressions"
    } else {
        "healthy"
    }
}

#[allow(dead_code)]
fn _stable_path(_path: &PathBuf) {}
