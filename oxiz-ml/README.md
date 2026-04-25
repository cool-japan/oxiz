# oxiz-ml — Machine Learning-Guided Heuristics for OxiZ

Machine learning-guided heuristics for the OxiZ SMT solver, providing adaptive branching, clause deletion, restart policies, and tactic selection driven by learned models trained on solving experience.

| Metric | Value |
|:-------|:------|
| Version | 0.2.1 |
| Status | Alpha |
| Release Date | 2026-04-25 |
| Tests | 151 passing |
| Source Files | 27 |
| Rust LoC | ~7,351 |
| Public API | 296 items |
| `todo!`/`unimplemented!` | 0 |

## Features

- **ML-Enhanced VSIDS Branching** — learned variable selection augments classical VSIDS scores with feature-based predictions
- **Adaptive Clause Deletion** — usefulness predictor scores learned clauses; poor clauses are pruned earlier to reduce memory pressure
- **Adaptive Restart Policies** — online-learned policy selects among restart schedules based on solver state signals
- **Formula Classification and Tactic Selection** — extracts formula features to pick the best solving strategy from a portfolio before search begins
- **Online Learning** — solver decisions and outcomes update models incrementally during a run
- **Offline Training** — batch trainer consumes collected solve traces to pre-train models for persistent deployment
- **Pure Rust** — no C/C++, no FFI; all model types (linear, decision tree, neural network) are implemented natively

## Architecture

| Module | Description |
|:-------|:------------|
| `branching` | ML-enhanced VSIDS: feature extraction, online learner, VSIDS integration |
| `clause_learning` | Clause usefulness predictor and ML-driven deletion policy |
| `models` | Internal model types: linear model, decision tree, neural network, tensor, activations, loss, optimizer |
| `restarts` | Adaptive restart policy: policy learner, online adaptation to solver state |
| `tactic` | Formula feature extraction, tactic portfolio, strategy selector |
| `training` | Data collection from solver runs, offline batch trainer, online learning pipeline |

## Quick Start

Add `oxiz-ml` to your workspace member and enable it from `oxiz-solver`:

```toml
[dependencies]
oxiz-ml = { workspace = true }
```

### Enabling ML Branching

```rust
use oxiz_ml::branching::{MlVsids, BranchingFeatures};
use oxiz_ml::training::online_learning::OnlineLearner;

// Build an ML-enhanced VSIDS heuristic
let learner = OnlineLearner::default();
let mut ml_vsids = MlVsids::new(learner);

// Feed solver state features and obtain a branching decision
let features = BranchingFeatures::from_solver_state(&state);
let decision = ml_vsids.select_literal(&features);
```

### Tactic Selection

```rust
use oxiz_ml::tactic::{FormulaFeatures, TacticSelector, TacticPortfolio};

let portfolio = TacticPortfolio::default();
let selector = TacticSelector::new(portfolio);

let features = FormulaFeatures::extract(&formula);
let tactic = selector.select(&features);
tactic.apply(&mut solver)?;
```

### Offline Training

```rust
use oxiz_ml::training::{DataCollector, OfflineTrainer};

// Collect solve traces
let mut collector = DataCollector::new();
collector.record_run(&solve_result);

// Train a model from collected data
let trainer = OfflineTrainer::default();
let model = trainer.train(collector.dataset())?;
model.save("branching_model.json")?;
```

## Benchmarks

Two criterion benchmark suites are included under `benches/`:

| Benchmark | Measures |
|:----------|:---------|
| `ml_overhead` | Wall-clock overhead introduced by ML prediction calls during a solve |
| `prediction_accuracy` | Branching prediction accuracy against a ground-truth optimal policy |

Run with:

```sh
cargo bench -p oxiz-ml
```

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
