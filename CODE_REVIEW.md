# Code Review and Improvement Suggestions

This document captures a comprehensive analysis of the **fin_ppl** project and outlines concrete suggestions for refactoring and improvement.

## 1) Project layout
- **Root**
  - `main.py`          ← training/testing entrypoint
  - `GruVTwo.py`       ← PPO implementation (Actor/Critic/Memory/Learning)
  - `TimFinEnv.py`     ← custom Gym portfolio‐allocation environment
  - `utils.py`         ← single function for learning‐curve plotting
  - `PostPlot.py`      ← ad-hoc log-concatenation & trend plotting
  - `requirements.txt`, `README.md`, `images/`, `tmp/`, Parquet data files
- **Data/**
  - `DataLoader.py`, `DataProcessor.py`, `DataValidation.py`, `LivePlot.py` (pipeline & validation)
- **Old/** ← legacy experiments (safe to archive or delete)

## 2) Data pipeline (DataLoader & DataProcessor)
**Findings**
- Fetches via yfinance + `ta` + sklearn, computes ~20 indicators
- Per-ticker loops for OBV, VWAP, ATR, etc. → slow on many tickers
- Scales all numeric columns except a small hard-coded exclude list
- `DataValidation.py` broken (wrong import, hard-wired paths)

- **Suggestions**
- Vectorize indicator computation (groupby-transform or library calls)
- Separate “fill/interpolate” vs “scale”; preserve unscaled copy for analysis (move scaling to pipeline)
- Consolidate pipeline steps into a single class/function with clear I/O and tests
- Parameterize indicators, sentiment, macro, and windowing via config or CLI (e.g., `--sentiment_file`, `--macro_file`, `--lookback_window`, `--skip_indicators`)
- Add placeholder columns for sub-agent features (`anomaly_score`, `short_term_pred`, `long_term_pred`, `sentiment_pred`) in `DataProcessor.compute_indicators`
- Fix DataValidation script, add assertions to ensure no NaNs in observations

## 3) Environment (TimFinEnv.py)
- **Findings**
- The intended objective of “dynamically allocating a portfolio” is unclear; the environment lacks a clear investment goal.
- Observation = `[market_features | lookback_mean/std | portfolio_features]` per stock
- Hard-coded feature count (e.g. `reshape(29,…)`) and magic numbers
- `action_space = Box(shape=(n_stocks,))`, but code treats last element as cash allocation
- Complex, partly commented reward logic with gating and penalties
- Absolute Windows paths for checkpoints/logs

**Suggestions**
- Compute obs and action shapes programmatically (no magic “29”)
- Explicitly include cash allocation: `Box(shape=(n_stocks+1,))`
- Simplify reward into named sub‐functions; add unit tests for each component
- Add type hints and shape/assertion checks in `reset`, `step`, `_get_observation`
- Remove hard-coded paths; configure via CLI or config file

## 4) PPO Agent and Networks (GruVTwo.py)
**Findings**
- Actor/Critic: GRU→FC producing Dirichlet policy and value estimate
- `PPOMemory` uses Python lists; potential memory buildup
- In `learn()`:
  - GAE implementation ok, but manual eps-clamp on log-probs has a likely bug (`clamp(max=-eps)`)
  - Global `training_log` DataFrame dumped to Parquet each learn call
  - Mixed `print()` calls and global list rather than structured logging

**Suggestions**
- Replace list-based memory with fixed buffers or torch tensors for efficiency
- Fix clamp logic: ensure only `min=eps`, not negative max
- Remove global log list; integrate TensorBoard or Weights & Biases for metrics
- Parameterize network hyperparameters via config/CLI
- Add dropout/layernorm and better initialization for stability
- Write tests for GAE correctness and PPO surrogate behavior on simple tasks

## 5) Training script (main.py)
**Findings**
- Hard-wired hyperparameters and absolute data/checkpoint paths
- Always calls `agent.load_models()` before training
- No random seed control → non-reproducible
- Mixed `print()` vs no dedicated logger

**Suggestions**
- Move all hyperparameters and file paths into argparse flags or a YAML config
- Introduce `--resume` to load checkpoints only when desired
- Add `--seed` argument to set `np.random.seed()` and `torch.manual_seed()`
- Use Python `logging` consistently with levels and handlers

## 6) Utilities & plotting
**Findings**
- Three separate plotting modules (`utils.py`, `LivePlot.py`, `PostPlot.py`) with overlap
- Hard-coded folder paths; infinite loop in `live_plot` requires manual interrupt

**Suggestions**
- Consolidate plotting into one module with configurable inputs
- Replace live‐plot loops with TensorBoard/Visdom integration
- Provide a single CLI command for post-training visualization

## 7) Project maintenance & hygiene
**Findings**
- `requirements.txt` missing `yfinance`, `ta`, `sklearn`, `seaborn`, `matplotlib`
- README usage examples don’t match current code (import paths)
- No unit tests, no CI, no packaging (`__init__.py`, setup)
- `Old/` directory is distracting clutter

**Suggestions**
- Add `tests/` with pytest; cover data, env, agent methods
- Introduce CI (GitHub Actions) for linting, testing, and optional nightly runs
- Restructure into a proper package (e.g. `finppl/agent.py`, `finppl/env.py`)
- Archive or remove `Old/` code

## 8) Roadmap—next refactoring phases
1. Establish configuration system (YAML + CLI); eliminate hard-coded paths/params
2. Package layout + dependencies fix + tests + CI (added pytest tests and GitHub Actions workflow)
3. Refactor DataProcessor for performance & clarity
4. Tidy Env interface: shape safety, reward clarity, unit tests
5. Clean up PPO code: memory buffers, logging → TensorBoard, fix clamp bug
6. Harmonize plotting module
7. Update README & examples for new CLI/package API

---
*Generated on* `$(date)`