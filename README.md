# Turbofan Engine - Remaining Useful Life (RUL) Prediction

This repository contains code and results for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using NASA turbofan datasets (FD001–FD004). The code implements data loading, preprocessing, Partial Least Squares (PLS) regression, Kernel PLS (KPLS) with several kernels, and hyperparameter optimization (grid/optimizer and genetic algorithms).

This README was generated from the source files in this folder. It summarizes the project structure, main scripts, data, usage examples, dependencies, and notes.

## Quick summary
- Goal: Train models on the training partitions and predict RUL on test partitions for FD001..FD004 datasets.
- Approaches included: classical PLS, Kernel PLS (RBF, Laplacian, Cauchy, polynomial), hyperparameter optimization (custom optimizer), and a genetic search for kernel gamma and latent variables.
- Outputs: CSV comparison files (predicted vs true RUL), plots (residuals, observed_vs_predicted, press/q2 curves), and per-dataset results saved under `Results/`.

## Folder layout

- `NASA-Turbofan-data/` — expected location of the raw NASA data files (train/test/RUL files). The code assumes files such as `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` exist in subfolders under here.
- `load_data.py` — helper script to load and inspect the raw text datasets and run a small PCA visualization.
- `project_pls.py` — main PLS pipeline: data loading, filtering (zero-variance sensors), rolling-window features (mean/max/min), RUL counting, normalization, PLS cross-validation, final training and evaluation, and plots.
- `project_kpls.py`, `project_kpls_genetic.py`, `project_kpls_optimizer.py` — Kernel PLS variants. They include KPLS fitting, genetic search (`run_genetic`) and an optimizer-based search strategy.
- `kernel.py` — custom kernels (Cauchy, polynomial) and helper functions for gamma bounds.
- `optimizer.py` — scalar/bounded optimizer that searches kernel parameters across latent variables using GroupKFold and returns Q² scores.
- `genetic.py` — genetic algorithm implementation that searches for kernel gamma and number of latent variables (n_lv) using cross-validated Q² as fitness.
- `print_data.py` — small utility for quickly printing dataset info.
- `optimizer.py`, `kernel.py`, `genetic.py` — supporting code for KPLS hyperparameter selection.
- `comparison_results.csv`, `test_comparison_results.csv` — example outputs containing Y_true and Y_pred values.
- `Results/` — directory used to save results (CSV and image files).

## Data description

The project uses the NASA turbofan engine degradation simulation datasets. Each input text file has rows that represent cycles and columns that include:

1) unit number
2) time (cycles)
3–5) three operating settings
6–26) sensor measurements (21 sensors) — source code refers to `sensor measurement 1..21`

The training partition contains full life runs (run to failure). The test partition contains truncated runs and separate RUL files containing ground-truth RUL per unit.

Files in the repo (examples):
- `FD001_train.txt`, `FD001_test.txt`, `RUL_FD001.txt` — for dataset FD001 (the code references these names under `NASA-Turbofan-data/data/`).


