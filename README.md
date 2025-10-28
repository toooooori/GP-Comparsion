# GP-Comparsion

This repository contains code for comparing different shape-constrained Gaussian process (GP) models, including monotonicity-constrained and convexity-constrained regression. It includes both MATLAB and R implementations, synthetic and real datasets, and experiment scripts for evaluation.

---

## Repository Structure

### Folders

- `CGP/`  
  MATLAB implementation of the custom constrained Gaussian Process (CGP): kernels, constrained inference, Gibbs sampling, utilities, and its own startup code. Used by scripts such as `CGP_monotone.m`, `CGP_monotone_2D.m`, and `CGP_convex.m`.

- `datasets_car/`  
  Car mileage/price datasets used in monotone regression experiments (e.g. “price should go down as mileage increases”).

- `datasets_cosh/`  
  Synthetic datasets generated from cosh-like shapes for testing smooth convex / monotone structure.

- `datasets_parabola/`  
  Synthetic parabola-shaped datasets for convex or monotone function experiments.

- `datasets_sigmoid/`  
  Synthetic sigmoid-shaped datasets for testing monotonicity with saturation / plateau behavior.

- `datasets_sine/`  
  Synthetic sine-wave datasets for stress-testing model behavior on oscillatory signals.

- `datasets_stepwise_convex/`  
  Stepwise / piecewise convex datasets to evaluate convexity-constrained regression.

- `datasets_stepwise_monotone/`  
  Stepwise / piecewise monotone datasets (including plateaus / jumps) to evaluate monotonicity-constrained regression.

- `gpstuff/`  
  The GPstuff toolbox (Gaussian process library). This is required by the monotone GP baseline implemented in `IP.m`.

- `plots/`  
  Plotting utilities and/or saved figures for fitted curves/surfaces, credible intervals, and diagnostics.

- `results_convex/`  
  Saved experiment outputs for convexity-constrained model runs.

- `results_convex_stats/`  
  Aggregated statistics and summaries computed from `results_convex`.

- `results_monotone/`  
  Saved experiment outputs for monotonicity-constrained model runs.

- `results_monotone_stats/`  
  Aggregated statistics and summaries computed from `results_monotone`.

---

### Files

- `.RData`  
  Saved R workspace image containing intermediate R objects/results.

- `.Rhistory`  
  R command history used to generate and inspect results.

- `BF_convex_1d.R`  
  Baseline (“BF”) model on 1D convex datasets, with evaluation.

- `BF_monotone_1d.R`  
  Baseline (“BF”) model on 1D monotone datasets.

- `BF_monotone_2d.R`  
  Baseline monotone model extended to 2D surfaces.

- `CGP_convex.asv`  
  MATLAB autosave/backup for `CGP_convex.m`.

- `CGP_convex.m`  
  Runs the custom constrained GP with convexity constraints, evaluates predictions, and reports metrics (MSE, coverage, etc.).

- `CGP_monotone.asv`  
  MATLAB autosave/backup for `CGP_monotone.m`.

- `CGP_monotone.m`  
  Runs the custom constrained GP with monotonicity constraints in 1D and evaluates predictive accuracy and uncertainty.

- `CGP_monotone_2D.m`  
  Runs the custom constrained GP with monotonicity constraints in 2D (e.g. enforcing monotonic behavior in multiple input dimensions).

- `GP_monotone_1d.R`  
  R implementation of a monotone GP model in 1D (Gaussian process with monotone shape constraint), used as a baseline against CGP.

- `IP.m`  
  MATLAB wrapper around a monotone GP built with GPstuff:
  - standardizes inputs/outputs,
  - adds virtual derivative observations via `gp_monotonic` to enforce monotonicity,
  - predicts on test points,
  - computes metrics (MSE, NLPD, coverage, interval width, runtime),
  - optionally plots and saves results.
  Requires GPstuff to be installed.

- `IP_2D.m`  
  Extension of `IP.m` to handle 2D monotonicity constraints (e.g. constraining one or both input dimensions to be monotone).

- `SR_convex_1d.R`  
  Alternative baseline (“SR”) for convex regression in 1D, typically a shape-restricted regression method.

- `SR_monotone_1d.R`  
  The same “SR” baseline for 1D monotone datasets.

- `baseline_convex.R`  
  Baseline convex regression model in R; fits and evaluates a convex-constrained function.

- `baseline_monotone.r`  
  Baseline monotone regression model in R; fits and evaluates a monotone-constrained function.

- `bf_2D_results1.csv`  
  Saved CSV containing metrics / summary results from 2D experiments.

- `cars-mbart.csv`  
  Real dataset (e.g. car mileage vs. price) used in monotone regression experiments.

- `data_generate.R`  
  R script for generating synthetic datasets for monotonicity experiments (e.g. stepwise monotone, sigmoid-like curves).

- `data_generate_convex.R`  
  R script for generating synthetic convex datasets.

- `fit_plot.m`  
  MATLAB plotting helper to visualize fitted means, credible intervals, and observed data.

- `main.m`  
  MATLAB entry script that ties the pipeline together:
  - loads or selects a dataset,
  - runs both IP and CGP,
  - computes summary metrics,
  - can generate plots.

- `simulate_convex.R`  
  R script to run repeated convex-regression simulations and collect performance statistics.

- `simulate_monotone.R`  
  R script to run repeated monotone-regression simulations and collect performance statistics.

- `test_convex.R`  
  R script applying the models to convex datasets and reporting predictive metrics.

- `test_monotone.R`  
  R script applying the models to monotone datasets and reporting predictive metrics.

