# GP-Comparsion

This repository contains code for comparing different shape-constrained Gaussian process (GP) models, including monotonicity-constrained and convexity-constrained regression. It includes both MATLAB and R implementations, synthetic and real datasets, and experiment scripts for evaluation.

---

## Repository Structure

### Folders

- `datasets_car/`  
  Car mileage/price datasets used in monotone regression experiments.

- `datasets_cosh/`  
  cosh datasets.

- `datasets_parabola/`  
  Parabola datasets.

- `datasets_sigmoid/`  
  Monotonic sigmoid datasets.

- `datasets_sine/`  
  Sine plus linear datasets.

- `datasets_stepwise_convex/`  
  Stepwise convex datasets.

- `datasets_stepwise_monotone/`  
  Stepwise monotone datasets.

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

- `BF_convex_1d.R`  
  Runs the BF with convexity constraints in 1D.

- `BF_monotone_1d.R`  
  Runs the BF with monotonicity constraints in 1D.

- `BF_monotone_2d.R`  
  Runs the BF with monotonicity constraints in 2D.

- `CGP_convex.m`  
  Runs the CGP with convexity constraints in 1D.
  Requires GPML to be installed.

- `CGP_monotone.m`  
  Runs the CGP with monotonicity constraints in 1D.
  Requires GPML to be installed.

- `CGP_monotone_2D.m`  
  Runs the CGP with monotonicity constraints in 2D.
  Requires GPML to be installed.

- `IP.m`  
  Runs the IP with monotonicity constraints in 1D.
  Requires GPstuff to be installed.

- `IP_2D.m`  
  Runs the IP with monotonicity constraints in 2D.
  Requires GPstuff to be installed.

- `SR_convex_1d.R`  
  Runs the SR with convexity constraints in 1D.

- `SR_monotone_1d.R`  
  Runs the SR with monotonicity constraints in 1D.

- `baseline_convex.R`  
  Baseline convex regression model.

- `baseline_monotone.r`  
  Baseline monotone regression model.

- `data_generate.R`  
  Generate synthetic datasets for monotonicity experiments.

- `data_generate_convex.R`  
  Generate synthetic datasets for convexity experiments.

- `fit_plot.m`  
  Plotting helper to visualize fitted means, credible intervals, and observed data.

- `main.m`  
  - Runs IP/CGP for monotonicity/convexity experiments and computes summary metrics.

- `simulate_convex.R`  
  Runs BF/SR/Baselines for monotonicity experiments and computes summary metrics.

- `simulate_monotone.R`  
  Runs BF/SR/Baselines for convexity experiments and computes summary metrics.

- `test_convex.R`  
  T tests of the performance for convexity experiments.

- `test_monotone.R`  
  T tests of the performance for monotonicity experiments.

