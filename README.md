# Why Loss Aversion May Be Overestimated

This repository contains the data and code for reproducing the results and figures reported in the paper *Why Loss Aversion May Be Overestimated*.

## Analysis workflow

| Script | Description |
|---|---|
| [`01_gamble_generation.ipynb`](src/01_gamble_generation.ipynb) | Generates and matches the three-outcome gambles used in the experiment. |
| [`02_glm_analysis.R`](src/02_glm_analysis.R) | Fits the Bayesian multilevel logistic regression of participants' choices. |
| [`03_cog_model_fitting.R`](src/03_cog_model_fitting.R) | Fits the hierarchical PT, CPT, and mean--variance--loss models. |
| [`04_cog_model_evaluation.R`](src/04_cog_model_evaluation.R) | Compares the cognitive models, summarizes their parameters, and produces posterior-predictive checks. |
| [`05_parameter_recovery.R`](src/05_parameter_recovery.R) | Runs the parameter-recovery analysis for the selected CPT model. |
| [`06_tom2007_reanalysis.R`](src/07_tom2007_reanalysis.R) | Reanalyzes the mixed-gamble data from Tom et al. (2007) using four alternative models. |

The scripts are numbered in the order of the main analysis workflow.
