# Results

This folder contains the curated result artifacts included with the repository for review and reproducibility. The files are copied from the local experiment and analysis pipeline after validation. Working outputs under `data/` remain ignored by Git.

## Folder Structure

```text
results/
├── responses/                  # Complete per-model response logs
└── analysis/                   # Derived analysis tables and final workbook
```

## Response Files

The `responses/` folder contains one CSV file per model run. These files are the primary model-level observations used to build the analysis datasets.

- `gemma4-31b_seed-2187_run.csv`: Responses from `gemma4:31b`.
- `gpt-oss-20b_seed-2187_run.csv`: Responses from `gpt-oss:20b`.
- `mistral-small3.2-24b_seed-2187_run.csv`: Responses from `mistral-small3.2:24b`.
- `qwen3.6-35b_seed-2187_run.csv`: Responses from `qwen3.6:35b`.

Each response file contains the run identifier, model name, dilemma metadata, both action options and their annotated values, the exact prompt, the parsed final answer, the selected action type, the model justification, the raw response text, and any recorded error.

## Analysis Files

The `analysis/` folder contains derived artifacts used for the paper's descriptive, bootstrap, and inferential analyses.

- `value_prevalence_summary.csv`: Per-model prevalence of each moral value in the selected options, including available counts, selected counts, prevalence, and rank.
- `contrastive_value_summary.csv`: Per-model contrastive selection rates for values that appear in only one of the two options for a dilemma, including contrastive counts and GLMM eligibility.
- `bootstrap_prevalence_ci.csv`: Bootstrap 95% confidence intervals for the value prevalence estimates.
- `bootstrap_contrastive_ci.csv`: Bootstrap 95% confidence intervals for the contrastive value selection rates.
- `bootstrap_model_differences.csv`: Pairwise model differences in contrastive selection rates with bootstrap confidence intervals.
- `glmm_global_results_adjusted.csv`: Global likelihood-ratio test results from the logistic mixed-effects models, including Benjamini-Hochberg adjusted p-values and significance flags.
- `glmm_pairwise_results.csv`: Pairwise post-hoc model comparisons for values with a significant global GLMM effect.
- `final_analysis_results.xlsx`: Consolidated workbook with the sheets `value_model_summary`, `global_glmm_results`, `pairwise_results`, and `bootstrap_model_differences`.

The CSV files are intended for direct inspection and programmatic reuse. The workbook provides the same core result layers in a reviewer-friendly format.
