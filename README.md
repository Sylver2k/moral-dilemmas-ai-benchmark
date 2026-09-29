# Moral Dilemmas AI Benchmark

Research code for the paper **"Value Alignment in Generative AI: An Analysis of Moral Value Preferences in Large Language Models"**.

The project prepares a sampled DailyDilemmas benchmark, prompts local or remote Ollama-compatible large language models with paired moral choices, parses the selected action, and builds analysis artifacts for descriptive summaries, bootstrap confidence intervals, GLMM input data, final workbooks, and figures.

This repository builds upon the **DailyDilemmas** dataset introduced by Chiu, Jiang, and Choi in *DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life* (ICLR 2025 spotlight; [arXiv:2410.02683](https://arxiv.org/abs/2410.02683)).

## Research Results

The following study compares the moral value preferences of four locally run LLMs (`qwen3.6:35b`, `gemma4:31b`, `mistral-small3.2:24b`, and `gpt-oss:20b`) on a stratified sample of 680 DailyDilemmas scenarios, producing 2,720 model decisions. Across models, the resulting value profiles were broadly similar. The values `self`, `honesty`, and `trust` appeared most often in the selected options, while the contrastive analysis showed especially high selection rates for options associated with `support`, `accountability`, `understanding`, `courage`, and `empathy`.

The inferential analysis found limited statistically robust differences between models. After correcting for multiple testing across 19 eligible values, only `accountability` showed a significant global model effect. For this value, `gemma4:31b` and `gpt-oss:20b` selected accountability-associated options more often than `mistral-small3.2:24b` and `qwen3.6:35b`. For all other tested values, observed descriptive differences did not survive correction for multiple comparisons.

## Repository Layout

```text
.
├── analysis.py                 # Analysis pipeline entry point
├── main.py                     # Experiment runner entry point
├── analysis/R/                 # R scripts for GLMM analyses
├── results/                    # Curated result artifacts
├── src/moral_dilemmas/         # Dataset, prompting, model, result, and analysis code
└── tests/                      # Python test suite
```

The working data directory is intentionally local-only:

```text
data/
├── raw/                        # Source DailyDilemmas CSV
├── processed/                  # Prepared experiment datasets
├── results/                    # Raw per-model run CSV files
└── analysis/                   # Derived analysis CSV/XLSX artifacts
```

## Environment Setup

Create and activate a Python environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The model runner expects an Ollama-compatible endpoint configured through environment variables. Create a local `.env` file such as:

```text
OLLAMA_HOST=http://localhost
OLLAMA_PORT=11434
OLLAMA_API_KEY=optional-token-if-required
```

## Running Experiments

Place the source dataset at:

```text
data/raw/Dilemmas_with_values_aggregated.csv
```

Then configure `MODEL_NAME`, `EXPERIMENT_RANDOM_SEED`, and related constants in `main.py`.

To run one configured model experiment:

```powershell
python main.py
```

By default, model outputs are written to:

```text
data/results/<model>_seed-<seed>_run.csv
```

Each result CSV contains the prompt metadata, selected final answer, selected action type, parsed justification, raw model response, and any parse or generation error.

## Building Analysis Artifacts

The Python analysis entry point combines per-model run files from `data/results/` and writes derived artifacts to `data/analysis/`.

```powershell
python analysis.py
```

The pipeline includes:

- `analysis_master.csv`: combined per-model result dataset
- `value_outcomes_long.csv`: one row per dilemma, model, and value
- `value_prevalence_summary.csv`: descriptive value prevalence summary
- `contrastive_value_summary.csv`: selection rates for contrastive values
- `glmm_input.csv`: R-ready GLMM input dataset

Additional functions in `analysis.py` can build bootstrap confidence intervals, the final Excel workbook, and publication-oriented figures once the prerequisite files exist.

Run the R GLMM scripts from `analysis/R/` after generating `data/analysis/glmm_input.csv`. The repository includes `renv.lock` and `renv/` settings for the R analysis environment.
