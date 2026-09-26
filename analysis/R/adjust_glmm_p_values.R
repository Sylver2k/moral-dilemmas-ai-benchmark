# Resolve project paths
project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(project_root, "data", "analysis", "glmm_global_results.csv"))) {
  project_root <- normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = TRUE)
}

input_path <- file.path(project_root, "data", "analysis", "glmm_global_results.csv")
output_path <- file.path(project_root, "data", "analysis", "glmm_global_results_adjusted.csv")
alpha <- 0.05

# Load raw GLMM results
results <- read.csv(input_path, stringsAsFactors = FALSE)
required_columns <- c(
  "value",
  "n_dilemmas",
  "n_observations",
  "lr_statistic",
  "df",
  "p_value",
  "convergence_ok",
  "singular_fit"
)
missing_columns <- setdiff(required_columns, names(results))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "), call. = FALSE)
}

raw_p_values <- results$p_value

# Correct global p-values for multiple testing
results$p_adjusted <- p.adjust(results$p_value, method = "BH")
results$significant <- results$p_adjusted < alpha

# Keep output columns compact and ordered
results <- results[
  ,
  c(
    "value",
    "n_dilemmas",
    "n_observations",
    "lr_statistic",
    "df",
    "p_value",
    "p_adjusted",
    "significant",
    "convergence_ok",
    "singular_fit"
  )
]

# Lightweight sanity checks
if (length(raw_p_values) != nrow(results)) {
  stop("Row count changed during p-value adjustment.", call. = FALSE)
}
if (!identical(raw_p_values, results$p_value)) {
  stop("Raw p_value column changed during p-value adjustment.", call. = FALSE)
}
valid_adjusted <- !is.na(results$p_adjusted)
if (!all(results$p_adjusted[valid_adjusted] >= 0 & results$p_adjusted[valid_adjusted] <= 1)) {
  stop("Adjusted p-values must be between 0 and 1.", call. = FALSE)
}
if (!all(results$significant[valid_adjusted] == (results$p_adjusted[valid_adjusted] < alpha))) {
  stop("Significance flag does not match adjusted p-value threshold.", call. = FALSE)
}

# Sort by adjusted evidence strength
results <- results[
  order(is.na(results$p_adjusted), results$p_adjusted, results$p_value, results$value),
]
row.names(results) <- NULL

# Write adjusted result table
write.csv(results, output_path, row.names = FALSE)

cat("Global tests:", nrow(results), "\n")
cat("BH correction applied\n")
cat("Significant after BH correction:", sum(results$significant, na.rm = TRUE), "\n")

significant_results <- results[results$significant %in% TRUE, ]
if (nrow(significant_results) > 0) {
  cat("Significant values:\n")
  for (row_index in seq_len(nrow(significant_results))) {
    cat(
      sprintf(
        "- %s: raw p=%.6g, adjusted p=%.6g\n",
        significant_results$value[[row_index]],
        significant_results$p_value[[row_index]],
        significant_results$p_adjusted[[row_index]]
      )
    )
  }
} else {
  cat("No global model effects remain significant after BH correction.\n")
}

cat("Results written to:", output_path, "\n")
