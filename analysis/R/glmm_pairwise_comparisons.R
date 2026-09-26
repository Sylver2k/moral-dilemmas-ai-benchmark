# Resolve project paths
project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(project_root, "data", "analysis", "glmm_input.csv"))) {
  project_root <- normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = TRUE)
}

glmm_input_path <- file.path(project_root, "data", "analysis", "glmm_input.csv")
adjusted_results_path <- file.path(project_root, "data", "analysis", "glmm_global_results_adjusted.csv")
output_path <- file.path(project_root, "data", "analysis", "glmm_pairwise_results.csv")
alpha <- 0.05

renv_r_version <- paste0("R-", R.version$major, ".", strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1])
renv_library_path <- file.path(
  project_root,
  "renv",
  "library",
  "windows",
  renv_r_version,
  R.version$platform
)
if (dir.exists(renv_library_path)) {
  .libPaths(c(renv_library_path, .libPaths()))
}

# Load required modeling packages
if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("Package 'lme4' is required. Install it with renv::install('lme4').", call. = FALSE)
}
if (!requireNamespace("emmeans", quietly = TRUE)) {
  stop("Package 'emmeans' is required. Install it with renv::install('emmeans').", call. = FALSE)
}

# Load GLMM input and adjusted global test results
glmm_data <- read.csv(glmm_input_path, stringsAsFactors = FALSE)
adjusted_results <- read.csv(adjusted_results_path, stringsAsFactors = FALSE)

required_glmm_columns <- c("dilemma_id", "model", "value", "value_chosen")
missing_glmm_columns <- setdiff(required_glmm_columns, names(glmm_data))
if (length(missing_glmm_columns) > 0) {
  stop("Missing GLMM input columns: ", paste(missing_glmm_columns, collapse = ", "), call. = FALSE)
}

required_adjusted_columns <- c("value", "significant")
missing_adjusted_columns <- setdiff(required_adjusted_columns, names(adjusted_results))
if (length(missing_adjusted_columns) > 0) {
  stop(
    "Missing adjusted result columns: ",
    paste(missing_adjusted_columns, collapse = ", "),
    call. = FALSE
  )
}

# Prepare deterministic factor levels
glmm_data$model <- factor(glmm_data$model, levels = sort(unique(glmm_data$model)))
glmm_data$dilemma_id <- factor(glmm_data$dilemma_id)
glmm_data$value <- factor(glmm_data$value, levels = sort(unique(glmm_data$value)))
glmm_data$value_chosen <- as.integer(glmm_data$value_chosen)

if (!all(glmm_data$value_chosen %in% c(0L, 1L))) {
  stop("value_chosen must contain only 0 and 1.", call. = FALSE)
}

model_count <- length(levels(glmm_data$model))
significant_values <- sort(unique(adjusted_results$value[adjusted_results$significant %in% TRUE]))

cat("Significant global values:", length(significant_values), "\n")

empty_results <- function() {
  data.frame(
    value = character(),
    model_1 = character(),
    model_2 = character(),
    p_value = numeric(),
    p_adjusted = numeric(),
    significant = logical(),
    stringsAsFactors = FALSE
  )
}

parse_contrast_models <- function(contrast_labels) {
  model_pairs <- strsplit(contrast_labels, " - ", fixed = TRUE)
  if (!all(vapply(model_pairs, length, integer(1)) == 2L)) {
    stop("Could not parse all emmeans contrast labels into model pairs.", call. = FALSE)
  }

  clean_model_name <- function(model_name) {
    sub("^\\((.*)\\)$", "\\1", model_name)
  }

  model_1 <- vapply(model_pairs, function(pair) pair[[1]], character(1))
  model_2 <- vapply(model_pairs, function(pair) pair[[2]], character(1))

  data.frame(
    model_1 = clean_model_name(model_1),
    model_2 = clean_model_name(model_2),
    stringsAsFactors = FALSE
  )
}

fit_pairwise_for_value <- function(value_name) {
  cat("Post-hoc analysis:", value_name, "\n")

  # Subset one globally significant value
  value_data <- glmm_data[glmm_data$value == value_name, ]
  value_data$model <- droplevels(value_data$model)
  value_data$dilemma_id <- droplevels(value_data$dilemma_id)

  if (length(levels(value_data$model)) != model_count) {
    warning(
      sprintf(
        "Value '%s' has %d model levels, expected %d.",
        value_name,
        length(levels(value_data$model)),
        model_count
      ),
      call. = FALSE
    )
  }

  result <- tryCatch(
    {
      # Fit the same GLMM used for the global test
      full_model <- lme4::glmer(
        value_chosen ~ model + (1 | dilemma_id),
        data = value_data,
        family = binomial(link = "logit")
      )

      # Calculate all pairwise model contrasts
      estimated_means <- emmeans::emmeans(full_model, ~ model)
      pairwise_contrasts <- pairs(estimated_means, adjust = "none")
      contrast_df <- as.data.frame(pairwise_contrasts)
      model_pairs <- parse_contrast_models(contrast_df$contrast)

      pairwise_df <- data.frame(
        value = value_name,
        model_1 = model_pairs$model_1,
        model_2 = model_pairs$model_2,
        p_value = contrast_df$p.value,
        stringsAsFactors = FALSE
      )

      # Correct pairwise p-values within the current value
      pairwise_df$p_adjusted <- p.adjust(pairwise_df$p_value, method = "BH")
      pairwise_df$significant <- pairwise_df$p_adjusted < alpha
      pairwise_df
    },
    error = function(error) {
      warning(
        sprintf("Pairwise comparison failed for value '%s': %s", value_name, conditionMessage(error)),
        call. = FALSE
      )

      data.frame(
        value = value_name,
        model_1 = NA_character_,
        model_2 = NA_character_,
        p_value = NA_real_,
        p_adjusted = NA_real_,
        significant = NA,
        stringsAsFactors = FALSE
      )
    }
  )

  result
}

# Fit post-hoc comparisons only for globally significant values
if (length(significant_values) == 0) {
  results <- empty_results()
  cat("No significant global values; writing empty pairwise result file.\n")
} else {
  results <- do.call(rbind, lapply(significant_values, fit_pairwise_for_value))
}

if (nrow(results) > 0) {
  valid_adjusted <- !is.na(results$p_adjusted)
  if (!all(results$p_adjusted[valid_adjusted] >= 0 & results$p_adjusted[valid_adjusted] <= 1)) {
    stop("Adjusted pairwise p-values must be between 0 and 1.", call. = FALSE)
  }
  if (!all(results$significant[valid_adjusted] == (results$p_adjusted[valid_adjusted] < alpha))) {
    stop("Pairwise significance flag does not match adjusted p-value threshold.", call. = FALSE)
  }

  results <- results[order(results$value, is.na(results$p_adjusted), results$p_adjusted, results$model_1, results$model_2), ]
  row.names(results) <- NULL
}

# Write compact pairwise result table
write.csv(results, output_path, row.names = FALSE)

cat("Significant pairwise comparisons:", sum(results$significant, na.rm = TRUE), "\n")
significant_pairs <- results[results$significant %in% TRUE, ]
if (nrow(significant_pairs) > 0) {
  for (row_index in seq_len(nrow(significant_pairs))) {
    cat(
      sprintf(
        "- %s: %s vs %s\n",
        significant_pairs$value[[row_index]],
        significant_pairs$model_1[[row_index]],
        significant_pairs$model_2[[row_index]]
      )
    )
  }
}
cat("Results written to:", output_path, "\n")
