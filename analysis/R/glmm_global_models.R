# Resolve project paths
project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
if (!file.exists(file.path(project_root, "data", "analysis", "glmm_input.csv"))) {
  project_root <- normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = TRUE)
}

input_path <- file.path(project_root, "data", "analysis", "glmm_input.csv")
output_path <- file.path(project_root, "data", "analysis", "glmm_global_results.csv")
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

# Load required modeling package
if (!requireNamespace("lme4", quietly = TRUE)) {
  stop(
    "Package 'lme4' is required. Install it with renv::install('lme4') ",
    "and then run renv::snapshot().",
    call. = FALSE
  )
}

# Load GLMM input data
data <- read.csv(input_path, stringsAsFactors = FALSE)
required_columns <- c("dilemma_id", "topic_group", "model", "value", "value_chosen")
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "), call. = FALSE)
}

# Prepare variables for mixed-effects modeling
data$model <- factor(data$model, levels = sort(unique(data$model)))
data$dilemma_id <- factor(data$dilemma_id)
data$value <- factor(data$value, levels = sort(unique(data$value)))
data$value_chosen <- as.integer(data$value_chosen)

if (!all(data$value_chosen %in% c(0L, 1L))) {
  stop("value_chosen must contain only 0 and 1.", call. = FALSE)
}

model_count <- length(levels(data$model))
values <- sort(unique(as.character(data$value)))

cat("Values to analyze:", length(values), "\n")
cat("Models:", model_count, "\n")

fit_value_model <- function(value_name, value_index, total_values) {
  cat(sprintf("[%d/%d] Fitting value: %s\n", value_index, total_values, value_name))

  # Subset one eligible value at a time
  value_data <- data[data$value == value_name, ]
  value_data$model <- droplevels(value_data$model)
  value_data$dilemma_id <- droplevels(value_data$dilemma_id)

  n_dilemmas <- length(unique(value_data$dilemma_id))
  n_observations <- nrow(value_data)

  if (n_observations != n_dilemmas * model_count) {
    warning(
      sprintf(
        "Value '%s' has %d observations for %d dilemmas and %d models.",
        value_name,
        n_observations,
        n_dilemmas,
        model_count
      ),
      call. = FALSE
    )
  }

  result <- tryCatch(
    {
      # Fit model including the LLM as fixed effect (full model)
      full_model <- lme4::glmer(
        value_chosen ~ model + (1 | dilemma_id),
        data = value_data,
        family = binomial(link = "logit")
      )

      # Fit null model with only the dilemma random intercept
      null_model <- lme4::glmer(
        value_chosen ~ 1 + (1 | dilemma_id),
        data = value_data,
        family = binomial(link = "logit")
      )

      # Compare full and null models via likelihood-ratio test
      lrt <- anova(null_model, full_model, test = "Chisq")
      lrt_row <- lrt[2, ]

      # Store compact diagnostics
      convergence_ok <- is.null(full_model@optinfo$conv$lme4$messages)
      singular_fit <- lme4::isSingular(full_model, tol = 1e-4)

      data.frame(
        value = value_name,
        n_dilemmas = n_dilemmas,
        n_observations = n_observations,
        lr_statistic = as.numeric(lrt_row[["Chisq"]]),
        df = as.numeric(lrt_row[["Df"]]),
        p_value = as.numeric(lrt_row[["Pr(>Chisq)"]]),
        convergence_ok = convergence_ok,
        singular_fit = singular_fit,
        stringsAsFactors = FALSE
      )
    },
    error = function(error) {
      warning(
        sprintf("Model failed for value '%s': %s", value_name, conditionMessage(error)),
        call. = FALSE
      )

      data.frame(
        value = value_name,
        n_dilemmas = n_dilemmas,
        n_observations = n_observations,
        lr_statistic = NA_real_,
        df = NA_real_,
        p_value = NA_real_,
        convergence_ok = FALSE,
        singular_fit = NA,
        stringsAsFactors = FALSE
      )
    }
  )

  result
}

# Fit one global model comparison per value
results <- do.call(
  rbind,
  lapply(seq_along(values), function(index) {
    fit_value_model(values[[index]], index, length(values))
  })
)

if (nrow(results) != length(values)) {
  stop("Expected one GLMM result row per value.", call. = FALSE)
}

# Sort by raw p-value, keeping failed fits at the end
results <- results[order(is.na(results$p_value), results$p_value, results$value), ]
row.names(results) <- NULL

# Write final compact result table
write.csv(results, output_path, row.names = FALSE)

cat("GLMM analysis completed\n")
cat("Results written to:", output_path, "\n")
cat("Models with convergence issues:", sum(!results$convergence_ok, na.rm = TRUE), "\n")
cat("Singular fits:", sum(results$singular_fit, na.rm = TRUE), "\n")
