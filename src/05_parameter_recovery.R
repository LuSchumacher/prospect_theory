library(tidyverse)
library(cmdstanr)
library(patchwork)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ---------------------------------------------------------------------------- #
# SETTINGS
# ---------------------------------------------------------------------------- #
NUM_SIMS <- 2

RECOVERY_DIR <- "../data/simulation_data/recovery_cpt_sign_separated"
TRUE_PARAMETER_FILE <- file.path(RECOVERY_DIR, "true_parameters.csv")
CHOICE_DIR <- file.path(RECOVERY_DIR, "choices")
SUMMARY_DIR <- file.path(RECOVERY_DIR, "fit_summaries")
DIAGNOSTIC_DIR <- file.path(RECOVERY_DIR, "diagnostics")

dir.create(RECOVERY_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CHOICE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(SUMMARY_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DIAGNOSTIC_DIR, recursive = TRUE, showWarnings = FALSE)

PARAMETER_LEVELS <- c("lambda", "alpha", "tau", "gamma")

PARAMETER_LABELS <- c(
  lambda = expression(lambda),
  alpha = expression(alpha),
  tau = expression(tau),
  gamma = expression(gamma)
)

EFFECT_LABELS <- c(
  b_lambda = expression(b[lambda]),
  b_alpha = expression(b[alpha]),
  b_tau = expression(b[tau]),
  b_gamma = expression(b[gamma])
)

# ---------------------------------------------------------------------------- #
# EMPIRICAL DESIGN AND MODEL
# ---------------------------------------------------------------------------- #
empirical_data <- read_csv(
  "../data/study_data_prepared.csv",
  show_col_types = FALSE
) |>
  mutate(
    subject_id = dense_rank(id),
    condition = if_else(gamble_type == "confounded", 1L, 2L)
  )

NUM_SUBJECTS <- n_distinct(empirical_data$subject_id)
NUM_TRIALS <- nrow(empirical_data)
SUBJECT_ID <- empirical_data$subject_id
CONDITION <- empirical_data$condition

OUTCOMES_A <- as.matrix(
  empirical_data[, c("outcome_a1", "outcome_a2", "outcome_a3")]
)

OUTCOMES_B <- as.matrix(
  empirical_data[, c("outcome_b1", "outcome_b2", "outcome_b3")]
)

cpt_model <- cmdstan_model(
  "../stan_models/model_2_2.stan",
  cpp_options = list(stan_threads = TRUE)
)

PARAMETER_NAMES <- c(
  "lambda", "alpha", "tau", "gamma",
  "lambda_out", "alpha_out", "tau_out", "gamma_out",
  "intercept_lambda", "intercept_alpha",
  "intercept_tau", "intercept_gamma",
  "b_lambda", "b_alpha", "b_tau", "b_gamma"
)

# ---------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------- #
softplus <- function(x) {
  log1p(exp(-abs(x))) + pmax(x, 0)
}

prelec_weight <- function(p, gamma) {
  exp(-(-log(p))^gamma)
}

get_cpt_utility <- function(outcomes, lambda, alpha, gamma) {
  values <- numeric(length(outcomes))
  gains <- outcomes >= 0
  values[gains] <- outcomes[gains]^alpha
  values[!gains] <- -lambda * abs(outcomes[!gains])^alpha
  
  values <- sort(values)
  weight_one_third <- prelec_weight(1 / 3, gamma)
  weight_two_thirds <- prelec_weight(2 / 3, gamma)
  
  weight_one_third * values[1] +
    (weight_two_thirds - weight_one_third) * values[2] +
    weight_one_third * values[3]
}

sample_parameters <- function(num_subjects, random_effect_sd = 0.25) {
  effect_coding <- c(-0.5, 0.5)
  
  # These ranges cover the empirically relevant region while avoiding
  # near-deterministic choices in most simulated datasets.
  intercept_lambda <- runif(1, 0.75, 2.50)
  intercept_alpha <- runif(1, -0.50, 0.50)
  intercept_tau <- runif(1, -1.50, 1.00)
  intercept_gamma <- runif(1, -0.40, 0.70)
  
  # The ranges include the condition effects estimated in the empirical data.
  b_lambda <- runif(1, -1.50, 1.50)
  b_alpha <- runif(1, -0.70, 0.70)
  b_tau <- runif(1, -1.50, 1.50)
  b_gamma <- runif(1, -0.70, 0.70)
  
  fixed_effects <- c(
    lambda = intercept_lambda,
    alpha = intercept_alpha,
    tau = intercept_tau,
    gamma = intercept_gamma
  )
  
  condition_effects <- c(
    lambda = b_lambda,
    alpha = b_alpha,
    tau = b_tau,
    gamma = b_gamma
  )
  
  parameter_matrices <- map2(
    fixed_effects,
    condition_effects,
    function(intercept, condition_effect) {
      random_intercept <- rnorm(num_subjects, 0, random_effect_sd)
      random_slope <- rnorm(num_subjects, 0, random_effect_sd)
      
      map_dfc(effect_coding, function(condition_code) {
        softplus(
          intercept +
            random_intercept +
            (condition_effect + random_slope) * condition_code
        )
      }) |>
        as.matrix() |>
        t()
    }
  )
  
  names(parameter_matrices) <- PARAMETER_LEVELS
  
  group_values <- map2(
    fixed_effects,
    condition_effects,
    ~ softplus(.x + .y * effect_coding)
  )
  
  tibble(
    subject = rep(seq_len(num_subjects), each = 2),
    condition = rep(1:2, times = num_subjects),
    lambda = as.vector(parameter_matrices$lambda),
    alpha = as.vector(parameter_matrices$alpha),
    tau = as.vector(parameter_matrices$tau),
    gamma = as.vector(parameter_matrices$gamma),
    intercept_lambda = intercept_lambda,
    intercept_alpha = intercept_alpha,
    intercept_tau = intercept_tau,
    intercept_gamma = intercept_gamma,
    b_lambda = b_lambda,
    b_alpha = b_alpha,
    b_tau = b_tau,
    b_gamma = b_gamma,
    mu_lambda = rep(group_values$lambda, times = num_subjects),
    mu_alpha = rep(group_values$alpha, times = num_subjects),
    mu_tau = rep(group_values$tau, times = num_subjects),
    mu_gamma = rep(group_values$gamma, times = num_subjects)
  )
}

make_parameter_matrix <- function(true_parameters, sim_id, parameter) {
  values <- true_parameters |>
    filter(.data$sim_id == sim_id) |>
    arrange(subject, condition) |>
    pull(all_of(parameter))
  
  stopifnot(length(values) == 2 * NUM_SUBJECTS)
  matrix(values, nrow = 2, ncol = NUM_SUBJECTS)
}

simulate_choices <- function(true_parameters, sim_id) {
  lambda <- make_parameter_matrix(true_parameters, sim_id, "lambda")
  alpha <- make_parameter_matrix(true_parameters, sim_id, "alpha")
  tau <- make_parameter_matrix(true_parameters, sim_id, "tau")
  gamma <- make_parameter_matrix(true_parameters, sim_id, "gamma")
  
  choices <- map_int(seq_len(NUM_TRIALS), function(i) {
    subject <- SUBJECT_ID[i]
    condition <- CONDITION[i]
    
    utility_a <- get_cpt_utility(
      OUTCOMES_A[i, ],
      lambda[condition, subject],
      alpha[condition, subject],
      gamma[condition, subject]
    )
    
    utility_b <- get_cpt_utility(
      OUTCOMES_B[i, ],
      lambda[condition, subject],
      alpha[condition, subject],
      gamma[condition, subject]
    )
    
    choice_probability <- plogis(
      tau[condition, subject] * (utility_b - utility_a)
    )
    
    rbinom(1, 1, choice_probability)
  })
  
  choices
}

make_stan_data <- function(choices) {
  list(
    T = NUM_TRIALS,
    N = NUM_SUBJECTS,
    subject_id = SUBJECT_ID,
    gamble_type = CONDITION,
    outcomes_a = OUTCOMES_A,
    outcomes_b = OUTCOMES_B,
    choice = choices
  )
}

cpt_init_fun <- function(chains = NUM_CHAINS, n = NUM_SUBJECTS) {
  map(seq_len(chains), function(chain) {
    list(
      intercept_lambda = rnorm(1, 1.5, 0.5),
      intercept_alpha = rnorm(1, 0, 0.5),
      intercept_tau = rnorm(1, 0, 0.5),
      intercept_gamma = rnorm(1, 0, 0.3),
      b_lambda = rnorm(1, 0, 0.3),
      b_alpha = rnorm(1, 0, 0.3),
      b_tau = rnorm(1, 0, 0.3),
      b_gamma = rnorm(1, 0, 0.3),
      sigma_lambda = rnorm(2, -1, 0.3),
      sigma_alpha = rnorm(2, -1, 0.3),
      sigma_tau = rnorm(2, -1, 0.3),
      sigma_gamma = rnorm(2, -1, 0.3),
      z_lambda = matrix(rnorm(n * 2), n, 2),
      z_alpha = matrix(rnorm(n * 2), n, 2),
      z_tau = matrix(rnorm(n * 2), n, 2),
      z_gamma = matrix(rnorm(n * 2), n, 2),
      Omega_lambda = diag(2),
      Omega_alpha = diag(2),
      Omega_tau = diag(2),
      Omega_gamma = diag(2)
    )
  })
}

summarise_fit <- function(fit) {
  draws <- fit$draws(
    variables = PARAMETER_NAMES,
    format = "draws_matrix"
  )
  
  estimates <- map_dfr(seq_len(ncol(draws)), function(column) {
    values <- draws[, column]
    
    tibble(
      variable = colnames(draws)[column],
      median = median(values),
      ci_lower = quantile(values, 0.025),
      ci_upper = quantile(values, 0.975)
    )
  })
  
  diagnostics <- fit$summary(variables = PARAMETER_NAMES) |>
    select(variable, rhat, ess_bulk, ess_tail)
  
  left_join(estimates, diagnostics, by = "variable")
}

safe_correlation <- function(x, y) {
  if (length(x) < 2 || sd(x) == 0 || sd(y) == 0) {
    return(NA_real_)
  }
  
  cor(x, y)
}

summarise_recovery <- function(data, grouping_variables) {
  data |>
    group_by(across(all_of(grouping_variables))) |>
    summarise(
      n = n(),
      correlation = safe_correlation(true_value, estimate),
      bias = mean(estimate - true_value),
      rmse = sqrt(mean((estimate - true_value)^2)),
      coverage = mean(ci_lower <= true_value & ci_upper >= true_value),
      mean_interval_width = mean(ci_upper - ci_lower),
      .groups = "drop"
    )
}

get_axis_limits <- function(data, facet_variable) {
  data |>
    group_by(across(all_of(facet_variable))) |>
    summarise(
      minimum = min(true_value, estimate, na.rm = TRUE),
      maximum = max(true_value, estimate, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      xmin = floor(minimum * 10) / 10,
      xmax = ceiling(maximum * 10) / 10,
      xmax = if_else(xmax <= xmin, xmin + 0.1, xmax)
    ) |>
    select(all_of(facet_variable), xmin, xmax)
}

make_recovery_plot <- function(
    data,
    facet_variable,
    facet_labels,
    x_axis_label = "True value",
    point_size = 1.5
) {
  axis_limits <- get_axis_limits(data, facet_variable)
  plot_data <- left_join(data, axis_limits, by = facet_variable)
  
  ggplot(plot_data, aes(x = true_value, y = estimate)) +
    geom_point(alpha = 0.55, size = point_size) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    geom_blank(aes(x = xmin, y = xmin)) +
    geom_blank(aes(x = xmax, y = xmax)) +
    geom_blank(aes(x = xmin, y = xmax)) +
    geom_blank(aes(x = xmax, y = xmin)) +
    facet_wrap(
      vars(!!sym(facet_variable)),
      nrow = 1,
      scales = "free",
      labeller = as_labeller(facet_labels, default = label_parsed)
    ) +
    scale_x_continuous(n.breaks = 5) +
    scale_y_continuous(n.breaks = 5) +
    labs(x = x_axis_label, y = "Estimated value") +
    ggthemes::theme_tufte(base_size = 14) +
    theme(
      axis.title.x = element_text(size = 16, margin = margin(t = 12)),
      axis.title.y = element_text(size = 16, margin = margin(r = 12)),
      axis.line = element_line(linewidth = 0.5, color = "#969696"),
      axis.ticks = element_line(color = "#969696"),
      strip.text = element_text(size = 16),
      panel.grid.major = element_line(color = scales::alpha("gray70", 0.3)),
      panel.grid.minor = element_line(color = scales::alpha("gray70", 0.15)),
      panel.background = element_blank(),
      panel.spacing = unit(1.2, "lines")
    )
}

# ---------------------------------------------------------------------------- #
# GENERATE OR LOAD TRUE PARAMETERS
# ---------------------------------------------------------------------------- #
if (file.exists(TRUE_PARAMETER_FILE)) {
  true_parameters <- read_csv(TRUE_PARAMETER_FILE, show_col_types = FALSE)
} else {
  true_parameters <- tibble()
}

existing_simulations <- if ("sim_id" %in% names(true_parameters)) {
  unique(true_parameters$sim_id)
} else {
  integer()
}
missing_simulations <- setdiff(seq_len(NUM_SIMS), existing_simulations)

if (length(missing_simulations) > 0) {
  new_parameters <- map_dfr(missing_simulations, function(sim_id) {
    sample_parameters(NUM_SUBJECTS) |>
      mutate(sim_id = sim_id, .before = 1)
  })
  
  true_parameters <- bind_rows(true_parameters, new_parameters) |>
    arrange(sim_id, subject, condition)
  
  write_csv(true_parameters, TRUE_PARAMETER_FILE)
}

true_parameters <- true_parameters |>
  filter(sim_id <= NUM_SIMS)

# ---------------------------------------------------------------------------- #
# SIMULATE DATA AND FIT THE MODEL
# ---------------------------------------------------------------------------- #
for (sim_id in seq_len(NUM_SIMS)) {
  summary_file <- file.path(
    SUMMARY_DIR,
    paste0("estimated_parameters_", sim_id, ".csv")
  )
  
  if (file.exists(summary_file)) {
    message("Skipping completed simulation ", sim_id)
    next
  }
  
  message("Simulation ", sim_id, " of ", NUM_SIMS)
  
  choice_file <- file.path(
    CHOICE_DIR,
    paste0("simulated_choices_", sim_id, ".csv")
  )
  
  if (file.exists(choice_file)) {
    choices <- read_csv(choice_file, show_col_types = FALSE) |>
      pull(choice)
  } else {
    choices <- simulate_choices(true_parameters, sim_id)
    
    tibble(trial = seq_len(NUM_TRIALS), choice = choices) |>
      write_csv(choice_file)
  }
  
  choice_diagnostics <- tibble(
    condition = CONDITION,
    choice = choices
  ) |>
    group_by(condition) |>
    summarise(
      choice_rate = mean(choice),
      .groups = "drop"
    ) |>
    mutate(sim_id = sim_id, .before = 1)
  
  write_csv(
    choice_diagnostics,
    file.path(DIAGNOSTIC_DIR, paste0("choice_rates_", sim_id, ".csv"))
  )
  
  fit <- cpt_model$sample(
    data = make_stan_data(choices),
    init = cpt_init_fun(),
    chains = 4,
    parallel_chains = 4,
    threads_per_chain = 2,
    iter_warmup = 1000,
    iter_sampling = 1000,
    adapt_delta = 0.85,
    max_treedepth = 8,
    refresh = 200
  )
  
  summarise_fit(fit) |>
    mutate(sim_id = sim_id, .before = 1) |>
    write_csv(summary_file)
  
  fit$diagnostic_summary() |>
    as_tibble() |>
    mutate(sim_id = sim_id, .before = 1) |>
    write_csv(
      file.path(DIAGNOSTIC_DIR, paste0("sampling_diagnostics_", sim_id, ".csv"))
    )
  
  rm(fit)
  gc()
}

# ---------------------------------------------------------------------------- #
# LOAD FIT SUMMARIES
# ---------------------------------------------------------------------------- #
summary_files <- list.files(
  SUMMARY_DIR,
  pattern = "^estimated_parameters_[0-9]+\\.csv$",
  full.names = TRUE
)

if (length(summary_files) == 0) {
  stop("No completed recovery fits were found.")
}

estimated_parameters <- map_dfr(summary_files, function(file) {
  read_csv(file, show_col_types = FALSE)
}) |>
  filter(sim_id <= NUM_SIMS)

# ---------------------------------------------------------------------------- #
# GROUP-LEVEL RECOVERY
# ---------------------------------------------------------------------------- #
group_estimates <- estimated_parameters |>
  filter(str_detect(variable, "^(lambda|alpha|tau|gamma)_out\\[")) |>
  mutate(
    base_param = str_match(
      variable,
      "^(lambda|alpha|tau|gamma)_out\\[(1|2)\\]$"
    )[, 2],
    condition = as.integer(str_match(
      variable,
      "^(lambda|alpha|tau|gamma)_out\\[(1|2)\\]$"
    )[, 3])
  ) |>
  transmute(
    sim_id,
    base_param,
    condition,
    estimate = median,
    ci_lower,
    ci_upper
  )

group_true <- true_parameters |>
  group_by(sim_id, condition) |>
  summarise(
    lambda = first(mu_lambda),
    alpha = first(mu_alpha),
    tau = first(mu_tau),
    gamma = first(mu_gamma),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = all_of(PARAMETER_LEVELS),
    names_to = "base_param",
    values_to = "true_value"
  )

group_recovery <- group_estimates |>
  inner_join(
    group_true,
    by = c("sim_id", "base_param", "condition")
  ) |>
  mutate(
    condition = recode(
      as.character(condition),
      `1` = "Aligned",
      `2` = "Opposed"
    ),
    base_param = factor(base_param, levels = PARAMETER_LEVELS)
  )

group_metrics <- summarise_recovery(
  group_recovery,
  c("base_param", "condition")
)

write_csv(group_metrics, file.path(RECOVERY_DIR, "group_recovery_metrics.csv"))

# ---------------------------------------------------------------------------- #
# CONDITION-EFFECT RECOVERY
# ---------------------------------------------------------------------------- #
effect_estimates <- estimated_parameters |>
  filter(str_detect(variable, "^b_(lambda|alpha|tau|gamma)$")) |>
  transmute(
    sim_id,
    term = variable,
    estimate = median,
    ci_lower,
    ci_upper
  )

effect_true <- true_parameters |>
  group_by(sim_id) |>
  summarise(
    b_lambda = first(b_lambda),
    b_alpha = first(b_alpha),
    b_tau = first(b_tau),
    b_gamma = first(b_gamma),
    .groups = "drop"
  ) |>
  pivot_longer(
    cols = starts_with("b_"),
    names_to = "term",
    values_to = "true_value"
  )

effect_recovery <- effect_estimates |>
  inner_join(effect_true, by = c("sim_id", "term")) |>
  mutate(
    term = factor(
      term,
      levels = c("b_lambda", "b_alpha", "b_tau", "b_gamma")
    )
  )

effect_metrics <- summarise_recovery(effect_recovery, "term")

write_csv(
  effect_metrics,
  file.path(RECOVERY_DIR, "condition_effect_recovery_metrics.csv")
)

# ---------------------------------------------------------------------------- #
# INDIVIDUAL-LEVEL RECOVERY
# ---------------------------------------------------------------------------- #
individual_matches <- str_match(
  estimated_parameters$variable,
  "^(lambda|alpha|tau|gamma)\\[(1|2),([0-9]+)\\]$"
)

individual_estimates <- estimated_parameters |>
  mutate(
    base_param = individual_matches[, 2],
    condition = as.integer(individual_matches[, 3]),
    id = as.integer(individual_matches[, 4])
  ) |>
  filter(!is.na(base_param)) |>
  transmute(
    sim_id,
    id,
    base_param,
    condition,
    estimate = median,
    ci_lower,
    ci_upper
  )

individual_true <- true_parameters |>
  rename(id = subject) |>
  pivot_longer(
    cols = all_of(PARAMETER_LEVELS),
    names_to = "base_param",
    values_to = "true_value"
  ) |>
  select(sim_id, id, base_param, condition, true_value)

individual_recovery <- individual_estimates |>
  inner_join(
    individual_true,
    by = c("sim_id", "id", "base_param", "condition")
  ) |>
  mutate(
    condition = recode(
      as.character(condition),
      `1` = "Aligned",
      `2` = "Opposed"
    ),
    base_param = factor(base_param, levels = PARAMETER_LEVELS)
  )

individual_metrics <- summarise_recovery(
  individual_recovery,
  c("base_param", "condition")
)

write_csv(
  individual_metrics,
  file.path(RECOVERY_DIR, "individual_recovery_metrics.csv")
)

# ---------------------------------------------------------------------------- #
# RECOVERY PLOTS
# ---------------------------------------------------------------------------- #
plot_group_aligned <- make_recovery_plot(
  filter(group_recovery, condition == "Aligned"),
  "base_param",
  PARAMETER_LABELS,
  x_axis_label = NULL
) +
  ggtitle("Aligned") +
  theme(plot.title = element_text(size = 18, hjust = 0.5))

plot_group_opposed <- make_recovery_plot(
  filter(group_recovery, condition == "Opposed"),
  "base_param",
  PARAMETER_LABELS
) +
  ggtitle("Opposed") +
  theme(plot.title = element_text(size = 18, hjust = 0.5))

group_plot <- plot_group_aligned / plot_group_opposed

ggsave(
  "../plots/parameter_recovery.pdf",
  group_plot,
  width = 10,
  height = 6,
  bg = "white"
)

effect_plot <- make_recovery_plot(
  effect_recovery,
  "term",
  EFFECT_LABELS,
  x_axis_label = "True condition coefficient"
)

ggsave(
  "../plots/parameter_recovery_condition_effects.pdf",
  effect_plot,
  width = 10,
  height = 3.5,
  bg = "white"
)

plot_individual_aligned <- make_recovery_plot(
  filter(individual_recovery, condition == "Aligned"),
  "base_param",
  PARAMETER_LABELS,
  x_axis_label = NULL,
  point_size = 0.35
) +
  ggtitle("Aligned") +
  theme(plot.title = element_text(size = 18, hjust = 0.5))

plot_individual_opposed <- make_recovery_plot(
  filter(individual_recovery, condition == "Opposed"),
  "base_param",
  PARAMETER_LABELS,
  point_size = 0.35
) +
  ggtitle("Opposed") +
  theme(plot.title = element_text(size = 18, hjust = 0.5))

individual_plot <- plot_individual_aligned / plot_individual_opposed

ggsave(
  "../plots/parameter_recovery_individual.pdf",
  individual_plot,
  width = 10,
  height = 6,
  bg = "white"
)

print(group_metrics, n = Inf)
print(effect_metrics, n = Inf)
print(individual_metrics, n = Inf)