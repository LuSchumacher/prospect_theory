library(tidyverse)
library(cmdstanr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

NUM_AGENTS <- 20

COLOR_PALETTE <- c(
  "Aligned" = "#c96016",
  "Opposed" = "#42686C"
)

RESULT_DIR <- "../data/simulation_data/cross_generative_simple"
FIT_DIR <- file.path(RESULT_DIR, "agent_estimates")

dir.create(FIT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create("../plots", recursive = TRUE, showWarnings = FALSE)

SIMULATION_SCENARIOS <- as_tibble(list(
  scenario = c(
    "cpt_only",
    "variance_only",
    "cpt_variance"
  ),
  scenario_label = c(
    "CPT only",
    "Variance only",
    "CPT + variance"
  ),
  lambda_true = c(2, 1, 2),
  alpha_true = c(1, 1, 1),
  gamma_true = c(1, 1, 1),
  beta_variance_true = c(0, 0.0075, 0.0075),
  tau_true = c(0.7, 0.7, 0.7)
))

choice_set <- read_csv(
  "../data/gamble_list.csv",
  show_col_types = FALSE
) |>
  filter(sanity_check == FALSE) |>
  mutate(
    gamble_type = if_else(
      lottery_type == "confounded",
      1L,
      2L
    )
  )

cpt_model <- cmdstan_model(
  "cpt_simulation_fit.stan",
  cpp_options = list(stan_threads = TRUE)
)

# ---------------------------------------------------------------------------- #
# GENERATING MODEL
# ---------------------------------------------------------------------------- #
prelec_w <- function(p, gamma) {
  exp(-(-log(p))^gamma)
}

get_cpt_utility <- function(outcomes, lambda, alpha, gamma) {
  values <- if_else(
    outcomes >= 0,
    outcomes^alpha,
    -lambda * abs(outcomes)^alpha
  ) |>
    sort()
  
  w1 <- prelec_w(1 / 3, gamma)
  w2 <- prelec_w(2 / 3, gamma)
  
  w1 * values[1] +
    (w2 - w1) * values[2] +
    (1 - w2) * values[3]
}

simulate_agent <- function(scenario) {
  choice_set |>
    mutate(
      utility_a = pmap_dbl(
        list(outcome_a1, outcome_a2, outcome_a3),
        ~ get_cpt_utility(
          c(...),
          scenario$lambda_true,
          scenario$alpha_true,
          scenario$gamma_true
        )
      ) - scenario$beta_variance_true * var_a^2,
      
      utility_b = pmap_dbl(
        list(outcome_b1, outcome_b2, outcome_b3),
        ~ get_cpt_utility(
          c(...),
          scenario$lambda_true,
          scenario$alpha_true,
          scenario$gamma_true
        )
      ) - scenario$beta_variance_true * var_b^2,
      
      choice_probability = plogis(
        scenario$tau_true *
          (utility_b - utility_a)
      ),
      
      choice = rbinom(
        n(),
        size = 1,
        prob = choice_probability
      )
    )
}

make_stan_data <- function(data) {
  list(
    T = nrow(data),
    gamble_type = data$gamble_type,
    outcomes_a = as.matrix(
      data[, c(
        "outcome_a1",
        "outcome_a2",
        "outcome_a3"
      )]
    ),
    outcomes_b = as.matrix(
      data[, c(
        "outcome_b1",
        "outcome_b2",
        "outcome_b3"
      )]
    ),
    choice = data$choice
  )
}

# ---------------------------------------------------------------------------- #
# SIMULATE AND FIT
# ---------------------------------------------------------------------------- #
reported_variables <- c(
  "lambda",
  "alpha",
  "tau",
  "gamma",
  "b_lambda",
  "b_alpha",
  "b_tau",
  "b_gamma",
  "delta_lambda",
  "delta_alpha",
  "delta_tau",
  "delta_gamma"
)

for (scenario_index in seq_len(nrow(SIMULATION_SCENARIOS))) {
  scenario_row <- SIMULATION_SCENARIOS[scenario_index, ]
  
  for (agent_id in seq_len(NUM_AGENTS)) {
    output_file <- file.path(
      FIT_DIR,
      paste0(
        scenario_row$scenario,
        "_",
        agent_id,
        ".csv"
      )
    )
    
    if (file.exists(output_file)) {
      message("Skipping existing fit: ", output_file)
      next
    }
    
    message(
      scenario_row$scenario_label,
      ": agent ",
      agent_id,
      " of ",
      NUM_AGENTS
    )
    
    simulated_data <- simulate_agent(scenario_row)
    
    fit <- cpt_model$sample(
      data = make_stan_data(simulated_data),
      init = 0,
      chains = 4,
      parallel_chains = 4,
      threads_per_chain = 2,
      iter_warmup = 1000,
      iter_sampling = 1000,
      adapt_delta = 0.9,
      max_treedepth = 10,
      refresh = 200
    )
    
    fit$summary(
      variables = reported_variables
    ) |>
      as_tibble() |>
      mutate(
        scenario = scenario_row$scenario,
        scenario_label = scenario_row$scenario_label,
        agent = agent_id,
        lambda_true = scenario_row$lambda_true,
        alpha_true = scenario_row$alpha_true,
        gamma_true = scenario_row$gamma_true,
        tau_true = scenario_row$tau_true,
        beta_variance_true = scenario_row$beta_variance_true,
        .before = 1
      ) |>
      write_csv(output_file)
    
    rm(fit)
    gc()
  }
}

# ---------------------------------------------------------------------------- #
# COMBINE RESULTS
# ---------------------------------------------------------------------------- #
estimate_files <- list.files(
  FIT_DIR,
  pattern = "\\.csv$",
  full.names = TRUE
)

if (length(estimate_files) == 0) {
  stop("No fitted-model results were found.")
}

results <- map_dfr(
  estimate_files,
  ~ read_csv(.x, show_col_types = FALSE)
) |>
  mutate(
    parameter = str_remove(
      variable,
      "\\[.*"
    ),
    condition = case_when(
      str_detect(variable, "\\[1\\]") ~ "Aligned",
      str_detect(variable, "\\[2\\]") ~ "Opposed",
      TRUE ~ NA_character_
    )
  )

write_csv(
  results,
  file.path(
    RESULT_DIR,
    "simulation_estimates.csv"
  )
)

summary_table <- results |>
  filter(!is.na(condition)) |>
  group_by(
    scenario_label,
    parameter,
    condition
  ) |>
  summarise(
    estimate = median(median),
    lower = quantile(median, 0.05),
    upper = quantile(median, 0.95),
    .groups = "drop"
  )

write_csv(
  summary_table,
  file.path(
    RESULT_DIR,
    "simulation_summary.csv"
  )
)

print(summary_table, n = Inf)

# ---------------------------------------------------------------------------- #
# MAIN FIGURE
# ---------------------------------------------------------------------------- #
lambda_results <- results |>
  filter(
    parameter == "lambda",
    !is.na(condition)
  ) |>
  mutate(
    scenario_label = factor(
      scenario_label,
      levels = SIMULATION_SCENARIOS$scenario_label
    ),
    condition = factor(
      condition,
      levels = c(
        "Aligned",
        "Opposed"
      )
    )
  )

lambda_truth <- SIMULATION_SCENARIOS |>
  transmute(
    scenario_label = factor(
      scenario_label,
      levels = SIMULATION_SCENARIOS$scenario_label
    ),
    lambda_true
  )

lambda_plot <- ggplot(
  lambda_results,
  aes(
    x = condition,
    y = median,
    color = condition
  )
) +
  geom_hline(
    data = lambda_truth,
    aes(yintercept = lambda_true),
    inherit.aes = FALSE,
    linetype = "dashed"
  ) +
  geom_line(
    aes(
      group = interaction(
        scenario_label,
        agent
      )
    ),
    color = "gray70"
  ) +
  geom_point(
    alpha = 0.5,
    position = position_jitter(
      width = 0.04
    )
  ) +
  stat_summary(
    fun = median,
    geom = "point",
    size = 3
  ) +
  facet_wrap(
    ~ scenario_label,
    nrow = 1
  ) +
  scale_color_manual(
    values = COLOR_PALETTE,
    guide = "none"
  ) +
  labs(
    x = NULL,
    y = expression(
      "Estimated " * lambda
    )
  ) +
  ggthemes::theme_tufte(
    base_size = 18
  )

ggsave(
  "../plots/simulation_lambda_attribution.pdf",
  lambda_plot,
  width = 10,
  height = 5
)