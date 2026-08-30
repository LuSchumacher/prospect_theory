library(tidyverse)
library(cmdstanr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

NUM_AGENTS <- 1  # Use 1 for testing; increase for the final simulation.
BASE_SEED <- 260826
COLOR_PALETTE <- c("Aligned" = "#c96016", "Opposed" = "#42686C")

RESULT_DIR <- "../data/simulation_data/cross_generative_simple"
FIT_DIR <- file.path(RESULT_DIR, "agent_estimates")
dir.create(FIT_DIR, recursive = TRUE, showWarnings = FALSE)

scenarios <- tribble(
  ~scenario_id,    ~scenario,         ~lambda, ~alpha, ~gamma, ~beta_variance, ~tau,
  "cpt_only",      "CPT only",              2,      1,      1,              0,  0.7,
  "variance_only", "Variance only",         1,      1,      1,         0.0075,  0.7,
  "cpt_variance",  "CPT + variance",        2,      1,      1,         0.0075,  0.7
)

# ---------------------------------------------------------------------------- #
# STIMULI
# ---------------------------------------------------------------------------- #
population_variance <- function(x) {
  mean((x - mean(x))^2)
}

choice_set <- read_csv("../data/gamble_list.csv", show_col_types = FALSE) |>
  filter(sanity_check == FALSE) |>
  mutate(
    trial = row_number(),
    gamble_type = if_else(lottery_type == "confounded", 1L, 2L),
    variance_a = pmap_dbl(
      list(outcome_a1, outcome_a2, outcome_a3),
      ~ population_variance(c(...))
    ),
    variance_b = pmap_dbl(
      list(outcome_b1, outcome_b2, outcome_b3),
      ~ population_variance(c(...))
    )
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
  w1 * values[1] + (w2 - w1) * values[2] + (1 - w2) * values[3]
}

simulate_agent <- function(scenario, agent_id) {
  set.seed(BASE_SEED + match(scenario$scenario_id, scenarios$scenario_id) * 1000 + agent_id)

  choice_set |>
    mutate(
      utility_a = pmap_dbl(
        list(outcome_a1, outcome_a2, outcome_a3),
        ~ get_cpt_utility(c(...), scenario$lambda, scenario$alpha, scenario$gamma)
      ) - scenario$beta_variance * variance_a,
      utility_b = pmap_dbl(
        list(outcome_b1, outcome_b2, outcome_b3),
        ~ get_cpt_utility(c(...), scenario$lambda, scenario$alpha, scenario$gamma)
      ) - scenario$beta_variance * variance_b,
      choice_probability = plogis(scenario$tau * (utility_b - utility_a)),
      choice = rbinom(n(), 1, choice_probability)
    )
}

make_stan_data <- function(data) {
  list(
    T = nrow(data),
    gamble_type = data$gamble_type,
    outcomes_a = as.matrix(data[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
    outcomes_b = as.matrix(data[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
    choice = data$choice
  )
}

# ---------------------------------------------------------------------------- #
# FIT REGULAR CPT TO EACH AGENT
# ---------------------------------------------------------------------------- #
cpt_model <- cmdstan_model(
  "cpt_simulation_fit.stan",
  cpp_options = list(stan_threads = TRUE)
)

reported_variables <- c(
  "lambda", "alpha", "tau", "gamma",
  "b_lambda", "b_alpha", "b_tau", "b_gamma",
  "delta_lambda", "delta_alpha", "delta_tau", "delta_gamma"
)

for (scenario_index in seq_len(nrow(scenarios))) {
  scenario <- scenarios[scenario_index, ]

  for (agent_id in seq_len(NUM_AGENTS)) {
    output_file <- file.path(
      FIT_DIR,
      paste0(scenario$scenario_id, "_", agent_id, ".csv")
    )

    if (file.exists(output_file)) {
      next
    }

    message(scenario$scenario, ": agent ", agent_id, " of ", NUM_AGENTS)
    simulated_data <- simulate_agent(scenario, agent_id)

    fit <- cpt_model$sample(
      data = make_stan_data(simulated_data),
      seed = BASE_SEED + scenario_index * 1000 + agent_id,
      chains = 4,
      parallel_chains = 4,
      threads_per_chain = 2,
      iter_warmup = 1000,
      iter_sampling = 1000,
      adapt_delta = 0.9,
      max_treedepth = 10,
      refresh = 200
    )

    fit$summary(variables = reported_variables) |>
      as_tibble() |>
      mutate(
        scenario_id = scenario$scenario_id,
        scenario = scenario$scenario,
        agent = agent_id,
        lambda_true = scenario$lambda,
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
results <- list.files(FIT_DIR, pattern = "\\.csv$", full.names = TRUE) |>
  map_dfr(~ read_csv(.x, show_col_types = FALSE)) |>
  mutate(
    parameter = str_remove(variable, "\\[.*"),
    condition = case_when(
      str_detect(variable, "\\[1\\]") ~ "Aligned",
      str_detect(variable, "\\[2\\]") ~ "Opposed",
      TRUE ~ NA_character_
    )
  )

write_csv(results, file.path(RESULT_DIR, "simulation_estimates.csv"))

summary_table <- results |>
  filter(!is.na(condition)) |>
  group_by(scenario, parameter, condition) |>
  summarise(
    estimate = median(median),
    lower = quantile(median, 0.05),
    upper = quantile(median, 0.95),
    .groups = "drop"
  )

write_csv(summary_table, file.path(RESULT_DIR, "simulation_summary.csv"))
print(summary_table, n = Inf)

# ---------------------------------------------------------------------------- #
# MAIN FIGURE
# ---------------------------------------------------------------------------- #
lambda_results <- results |>
  filter(parameter == "lambda") |>
  mutate(
    scenario = factor(scenario, levels = scenarios$scenario),
    condition = factor(condition, levels = c("Aligned", "Opposed"))
  )

lambda_truth <- scenarios |>
  transmute(
    scenario = factor(scenario, levels = scenarios$scenario),
    lambda_true = lambda
  )

lambda_plot <- ggplot(
  lambda_results,
  aes(x = condition, y = median, color = condition)
) +
  geom_hline(
    data = lambda_truth,
    aes(yintercept = lambda_true),
    inherit.aes = FALSE,
    linetype = "dashed"
  ) +
  geom_line(aes(group = interaction(scenario, agent)), color = "gray70") +
  geom_point(alpha = 0.5, position = position_jitter(width = 0.04)) +
  stat_summary(fun = median, geom = "point", size = 3) +
  facet_wrap(~ scenario, nrow = 1) +
  scale_color_manual(values = COLOR_PALETTE, guide = "none") +
  labs(x = NULL, y = expression("Estimated " * lambda)) +
  ggthemes::theme_tufte(base_size = 18)

ggsave(
  "../plots/simulation_lambda_attribution.pdf",
  lambda_plot,
  width = 10,
  height = 5
)
