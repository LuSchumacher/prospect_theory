library(tidyverse)
library(magrittr)
library(LaplacesDemon)
library(cmdstanr)
library(patchwork)
library(posterior)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

NUM_SIMS <- 50

empirical_data <- read_csv("../data/study_data_prepared.csv") %>% 
  mutate(
    correct_choice = ifelse(ev_diff < 0, 1, 0),
    correct_choice = ifelse(ev_diff == 0, NA, correct_choice)
  )

NUM_SUBJECTS <- length(unique(empirical_data$id))
ID_VECTOR <- empirical_data$id
CONDITION <- ifelse(empirical_data$gamble_type == "confounded", 1, 2)

EMPIRICAL_OUTCOMES_A <- cbind(
  empirical_data$outcome_a1,
  empirical_data$outcome_a2,
  empirical_data$outcome_a3
)

EMPIRICAL_OUTCOMES_B <- cbind(
  empirical_data$outcome_b1,
  empirical_data$outcome_b2,
  empirical_data$outcome_b3
)

EMPIRICAL_CORRECT_CHOICE <- empirical_data$correct_choice

gamble_set <- read_csv("../data/gamble_list.csv") %>% 
  filter(sanity_check == FALSE) %>% 
  mutate(
    correct_choice = ifelse(ev_diff < 0, 1, 0),
    correct_choice = ifelse(ev_diff == 0, NA, correct_choice)
  )

confounded_gambles <- gamble_set %>% 
  filter(lottery_type == "confounded")

unconfounded_gambles <- gamble_set %>% 
  filter(lottery_type == "unconfounded")

CONFOUNDED_OUTCOMES_A <- cbind(
  confounded_gambles$outcome_a1,
  confounded_gambles$outcome_a2,
  confounded_gambles$outcome_a3
)

CONFOUNDED_OUTCOMES_B <- cbind(
  confounded_gambles$outcome_b1,
  confounded_gambles$outcome_b2,
  confounded_gambles$outcome_b3
)

CONFOUNDED_CORRECT_CHOICE <- confounded_gambles$correct_choice

UNCONFOUNDED_OUTCOMES_A <- cbind(
  unconfounded_gambles$outcome_a1,
  unconfounded_gambles$outcome_a2,
  unconfounded_gambles$outcome_a3
)

UNCONFOUNDED_OUTCOMES_B <- cbind(
  unconfounded_gambles$outcome_b1,
  unconfounded_gambles$outcome_b2,
  unconfounded_gambles$outcome_b3
)

UNCONFOUNDED_CORRECT_CHOICE <- unconfounded_gambles$correct_choice

cpt_model <- cmdstan_model(
  '../stan_models/model_2_2.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

# ---------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------- #
get_prelec_weights <- function(p, gamma) {
  exp(-(-log(p))^gamma)
}

get_cpt_utility <- function(outcomes, lambda, alpha, gamma) {
  stopifnot(length(outcomes) == 3)
  
  # value function
  v <- numeric(3)
  for (k in 1:3) {
    if (outcomes[k] >= 0) {
      v[k] <- outcomes[k]^alpha
    } else {
      v[k] <- -lambda * abs(outcomes[k])^alpha
    }
  }
  
  # sort by value (ascending)
  idx <- order(v)
  v <- v[idx]
  
  # decision weights
  w1 <- get_prelec_weights(1/3, gamma)
  w2 <- get_prelec_weights(2/3, gamma)
  w3 <- 1
  
  utility <- w1 * v[1] + (w2 - w1) * v[2] + (w3 - w2) * v[3]
  return(utility)
}

simulate_choices <- function(
    outcomes_a,
    outcomes_b,
    correct_choice,
    id_vector,
    condition,
    lambda,
    alpha,
    tau,
    gamma
) {
  N <- nrow(outcomes_a)
  choices <- integer(N)
  
  for (i in 1:N) {
    utility_a <- get_cpt_utility(
      outcomes_a[i, ],
      lambda[id_vector[i], condition[i]],
      alpha[id_vector[i], condition[i]],
      gamma[id_vector[i], condition[i]]
    )
    utility_b <- get_cpt_utility(
      outcomes_b[i, ],
      lambda[id_vector[i], condition[i]],
      alpha[id_vector[i], condition[i]],
      gamma[id_vector[i], condition[i]]
    )
    
    logit_p <- tau * (utility_b - utility_a)
    p <- plogis(logit_p)
    choices[i] <- rbinom(1, 1, p)  # 1 = choose B
  }
  
  correct <- as.numeric(choices == correct_choice)
  
  sim_data <- tibble(
    id = id_vector,
    condition = condition,
    outcome_a1 = outcomes_a[ , 1],
    outcome_a2 = outcomes_a[ , 2],
    outcome_a3 = outcomes_a[ , 3],
    outcome_b1 = outcomes_b[ , 1],
    outcome_b2 = outcomes_b[ , 2],
    outcome_b3 = outcomes_b[ , 3],
    choice = choices,
    correct_choice = correct_choice,
    correct = correct
  )
  
  return(sim_data)
}

sample_parameters <- function(num_subjects, sigma = 0.25) {
  effect_coding <- c(-0.5, 0.5)
  
  intercept_lambda <- runif(1, 0.75, 2.75)
  intercept_alpha  <- runif(1, -0.8, 0.25)
  intercept_tau    <- runif(1, -0.25, 5.5)
  intercept_gamma  <- runif(1, 0.05, 0.3)
  
  b_lambda <- rnorm(1, 0, 0.25)
  b_alpha  <- rnorm(1, 0, 0.25)
  b_tau    <- rnorm(1, 0, 0.25)
  b_gamma  <- rnorm(1, 0, 0.25)
  
  mu_lambda <- log1p(exp(intercept_lambda + b_lambda * effect_coding))
  mu_alpha  <- log1p(exp(intercept_alpha + b_alpha * effect_coding))
  mu_tau    <- log1p(exp(intercept_tau + b_tau * effect_coding))
  mu_gamma  <- log1p(exp(intercept_gamma + b_gamma * effect_coding))
  
  # ---- subject random effects ----
  z_lambda <- matrix(rnorm(2 * num_subjects), 2, num_subjects)
  z_alpha  <- matrix(rnorm(2 * num_subjects), 2, num_subjects)
  z_tau    <- matrix(rnorm(2 * num_subjects), 2, num_subjects)
  z_gamma  <- matrix(rnorm(2 * num_subjects), 2, num_subjects)
  
  u_lambda <- sigma * z_lambda
  u_alpha  <- sigma * z_alpha
  u_tau    <- sigma * z_tau
  u_gamma  <- sigma * z_gamma
  
  lambda <- matrix(0, 2, num_subjects)
  alpha  <- matrix(0, 2, num_subjects)
  tau    <- matrix(0, 2, num_subjects)
  gamma  <- matrix(0, 2, num_subjects)
  
  for (c in 1:2) {
    for (s in 1:num_subjects) {
      lambda[c, s] <- log1p(exp(
        intercept_lambda +
          u_lambda[1, s] +
          (b_lambda + u_lambda[2, s]) * effect_coding[c]
      ))
      alpha[c, s] <- log1p(exp(
        intercept_alpha +
          u_alpha[1, s] +
          (b_alpha + u_alpha[2, s]) * effect_coding[c]
      ))
      tau[c, s] <- log1p(exp(
        intercept_tau +
          u_tau[1, s] +
          (b_tau + u_tau[2, s]) * effect_coding[c]
      ))
      gamma[c, s] <- log1p(exp(
        intercept_gamma +
          u_gamma[1, s] +
          (b_gamma + u_gamma[2, s]) * effect_coding[c]
      ))
    }
  }
  
  param_draws <- tibble(
    subject = rep(1:num_subjects, each = 2),
    condition = rep(1:2, num_subjects),
    lambda = as.vector(lambda),
    alpha = as.vector(alpha),
    tau = as.vector(tau),
    gamma = as.vector(gamma),
    intercept_lambda = intercept_lambda,
    intercept_alpha = intercept_alpha,
    intercept_tau = intercept_tau,
    intercept_gamma = intercept_gamma,
    b_lambda = b_lambda,
    b_alpha = b_alpha,
    b_tau = b_tau,
    b_gamma = b_gamma,
    mu_lambda = rep(mu_lambda, times = num_subjects),
    mu_alpha = rep(mu_alpha, times = num_subjects),
    mu_tau = rep(mu_tau, times = num_subjects),
    mu_gamma = rep(mu_gamma, times = num_subjects)
  )
  
  return(param_draws)

}

cpt_init_fun <- function(chains = 4, n = N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      intercept_lambda = rnorm(1, 2, 0.5),
      intercept_alpha  = rnorm(1, 0, 0.5),
      intercept_tau    = rnorm(1, 0.5, 0.5),
      intercept_gamma  = rnorm(1, 0, 0.3),
      b_lambda = rnorm(1, 0, 0.2),
      b_alpha  = rnorm(1, 0, 0.2),
      b_tau    = rnorm(1, 0, 0.2),
      b_gamma  = rnorm(1, 0, 0.2),
      sigma_lambda = rnorm(2, 0, 0.5),
      sigma_alpha  = rnorm(2, 0, 0.5),
      sigma_tau    = rnorm(2, 0, 0.5),
      sigma_gamma  = rnorm(2, 0, 0.5),
      z_lambda = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      z_alpha  = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      z_tau    = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      z_gamma  = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      Omega_lambda = diag(2),
      Omega_alpha  = diag(2),
      Omega_tau    = diag(2),
      Omega_gamma  = diag(2)
    )
  }
  return(inits)
}

# ---------------------------------------------------------------------------- #
# SAMPLING PARAMETERS
# ---------------------------------------------------------------------------- #
prior_draws <- tibble()
for (i in seq_len(NUM_SIMS)) {
  draws <- sample_parameters(NUM_SUBJECTS)
  draws$sim_id <- i
  prior_draws <- prior_draws %>% 
    bind_rows(draws)
}

# write_csv(prior_draws, "../data/simulation_data/true_parameter.csv")

hist(prior_draws$lambda, breaks=100)
hist(prior_draws$alpha, breaks=100)
hist(prior_draws$tau, breaks=100)
hist(prior_draws$gamma, breaks=100)

# ---------------------------------------------------------------------------- #
# DATA SIMULATION
# ---------------------------------------------------------------------------- #
prior_draws <- read_csv("../data/simulation_data/true_parameter.csv")

sim_data <- tibble()

for (i in seq_len(NUM_SIMS)) {
  lambda <- cbind(
    prior_draws$lambda[prior_draws$sim_id == i & prior_draws$condition == 1],
    prior_draws$lambda[prior_draws$sim_id == i & prior_draws$condition == 2]
  )
  alpha <- cbind(
    prior_draws$alpha[prior_draws$sim_id == i & prior_draws$condition == 1],
    prior_draws$alpha[prior_draws$sim_id == i & prior_draws$condition == 2]
  )
  tau <- cbind(
    prior_draws$tau[prior_draws$sim_id == i & prior_draws$condition == 1],
    prior_draws$tau[prior_draws$sim_id == i & prior_draws$condition == 2]
  )
  gamma <- cbind(
    prior_draws$gamma[prior_draws$sim_id == i & prior_draws$condition == 1],
    prior_draws$gamma[prior_draws$sim_id == i & prior_draws$condition == 2]
  )
  
  tmp_data <- simulate_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    EMPIRICAL_CORRECT_CHOICE,
    ID_VECTOR,
    CONDITION,
    lambda, alpha, tau, gamma
  )
  
  tmp_data$sim_id <- i
  
  sim_data <- sim_data %>% 
    bind_rows(tmp_data)
}

summary <- sim_data %>% 
  group_by(id, sim_id) %>% 
  summarise(acc = mean(correct, na.rm=TRUE))

hist(summary$acc, breaks=100)

# write_csv(sim_data, "../data/simulation_data/sim_data.csv")

# ---------------------------------------------------------------------------- #
# MODEL FITTING
# ---------------------------------------------------------------------------- #
sim_data <- read_csv("../data/simulation_data/sim_data.csv")

param_names <- c(
  "lambda", "alpha", "tau", "gamma",
  "lambda_out", "alpha_out", "tau_out", "gamma_out",
  "intercept_lambda", "intercept_alpha", "intercept_tau", "intercept_gamma",
  "b_lambda", "b_alpha", "b_tau", "b_gamma"
)

for (i in 2:NUM_SIMS) {
  tmp_data <- sim_data %>% 
    filter(sim_id == i)
  
  N <- length(unique(tmp_data$id))
  `T` <- nrow(tmp_data)
  stan_data = list(
    `T`         = `T`,
    N           = N,
    subject_id  = tmp_data$id,
    gamble_type = tmp_data$condition,
    outcomes_a   = as.matrix(tmp_data[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
    outcomes_b   = as.matrix(tmp_data[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
    choice      = tmp_data$choice
  )
  
  model_fit <- cpt_model$sample(
    data = stan_data,
    init = cpt_init_fun(),
    max_treedepth = 8,
    adapt_delta = 0.8,
    refresh = 100,
    iter_sampling = 1000,
    iter_warmup = 1000,
    chains = 4,
    parallel_chains = 4,
    threads_per_chain = 2,
    save_warmup = TRUE
  )
  
  model_fit$draws(
    variables = param_names,
    format = "df"
  ) %>%
    as_tibble() %>%
    write_csv(
      paste0("../data/simulation_data/recovery/estimated_params_", i, ".csv")
    )
}

# ---------------------------------------------------------------------------- #
# EVALUATION
# ---------------------------------------------------------------------------- #
true_params <- read_csv("../data/simulation_data/true_parameter.csv")

# post_medians <- model_fit$draws(
#   variables = PARAM_NAMES,
#   format = "df"
# ) %>%
#   as_tibble() %>% 
#   select(-c(.draw, .iteration, .chain)) %>% 
#   summarise(across(everything(), median)) %>%
#   pivot_longer(
#     cols = everything(),
#     names_to = c("parameter", "condition"),
#     names_pattern = "(.*)\\[(.*)\\]",
#     values_to = "median"
#   ) %>%
#   mutate(
#     parameter = str_remove(parameter, "_out$")
#   )
# 
# true_params %>% 
#   filter(sim_id == 1, subject == 1) %>% 
#   select(starts_with("mu_"))
