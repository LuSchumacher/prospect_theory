library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)
library(posterior)
library(loo)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ---------------------------------------------------------------------------- #
# DATA AND MODEL PREPARATION
# ---------------------------------------------------------------------------- #
df <- read_csv('../data/study_data_prepared.csv')

N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`         = `T`,
  N           = N,
  subject_id  = df$id,
  gamble_type = ifelse(df$gamble_type == "confounded", 1, 2),
  outcomes_a   = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcomes_b   = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice      = df$resp
)

pt_init_fun <- function(chains = 4, n=N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      intercept_lambda = rnorm(1, log(2), 0.5),
      intercept_alpha  = rnorm(1, 1.0, 1.75),
      intercept_tau    = rnorm(1, 0.5, 0.5),
      b_lambda         = rnorm(1, 0.0, 0.5),
      b_alpha          = rnorm(1, 0.0, 0.5),
      b_tau            = rnorm(1, 0.0, 0.5),
      sigma_lambda     = rnorm(1, 0, 0.5),
      sigma_alpha      = rnorm(1, 0, 0.5),
      sigma_tau        = rnorm(1, 0, 0.5),
      z_lambda         = rnorm(n, 0, 1),
      z_alpha          = rnorm(n, 0, 1),
      z_tau            = rnorm(n, 0, 1)
    )
  }
  return(inits)
}

pt_init_fun_2 <- function(chains = 4, n = N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      intercept_lambda = rnorm(1, 2, 0.5),
      intercept_alpha = rnorm(1, 0, 0.5),
      intercept_tau = rnorm(1, 0.5, 0.5),
      b_lambda = rnorm(1, 0, 0.2),
      b_alpha = rnorm(1, 0, 0.2),
      b_tau = rnorm(1, 0, 0.2),
      sigma_lambda = rnorm(2, 0, 0.5),
      sigma_alpha = rnorm(2, 0, 0.5),
      sigma_tau = rnorm(2, 0, 0.5),
      z_lambda = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      z_alpha = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      z_tau = matrix(rnorm(n * 2, 0, 0.5), n, 2),
      Omega_lambda = diag(2),
      Omega_alpha = diag(2),
      Omega_tau = diag(2)
    )
  }
  return(inits)
}

cpt_init_fun <- function(chains = 4, n = N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      intercept_lambda = rnorm(1, log(2), 0.5),
      intercept_alpha  = rnorm(1, 0.0, 0.5),
      intercept_tau    = rnorm(1, 0.0, 0.5),
      intercept_gamma  = rnorm(1, 0.0, 0.3),
      b_lambda = rnorm(1, 0.0, 0.3),
      b_alpha  = rnorm(1, 0.0, 0.3),
      b_tau    = rnorm(1, 0.0, 0.3),
      b_gamma  = rnorm(1, 0.0, 0.3),
      sigma_lambda = rnorm(1, 0, 0.5),
      sigma_alpha  = rnorm(1, 0, 0.5),
      sigma_tau    = rnorm(1, 0, 0.5),
      sigma_gamma  = rnorm(1, 0, 0.5),
      z_lambda = rnorm(n, 0, 1),
      z_alpha  = rnorm(n, 0, 1),
      z_tau    = rnorm(n, 0, 1),
      z_gamma  = rnorm(n, 0, 1)
    )
  }
  return(inits)
}

cpt_init_fun_2 <- function(chains = 4, n = N) {
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

mvl_init_fun <- function(chains = 4, n = N) {
  map(seq_len(chains), function(chain) {
    list(
      intercept_b_loss = rnorm(1, 0, 0.5),
      intercept_b_var = rnorm(1, 0, 0.005),
      intercept_tau = rnorm(1, 0.5, 0.5),
      beta_b_loss = rnorm(1, 0, 0.2),
      beta_b_var = rnorm(1, 0, 0.002),
      beta_tau = rnorm(1, 0, 0.2),
      sigma_b_loss = rnorm(1, 0, 0.5),
      sigma_b_var = rnorm(1, -5, 0.5),
      sigma_tau = rnorm(1, 0, 0.5),
      z_b_loss = rnorm(n),
      z_b_var = rnorm(n),
      z_tau = rnorm(n)
    )
  })
}

mvl_init_fun_2 <- function(chains = 4, n = N) {
  map(seq_len(chains), function(chain) {
    list(
      intercept_b_loss = rnorm(1, 0, 0.5),
      intercept_b_var = rnorm(1, 0, 0.005),
      intercept_tau = rnorm(1, 0.5, 0.5),
      beta_b_loss = rnorm(1, 0, 0.2),
      beta_b_var = rnorm(1, 0, 0.002),
      beta_tau = rnorm(1, 0, 0.2),
      sigma_b_loss = rnorm(2, 0, 0.5),
      sigma_b_var = rnorm(2, -5, 0.5),
      sigma_tau = rnorm(2, 0, 0.5),
      z_b_loss = matrix(rnorm(n * 2), nrow = n, ncol = 2),
      z_b_var = matrix(rnorm(n * 2), nrow = n, ncol = 2),
      z_tau = matrix(rnorm(n * 2), nrow = n, ncol = 2),
      Omega_b_loss = diag(2),
      Omega_b_var = diag(2),
      Omega_tau = diag(2)
    )
  })
}

M1_PARAM_NAMES <- c(
  "lambda_out[1]", "lambda_out[2]", "b_lambda",
  "alpha_out[1]", "alpha_out[2]", "b_alpha",
  "tau_out[1]", "tau_out[2]", "b_tau"
)

M2_PARAM_NAMES <- c(
  "lambda_out[1]", "lambda_out[2]", "b_lambda",
  "alpha_out[1]", "alpha_out[2]", "b_alpha",
  "tau_out[1]", "tau_out[2]", "b_tau",
  "gamma_out[1]", "gamma_out[2]", "b_gamma"
)

M3_PARAM_NAMES <- c(
  "b_loss_out[1]", "b_loss_out[2]", "beta_b_loss",
  "b_var_out[1]", "b_var_out[2]", "beta_b_var",
  "tau_out[1]", "tau_out[2]", "beta_tau"
)

model_1_0 <- cmdstan_model(
  '../stan_models/model_1_0.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

model_1_1 <- cmdstan_model(
  '../stan_models/model_1_1.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)
model_1_2 <- cmdstan_model(
  '../stan_models/model_1_2.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)
model_1_3 <- cmdstan_model(
  '../stan_models/model_1_3.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

model_2_0 <- cmdstan_model(
  '../stan_models/model_2_0.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

model_2_1 <- cmdstan_model(
  '../stan_models/model_2_1.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)
model_2_2 <- cmdstan_model(
  '../stan_models/model_2_2.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)
model_2_3 <- cmdstan_model(
  '../stan_models/model_2_3.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

model_3_0 <- cmdstan_model(
  '../stan_models/model_3_0.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

model_3_1 <- cmdstan_model(
  '../stan_models/model_3_1.stan',
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

# ---------------------------------------------------------------------------- #
# PT MODEL FITTING
# ---------------------------------------------------------------------------- #
fit_model_1_0 <- model_1_0$sample(
  data = stan_data,
  init = pt_init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_1_0$save_object("../fits/fit_model_1_0.rds")

fit_model_1_1 <- model_1_1$sample(
  data = stan_data,
  init = pt_init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_1_1$save_object("../fits/fit_model_1_1.rds")

fit_model_1_2 <- model_1_2$sample(
  data = stan_data,
  init = pt_init_fun_2(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_1_2$save_object("../fits/fit_model_1_2.rds")

fit_model_1_3 <- model_1_3$sample(
  data = stan_data,
  init = pt_init_fun_2(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_1_3$save_object("../fits/fit_model_1_3.rds")

# ---------------------------------------------------------------------------- #
# CPT MODEL FITTING
# ---------------------------------------------------------------------------- #
fit_model_2_0 <- model_2_0$sample(
  data = stan_data,
  init = cpt_init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_2_0$save_object("../fits/fit_model_2_0.rds")

fit_model_2_1 <- model_2_1$sample(
  data = stan_data,
  init = cpt_init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_2_1$save_object("../fits/fit_model_2_1.rds")

fit_model_2_2 <- model_2_2$sample(
  data = stan_data,
  init = cpt_init_fun_2(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_2_2$save_object("../fits/fit_model_2_2.rds")

fit_model_2_3 <- model_2_3$sample(
  data = stan_data,
  init = cpt_init_fun_2(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_2_3$save_object("../fits/fit_model_2_3.rds")

# ---------------------------------------------------------------------------- #
# MVL MODEL FITTING
# ---------------------------------------------------------------------------- #
fit_model_3_0 <- model_3_0$sample(
  data = stan_data,
  init = mvl_init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_3_0$save_object("../fits/fit_model_3_0.rds")


fit_model_3_1 <- model_3_1$sample(
  data = stan_data,
  init = mvl_init_fun_2(),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_model_3_1$save_object("../fits/fit_model_3_1.rds")
