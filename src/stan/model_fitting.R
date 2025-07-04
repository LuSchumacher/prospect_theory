library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv('../../data/pilot_2_data_prepped_excluded.csv')
df <- df %>% 
  filter(id == 1)

N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`        = `T`,
  # N          = N,
  # subject_id = df$id,
  condition  = ifelse(df$condition == "old", 0, 1),
  outcome_a  = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b  = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice    = df$resp
)

init_fun <- function(chains = 4, N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      mu_lambda_0 = 2 + runif(1, -0.5, 0.5),
      mu_lambda_1 = 0 + runif(1, -0.2, 0.2),
      mu_alpha_0  = 1 + runif(1, -0.5, 0.5),
      mu_alpha_1  = 0 + runif(1, -0.2, 0.2),
      mu_tau_0    = 2 + runif(1, -0.5, 0.5),
      mu_tau_1    = 0 + runif(1, -0.2, 0.2),
      
      sigma_lambda_0 = runif(1, 0.2, 0.5),
      sigma_lambda_1 = runif(1, 0.2, 0.5),
      sigma_alpha_0  = runif(1, 0.2, 0.5),
      sigma_alpha_1  = runif(1, 0.2, 0.5),
      sigma_tau_0    = runif(1, 0.2, 0.5),
      sigma_tau_1    = runif(1, 0.2, 0.5),
      
      lambda_0 = rnorm(N, 2, 0.2),
      lambda_1 = rnorm(N, 0, 0.1),
      alpha_0  = rnorm(N, 1, 0.1),
      alpha_1  = rnorm(N, 0, 0.1),
      tau_0    = rnorm(N, 2, 0.2),
      tau_1    = rnorm(N, 0, 0.1)
    )
  }
  return(inits)
}

init_fun <- function(chains = 4) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      lambda_0 = runif(1, 0.5, 2.5),
      lambda_1 = runif(1, 0.5, 2.5),
      alpha_0  = runif(1, 0.3, 0.9),
      alpha_1  = runif(1, 0.3, 0.9),
      tau_0    = runif(1, 0.2, 1.5),
      tau_1    = runif(1, 0.2, 1.5)
    )
  }
  return(inits)
}


# PARAM_NAMES <- c(
#   "lambda_mean_0", "alpha_mean_0", "tau_mean_0",
#   "lambda_mean_1", "alpha_mean_1", "tau_mean_1",
#   "delta_lambda", "delta_alpha", "delta_tau"
# )

pt_model <- cmdstan_model(
  'pt_single.stan',
  cpp_options = list(stan_threads = T)
)

fit_pt_model <- pt_model$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 15,
  adapt_delta = 0.95,
  refresh = 50,
  iter_sampling = 1000,
  iter_warmup = 1000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_pt_model$summary()
mcmc_trace(fit_pt_model$draws(inc_warmup = TRUE), n_warmup = 200, pars=PARAM_NAMES)



