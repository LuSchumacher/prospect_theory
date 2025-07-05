library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv('../../data/pilot_2_data_prepped_excluded.csv')
# df <- df %>% 
#   filter(id == 1)

N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`        = `T`,
  N          = N,
  subject_id = df$id,
  condition  = ifelse(df$condition == "old", 0, 1),
  outcome_a  = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b  = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice    = df$resp
)

# init_fun <- function(chains = 4, N) {
#   inits <- vector("list", chains)
#   for (i in 1:chains) {
#     inits[[i]] <- list(
#       mu_lambda_0 = 2 + runif(1, -0.5, 0.5),
#       mu_lambda_1 = 0 + runif(1, -0.2, 0.2),
#       mu_alpha_0  = 1 + runif(1, -0.5, 0.5),
#       mu_alpha_1  = 0 + runif(1, -0.2, 0.2),
#       mu_tau_0    = 2 + runif(1, -0.5, 0.5),
#       mu_tau_1    = 0 + runif(1, -0.2, 0.2),
#       
#       sigma_lambda_0 = runif(1, 0.2, 0.5),
#       sigma_lambda_1 = runif(1, 0.2, 0.5),
#       sigma_alpha_0  = runif(1, 0.2, 0.5),
#       sigma_alpha_1  = runif(1, 0.2, 0.5),
#       sigma_tau_0    = runif(1, 0.2, 0.5),
#       sigma_tau_1    = runif(1, 0.2, 0.5),
#       
#       lambda_0 = rnorm(N, 2, 0.2),
#       lambda_1 = rnorm(N, 0, 0.1),
#       alpha_0  = rnorm(N, 1, 0.1),
#       alpha_1  = rnorm(N, 0, 0.1),
#       tau_0    = rnorm(N, 2, 0.2),
#       tau_1    = rnorm(N, 0, 0.1)
#     )
#   }
#   return(inits)
# }
# 
init_fun <- function(chains = 4) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      mu_lambda_0    = rnorm(1, log(2), 0.5),
      mu_lambda_1    = rnorm(1, log(2), 0.5),
      mu_alpha_0     = rnorm(1, 1.0, 1.75),
      mu_alpha_1     = rnorm(1, 1.0, 1.75),
      mu_tau_0       = rnorm(1, 0.5, 0.5),
      mu_tau_1       = rnorm(1, 0.5, 0.5),
      sigma_lambda_0 = rnorm(1, 0, 2),
      sigma_lambda_1 = rnorm(1, 0, 2),
      sigma_alpha_0  = rnorm(1, 0, 2),
      sigma_alpha_1  = rnorm(1, 0, 2),
      sigma_tau_0    = rnorm(1, 0, 2),
      sigma_tau_1    = rnorm(1, 0, 2),
      z_lambda_0     = rnorm(N, 0, 1),
      z_lambda_1     = rnorm(N, 0, 1),
      z_alpha_0      = rnorm(N, 0, 1),
      z_alpha_1      = rnorm(N, 0, 1),
      z_tau_0        = rnorm(N, 0, 1),
      z_tau_1        = rnorm(N, 0, 1)
    )
  }
  return(inits)
}


# init_fun <- function(chains = 4) {
#   inits <- vector("list", chains)
#   for (i in 1:chains) {
#     inits[[i]] <- list(
#       lambda_0_raw = rnorm(1, log(2), 0.5),
#       lambda_1_raw = rnorm(1, log(2), 0.5),
#       alpha_0_raw  = rnorm(1, 1.0, 1.75),
#       alpha_1_raw  = rnorm(1, 1.0, 1.75),
#       tau_0_raw    = rnorm(1, 0.5, 0.5),
#       tau_1_raw    = rnorm(1, 0.5, 0.5)
#     )
#   }
#   return(inits)
# }


# PARAM_NAMES <- c(
#   "lambda_0", "alpha_0", "tau_0",
#   "lambda_1", "alpha_1", "tau_1"
# )
# 
PARAM_NAMES <- c(
  "lambda_0_out", "alpha_0_out", "tau_0_out",
  "lambda_1_out", "alpha_1_out", "tau_1_out"
)

pt_model <- cmdstan_model(
  'pt_single_new.stan',
  cpp_options = list(stan_threads = T)
)

fit_pt_model <- pt_model$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 15,
  adapt_delta = 0.95,
  refresh = 50,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_pt_model$summary(variables = PARAM_NAMES)
mcmc_trace(
  fit_pt_model$draws(inc_warmup = TRUE),
  n_warmup = 2000,
  pars=PARAM_NAMES
)

y_rep <- fit_pt_model$draws("y_rep", format = "matrix")
y_obs <- df$choice

NUM_PP_DRAWS <- 500
idx <- sample(1:8000, NUM_PP_DRAWS)

pred_data <- y_rep %>% 
  as_tibble() %>% 
  mutate(draw = 1:8000) %>% 
  filter(draw %in% idx) %>% 
  pivot_longer(-draw, names_to = "trial", values_to = "resp") %>% 
  mutate(
    trial = str_extract(trial, "(?<=\\[)\\d+(?=\\])"),
    condition = rep(df$condition, times=NUM_PP_DRAWS),
    ev_diff = rep(df$ev_diff, times=NUM_PP_DRAWS)
  )

pred_summary <- pred_data %>% 
  group_by(condition, ev_diff) %>% 
  summarise(mean_resp = mean(resp))

emp_summary <- df %>% 
  group_by(condition, ev_diff) %>% 
  summarise(mean_resp = mean(resp))

pred_summary %>% 
  ggplot(aes(x = ev_diff, y = mean_resp, color = condition)) +
  geom_point() +
  geom_point(data = emp_summary, color="black")

