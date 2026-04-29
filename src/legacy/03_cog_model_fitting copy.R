library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)
library(posterior)
library(loo)

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#27374D', '#B70404')

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
  gamble_type = ifelse(df$gamble_type == "confounded", 0, 1),
  outcome_a   = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b   = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice      = df$resp
)

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

param_names_separate <- c(
  "lambda_0_out", "lambda_1_out", "alpha_0_out",
  "alpha_1_out", "tau_0_out", "tau_1_out"
)

pt_model_separate_params <- cmdstan_model(
  'pt_model_separate_params.stan',
  cpp_options = list(stan_threads = T)
)
pt_model_separate_params_inversed <- cmdstan_model(
  'pt_model_separate_params_inversed.stan',
  cpp_options = list(stan_threads = T)
)

################################################################################
# MODEL FITTING
################################################################################
fit_pt_model_separate_params <- pt_model_separate_params$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 5,
  adapt_delta = 0.85,
  refresh = 50,
  iter_sampling = 1500,
  iter_warmup = 1500,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

# mcmc_trace(
#   fit_pt_model_separate_params$draws(inc_warmup = TRUE),
#   n_warmup = 2000,
#   pars=param_names_separate
# )
fit_pt_model_separate_params$save_object("../fits/fit_pt_model_separate_params.rds")

################################################################################
fit_pt_model_separate_params_inversed <- pt_model_separate_params_inversed$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 5,
  adapt_delta = 0.85,
  refresh = 50,
  iter_sampling = 1500,
  iter_warmup = 1500,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

# mcmc_trace(
#   fit_pt_model_separate_params_inversed$draws(inc_warmup = TRUE),
#   n_warmup = 2000,
#   pars=param_names_separate
# )
fit_pt_model_separate_params_inversed$save_object("../fits/fit_pt_model_separate_params_inversed.rds")

################################################################################
# PT REGRESSION APPROACH
################################################################################
N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`         = `T`,
  N           = N,
  subject_id  = df$id,
  gamble_type = ifelse(df$gamble_type == "confounded", 1, 2),
  outcome_a   = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b   = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice      = df$resp
)

init_fun <- function(chains = 4) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      intercept_lambda = rnorm(1, log(2), 0.5),
      intercept_alpha  = rnorm(1, 1.0, 1.75),
      intercept_tau    = rnorm(1, 0.5, 0.5),
      beta_lambda      = rnorm(1, 0.0, 0.5),
      beta_alpha       = rnorm(1, 0.0, 0.5),
      beta_tau         = rnorm(1, 0.0, 0.5),
      sigma_lambda     = rnorm(1, 0, 0.5),
      sigma_alpha      = rnorm(1, 0, 0.5),
      sigma_tau        = rnorm(1, 0, 0.5),
      z_lambda         = rnorm(N, 0, 1),
      z_alpha          = rnorm(N, 0, 1),
      z_tau            = rnorm(N, 0, 1)
    )
  }
  return(inits)
}

param_names_regression <- c(
  "lambda_out[1]", "lambda_out[2]", "intercept_lambda", "beta_lambda",
  "alpha_out[1]", "alpha_out[2]", "intercept_alpha", "beta_alpha",
  "tau_out[1]", "tau_out[2]", "intercept_tau", "beta_tau"
)

pt_model_regression <- cmdstan_model(
  'pt_model_regression.stan',
  cpp_options = list(stan_threads = T)
)
pt_model_regression_inversed <- cmdstan_model(
  'pt_model_regression_inversed.stan',
  cpp_options = list(stan_threads = T)
)

################################################################################
# MODEL FITTING
################################################################################
fit_pt_model_regression <- pt_model_regression$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 10,
  adapt_delta = 0.9,
  refresh = 50,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

# mcmc_trace(
#   fit_pt_model_regression$draws(inc_warmup = TRUE),
#   n_warmup = 1000,
#   pars=param_names_regression
# )
fit_pt_model_regression$save_object("../fits/fit_pt_model_regression.rds")

################################################################################
fit_pt_model_regression_inversed <- pt_model_regression_inversed$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 5,
  adapt_delta = 0.85,
  refresh = 50,
  iter_sampling = 1500,
  iter_warmup = 1500,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

# mcmc_trace(
#   fit_pt_model_regression_inversed$draws(inc_warmup = TRUE),
#   n_warmup = 1000,
#   pars=PARAM_NAMES
# )
fit_pt_model_regression_inversed$save_object("../fits/fit_pt_model_regression_inversed.rds")

################################################################################
# MODEL EVALUATION
################################################################################
fit_pt_model_separate_params$summary(variables = param_names_separate)
fit_pt_model_separate_params_inversed$summary(variables = param_names_separate)
fit_pt_model_regression$summary(variables = param_names_regression)
fit_pt_model_regression_inversed$summary(variables = param_names_regression)

log_lik_separate <- fit_pt_model_separate_params$draws("log_lik", format = "matrix")
log_lik_separate_inv <- fit_pt_model_separate_params_inversed$draws("log_lik", format = "matrix")
log_lik_regression <- fit_pt_model_regression$draws("log_lik", format = "matrix")
log_lik_regression_inv <- fit_pt_model_regression_inversed$draws("log_lik", format = "matrix")

loo_separate <- loo(log_lik_separate)
loo_separate_inv <- loo(log_lik_separate_inv)
loo_regression <- loo(log_lik_regression)
loo_regression_inv <- loo(log_lik_regression_inv)

loo_compare(loo_separate, loo_separate_inv, loo_regression, loo_regression_inv)

compute_bf <- function(draws, prior_mean = 0, prior_sd = 0.5) {
  prior_dens_0 <- dnorm(0, mean = prior_mean, sd = prior_sd)
  
  dens <- density(draws, n = 2^14)  # very fine grid
  posterior_dens_0 <- approx(dens$x, dens$y, xout = 0, rule = 2)$y
  
  BF01 <- prior_dens_0 / posterior_dens_0
  BF10 <- 1 / BF01
  
  list(
    prior_dens_0 = prior_dens_0,
    posterior_dens_0 = posterior_dens_0,
    BF01 = BF01,
    BF10 = BF10
  )
}

compute_bf_logspline <- function(draws, prior_mean = 0, prior_sd = 0.5) {
  prior_dens_0 <- dnorm(0, mean = prior_mean, sd = prior_sd)
  
  fit <- logspline(draws)  # flexible density estimate
  posterior_dens_0 <- dlogspline(0, fit)
  
  BF01 <- prior_dens_0 / posterior_dens_0
  BF10 <- 1 / BF01
  
  list(
    prior_dens_0 = prior_dens_0,
    posterior_dens_0 = posterior_dens_0,
    BF01 = BF01,
    BF10 = BF10
  )
}

# corrected Savage–Dickey using KDE
compute_bf_sd <- function(draws, prior_mean = 0, prior_sd = 0.5, kde_n = 2^14) {
  # densities at 0
  prior_dens_0 <- dnorm(0, mean = prior_mean, sd = prior_sd)
  
  dens <- density(draws, n = kde_n)
  post_dens_0_kde <- approx(dens$x, dens$y, xout = 0, rule = 2)$y
  
  # normal-approximation cross-check for posterior
  m <- mean(draws); s <- sd(draws)
  post_dens_0_norm <- dnorm(0, mean = m, sd = s)
  
  BF01 <- post_dens_0_kde / prior_dens_0
  BF10 <- 1 / BF01
  
  list(
    BF01 = BF01, BF10 = BF10,
    prior_dens_0 = prior_dens_0,
    post_dens_0_kde = post_dens_0_kde,
    post_dens_0_norm = post_dens_0_norm,
    post_mean = m, post_sd = s
  )
}

fit_pt_model_regression <- readRDS("../fits/fit_pt_model_regression.rds")

# assuming your fit object is called "fit"
draws <- as_draws_df(fit_pt_model_regression)

res_lambda <- compute_bf_sd(draws$beta_lambda, prior_sd = 0.5)
res_alpha  <- compute_bf_sd(draws$beta_alpha,  prior_sd = 0.5)
res_tau    <- compute_bf_sd(draws$beta_tau,    prior_sd = 0.5)

bf_table <- tibble(
  parameter = c("beta_lambda","beta_alpha","beta_tau"),
  BF01 = c(res_lambda$BF01, res_alpha$BF01, res_tau$BF01),
  BF10 = c(res_lambda$BF10, res_alpha$BF10, res_tau$BF10),
  prior_dens_0 = c(res_lambda$prior_dens_0, res_alpha$prior_dens_0, res_tau$prior_dens_0),
  post_dens_0_kde = c(res_lambda$post_dens_0_kde, res_alpha$post_dens_0_kde, res_tau$post_dens_0_kde),
  post_dens_0_norm = c(res_lambda$post_dens_0_norm, res_alpha$post_dens_0_norm, res_tau$post_dens_0_norm),
  post_mean = c(res_lambda$post_mean, res_alpha$post_mean, res_tau$post_mean),
  post_sd   = c(res_lambda$post_sd, res_alpha$post_sd, res_tau$post_sd)
)

COLOR_PALETTE <- c(confounded = '#27374D', unconfounded = '#B70404')

PARAM_NAMES <- c(
  "lambda_out[1]", "lambda_out[2]", "beta_lambda",
  "alpha_out[1]",  "alpha_out[2]",  "beta_alpha",
  "tau_out[1]",    "tau_out[2]",    "beta_tau"
)

# tidy posterior draws
draws_tidy <- draws %>%
  select(all_of(PARAM_NAMES)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    family = case_when(
      grepl("lambda", parameter) ~ "lambda",
      grepl("alpha", parameter)  ~ "alpha",
      grepl("tau", parameter)    ~ "tau"
    ),
    panel = case_when(
      grepl("out", parameter)  ~ "PT model parameters",
      grepl("beta", parameter) ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      parameter %in% c("lambda_out[1]","alpha_out[1]","tau_out[1]") ~ "confounded",
      parameter %in% c("lambda_out[2]","alpha_out[2]","tau_out[2]") ~ "unconfounded",
      TRUE ~ NA_character_
    ),
    # force desired column order: params in col 1, betas in col 2
    panel = factor(panel, levels = c("PT model parameters", "Effect of gamble type"))
  )

# plotting
FONT_SCALER <- 0
pt_model_params <- ggplot() +
  geom_density(
    data = draws_tidy %>% filter(panel == "PT model parameters"),
    aes(x = value, fill = gamble_type),
    alpha = 0.75, color = NA
  ) +
  scale_fill_manual(values = COLOR_PALETTE, name = "Gamble type") +
  geom_density(
    data = draws_tidy %>% filter(panel == "Effect of gamble type"),
    aes(x = value),
    fill = "darkgray", alpha = 0.75, color = NA
  ) +
  facet_grid(
    family ~ panel,
    scales = "free",
    labeller = labeller(
      family = label_parsed,   # Greek letters
      panel  = label_value     # plain text
    )
  ) +
  geom_vline(
    data = draws_tidy %>% filter(panel == "Effect of gamble type"),
    aes(xintercept = 0),
    linetype = "dashed", inherit.aes = FALSE
  ) +
  labs(x = "Value", y = "Density") +
  ggthemes::theme_tufte(base_size = FONT_SIZE_2) +
  theme(
    axis.title.x = element_text(margin = margin(t = 12)),
    axis.title.y = element_text(margin = margin(r = 12)),
    axis.line = element_line(linewidth = 0.5, color = "#969696"),
    axis.ticks = element_line(color = "#969696"),
    axis.text.x = element_text(size = FONT_SIZE_3 - FONT_SCALER, vjust = 0.5),
    axis.text.y = element_text(size = FONT_SIZE_3 - FONT_SCALER),
    strip.text.x = element_text(size = FONT_SIZE_2 - FONT_SCALER),
    strip.text.y = element_text(size = FONT_SIZE_2 - FONT_SCALER, hjust = 0, angle = 0),
    panel.grid.major = element_line(color = scales::alpha("gray70", 0.3)),
    panel.grid.minor = element_line(color = scales::alpha("gray70", 0.15)),
    panel.background = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    legend.position = "bottom",
    legend.margin = margin(t = -5, r = 0, b = 0, l = 0),
    legend.spacing.y = unit(0.2, "cm"),
  )

ggsave(
  '../plots/pt_model_params.pdf',
  pt_model_params,
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)



################################################################################
# POSTERIOR RE-SIMULATION
################################################################################
y_rep <- fit_pt_model_regression$draws("y_rep", format = "matrix")
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
    lottery_type = rep(df$gamble_type, times=NUM_PP_DRAWS),
    ev_diff = rep(df$ev_diff, times=NUM_PP_DRAWS)
  )

pred_summary <- pred_data %>%
  group_by(draw, lottery_type, ev_diff) %>% 
  summarise(
    resp_mean = mean(resp),   # mean within each draw
    .groups = "drop"
  ) %>%
  group_by(lottery_type, ev_diff) %>%
  summarise(
    mean_resp = mean(resp_mean),             # mean across draws
    lower = quantile(resp_mean, 0.025),     # 95% CI across draws
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(gamble_type = lottery_type)

emp_summary <- df %>% 
  group_by(gamble_type, ev_diff) %>% 
  summarise(mean_resp = mean(resp), .groups = "drop")

ggplot(pred_summary, aes(
  x = ev_diff, y = mean_resp,
  color = gamble_type,
  fill = gamble_type
  )) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, color = NA) +
  geom_line(
    data = emp_summary,
    aes(
      group = gamble_type,
      color = gamble_type
    ),
    linetype = "dashed",
    linewidth = 1
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_point(
    data = emp_summary,
    aes(color = gamble_type),
    size = 2.5
  ) +
  scale_fill_manual(values = COLOR_PALETTE, guide = "none") +
  scale_color_manual(values = COLOR_PALETTE) +
  scale_x_continuous(breaks = unique(df$ev_diff)) +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Difference in EV",
    y = "P(choose lower losses option)",
    color = "Gamble type"
  ) +
  ggthemes::theme_tufte() + 
  theme(
    axis.line = element_line(size = .5, color = "#969696"),
    axis.ticks = element_line(color = "#969696"),
    axis.text.x = element_text(size = FONT_SIZE_3,
                               vjust = 0.5),
    axis.text.y = element_text(size = FONT_SIZE_3),
    strip.text.x = element_text(size = FONT_SIZE_2),
    strip.text.y = element_text(size = FONT_SIZE_2, angle = 0),
    text = element_text(size = FONT_SIZE_2),
    plot.title = element_text(size = FONT_SIZE_1,
                              hjust = 0.5,
                              face = 'bold'),
    panel.grid = element_line(color = "#969696",
                              size = 0.2,
                              linetype = 1),
    legend.spacing.y = unit(0.25, 'cm'),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 5, l = 0))
  )

ggsave(
  '../plots/pt_model_pp_check.pdf',
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)
