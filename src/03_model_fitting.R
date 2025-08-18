library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#27374D', '#B70404')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

################################################################################
# DATA AND MODEL PREPARATION
################################################################################
df <- read_csv('../data/study_data_prepared.csv')

N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`        = `T`,
  N          = N,
  subject_id = df$id,
  condition  = ifelse(df$lottery_type == "confounded", 0, 1),
  outcome_a  = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b  = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice    = df$resp
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

PARAM_NAMES <- c(
  "lambda_0_out", "alpha_0_out", "tau_0_out",
  "lambda_1_out", "alpha_1_out", "tau_1_out"
)

pt_model <- cmdstan_model(
  'pt_model_separate_params.stan',
  cpp_options = list(stan_threads = T)
)

################################################################################
# MODEL FITTING
################################################################################
fit_pt_model <- pt_model$sample(
  data = stan_data,
  init = init_fun(),
  max_treedepth = 5,
  adapt_delta = 0.85,
  refresh = 50,
  iter_sampling = 1000,
  iter_warmup = 1000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2,
  save_warmup = TRUE
)

fit_pt_model$summary(variables = PARAM_NAMES)
mcmc_trace(
  fit_pt_model$draws(inc_warmup = TRUE),
  n_warmup = 1000,
  pars=PARAM_NAMES
)
fit_pt_model$save_object("../fits/fit_pt_model_separate_params.stan.rds")

################################################################################
# POSTERIOR RE-SIMULATION
################################################################################
y_rep <- fit_pt_model$draws("y_rep", format = "matrix")
y_obs <- df$choice

NUM_PP_DRAWS <- 500
idx <- sample(1:4000, NUM_PP_DRAWS)

pred_data <- y_rep %>% 
  as_tibble() %>% 
  mutate(draw = 1:4000) %>% 
  filter(draw %in% idx) %>% 
  pivot_longer(-draw, names_to = "trial", values_to = "resp") %>% 
  mutate(
    trial = str_extract(trial, "(?<=\\[)\\d+(?=\\])"),
    lottery_type = rep(df$lottery_type, times=NUM_PP_DRAWS),
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
  )

emp_summary <- df %>% 
  group_by(lottery_type, ev_diff) %>% 
  summarise(mean_resp = mean(resp), .groups = "drop")

ggplot(pred_summary, aes(
  x = ev_diff, y = mean_resp,
  color = lottery_type,
  fill = lottery_type
  )) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, color = NA) +
  geom_line(
    data = emp_summary,
    aes(
      group = lottery_type,
      color = lottery_type
    ),
    linetype = "dashed",
    linewidth = 1
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_point(
    data = emp_summary,
    aes(color = lottery_type),
    size = 2.5
  ) +
  scale_fill_manual(values = COLOR_PALETTE, guide = "none") +
  scale_color_manual(values = COLOR_PALETTE) +
  scale_x_continuous(breaks = unique(df$ev_diff)) +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Difference in EV",
    y = "P(choose risky)",
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


