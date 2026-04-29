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
  gamble_type  = ifelse(df$gamble_type == "confounded", 1, 2),
  outcome_a  = as.matrix(df[, c("outcome_a1", "outcome_a2", "outcome_a3")]),
  outcome_b  = as.matrix(df[, c("outcome_b1", "outcome_b2", "outcome_b3")]),
  choice    = df$resp
)

init_fun <- function(chains = 4, N) {
  inits <- vector("list", chains)
  for (i in 1:chains) {
    inits[[i]] <- list(
      # population intercepts
      intercept_b_loss = rnorm(1, 0, 0.5),
      intercept_b_var  = rnorm(1, 0, 0.5),
      intercept_tau    = rnorm(1, 0.5, 0.5),
      # condition effects (betas)
      beta_b_loss = rnorm(1, 0, 0.2),
      beta_b_var  = rnorm(1, 0, 0.2),
      beta_tau    = rnorm(1, 0, 0.2),
      # population STDs (in unconstrained space)
      sigma_b_loss = rnorm(1, 0, 1),
      sigma_b_var  = rnorm(1, 0, 1),
      sigma_tau    = rnorm(1, 0, 1),
      # latent subject variables
      z_b_loss = rnorm(N, 0, 1),
      z_b_var  = rnorm(N, 0, 1),
      z_tau    = rnorm(N, 0, 1)
    )
  }
  return(inits)
}

PARAM_NAMES <- c(
  "b_loss_out[1]", "b_loss_out[2]",
  "b_var_out[1]",  "b_var_out[2]",
  "tau_out[1]",    "tau_out[2]"
)

mvl_model <- cmdstan_model(
  'mvl_model.stan',
  cpp_options = list(stan_threads = T)
)

################################################################################
# MODEL FITTING
################################################################################
fit_mvl_model <- mvl_model$sample(
  data = stan_data,
  init = init_fun(N = N),
  max_treedepth = 10,
  adapt_delta = 0.85,
  refresh = 50,
  iter_sampling = 1000,
  iter_warmup = 1000,
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

fit_mvl_model$save_object("../fits/fit_mvl_model.rds")

################################################################################
# MODEL EVALUATION
################################################################################
draws <- as_draws_df(fit_mvl_model)

PARAM_NAMES <- c(
  "b_loss_out[1]", "b_loss_out[2]", "beta_b_loss",
  "b_var_out[1]",  "b_var_out[2]", "beta_b_var",
  "tau_out[1]",    "tau_out[2]", "beta_tau"
)

# tidy posterior draws
draws_tidy <- draws %>%
  select(all_of(PARAM_NAMES)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    family = case_when(
      grepl("b_loss", parameter) ~ "b_loss",
      grepl("b_var", parameter)  ~ "b_var",
      grepl("tau", parameter)    ~ "tau"
    ),
    panel = case_when(
      grepl("out", parameter)  ~ "MVL model parameters",
      grepl("beta", parameter) ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      parameter %in% c("b_loss_out[1]","b_var_out[1]","tau_out[1]") ~ "confounded",
      parameter %in% c("b_loss_out[2]","b_var_out[2]","tau_out[2]") ~ "unconfounded",
      TRUE ~ NA_character_
    ),
    # force desired column order: params in col 1, betas in col 2
    panel = factor(panel, levels = c("MVL model parameters", "Effect of gamble type"))
  )

# plotting
FONT_SCALER <- 0
mvl_model_params <- ggplot() +
  geom_density(
    data = draws_tidy %>% filter(panel == "MVL model parameters"),
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
  '../plots/mvl_model_parameters.pdf',
  mvl_model_params,
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)

################################################################################
# POSTERIOR RE-SIMULATION
################################################################################
y_rep <- fit_mvl_model$draws("y_rep", format = "matrix")
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
  '../plots/mvl_model_pp_check.pdf',
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)

