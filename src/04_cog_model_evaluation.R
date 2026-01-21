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
COLOR_PALETTE <- c('#c96016', '#42686C')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(mc.cores = 8)

emp_data <- read_csv('../data/study_data_prepared.csv')

models <- c(
  "fit_model_1_0", "fit_model_1_1", "fit_model_1_2", "fit_model_1_3",
  "fit_model_2_0", "fit_model_2_1", "fit_model_2_2", "fit_model_2_3",
  "fit_model_3_0", "fit_model_3_1"
)

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
  "tau_out[1]", "tau_out[2]", "b_tau"
)

# ---------------------------------------------------------------------------- #
# MODEL COMPARISON
# ---------------------------------------------------------------------------- #
loos <- list()
for (i in seq_along(models)) {
  model <- models[i]
  model_fit <- readRDS(paste0("../fits/", model, ".rds"))
  log_lik <- model_fit$draws("log_lik", format = "draws_array")
  loo_model <- loo(log_lik, cores = getOption("mc.cores", 8))
  loos[[i]] <- loo_model
  rm(model_fit)
  gc()
}

model_comparison <- loo_compare(
  loos[[1]], loos[[2]], loos[[3]], loos[[4]],
  loos[[5]], loos[[6]], loos[[7]], loos[[8]],
  loos[[9]], loos[[10]]
  )

# ---------------------------------------------------------------------------- #
# POSTERIOR ESTIMATES
# ---------------------------------------------------------------------------- #
fit_model_2_2 <- readRDS(paste0("../fits/", models[7], ".rds"))
draws <- fit_model_2_2$draws(variables = M2_PARAM_NAMES)
draws_df <- as_draws_df(draws)

draws_tidy <- draws_df %>%
  select(all_of(M2_PARAM_NAMES)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    family = case_when(
      grepl("lambda", parameter) ~ "lambda",
      grepl("alpha", parameter)  ~ "alpha",
      grepl("tau", parameter)    ~ "tau",
      grepl("gamma", parameter)    ~ "gamma"
    ),
    panel = case_when(
      grepl("out", parameter)  ~ "CPT model parameters",
      grepl("b_", parameter) ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      parameter %in% c("lambda_out[1]","alpha_out[1]","tau_out[1]", "gamma_out[1]") ~ "confounded",
      parameter %in% c("lambda_out[2]","alpha_out[2]","tau_out[2]", "gamma_out[2]") ~ "unconfounded",
      TRUE ~ NA_character_
    ),
    panel = factor(panel, levels = c("CPT model parameters", "Effect of gamble type")),
    family = forcats::fct_relevel(family, "lambda", "alpha", "tau", "gamma")
  )

# plotting
FONT_SCALER <- 0
cpt_model_params <- ggplot() +
  geom_density(
    data = draws_tidy %>% filter(panel == "CPT model parameters"),
    aes(x = value, fill = gamble_type),
    alpha = 1, color = NA
  ) +
  scale_fill_manual(values = COLOR_PALETTE, name = "Gamble type") +
  geom_density(
    data = draws_tidy %>% filter(panel == "Effect of gamble type"),
    aes(x = value),
    fill = "darkgray", alpha = 1, color = NA
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
  '../plots/param_estimates_model_2_2.pdf',
  cpt_model_params,
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)


draws <- fit_model_1_1$draws(variables = M1_PARAM_NAMES)
draws_df <- as_draws_df(draws)

draws_tidy <- draws_df %>%
  select(all_of(M1_PARAM_NAMES)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    family = case_when(
      grepl("lambda", parameter) ~ "lambda",
      grepl("alpha", parameter)  ~ "alpha",
      grepl("tau", parameter)    ~ "tau"
    ),
    panel = case_when(
      grepl("out", parameter)  ~ "PT model parameters",
      grepl("b_", parameter) ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      parameter %in% c("lambda_out[1]","alpha_out[1]","tau_out[1]") ~ "confounded",
      parameter %in% c("lambda_out[2]","alpha_out[2]","tau_out[2]") ~ "unconfounded",
      TRUE ~ NA_character_
    ),
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
  '../plots/param_estimates_model_1_1.pdf',
  pt_model_params,
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)

# ---------------------------------------------------------------------------- #
# POSTERIOR RE-SIMULATION
# ---------------------------------------------------------------------------- #
y_rep <- fit_model_1$draws("y_rep", format = "matrix")
y_obs <- emp_data$choice

NUM_PP_DRAWS <- 50
idx <- sample(1:8000, NUM_PP_DRAWS)

pred_data <- y_rep %>% 
  as_tibble() %>% 
  mutate(draw = 1:8000) %>% 
  filter(draw %in% idx) %>% 
  pivot_longer(-draw, names_to = "trial", values_to = "resp") %>% 
  mutate(
    trial = str_extract(trial, "(?<=\\[)\\d+(?=\\])"),
    lottery_type = rep(emp_data$gamble_type, times=NUM_PP_DRAWS),
    ev_diff = rep(emp_data$ev_diff, times=NUM_PP_DRAWS)
  )

pred_summary <- pred_data %>%
  group_by(draw, lottery_type, ev_diff) %>% 
  summarise(
    resp_mean = mean(resp), 
    .groups = "drop"
  ) %>%
  group_by(lottery_type, ev_diff) %>%
  summarise(
    mean_resp = mean(resp_mean),
    lower = quantile(resp_mean, 0.025),
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(gamble_type = lottery_type)

emp_summary <- emp_data %>% 
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
  scale_x_continuous(breaks = unique(emp_data$ev_diff)) +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Difference in EV",
    y = "P(choose lower losses option)",
    color = "Gamble type"
  ) +
  ggthemes::theme_tufte() + 
  theme(
    axis.line = element_line(linewidth = .5, color = "#969696"),
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
                              linewidth = 0.2,
                              linetype = 1),
    legend.spacing.y = unit(0.25, 'cm'),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 5, l = 0))
  )

ggsave(
  '../plots/pt_model_1_pp_check.pdf',
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)


y_rep <- fit_model_1_1$draws("y_rep", format = "matrix")

pred_data <- y_rep %>% 
  as_tibble() %>% 
  mutate(draw = 1:8000) %>% 
  filter(draw %in% idx) %>% 
  pivot_longer(-draw, names_to = "trial", values_to = "resp") %>% 
  mutate(
    trial = str_extract(trial, "(?<=\\[)\\d+(?=\\])"),
    lottery_type = rep(emp_data$gamble_type, times=NUM_PP_DRAWS),
    ev_diff = rep(emp_data$ev_diff, times=NUM_PP_DRAWS)
  )

pred_summary <- pred_data %>%
  group_by(draw, lottery_type, ev_diff) %>% 
  summarise(
    resp_mean = mean(resp), 
    .groups = "drop"
  ) %>%
  group_by(lottery_type, ev_diff) %>%
  summarise(
    mean_resp = mean(resp_mean),
    lower = quantile(resp_mean, 0.025),
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(gamble_type = lottery_type)


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
  scale_x_continuous(breaks = unique(emp_data$ev_diff)) +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  labs(
    x = "Difference in EV",
    y = "P(choose lower losses option)",
    color = "Gamble type"
  ) +
  ggthemes::theme_tufte() + 
  theme(
    axis.line = element_line(linewidth = .5, color = "#969696"),
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
                              linewidth = 0.2,
                              linetype = 1),
    legend.spacing.y = unit(0.25, 'cm'),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 5, l = 0))
  )

ggsave(
  '../plots/pt_model_1_1_pp_check.pdf',
  device = 'pdf', dpi = 300,
  width = 10, height = 6
)


