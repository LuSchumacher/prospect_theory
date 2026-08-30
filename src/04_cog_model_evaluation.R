library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)
library(posterior)
library(loo)

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#c96016', '#42686C')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
options(mc.cores = 8)

emp_data <- read_csv("../data/study_data_prepared.csv") %>%
  mutate(ev_diff_sl = -ev_diff)

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
  "tau_out[1]", "tau_out[2]", "beta_tau"
)

# ---------------------------------------------------------------------------- #
# MODEL COMPARISON
# ---------------------------------------------------------------------------- #
loos <- set_names(vector("list", length(models)), models)

for (model in models) {
  model_fit <- readRDS(file.path("../fits", paste0(model, ".rds")))
  log_lik <- model_fit$draws("log_lik", format = "draws_array")
  loos[[model]] <- loo(log_lik, cores = getOption("mc.cores", 8))
  rm(model_fit, log_lik)
  gc()
}

model_comparison <- loo_compare(loos)

model_comparison %>%
  as_tibble() %>% 
  mutate(across(where(is.numeric), ~ round(.x, 2))) %>%
  write_csv("../data/model_comparison.csv")

# ---------------------------------------------------------------------------- #
# POSTERIOR ESTIMATES
# ---------------------------------------------------------------------------- #
fit_model <- readRDS("../fits/fit_model_2_2.rds")
draws <- fit_model$draws(variables = M2_PARAM_NAMES)
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
      grepl("out", parameter)  ~ "CPT parameters",
      grepl("b_", parameter) ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      parameter %in% c("lambda_out[1]","alpha_out[1]","tau_out[1]", "gamma_out[1]") ~ "Aligned",
      parameter %in% c("lambda_out[2]","alpha_out[2]","tau_out[2]", "gamma_out[2]") ~ "Opposed",
      TRUE ~ NA_character_
    ),
    panel = factor(panel, levels = c("CPT parameters", "Effect of gamble type")),
    family = forcats::fct_relevel(family, "lambda", "alpha", "tau", "gamma")
  )

estimate_summary <- draws_tidy %>%
  group_by(parameter) %>%
  summarise(
    median = median(value),
    ci_lower = quantile(value, 0.025),
    ci_upper = quantile(value, 0.975),
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))

calc_BF <- function(posterior_samples, prior_sd = 0.5) {
  bandwidth <- bw.nrd0(posterior_samples)
  
  log_kernel <- dnorm(
    0,
    mean = posterior_samples,
    sd = bandwidth,
    log = TRUE
  )
  
  maximum <- max(log_kernel)
  
  log_posterior_at_zero <- maximum +
    log(mean(exp(log_kernel - maximum)))
  
  log_prior_at_zero <- dnorm(
    0,
    mean = 0,
    sd = prior_sd,
    log = TRUE
  )
  
  exp(log_prior_at_zero - log_posterior_at_zero)
}

compute_BFs <- function(fit, param_names, prior_sd = 0.5) {
  sapply(param_names, function(param) {
    samples <- as.numeric(fit$draws(variables = param))
    calc_BF(samples, prior_sd)
  })
}

effect_parameters <- c(
  "b_lambda",
  "b_alpha",
  "b_tau",
  "b_gamma"
)

bf_summary <- compute_BFs(
  fit_model,
  effect_parameters,
  prior_sd = 0.5
)

bf_summary

zero_lines <- draws_tidy %>%
  filter(panel == "Effect of gamble type") %>%
  distinct(family, panel)

cpt_model_params <- ggplot() +
  geom_density(
    data = draws_tidy %>% filter(panel == "CPT parameters"),
    aes(x = value, fill = gamble_type),
    alpha = 0.9,
    color = NA
  ) +
  geom_density(
    data = draws_tidy %>% filter(panel == "Effect of gamble type"),
    aes(x = value),
    fill = "darkgray",
    alpha = 1,
    color = NA
  ) +
  geom_vline(
    data = zero_lines,
    aes(xintercept = 0),
    linetype = "dashed",
    inherit.aes = FALSE
  ) +
  scale_fill_manual(values = COLOR_PALETTE, name = "Gamble type") +
  ggh4x::facet_grid2(
    family ~ panel,
    labeller = labeller(
      family = label_parsed,
      panel = label_value
    ),
    scales = "free",
    independent = "all"
  ) +
  labs(x = "Value", y = "Density") +
  ggthemes::theme_tufte(base_size = FONT_SIZE_2) +
  theme(
    axis.title.x = element_text(margin = margin(t = 12)),
    axis.title.y = element_text(margin = margin(r = 12)),
    axis.line = element_line(linewidth = 0.5, color = "#969696"),
    axis.ticks = element_line(color = "#969696"),
    axis.text.x = element_text(size = FONT_SIZE_3, vjust = 0.5),
    axis.text.y = element_text(size = FONT_SIZE_3),
    strip.text.x = element_text(size = FONT_SIZE_2),
    strip.text.y = element_text(
      size = FONT_SIZE_2,
      hjust = 0,
      angle = 0
    ),
    panel.grid.major = element_line(
      color = scales::alpha("gray70", 0.3)
    ),
    panel.grid.minor = element_line(
      color = scales::alpha("gray70", 0.15)
    ),
    panel.background = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    legend.position = "bottom",
    legend.margin = margin(t = -5, r = 0, b = 0, l = 0),
    legend.spacing.y = unit(0.2, "cm")
  )

ggsave(
  "../plots/cpt_model_posteriors.pdf",
  cpt_model_params,
  device = "pdf",
  width = 11,
  height = 9
)

# -------------------------------------------------------------------------
# POSTERIOR RESIMULATION
# -------------------------------------------------------------------------
NUM_PP_DRAWS <- 100
VAR_BIN_WIDTH <- 1
SKEW_BIN_WIDTH <- 0.2

y_rep <- fit_model$draws(
  variables = "y_rep",
  format = "matrix"
)

draw_ids <- sample(
  seq_len(nrow(y_rep)),
  size = min(NUM_PP_DRAWS, nrow(y_rep)),
  replace = FALSE
)

# HELPER FUNCTIONS
# ----------------
format_bin_value <- function(x) {
  x <- round(x, digits = 3)
  sub(
    pattern = "\\.?0+$",
    replacement = "",
    x = format(
      x,
      trim = TRUE,
      scientific = FALSE
    )
  )
}

summarise_pp <- function(data, x_var) {
  data %>%
    filter(!is.na(.data[[x_var]])) %>%
    group_by(draw, gamble_type, .data[[x_var]]) %>%
    summarise(
      resp_mean = mean(resp_pred, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(gamble_type, .data[[x_var]]) %>%
    summarise(
      mean_resp = mean(resp_mean, na.rm = TRUE),
      lower = quantile(resp_mean, probs = 0.025, na.rm = TRUE),
      upper = quantile(resp_mean, probs = 0.975, na.rm = TRUE),
      .groups = "drop"
    )
}

summarise_empirical <- function(data, x_var) {
  data %>%
    filter(!is.na(.data[[x_var]])) %>%
    group_by(id, gamble_type, .data[[x_var]]) %>%
    summarise(
      choice_prop = mean(resp, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(gamble_type, .data[[x_var]]) %>%
    summarise(
      mean_resp = mean(choice_prop,na.rm = TRUE),
      n_participants = n(),
      se_resp = sd(choice_prop, na.rm = TRUE) / sqrt(n_participants),
      .groups = "drop"
    )
}

make_pp_plot <- function(
    x_var,
    x_label,
    x_breaks = waiver(),
    x_labels = waiver()
) {
  pred_summary <- summarise_pp(pred_data, x_var)
  emp_summary <- summarise_empirical(emp_data, x_var)
  
  ggplot(
    pred_summary,
    aes(
      x = .data[[x_var]],
      y = mean_resp,
      color = gamble_type,
      fill = gamble_type,
      group = gamble_type
    )
  ) +
    geom_ribbon(
      aes(ymin = lower, ymax = upper),
      alpha = 0.4,
      color = NA
    ) +
    geom_line(
      data = emp_summary,
      aes(
        x = .data[[x_var]],
        y = mean_resp,
        color = gamble_type,
        group = gamble_type
      ),
      linetype = "dashed",
      linewidth = 1
    ) +
    geom_errorbar(
      data = emp_summary,
      aes(
        x = .data[[x_var]],
        y = mean_resp,
        ymin = pmax(0, mean_resp - 1.96 * se_resp),
        ymax = pmin(1, mean_resp + 1.96 * se_resp),
        color = gamble_type
      ),
      width = 0.05,
      linewidth = 1
    ) +
    geom_point(
      data = emp_summary,
      aes(
        x = .data[[x_var]],
        y = mean_resp,
        color = gamble_type
      ),
      size = 2.5
    ) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
    scale_fill_manual(
      values = COLOR_PALETTE,
      guide = "none"
    ) +
    scale_color_manual(values = COLOR_PALETTE) +
    scale_x_continuous(
      breaks = x_breaks,
      labels = x_labels
    ) +
    scale_y_continuous(
      limits = c(-0.025, 1.025),
      breaks = seq(0, 1, by = 0.2)
    ) +
    labs(
      x = x_label,
      y = "P(choose smaller-loss option)",
      color = "Gamble type"
    ) +
    ggthemes::theme_tufte(base_size = FONT_SIZE_2) +
    theme(
      axis.title.x = element_text(margin = margin(t = 12)),
      axis.title.y = element_text(margin = margin(r = 12)),
      axis.line = element_line(linewidth = 0.5, color = "#969696"),
      axis.ticks = element_line(color = "#969696"),
      axis.text.x = element_text(size = FONT_SIZE_3, vjust = 0.5),
      axis.text.y = element_text(size = FONT_SIZE_3),
      panel.grid.major = element_line(
        color = scales::alpha("gray70", 0.3)
      ),
      panel.grid.minor = element_line(
        color = scales::alpha("gray70", 0.15)
      ),
      panel.background = element_blank(),
      panel.spacing = unit(1.2, "lines"),
      legend.position = "bottom",
      legend.margin = margin(t = -5),
      legend.spacing.y = unit(0.2, "cm")
    )
}


# BINNING
# -------
max_abs_var_diff <- max(
  abs(emp_data$var_diff),
  na.rm = TRUE
)

max_var_break <- ceiling(
  max_abs_var_diff / VAR_BIN_WIDTH
) * VAR_BIN_WIDTH

if (max_var_break == 0) {
  max_var_break <- VAR_BIN_WIDTH
}

var_breaks <- seq(
  from = 0,
  to = max_var_break,
  by = VAR_BIN_WIDTH
)

var_bin_midpoints <- head(var_breaks, -1) +
  diff(var_breaks) / 2

var_bin_labels <- paste0(
  format_bin_value(head(var_breaks, -1)),
  "–",
  format_bin_value(tail(var_breaks, -1))
)

min_skew_diff <- min(
  emp_data$skew_diff,
  na.rm = TRUE
)

max_skew_diff <- max(
  emp_data$skew_diff,
  na.rm = TRUE
)

min_skew_break <- floor(
  min_skew_diff / SKEW_BIN_WIDTH
) * SKEW_BIN_WIDTH

max_skew_break <- ceiling(
  max_skew_diff / SKEW_BIN_WIDTH
) * SKEW_BIN_WIDTH

if (min_skew_break == max_skew_break) {
  max_skew_break <- min_skew_break + SKEW_BIN_WIDTH
}

skew_breaks <- seq(
  from = min_skew_break,
  to = max_skew_break,
  by = SKEW_BIN_WIDTH
)

skew_bin_midpoints <- head(skew_breaks, -1) +
  diff(skew_breaks) / 2

skew_bin_labels <- paste0(
  format_bin_value(head(skew_breaks, -1)),
  "–",
  format_bin_value(tail(skew_breaks, -1))
)

emp_data <- emp_data %>%
  mutate(
    abs_var_diff = abs(var_diff),
    var_diff_bin_id = cut(
      abs_var_diff,
      breaks = var_breaks,
      include.lowest = TRUE,
      right = TRUE,
      labels = FALSE
    ),
    var_diff_bin = var_bin_midpoints[var_diff_bin_id],
    skew_diff_bin_id = cut(
      skew_diff,
      breaks = skew_breaks,
      include.lowest = TRUE,
      right = TRUE,
      labels = FALSE
    ),
    
    skew_diff_bin = skew_bin_midpoints[skew_diff_bin_id]
  )

trial_info <- emp_data %>%
  mutate(trial = row_number()) %>%
  select(
    trial, id, gamble_type, resp, ev_diff_sl,
    loss_diff, var_diff, abs_var_diff,
    var_diff_bin, skew_diff, skew_diff_bin
  )

# PREPARE DRAWS
# -------------
pred_data <- y_rep[draw_ids, , drop = FALSE] %>%
  as_tibble() %>%
  mutate(draw = seq_len(n())) %>%
  pivot_longer(
    cols = -draw,
    names_to = "trial",
    values_to = "resp_pred"
  ) %>%
  mutate(trial = as.integer(str_extract(trial, "(?<=\\[)\\d+(?=\\])"))) %>%
  left_join(trial_info, by = "trial") |> 
  mutate(gamble_type = ifelse(gamble_type == "confounded", "Aligned", "Opposed"))

emp_data <- emp_data |> 
  mutate(gamble_type = ifelse(gamble_type == "confounded", "Aligned", "Opposed"))

# PLOTTING
# ------------
plot_ev_diff <- make_pp_plot(
  x_var = "ev_diff_sl",
  x_label = expression(Delta * EV ~ "(smaller-loss minus larger-loss option)"),
  x_breaks = sort(unique(emp_data$ev_diff_sl))
)

plot_loss_diff <- make_pp_plot(
  x_var = "loss_diff",
  x_label = "Difference in loss magnitude"
)

plot_var_diff <- make_pp_plot(
  x_var = "var_diff_bin",
  x_label = "Absolute difference in std. dev.",
  x_breaks = var_bin_midpoints,
  x_labels = var_bin_labels
)

plot_skew_diff <- make_pp_plot(
  x_var = "skew_diff_bin",
  x_label = "Difference in skewness",
  x_breaks = skew_bin_midpoints,
  x_labels = skew_bin_labels
)

ggsave(
  "../plots/cpt_model_pp_check.pdf",
  plot_ev_diff,
  device = 'pdf', dpi = 300,
  width = 9, height = 6
)

# ---------------------------------------------------------------------------- #
# POSTERIOR ESTIMATES (MVL MODEL)
# ---------------------------------------------------------------------------- #
fit_model <- readRDS(paste0("../fits/", models[10], ".rds"))
draws <- fit_model$draws(variables = M3_PARAM_NAMES)
draws_df <- as_draws_df(draws)

draws_tidy_mvl <- draws_df %>%
  select(all_of(M3_PARAM_NAMES)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
  mutate(
    family = case_when(
      grepl("loss", parameter) ~ "loss",
      grepl("var", parameter)  ~ "variance",
      grepl("tau", parameter)  ~ "tau"
    ),
    panel = case_when(
      grepl("_out\\[", parameter) ~ "MVL parameters",
      TRUE                        ~ "Effect of gamble type"
    ),
    gamble_type = case_when(
      grepl("\\[1\\]$", parameter) ~ "Aligned",
      grepl("\\[2\\]$", parameter) ~ "Opposed",
      TRUE                         ~ NA_character_
    ),
    panel = factor(panel, levels = c("MVL parameters", "Effect of gamble type")),
    family = forcats::fct_relevel(family, "loss", "variance", "tau")
  )

mvl_estimate_summary <- draws_tidy_mvl %>% 
  group_by(parameter) %>% 
  summarise(
    median = median(value),
    ci_lower = quantile(value, 0.025),
    ci_upper = quantile(value, 0.975)
  ) %>% 
  mutate(across(where(is.numeric), round, 2))


calc_BF <- function(posterior_samples, prior_sd = 0.5) {
  bandwidth <- bw.nrd0(posterior_samples)
  log_kernel <- dnorm(
    0,
    mean = posterior_samples,
    sd = bandwidth,
    log = TRUE
  )
  
  maximum <- max(log_kernel)
  log_posterior_at_zero <- maximum +
    log(mean(exp(log_kernel - maximum)))
  
  log_prior_at_zero <- dnorm(
    0,
    mean = 0,
    sd = prior_sd,
    log = TRUE
  )
  
  exp(log_prior_at_zero - log_posterior_at_zero)
}

compute_BFs <- function(fit, param_names, prior_sd = 0.5) {
  sapply(param_names, function(param) {
    samples <- as.numeric(fit$draws(param))
    calc_BF(samples, prior_sd)
  })
}

mvl_effect_names <- c("beta_b_loss", "beta_b_var", "beta_tau")
mvl_prior_sds <- c(0.5, 0.005, 0.5)

mvl_bf_summary <- map2_dbl(
  mvl_effect_names,
  mvl_prior_sds,
  ~ calc_BF(as.numeric(fit_model$draws(.x)), prior_sd = .y)
) %>%
  set_names(mvl_effect_names) %>%
  round(2)

mvl_model_params <- ggplot() +
  geom_density(
    data = draws_tidy_mvl %>% filter(panel == "MVL parameters"),
    aes(x = value, fill = gamble_type),
    alpha = 0.9, color = NA
  ) +
  scale_fill_manual(values = COLOR_PALETTE, name = "Gamble type") +
  geom_density(
    data = draws_tidy_mvl %>% filter(panel == "Effect of gamble type"),
    aes(x = value),
    fill = "darkgray", alpha = 1, color = NA
  ) +
  ggh4x::facet_grid2(
    family ~ panel,
    labeller = labeller(
      family = label_parsed,
      panel  = label_value
    ),
    scales = "free",
    independent = "all"
  ) +
  geom_vline(
    data = draws_tidy_mvl %>% filter(panel == "Effect of gamble type"),
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
    axis.text.x = element_text(size = FONT_SIZE_3, vjust = 0.5),
    axis.text.y = element_text(size = FONT_SIZE_3),
    strip.text.x = element_text(size = FONT_SIZE_2),
    strip.text.y = element_text(size = FONT_SIZE_2, hjust = 0, angle = 0),
    panel.grid.major = element_line(color = scales::alpha("gray70", 0.3)),
    panel.grid.minor = element_line(color = scales::alpha("gray70", 0.15)),
    panel.background = element_blank(),
    panel.spacing = unit(1.2, "lines"),
    legend.position = "bottom",
    legend.margin = margin(t = -5, r = 0, b = 0, l = 0),
    legend.spacing.y = unit(0.2, "cm"),
  )

ggsave(
  '../plots/mvl_model_posteriors.pdf',
  mvl_model_params,
  device = 'pdf', dpi = 300,
  width = 11, height = 9
)
