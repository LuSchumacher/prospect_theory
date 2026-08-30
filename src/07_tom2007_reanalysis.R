library(tidyverse)
library(cmdstanr)
library(loo)
library(patchwork)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c(
  "Loss aversion" = "#c96016",
  "Mean-SD" = "#CC79A7",
  "Mean-variance" = "#D9AE00",
  "Shifted curvature" = "#42686C"
)
MODEL_LINETYPES <- c(
  "Loss aversion" = "solid",
  "Mean-SD" = "dashed",
  "Mean-variance" = "dotdash",
  "Shifted curvature" = "dotted"
)

CI_LOWER <- 0.025
CI_UPPER <- 0.975

df <- read_csv("../data/tom2007_data.csv") |> 
  mutate(
    ev_b = (outcome_b1 + outcome_b2) / 2,
    loss_magnitude = abs(outcome_b2),
    variance_b = (outcome_b1 - outcome_b2)^2 / 4
  )

make_inits <- function(model, chains = 4, n = N) {
  map(seq_len(chains), function(chain) {
    if (model == "lambda") {
      list(
        mu_lambda = rnorm(1, 2, 0.5),
        mu_tau = rnorm(1, 0.5, 0.5),
        sigma_lambda = runif(1, 0.1, 0.75),
        sigma_tau = runif(1, 0.1, 0.75),
        z_lambda = rnorm(n),
        z_tau = rnorm(n)
      )
    } else if (model == "alpha") {
      list(
        mu_alpha = rnorm(1, 0, 0.5),
        mu_tau = rnorm(1, 0.5, 0.5),
        sigma_alpha = runif(1, 0.1, 0.75),
        sigma_tau = runif(1, 0.1, 0.75),
        z_alpha = rnorm(n),
        z_tau = rnorm(n)
      )
    } else if (model == "mean_variance") {
      list(
        mu_beta_variance = rnorm(1, -4, 0.5),
        mu_tau = rnorm(1, 0.5, 0.5),
        sigma_beta_variance = runif(1, 0.1, 0.75),
        sigma_tau = runif(1, 0.1, 0.75),
        z_beta_variance = rnorm(n),
        z_tau = rnorm(n)
      )
    } else {
      stop("Unknown model: ", model)
    }
  })
}

get_inits_mean_sd <- function(chains = 4, n = N) {
  map(seq_len(chains), function(chain) {
    list(
      mu_kappa = rnorm(1, 0.7, 0.5),
      mu_tau = rnorm(1, 0.5, 0.5),
      sigma_kappa = runif(1, 0.1, 0.75),
      sigma_tau = runif(1, 0.1, 0.75),
      z_kappa = rnorm(n),
      z_tau = rnorm(n)
    )
  })
}

model_free_lambda <- cmdstan_model(
  "../stan_models/pt_model_free_lambda.stan",
  cpp_options = list(stan_threads = TRUE)
)

model_free_alpha <- cmdstan_model(
  "../stan_models/pt_model_free_alpha.stan",
  cpp_options = list(stan_threads = TRUE)
)

model_mean_variance <- cmdstan_model(
  "../stan_models/mean_variance_model.stan",
  cpp_options = list(stan_threads = TRUE)
)

model_mean_sd <- cmdstan_model(
  "../stan_models/mean_standard_deviation_model.stan",
  cpp_options = list(stan_threads = TRUE),
  force_recompile = TRUE
)

# ---------------------------------------------------------------------------- #
# FREE LAMBDA
# ---------------------------------------------------------------------------- #
N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`         = `T`,
  N           = N,
  subject_id  = df$id,
  outcomes_b  = as.matrix(df[, c("outcome_b1", "outcome_b2")]),
  choice      = df$choice
)

fit_free_lambda <- model_free_lambda$sample(
  data = stan_data,
  init = make_inits("lambda"),
  max_treedepth = 10,
  adapt_delta = 0.9,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2
)

fit_free_lambda$save_object("../fits/fit_free_lambda.rds")

# ---------------------------------------------------------------------------- #
# FREE ALPHA
# ---------------------------------------------------------------------------- #
MAX_LOSS = 20

N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`         = `T`,
  N           = N,
  subject_id  = df$id,
  outcomes_b  = as.matrix(df[, c("outcome_b1", "outcome_b2")]) + MAX_LOSS,
  choice      = df$choice
)

fit_free_alpha <- model_free_alpha$sample(
  data = stan_data,
  init = make_inits("alpha"),
  max_treedepth = 10,
  adapt_delta = 0.9,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2
)
  
fit_free_alpha$save_object("../fits/fit_free_alpha.rds")

# ---------------------------------------------------------------------------- #
# MEAN VARIANCE
# ---------------------------------------------------------------------------- #
N <- length(unique(df$id))
`T` <- nrow(df)
stan_data = list(
  `T`         = `T`,
  N           = N,
  subject_id  = df$id,
  outcomes_b  = as.matrix(df[, c("outcome_b1", "outcome_b2")]),
  choice      = df$choice
)

fit_mean_variance <- model_mean_variance$sample(
  data = stan_data,
  init = make_inits("mean_variance"),
  max_treedepth = 10,
  adapt_delta = 0.9,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2
)

fit_mean_variance$save_object("../fits/fit_mean_variance.rds")

# ---------------------------------------------------------------------------- #
# MEAN STD
# ---------------------------------------------------------------------------- #
fit_mean_sd <- model_mean_sd$sample(
  data = stan_data,
  init = get_inits_mean_sd(),
  max_treedepth = 10,
  adapt_delta = 0.9,
  refresh = 100,
  iter_sampling = 2000,
  iter_warmup = 2000,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 2
)

fit_mean_sd$save_object("../fits/fit_mean_sd.rds")

# ---------------------------------------------------------------------------- #
# PARAMETER ESTIMATES
# ---------------------------------------------------------------------------- #
fit_free_lambda <- readRDS("../fits/fit_free_lambda.rds")
fit_free_alpha <- readRDS("../fits/fit_free_alpha.rds")
fit_mean_variance <- readRDS("../fits/fit_mean_variance.rds")
fit_mean_sd <- readRDS("../fits/fit_mean_sd.rds")

summarise_group_parameters <- function(fit, variables) {
  fit$draws(variables = variables) |>
    posterior::summarise_draws(
      median = ~ median(.x),
      ci_lower = ~ quantile(.x, 0.025, names = FALSE),
      ci_upper = ~ quantile(.x, 0.975, names = FALSE),
      rhat = posterior::rhat,
      ess_bulk = posterior::ess_bulk,
      ess_tail = posterior::ess_tail
    )
}

parameter_estimates <- bind_rows(
  summarise_group_parameters(
    fit_free_lambda,
    c("lambda_group_location", "tau_group_location")
  ) |>
    mutate(model = "Loss aversion"),
  summarise_group_parameters(
    fit_mean_sd,
    c(
      "kappa_group_location",
      "lambda_implied_group_location",
      "tau_group_location",
      "tau_lambda_implied_group_location"
    )
  ) |>
    mutate(model = "Mean-SD"),
  summarise_group_parameters(
    fit_mean_variance,
    c(
      "beta_variance_group_location",
      "tau_group_location"
    )
  ) |>
    mutate(model = "Mean-variance"),
  summarise_group_parameters(
    fit_free_alpha,
    c("alpha_group_location", "tau_group_location")
  ) |>
    mutate(model = "Shifted curvature")
) |>
  select(model, everything()) |>
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 2)
    )
  )

print(parameter_estimates, n = Inf)

write_csv(
  parameter_estimates,
  "../data/tom2007_parameter_estimates.csv"
)

# ---------------------------------------------------------------------------- #
# MODEL COMPARISON
# ---------------------------------------------------------------------------- #
fits <- list(
  "Free lambda" = fit_free_lambda,
  "Mean-SD" = fit_mean_sd,
  "Mean-variance" = fit_mean_variance,
  "Free alpha" = fit_free_alpha
)

loos <- map(
  fits,
  ~ .x$loo(
    variables = "log_lik",
    r_eff = TRUE,
    cores = 4,
    save_psis = TRUE
  )
)

model_comp <- loo_compare(loos)
print(model_comp, simplify = FALSE)
write.csv(
  as.data.frame(model_comp),
  "../data/tom2007_loo_comparison.csv",
  row.names = TRUE
)

# ---------------------------------------------------------------------------- #
# PP CHECK
# ---------------------------------------------------------------------------- #
EV_BIN_WIDTH <- 2
VARIANCE_BIN_WIDTH <- 50

bin_numeric <- function(x, width) {
  floor(x / width) * width + width / 2
}

plot_df <- df |>
  mutate(
    ev_b = bin_numeric(ev_b, EV_BIN_WIDTH),
    variance_b = bin_numeric(variance_b, VARIANCE_BIN_WIDTH)
  )

summarise_empirical <- function(data, x_var) {
  data |>
    group_by(id, .data[[x_var]]) |>
    summarise(choice_rate = mean(choice), .groups = "drop") |>
    group_by(.data[[x_var]]) |>
    summarise(choice_mean = mean(choice_rate), .groups = "drop")
}

summarise_predictions <- function(yrep, data, x_var, model_name) {
  x_values <- sort(unique(data[[x_var]]))
  
  map_dfr(x_values, function(x_value) {
    trial_indices <- which(data[[x_var]] == x_value)
    
    indices_by_subject <- split(
      trial_indices,
      data$id[trial_indices]
    )
    
    subject_rates <- vapply(
      indices_by_subject,
      function(indices) {
        rowMeans(yrep[, indices, drop = FALSE])
      },
      numeric(nrow(yrep))
    )
    
    if (is.null(dim(subject_rates))) {
      predicted_rates <- subject_rates
    } else {
      predicted_rates <- rowMeans(subject_rates)
    }
    
    tibble(
      x = x_value,
      prediction = median(predicted_rates),
      lower = quantile(predicted_rates, CI_LOWER),
      upper = quantile(predicted_rates, CI_UPPER),
      model = model_name
    )
  }) |>
    rename(!!x_var := x)
}

yrep <- map(
  fits,
  ~ .x$draws(variables = "y_rep", format = "draws_matrix")
)

x_variables <- c("ev_b", "loss_magnitude", "variance_b")

empirical <- map(x_variables, ~ summarise_empirical(plot_df, .x)) |>
  set_names(x_variables)

MODEL_LABELS <- c(
  "Free lambda" = "Loss aversion",
  "Mean-SD" = "Mean-SD",
  "Mean-variance" = "Mean-variance",
  "Free alpha" = "Shifted curvature"
)

predictions <- map(
  x_variables,
  function(x_var) {
    imap_dfr(
      yrep,
      ~ summarise_predictions(.x, plot_df, x_var, .y)
    ) |>
      mutate(
        model = recode(model, !!!MODEL_LABELS)
      )
  }
) |>
  set_names(x_variables)


make_plot <- function(
    x_var, x_label, x_breaks = waiver(), show_y_label = FALSE
) {
  ggplot() +
    geom_ribbon(
      data = predictions[[x_var]],
      aes(
        x = .data[[x_var]], ymin = lower, ymax = upper,
        fill = model, group = model
      ),
      alpha = 0.25
    ) +
    geom_line(
      data = predictions[[x_var]],
      aes(
        x = .data[[x_var]], y = prediction, color = model,
        linetype = model, group = model
      ),
      linewidth = 0.9
    ) +
    geom_line(
      data = empirical[[x_var]],
      aes(x = .data[[x_var]], y = choice_mean),
      color = "black", linetype = "dashed", linewidth = 0.7
    ) +
    geom_point(
      data = empirical[[x_var]],
      aes(x = .data[[x_var]], y = choice_mean),
      color = "black", size = 1.8
    ) +
    geom_hline(yintercept = 0.5, linetype = "dotted") +
    scale_fill_manual(name = "Model", values = COLOR_PALETTE) +
    scale_color_manual(name = "Model", values = COLOR_PALETTE) +
    scale_linetype_manual(name = "Model", values = MODEL_LINETYPES) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_continuous(breaks = x_breaks) +
    labs(
      x = x_label,
      y = if (show_y_label) "P(accept gamble)" else NULL,
      fill = "Model", color = "Model", linetype = "Model"
    ) +
    ggthemes::theme_tufte(base_size = FONT_SIZE_2) +
    theme(
      axis.title.x = element_text(margin = margin(t = 12)),
      axis.title.y = element_text(margin = margin(r = 12)),
      axis.line = element_line(linewidth = 0.5, color = "#969696"),
      axis.ticks = element_line(color = "#969696"),
      axis.text.x = element_text(size = FONT_SIZE_3, vjust = 0.5),
      axis.text.y = element_text(size = FONT_SIZE_3),
      panel.grid.major = element_line(color = scales::alpha("gray70", 0.3)),
      panel.grid.minor = element_line(color = scales::alpha("gray70", 0.15)),
      panel.background = element_blank(),
      panel.spacing = unit(1.2, "lines"),
      legend.position = "bottom"
    )
}

plot_ev <- make_plot(
  "ev_b",
  "EV of gamble",
  x_breaks = sort(unique(plot_df$ev_b)),
  show_y_label = TRUE
)

plot_loss <- make_plot(
  "loss_magnitude",
  "Loss magnitude"
)

plot_variance <- make_plot(
  "variance_b",
  "Variance of gamble bin"
)

tom2007_ppc <-
  plot_ev + plot_loss + plot_variance +
  plot_layout(nrow = 1, guides = "collect") &
  theme(legend.position = "bottom")

ggsave(
  "../plots/tom2007_reanalysis.pdf",
  plot_ev,
  device = "pdf",
  dpi = 300,
  width = 9, height = 6
)
