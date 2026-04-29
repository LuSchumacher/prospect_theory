library(tidyverse)
library(magrittr)
library(cmdstanr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

emp_data <- read_csv('../data/study_data_prepared.csv')

ID_VECTOR <- emp_data$id
NUM_SUBJECT <- length(unique(ID_VECTOR))
CONDITION <- ifelse(emp_data$gamble_type == "confounded", 1, 2)
EV_DIFFS <- emp_data$ev_diff

EMPIRICAL_OUTCOMES_A <- cbind(
  emp_data$outcome_a1,
  emp_data$outcome_a2,
  emp_data$outcome_a3
)

EMPIRICAL_OUTCOMES_B <- cbind(
  emp_data$outcome_b1,
  emp_data$outcome_b2,
  emp_data$outcome_b3
)

cpt_model <- readRDS("../fits/fit_model_2_2.rds")
mvl_model <- readRDS("../fits/fit_model_3_1.rds")

CPT_PARAMS <- c("lambda", "alpha", "tau", "gamma")
MVL_PARAMS <- c("b_loss", "b_var", "tau")

NUM_SIM <- 100

SCALE <- 2
FONT_SIZE_1 <- 22 - SCALE
FONT_SIZE_2 <- 20 - SCALE
FONT_SIZE_3 <- 18 - SCALE
COLOR_PALETTE <- c('#c96016', '#42686C')

# ---------------------------------------------------------------------------- #
# EXTRACT INDIVIDUAL PARAMETER DRAWS
# ---------------------------------------------------------------------------- #
cpt_post_indiv <- cpt_model$draws(CPT_PARAMS, format = "df") %>%
  pivot_longer(
    cols = -c(.draw, .iteration, .chain),
    names_to = c("param", "condition", "id"),
    names_pattern = "(.*)\\[(.*),(.*)\\]"
  ) %>%
  pivot_wider(
    names_from = param,
    values_from = value
  ) %>%
  mutate(
    condition = as.integer(condition),
    id = as.integer(id)
  ) %>% 
  select(-c(.iteration, .chain)) %>% 
  rename(draw = .draw)

mvl_post_indiv <- mvl_model$draws(MVL_PARAMS, format = "df") %>%
  pivot_longer(
    cols = -c(.draw, .iteration, .chain),
    names_to = c("param", "condition", "id"),
    names_pattern = "(.*)\\[(.*),(.*)\\]"
  ) %>%
  pivot_wider(
    names_from = param,
    values_from = value
  ) %>%
  mutate(
    id = as.integer(id),
    condition = as.integer(condition)
  ) %>% 
  select(-c(.iteration, .chain)) %>% 
  rename(draw = .draw)

# ---------------------------------------------------------------------------- #
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------- #
get_prelec_weights <- function(p, gamma) {
  exp(-(-log(p))^gamma)
}

get_cpt_utility <- function(x, alpha, lambda, gamma) {
  v <- numeric(3)
  
  for (k in 1:3) {
    if (x[k] >= 0) {
      v[k] <- x[k]^alpha
    } else {
      v[k] <- -lambda * abs(x[k])^alpha
    }
  }
  
  v <- sort(v)
  w1 <- get_prelec_weights(1/3, gamma)
  w2 <- get_prelec_weights(2/3, gamma)
  w3 <- 1
  
  utility <-
    w1 * v[1] +
    (w2 - w1) * v[2] +
    (w3 - w2) * v[3]
  
  return(utility)
}

simulate_cpt_choices <- function(
    outcomes_a,
    outcomes_b,
    id,
    condition,
    parameters
) {
  
  N <- nrow(outcomes_a)
  choices <- integer(N)
  
  for (i in seq_len(N)) {
    
    s <- id[i]
    c <- condition[i]
    
    utility_a <- get_cpt_utility(
      outcomes_a[i, ],
      parameters$alpha[parameters$id == s & parameters$condition == c],
      parameters$lambda[parameters$id == s & parameters$condition == c],
      parameters$gamma[parameters$id == s & parameters$condition == c]
    )
    
    utility_b <- get_cpt_utility(
      outcomes_b[i, ],
      parameters$alpha[parameters$id == s & parameters$condition == c],
      parameters$lambda[parameters$id == s & parameters$condition == c],
      parameters$gamma[parameters$id == s & parameters$condition == c]
    )
    
    tau <- parameters$tau[parameters$id == s & parameters$condition == c]
    
    p <- plogis(tau * (utility_b - utility_a))
    
    choices[i] <- rbinom(1, 1, p)
  }
  
  return(choices)
}

get_mvl_utility <- function(x, b_var, b_loss) {
  mean(x) - b_var * sd(x) - b_loss * abs(min(x))
}

simulate_mvl_choices <- function(
    outcomes_a,
    outcomes_b,
    id,
    condition,
    parameters
) {
  
  N <- nrow(outcomes_a)
  choices <- integer(N)
  
  for (i in seq_len(N)) {
    
    s <- id[i]
    c <- condition[i]
    
    utility_a <- get_mvl_utility(
      outcomes_a[i, ],
      parameters$b_var[parameters$id == s & parameters$condition == c],
      parameters$b_loss[parameters$id == s & parameters$condition == c]
    )
    
    utility_b <- get_mvl_utility(
      outcomes_b[i, ],
      parameters$b_var[parameters$id == s & parameters$condition == c],
      parameters$b_loss[parameters$id == s & parameters$condition == c]
    )
    
    tau <- parameters$tau[parameters$id == s & parameters$condition == c]
    
    p <- plogis(tau * (utility_b - utility_a))
    
    choices[i] <- rbinom(1, 1, p)
  }
  
  return(choices)
}

# ---------------------------------------------------------------------------- #
# RESIMULATION
# ---------------------------------------------------------------------------- #
set.seed(123)

cpt_draw_ids <- unique(cpt_post_indiv$draw)
mvl_draw_ids <- unique(mvl_post_indiv$draw)

cpt_draw_sample <- sample(cpt_draw_ids, NUM_SIM, replace = FALSE)
mvl_draw_sample <- sample(mvl_draw_ids, NUM_SIM, replace = FALSE)

cpt_sim_data <- tibble()
mvl_sim_data <- tibble()

for (sim in seq_len(NUM_SIM)) {
  draw_cpt <- cpt_draw_sample[sim]
  draw_mvl <- mvl_draw_sample[sim]
  
  params_cpt <- cpt_post_indiv %>% filter(draw == draw_cpt)
  params_mvl <- mvl_post_indiv %>% filter(draw == draw_mvl)

  cpt_choices <- simulate_cpt_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    CONDITION,
    params_cpt
  )
  
  mvl_choices <- simulate_mvl_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    CONDITION,
    params_mvl
  )
  
  cpt_sim_data <- bind_rows(
    cpt_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_cpt,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = cpt_choices,
      ev_diff = EV_DIFFS,
      model = "CPT"
    )
  )
  
  mvl_sim_data <- bind_rows(
    mvl_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_mvl,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = mvl_choices,
      ev_diff = EV_DIFFS,
      model = "MVL"
    )
  )
}

regular_ppc_data <- bind_rows(cpt_sim_data, mvl_sim_data)

cpt_sim_data <- tibble()
mvl_sim_data <- tibble()

for (sim in seq_len(NUM_SIM)) {
  draw_cpt <- cpt_draw_sample[sim]
  draw_mvl <- mvl_draw_sample[sim]
  
  params_cpt <- cpt_post_indiv %>% filter(draw == draw_cpt)
  params_mvl <- mvl_post_indiv %>% filter(draw == draw_mvl)
  
  cpt_choices <- simulate_cpt_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    rep(2, length(CONDITION)),
    params_cpt
  )
  
  mvl_choices <- simulate_mvl_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    rep(2, length(CONDITION)),
    params_mvl
  )
  
  cpt_sim_data <- bind_rows(
    cpt_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_cpt,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = cpt_choices,
      ev_diff = EV_DIFFS,
      model = "CPT"
    )
  )
  
  mvl_sim_data <- bind_rows(
    mvl_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_mvl,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = mvl_choices,
      ev_diff = EV_DIFFS,
      model = "MVL"
    )
  )
}

unconfound_only_ppc_data <- bind_rows(cpt_sim_data, mvl_sim_data)

cpt_sim_data <- tibble()
mvl_sim_data <- tibble()

for (sim in seq_len(NUM_SIM)) {
  draw_cpt <- cpt_draw_sample[sim]
  draw_mvl <- mvl_draw_sample[sim]
  
  params_cpt <- cpt_post_indiv %>% filter(draw == draw_cpt)
  params_mvl <- mvl_post_indiv %>% filter(draw == draw_mvl)
  
  cpt_choices <- simulate_cpt_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    rep(1, length(CONDITION)),
    params_cpt
  )
  
  mvl_choices <- simulate_mvl_choices(
    EMPIRICAL_OUTCOMES_A,
    EMPIRICAL_OUTCOMES_B,
    ID_VECTOR,
    rep(1, length(CONDITION)),
    params_mvl
  )
  
  cpt_sim_data <- bind_rows(
    cpt_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_cpt,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = cpt_choices,
      ev_diff = EV_DIFFS,
      model = "CPT"
    )
  )
  
  mvl_sim_data <- bind_rows(
    mvl_sim_data,
    tibble(
      sim_id = sim,
      draw = draw_mvl,
      id = ID_VECTOR,
      condition = CONDITION,
      choice = mvl_choices,
      ev_diff = EV_DIFFS,
      model = "MVL"
    )
  )
}

confound_only_ppc_data <- bind_rows(cpt_sim_data, mvl_sim_data)

# ---------------------------------------------------------------------------- #
# PLOTTING
# ---------------------------------------------------------------------------- #
emp_summary <- emp_data %>% 
  group_by(id, gamble_type, ev_diff) %>% 
  summarise(
    choice_prop = mean(resp),
    .groups = "drop"
  ) %>% 
  group_by(gamble_type, ev_diff) %>% 
  summarise(
    mean_resp = mean(choice_prop),
    std_resp = sd(choice_prop) / sqrt(NUM_SUBJECT - 1),
    .groups = "drop"
  ) %>% 
  mutate(
    gamble_type = ifelse(
      gamble_type == "confounded",
      "Confounded",
      "Unconfounded"
    )
  )

regular_pred_summary <- regular_ppc_data %>%
  group_by(sim_id, model, condition, ev_diff) %>% 
  summarise(
    resp_mean = mean(choice), 
    .groups = "drop"
  ) %>%
  group_by(model, condition, ev_diff) %>%
  summarise(
    mean_resp = median(resp_mean),
    lower = quantile(resp_mean, 0.025),
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(
    gamble_type = condition,
    gamble_type = ifelse(gamble_type == 1, "Confounded", "Unconfounded"),
    sim_type = "Regular PPC"
  )

unconfound_only_pred_summary <- unconfound_only_ppc_data %>%
  group_by(sim_id, model, condition, ev_diff) %>% 
  summarise(
    resp_mean = mean(choice), 
    .groups = "drop"
  ) %>%
  group_by(model, condition, ev_diff) %>%
  summarise(
    mean_resp = median(resp_mean),
    lower = quantile(resp_mean, 0.025),
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(
    gamble_type = condition,
    gamble_type = ifelse(gamble_type == 1, "Confounded", "Unconfounded"),
    sim_type = "Unconfounded only PPC"
  )

confound_only_pred_summary <- confound_only_ppc_data %>%
  group_by(sim_id, model, condition, ev_diff) %>% 
  summarise(
    resp_mean = mean(choice), 
    .groups = "drop"
  ) %>%
  group_by(model, condition, ev_diff) %>%
  summarise(
    mean_resp = median(resp_mean),
    lower = quantile(resp_mean, 0.025),
    upper = quantile(resp_mean, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(
    gamble_type = condition,
    gamble_type = ifelse(gamble_type == 1, "Confounded", "Unconfounded"),
    sim_type = "Confounded only PPC"
  )

pred_summary <- bind_rows(
  regular_pred_summary,
  unconfound_only_pred_summary,
  confound_only_pred_summary
) %>% 
  mutate(sim_type = factor(sim_type, levels = c(
    "Regular PPC",
    "Unconfounded only PPC",
    "Confounded only PPC"
  )))

dodge <- position_dodge(width = 0.5)
breaks_vals <- sort(unique(emp_data$ev_diff))

ggplot(pred_summary, aes(
  x = ev_diff, y = mean_resp,
  color = gamble_type,
  fill = gamble_type
)) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper),
    alpha = 0.4,
    color = NA,
    position = dodge
  ) +
  geom_line(
    data = emp_summary,
    aes(group = gamble_type, color = gamble_type),
    linetype = "dashed",
    linewidth = 1,
    position = dodge
  ) +
  facet_grid(model ~ sim_type) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_errorbar(
    data = emp_summary,
    aes(
      ymin = pmax(0, mean_resp - std_resp),
      ymax = pmin(1, mean_resp + std_resp),
      color = gamble_type
    ),
    width = 0.05,
    linewidth = 1,
    position = dodge
  ) +
  geom_point(
    data = emp_summary,
    aes(color = gamble_type),
    size = 2.5,
    position = dodge
  ) +
  scale_fill_manual(values = COLOR_PALETTE, guide = "none") +
  scale_color_manual(values = COLOR_PALETTE) +
  scale_x_continuous(
    breaks = breaks_vals,
    labels = rev(-breaks_vals)
  ) +
  scale_y_continuous(
    limits = c(-0.025, 1.025),
    breaks = seq(0, 1, 0.2)
  ) +
  labs(
    x = "Difference in EV",
    y = "P(choose lower losses option)",
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
  '../plots/prediction_study_ppc.pdf',
  device = 'pdf', dpi = 300,
  width = 16, height = 7
)
