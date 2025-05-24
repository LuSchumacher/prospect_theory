library(tidyverse)
library(magrittr)
library(brms)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#27374D', '#B70404')

path <- "pilot_data_2/"
files <- list.files(path, pattern = ".csv")

df <- read_csv(paste0(path, files))

payment_df <- df %>% 
  select(PROLIFIC_PID, bonus_payment) %>% 
  drop_na(bonus_payment)

attention_check <- df %>% 
  select(
    PROLIFIC_PID,
    mouse_resp_3.clicked_name,
    text_response_box.text
  ) %>% 
  filter(
    !is.na(mouse_resp_3.clicked_name) | !is.na(text_response_box.text)
  )

sanity_check <- df %>% 
  filter(sanity_check == TRUE) %>% 
  select(
    PROLIFIC_PID,
    mouse_resp.clicked_name
  )

df %<>% 
  filter(
    PROLIFIC_PID != "66e0dc819ca421d2e6da6d67",
    PROLIFIC_PID != "61501f8fdda44b1783c3556b",
    PROLIFIC_PID != "6129ed2fa30b593821361cb3",
    PROLIFIC_PID != "66bf8e9849e9d38db3d8d40e",
    PROLIFIC_PID != "66d9fc2d2b46f520118ea65f",
    PROLIFIC_PID != "65577234800b1fea9ae3cf4d",
    PROLIFIC_PID != "67237e8fe4d32fbb53aabc67",
    PROLIFIC_PID != "66e7209b6ee5ee81f22afeb5"
  ) %>%
  mutate(id = dense_rank(PROLIFIC_PID)) %>% 
  select(
    id, age, sex, trials.thisN,
    mouse_resp.clicked_name,
    mouse_resp.time,
    outcome_a1:skew_diff,
    sanity_check
  ) %>% 
  rename(
    choice = mouse_resp.clicked_name,
    rt = mouse_resp.time,
    trial = trials.thisN
  ) %>% 
  arrange(id) %>% 
  drop_na(choice) %>% 
  mutate(
    rt = str_replace_all(rt, "[^0-9.]", ""),
    rt = as.numeric(rt),
    choice = str_replace(choice, "^[^ab]*([ab]).*", "\\1"),
    resp = ifelse(choice == "a", 0, 1),
    condition = ifelse(var_diff < 0, "new", "old")
  ) %>% 
  relocate(resp, .after = choice) %>% 
  relocate(condition, .after = trial) %>% 
  filter(sanity_check == FALSE) %>% 
  mutate(
    condition = as.factor(condition)
  )
  # group_by(id) %>% 
  # mutate(trial = 1:172) %>% 
  # ungroup()

# write_csv(df, "pilot_data_prepped.csv")

summary <- df %>% 
  # filter(id != 5) %>% 
  group_by(id, condition, ev_diff) %>% 
  summarise(mean_resp = mean(resp)) %>% 
  ungroup() %>% 
  group_by(condition, ev_diff) %>% 
  summarise(
    mean = mean(mean_resp),
    std = sd(mean_resp)
  ) %>% 
  mutate(ev_diff = as.factor(ev_diff))

summary %>% 
  ggplot(aes(x = ev_diff, y = mean, colour = condition)) + 
  geom_pointrange(
    aes(ymin = mean - std, ymax = mean + std),
    position = position_dodge(width = 0.25),
    size = 0.75, linewidth = 1
  ) +
  geom_line(
    aes(group = condition),
    position = position_dodge(width = 0.25),
    linetype = "dashed",
    linewidth = 0.8
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = COLOR_PALETTE) +
  labs(
    x = "Difference in EV",
    y = "Proportion of choosing B",
    color = "Lottery type"
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


ggsave("../plots/pilot_data_plot.pdf", width = 8, height = 5)


summary <- df %>% 
  group_by(id, condition, ev_diff) %>% 
  summarise(mean_resp = mean(resp))


summary %>% 
  ggplot(aes(x = ev_diff, y = mean_resp, colour = condition)) + 
  geom_point(
    position = position_dodge(width = 0.25),
    size = 1
  ) +
  geom_line(
    aes(group = condition),
    position = position_dodge(width = 0.25),
    linewidth = 0.8
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  scale_color_manual(values = COLOR_PALETTE) +
  labs(
    x = "Difference in EV",
    y = "Proportion of choosing B",
    color = "Lottery type"
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
  ) + 
  facet_wrap(~id)


model_formula <- resp ~ ev_diff * condition + (ev_diff * condition | id)

model_priors <- prior(normal(0, 1.0), class = b)
model_fit <- brm(
  formula = model_formula,
  data = df,
  family = bernoulli("logit"),
  prior = model_priors,
  iter = 4000,
  cores = 4,
  chains = 4,
  sample_prior = "yes"
)

bf <- 1 / hypothesis(model_fit, "conditionold = 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(model_fit, "conditionold > 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(model_fit, "ev_diff = 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(model_fit, "ev_diff < 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(model_fit, "ev_diff:conditionold = 0")$hypothesis["Evid.Ratio"]


model_fit
conditional_effects(model_fit)

# 
# plot(hypothesis(model_fit, "ev_diff < 0"))
plot(hypothesis(model_fit, "conditionold > 0"))
# plot(hypothesis(model_fit, "ev_diff:conditionold < 0"))

pp_check(model_fit, ndraws=100)
