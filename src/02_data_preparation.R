library(tidyverse)
library(magrittr)
library(brms)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#27374D', '#B70404')

IDS_TO_EXCLUDE = c(
  "60a033197af9369e76946104", "615e233e180ad299aaea710c", 
  "6666fb1567243aa56ed8cffa", "5fe7bc4ec39215684de426f4",
  "676aa6bfdda58087a94af1ea", "5e70bd5480f43a0009625d4c",
  "6724b9788642a492d3a55188"
)
SANITY_CHECK_THRESHOLD <- 3

################################################################################
# DATA PREPARATION
################################################################################

path <- "../data/study_data/"
files <- list.files(path, pattern = ".csv")

df <- tibble()
df_bonus_payment <- tibble()
sanity_fail_counter <- 0
for (i in seq_along(files)) {
  if (files[i] %in% IDS_TO_EXCLUDE) {
    next
  }
  tmp <- read_csv(paste0(path, files[i]))
  
  # record bonus payment
  bonus <- na.omit(tmp$bonus_payment)
  bonus <- bonus[1]
  tmp_bonus <-tibble(
    id = unique(tmp$PROLIFIC_PID),
    bonus_payment = bonus
  )
  df_bonus_payment <- bind_rows(df_bonus_payment, tmp_bonus)
  
  # check if sanity check is passed
  sanity_check <- tmp %>%
    filter(sanity_check == TRUE) %>%
    select(
      mouse_resp.clicked_name
    ) %>% 
    mutate(correct = ifelse(
      mouse_resp.clicked_name == '["coins_b_3","coins_b_3"]', 1, 0
    ))
  if (sum(sanity_check$correct) < SANITY_CHECK_THRESHOLD) {
    sanity_fail_counter <- sanity_fail_counter + 1
    next
  }
  
  # data merge
  tmp %<>%
    select(
      PROLIFIC_PID, age, sex, trials.thisN, lottery_type,
      mouse_resp.clicked_name, mouse_resp.time,
      outcome_a1:skew_diff, sanity_check
    )
  df <- bind_rows(df, tmp)
}

df %<>%
  rename(
    choice = mouse_resp.clicked_name,
    rt = mouse_resp.time,
    trial = trials.thisN,
    gamble_type = lottery_type
  ) %>% 
  drop_na(choice) %>% 
  mutate(
    rt = str_replace_all(rt, "[^0-9.]", ""),
    rt = as.numeric(rt),
    choice = str_replace(choice, "^[^ab]*([ab]).*", "\\1"),
    resp = ifelse(choice == "a", 0, 1)
  ) %>% 
  relocate(resp, .after = choice) %>% 
  relocate(gamble_type, .after = trial) %>% 
  filter(sanity_check == FALSE) %>% 
  mutate(
    gamble_type = as.factor(gamble_type),
    id = dense_rank(PROLIFIC_PID)
  ) %>% 
  select(-c(sanity_check, PROLIFIC_PID)) %>% 
  relocate(id) %>% 
  arrange(id)

write_csv(df, "../data/study_data_prepared.csv")

################################################################################
# RT SUMMARY STATISTICS
################################################################################

rt_summary <- df %>% 
  group_by(id, gamble_type) %>% 
  summarise(
    mean_rt = mean(rt),
    median_rt = median(rt),
    min_rt = min(rt),
    max_rt = max(rt)
  )

################################################################################
# PLOT: CHOICE PROPORTION AND EV DIFFERENCE
################################################################################

summary <- df %>% 
  group_by(id, gamble_type, ev_diff) %>% 
  summarise(mean_resp = mean(resp)) %>% 
  ungroup() %>% 
  group_by(gamble_type, ev_diff) %>% 
  summarise(
    mean = mean(mean_resp),
    std = sd(mean_resp)
  ) %>% 
  mutate(ev_diff = as.factor(ev_diff))

summary %>% 
  ggplot(aes(x = ev_diff, y = mean, colour = gamble_type)) + 
  geom_pointrange(
    aes(ymin = mean - std, ymax = mean + std),
    position = position_dodge(width = 0.25),
    size = 0.75, linewidth = 1
  ) +
  geom_line(
    aes(group = gamble_type),
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
                              size = 0.2,
                              linetype = 1),
    legend.spacing.y = unit(0.25, 'cm'),
    axis.title.y = element_text(margin = margin(t = 0, r = 15, b = 0, l = 0)),
    axis.title.x = element_text(margin = margin(t = 15, r = 0, b = 5, l = 0))
  )

ggsave("../plots/empirical_data.pdf", width = 12, height = 10)

################################################################################
# PLOT: CHOICE PROPORTION AND EV DIFFERENCE PER INDIVIDUAL
################################################################################

summary <- df %>% 
  group_by(id, gamble_type, ev_diff) %>% 
  summarise(mean_resp = mean(resp))

summary %>% 
  ggplot(aes(x = ev_diff, y = mean_resp, colour = gamble_type)) + 
  geom_point(
    position = position_dodge(width = 0.25),
    size = 1
  ) +
  geom_line(
    aes(group = gamble_type),
    position = position_dodge(width = 0.25),
    linewidth = 0.8
  ) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.2)) +
  scale_x_continuous(breaks = unique(df$ev_diff)) +
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

ggsave("../plots/empirical_data_subjects.pdf", width = 12, height = 10)
