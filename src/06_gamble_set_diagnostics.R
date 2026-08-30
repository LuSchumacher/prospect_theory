library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

OUTPUT_DIR <- "../data/stimulus_validity"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

gamble_set <- read_csv("../data/gamble_list.csv")

row_skewness <- function(x) {
  centered <- x - rowMeans(x)
  rowMeans(centered^3) / rowMeans(centered^2)^(3 / 2)
}

outcomes_a <- as.matrix(gamble_set[, c("outcome_a1", "outcome_a2", "outcome_a3")])
outcomes_b <- as.matrix(gamble_set[, c("outcome_b1", "outcome_b2", "outcome_b3")])

ev_a <- rowMeans(outcomes_a)
ev_b <- rowMeans(outcomes_b)
variance_a <- rowMeans((outcomes_a - ev_a)^2)
variance_b <- rowMeans((outcomes_b - ev_b)^2)
sd_a <- sqrt(variance_a)
sd_b <- sqrt(variance_b)
minimum_a <- apply(outcomes_a, 1, min)
minimum_b <- apply(outcomes_b, 1, min)
maximum_a <- apply(outcomes_a, 1, max)
maximum_b <- apply(outcomes_b, 1, max)
range_a <- maximum_a - minimum_a
range_b <- maximum_b - minimum_b
skewness_a <- row_skewness(outcomes_a)
skewness_b <- row_skewness(outcomes_b)

stimulus_attributes <- gamble_set |>
  transmute(
    stimulus_id = row_number(),
    condition = recode(
      lottery_type,
      "confounded" = "Aligned",
      "unconfounded" = "Opposed",
      .default = NA_character_
    ),
    sanity_check,
    across(starts_with("outcome_")),
    loss_magnitude_difference = abs(minimum_a) - abs(minimum_b),
    variance_difference = variance_b - variance_a,
    standard_deviation_difference = sd_b - sd_a,
    expected_value_difference = round(ev_b - ev_a, 2),
    skewness_difference = skewness_b - skewness_a,
    range_difference = range_b - range_a,
    minimum_larger_loss_option = minimum_a,
    minimum_smaller_loss_option = minimum_b,
    maximum_larger_loss_option = maximum_a,
    maximum_smaller_loss_option = maximum_b,
    expected_value_larger_loss_option = ev_a,
    expected_value_smaller_loss_option = ev_b,
    variance_larger_loss_option = variance_a,
    variance_smaller_loss_option = variance_b,
    standard_deviation_larger_loss_option = sd_a,
    standard_deviation_smaller_loss_option = sd_b,
    skewness_larger_loss_option = skewness_a,
    skewness_smaller_loss_option = skewness_b,
    range_larger_loss_option = range_a,
    range_smaller_loss_option = range_b
  )

if (anyNA(stimulus_attributes$condition)) stop("Unexpected value in lottery_type.")

experimental_stimuli <- stimulus_attributes |>
  filter(!sanity_check) |>
  mutate(condition = factor(condition, levels = c("Aligned", "Opposed")))

# Option A is the larger-loss option and option B is the smaller-loss option.
one_loss_two_gains <-
  rowSums(outcomes_a[!gamble_set$sanity_check, ] < 0) == 1 &
  rowSums(outcomes_a[!gamble_set$sanity_check, ] > 0) == 2 &
  rowSums(outcomes_b[!gamble_set$sanity_check, ] < 0) == 1 &
  rowSums(outcomes_b[!gamble_set$sanity_check, ] > 0) == 2

b_is_smaller_loss <-
  abs(experimental_stimuli$minimum_smaller_loss_option) <
  abs(experimental_stimuli$minimum_larger_loss_option)

aligned_correct <- experimental_stimuli |>
  filter(condition == "Aligned") |>
  pull(variance_difference) < 0

opposed_correct <- experimental_stimuli |>
  filter(condition == "Opposed") |>
  pull(variance_difference) > 0

stopifnot(
  all(one_loss_two_gains),
  all(b_is_smaller_loss),
  all(aligned_correct),
  all(opposed_correct)
)

# ---------------------------------------------------------------------------- #
# CONSTRUCTION CHECKS
# ---------------------------------------------------------------------------- #
validity_checks <- tibble(
  check = c(
    "Experimental stimuli",
    "Sanity-check stimuli",
    "Aligned experimental stimuli",
    "Opposed experimental stimuli",
    "Every option has one loss and two gains",
    "Option B is always the smaller-loss option",
    "Aligned variance ordering is correct",
    "Opposed variance ordering is correct"
  ),
  value = c(
    nrow(experimental_stimuli),
    sum(stimulus_attributes$sanity_check),
    sum(experimental_stimuli$condition == "Aligned"),
    sum(experimental_stimuli$condition == "Opposed"),
    all(one_loss_two_gains),
    all(b_is_smaller_loss),
    all(aligned_correct),
    all(opposed_correct)
  )
)

expected_value_balance <- experimental_stimuli |>
  count(condition, expected_value_difference) |>
  pivot_wider(names_from = condition, values_from = n, values_fill = 0) |>
  mutate(difference = Opposed - Aligned)

# ---------------------------------------------------------------------------- #
# APPENDIX SUMMARY TABLE
# ---------------------------------------------------------------------------- #
ATTRIBUTE_LABELS <- c(
  "loss_magnitude_difference" = "Loss-magnitude difference (larger minus smaller)",
  "variance_difference" = "Variance difference (smaller-loss minus larger-loss option)",
  "standard_deviation_difference" = "Standard-deviation difference (smaller-loss minus larger-loss option)",
  "expected_value_difference" = "Expected-value difference (smaller-loss minus larger-loss option)",
  "skewness_difference" = "Skewness difference (smaller-loss minus larger-loss option)",
  "range_difference" = "Range difference (smaller-loss minus larger-loss option)",
  "minimum_larger_loss_option" = "Minimum outcome of larger-loss option",
  "minimum_smaller_loss_option" = "Minimum outcome of smaller-loss option",
  "maximum_larger_loss_option" = "Maximum outcome of larger-loss option",
  "maximum_smaller_loss_option" = "Maximum outcome of smaller-loss option"
)

summary_attributes <- names(ATTRIBUTE_LABELS)

stimulus_summary <- experimental_stimuli |>
  select(condition, all_of(summary_attributes)) |>
  pivot_longer(-condition, names_to = "attribute", values_to = "value") |>
  group_by(condition, attribute) |>
  summarise(
    n = n(),
    mean = mean(value),
    sd = sd(value),
    minimum = min(value),
    maximum = max(value),
    .groups = "drop"
  ) |>
  mutate(
    attribute_label = unname(ATTRIBUTE_LABELS[attribute]),
    attribute = factor(attribute, levels = summary_attributes)
  ) |>
  arrange(attribute, condition) |>
  select(condition, attribute, attribute_label, n, mean, sd, minimum, maximum)

stimulus_summary_rounded <- stimulus_summary |>
  mutate(across(c(mean, sd, minimum, maximum), ~ round(.x, 2)))

stimulus_summary_appendix <- stimulus_summary |>
  transmute(
    Condition = condition,
    Attribute = attribute_label,
    N = n,
    `M (SD)` = sprintf("%.2f (%.2f)", mean, sd),
    `Min--max` = sprintf("%.2f--%.2f", minimum, maximum)
  )

# ---------------------------------------------------------------------------- #
# CORRELATIONS
# ---------------------------------------------------------------------------- #
make_correlation_matrix <- function(data) {
  data |>
    select(all_of(summary_attributes)) |>
    cor(use = "pairwise.complete.obs")
}

correlations <- list(
  Aligned = experimental_stimuli |>
    filter(condition == "Aligned") |>
    make_correlation_matrix(),
  Opposed = experimental_stimuli |>
    filter(condition == "Opposed") |>
    make_correlation_matrix(),
  Overall = make_correlation_matrix(experimental_stimuli)
)

correlation_matrix_to_tibble <- function(x) {
  x |>
    round(2) |>
    as.data.frame() |>
    rownames_to_column("attribute") |>
    as_tibble()
}

correlations_long <- imap_dfr(
  correlations,
  function(x, condition_name) {
    as.data.frame(as.table(x)) |>
      as_tibble() |>
      transmute(
        condition = condition_name,
        attribute_1 = as.character(Var1),
        attribute_2 = as.character(Var2),
        correlation = Freq,
        index_1 = match(attribute_1, summary_attributes),
        index_2 = match(attribute_2, summary_attributes)
      ) |>
      filter(index_1 < index_2) |>
      select(-index_1, -index_2)
  }
) |>
  mutate(correlation = round(correlation, 2))

# ---------------------------------------------------------------------------- #
# CORRELATION HEATMAP
# ---------------------------------------------------------------------------- #
SHORT_LABELS <- c(
  "loss_magnitude_difference" = "Loss magnitude",
  "variance_difference" = "Variance",
  "standard_deviation_difference" = "SD",
  "expected_value_difference" = "EV",
  "skewness_difference" = "Skewness",
  "range_difference" = "Range",
  "minimum_larger_loss_option" = "Minimum, larger loss",
  "minimum_smaller_loss_option" = "Minimum, smaller loss",
  "maximum_larger_loss_option" = "Maximum, larger loss",
  "maximum_smaller_loss_option" = "Maximum, smaller loss"
)

correlation_plot_data <- correlations_long |>
  filter(condition != "Overall") |>
  mutate(
    attribute_1 = factor(
      attribute_1,
      levels = summary_attributes,
      labels = SHORT_LABELS[summary_attributes]
    ),
    attribute_2 = factor(
      attribute_2,
      levels = rev(summary_attributes),
      labels = rev(SHORT_LABELS[summary_attributes])
    )
  )

correlation_plot <- ggplot(
  correlation_plot_data,
  aes(x = attribute_1, y = attribute_2, fill = correlation)
) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 3) +
  facet_wrap(~ condition) +
  scale_fill_gradient2(
    low = "#42686C",
    mid = "white",
    high = "#c96016",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Pearson r"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    strip.text = element_text(size = 13),
    legend.position = "bottom"
  )

# ---------------------------------------------------------------------------- #
# SAVE AND PRINT
# ---------------------------------------------------------------------------- #
write_csv(stimulus_attributes, file.path(OUTPUT_DIR, "stimulus_attributes_all_trials.csv"))
write_csv(experimental_stimuli, file.path(OUTPUT_DIR, "stimulus_attributes_experimental_trials.csv"))
write_csv(validity_checks, file.path(OUTPUT_DIR, "stimulus_validity_checks.csv"))
write_csv(expected_value_balance, file.path(OUTPUT_DIR, "expected_value_balance.csv"))
write_csv(stimulus_summary_rounded, file.path(OUTPUT_DIR, "stimulus_summary_by_condition.csv"))
write_csv(stimulus_summary_appendix, file.path(OUTPUT_DIR, "stimulus_summary_appendix.csv"))
write_csv(correlation_matrix_to_tibble(correlations$Aligned), file.path(OUTPUT_DIR, "stimulus_correlations_aligned.csv"))
write_csv(correlation_matrix_to_tibble(correlations$Opposed), file.path(OUTPUT_DIR, "stimulus_correlations_opposed.csv"))
write_csv(correlation_matrix_to_tibble(correlations$Overall), file.path(OUTPUT_DIR, "stimulus_correlations_overall.csv"))
write_csv(correlations_long, file.path(OUTPUT_DIR, "stimulus_correlations_long.csv"))

ggsave(
  "../plots/stimulus_correlation_heatmap.pdf",
  correlation_plot,
  device = "pdf",
  width = 14,
  height = 8
)

print(validity_checks, n = Inf)
print(expected_value_balance, n = Inf)
print(stimulus_summary_rounded, n = Inf)
print(correlation_plot_data, n = Inf)
