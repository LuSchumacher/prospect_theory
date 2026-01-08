library(tidyverse)
library(magrittr)
library(cmdstanr)
library(bayesplot)
library(posterior)

FONT_SIZE_1 <- 22
FONT_SIZE_2 <- 20
FONT_SIZE_3 <- 18
COLOR_PALETTE <- c('#27374D', '#B70404')

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv('../data/study_data_prepared.csv') %>% 
  mutate(gamble_type = as_factor(gamble_type))

contrasts(df$gamble_type) <- rbind(-0.5, 0.5)

model_formula <- resp ~ ev_diff * gamble_type + (ev_diff * gamble_type | id)
model_priors <- prior(normal(0, 1.0), class = b)

glm_fit <- brm(
  formula = model_formula,
  data = df,
  family = bernoulli(),
  prior = model_priors,
  iter = 8000,
  cores = 4,
  chains = 4,
  control = list(adapt_delta = 0.85),
  sample_prior = "yes",
  file = "../fits/glm_fit" 
)

conditional_effects(glm_fit)

bf <- 1 / hypothesis(glm_fit, "gamble_typeunconfounded = 0")$hypothesis["Evid.Ratio"]
bf <- hypothesis(glm_fit, "gamble_typeunconfounded < 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(glm_fit, "ev_diff = 0")$hypothesis["Evid.Ratio"]
bf <- hypothesis(glm_fit, "ev_diff < 0")$hypothesis["Evid.Ratio"]
bf <- 1 / hypothesis(glm_fit, "ev_diff:gamble_typeunconfounded = 0")$hypothesis["Evid.Ratio"]

plot(hypothesis(glm_fit, "gamble_typeunconfounded < 0"))
plot(hypothesis(glm_fit, "ev_diff < 0"))
plot(hypothesis(glm_fit, "ev_diff:gamble_typeunconfounded = 0"))

pp_check(glm_fit, ndraws=50)
