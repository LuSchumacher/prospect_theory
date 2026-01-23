library(tidyverse)
library(brms)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df <- read_csv('../data/study_data_prepared.csv') %>% 
  mutate(
    gamble_type = as_factor(gamble_type),
    ev_diff = ev_b - ev_a
  )

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

bf <- 1 / hypothesis(glm_fit, "ev_diff = 0")$hypothesis["Evid.Ratio"]
bf
bf <- 1 / hypothesis(glm_fit, "ev_diff:gamble_type1 = 0")$hypothesis["Evid.Ratio"]
bf

bf <- 1 / hypothesis(glm_fit, "gamble_typeunconfounded = 0")$hypothesis["Evid.Ratio"]
bf <- hypothesis(glm_fit, "gamble_typeunconfounded < 0")$hypothesis["Evid.Ratio"]

bf <- hypothesis(glm_fit, "ev_diff < 0")$hypothesis["Evid.Ratio"]


plot(hypothesis(glm_fit, "ev_diff:gamble_type1 = 0"))
plot(hypothesis(glm_fit, "ev_diff < 0"))
plot(hypothesis(glm_fit, "ev_diff:gamble_typeunconfounded = 0"))

pp_check(glm_fit, ndraws=50)
