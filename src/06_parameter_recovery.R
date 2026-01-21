library(tidyverse)
library(magrittr)
library(LaplacesDemon)
library(cmdstanr)
library(patchwork)
library(posterior)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

get_prelec_weights <- function(gamma, probs = c(1/3, 2/3)) {
  weights <- exp(-(-log(probs))^gamma)
  return(c(weights, 1))
}

get_pt_utility <- function(outcomes, alpha, lambda) {
  utility <- 0
  K <- length(outcomes)
  for (k in seq_len(K)) {
    if (outcomes[k] >= 0) {
      utility <- utility + outcomes[k]^alpha
    } else {
      utility <- utility - lambda * abs(outcomes[k])^alpha
    }
  }
  return(utility / K)
}

sample_pt <- function(outcomes, alpha, lambda, tau) {
  outcomes_a <- outcomes[1:3]
  outcomes_b <- outcomes[4:6]
  utility_a <- get_pt_utility(outcomes_a, alpha, lambda)
  utility_b <- get_pt_utility(outcomes_b, alpha, lambda)
  logit_p <- tau * (utility_b - utility_a)
  return(rbinom(1, 1, logit_p))
}









