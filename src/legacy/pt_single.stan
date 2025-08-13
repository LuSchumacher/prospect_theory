functions {
  real get_pt_utility(vector x, real alpha, real lambda) {
    real u = 0;
    int K = num_elements(x);
    for (k in 1:K) {
      real val = x[k];
      if (val >= 0)
        u += pow(val, alpha);
      else
        u += -lambda * pow(abs(val), alpha);
    }
    return u / K;
  }
  real inverse_utility(real u, real alpha, real lambda) {
    if (u >= 0) {
      return pow(u, 1 / alpha);
    } else {
      return -pow(abs(u) / lambda, 1 / alpha);
    }
  }
}

data {
  int<lower=1> T;
  array[T] int<lower=0, upper=1> condition;
  array[T] int<lower=0, upper=1> choice;
  matrix[T, 3] outcome_a;
  matrix[T, 3] outcome_b;
}

parameters {
  real lambda_0_raw;
  real lambda_1_raw;
  real alpha_0_raw;
  real alpha_1_raw;
  real tau_0_raw;
  real tau_1_raw;
}

transformed parameters {
  real<lower=0>          lambda_0 = exp(lambda_0_raw);
  real<lower=0>          lambda_1 = exp(lambda_1_raw);
  real<lower=0, upper=1> alpha_0  = inv_logit(alpha_0_raw);
  real<lower=0, upper=1> alpha_1  = inv_logit(alpha_1_raw);
  real<lower=0>          tau_0    = exp(tau_0_raw);
  real<lower=0>          tau_1    = exp(tau_1_raw);
}

model {
  lambda_0_raw ~ normal(log(2), 0.5);
  lambda_1_raw ~ normal(log(2), 0.5);
  alpha_0_raw  ~ normal(1.0, 1.75); 
  alpha_1_raw  ~ normal(1.0, 1.75);
  tau_0_raw    ~ normal(0.5, 0.5);
  tau_1_raw    ~ normal(0.5, 0.5);

  for (t in 1:T) {
    real lambda = condition[t] == 0 ? lambda_0 : lambda_1;
    real alpha  = condition[t] == 0 ? alpha_0  : alpha_1;
    real tau    = condition[t] == 0 ? tau_0    : tau_1;

    real u_a_raw = get_pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real u_b_raw = get_pt_utility(to_vector(outcome_b[t]), alpha, lambda);
    // real u_a = inverse_utility(u_a_raw, alpha, lambda);
    // real u_b = inverse_utility(u_b_raw, alpha, lambda);

    choice[t] ~ bernoulli_logit(tau * (u_b_raw - u_a_raw));
  }
}

generated quantities {
  real lambda_0_out = exp(lambda_0_raw);
  real lambda_1_out = exp(lambda_1_raw);
  real alpha_0_out = inv_logit(alpha_0_raw);
  real alpha_1_out = inv_logit(alpha_1_raw);
  real tau_0_out = exp(tau_0_raw);
  real tau_1_out = exp(tau_1_raw);
}

