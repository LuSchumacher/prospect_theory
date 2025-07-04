functions {
  real pt_utility(vector x, real alpha, real lambda) {
    real u = 0;
    int K = num_elements(x);
    for (k in 1:K) {
      real val = x[k];
      if (val >= 0)
        u += pow(val, alpha);
      else
        u += -lambda * pow(abs(val), alpha);  // fix for negative values
    }
    return u / K;
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
  real<lower=0> lambda_0;
  real<lower=0> lambda_1;
  real<lower=0, upper=1> alpha_0;
  real<lower=0, upper=1> alpha_1;
  real<lower=0> tau_0;
  real<lower=0> tau_1;
}

model {
  lambda_0 ~ lognormal(log(1.5), 0.5);
  lambda_1 ~ lognormal(log(1.5), 0.5);
  alpha_0  ~ beta(2, 1);
  alpha_1  ~ beta(2, 1);
  tau_0    ~ gamma(1.5, 0.5);
  tau_1    ~ gamma(1.5, 0.5);

  for (t in 1:T) {
    real lambda = condition[t] == 0 ? lambda_0 : lambda_1;
    real alpha  = condition[t] == 0 ? alpha_0  : alpha_1;
    real tau    = condition[t] == 0 ? tau_0    : tau_1;

    real u_a = pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real u_b = pt_utility(to_vector(outcome_b[t]), alpha, lambda);

    choice[t] ~ bernoulli_logit(tau * (u_b - u_a));
  }
}
