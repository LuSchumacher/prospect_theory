functions {
  real pt_utility(vector x, real alpha, real lambda) {
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
}

data {
  int<lower=1>                   T;          // Number of trials
  int<lower=1>                   N;          // Number of subjects
  array[T] int<lower=1>          subject_id; // Subject index per trial
  array[T] int<lower=0, upper=1> condition;  // 0 = old, 1 = new
  matrix[T, 3]                   outcome_a;  // A option outcomes
  matrix[T, 3]                   outcome_b;  // B option outcomes
  array[T] int<lower=0, upper=1> choice;     // 0 = chose A, 1 = chose B
}

parameters {
  // Group-level means (on unconstrained scale if needed)
  real<lower=0> mu_lambda_old;
  real<lower=0> mu_lambda_new;
  real<lower=0, upper=1> mu_alpha_old;
  real<lower=0, upper=1> mu_alpha_new;
  real<lower=0> mu_tau_old;
  real<lower=0> mu_tau_new;

  // Group-level standard deviations
  real<lower=0> sigma_lambda_old;
  real<lower=0> sigma_lambda_new;
  real<lower=0> sigma_alpha_old;
  real<lower=0> sigma_alpha_new;
  real<lower=0> sigma_tau_old;
  real<lower=0> sigma_tau_new;

  // Subject-level parameters
  vector<lower=0>[N] lambda_old;
  vector<lower=0>[N] lambda_new;
  vector<lower=0, upper=1>[N] alpha_old;
  vector<lower=0, upper=1>[N] alpha_new;
  vector<lower=0>[N] tau_old;
  vector<lower=0>[N] tau_new;
}

model {
  // Priors on group means
  mu_lambda_0 ~ lognormal(log(1.5), 0.5);
  mu_lambda_1 ~ lognormal(log(1.5), 0.5);
  mu_alpha_0  ~ beta(2, 1);
  mu_alpha_1  ~ beta(2, 1);
  mu_tau_0    ~ gamma(1.5, 0.5);
  mu_tau_1    ~ gamma(1.5, 0.5);

  // Priors on group-level std dev
  sigma_lambda_0 ~ normal(0, 1);
  sigma_lambda_1 ~ normal(0, 1);
  sigma_alpha_0  ~ normal(0, 1);
  sigma_alpha_1  ~ normal(0, 1);
  sigma_tau_0    ~ normal(0, 1);
  sigma_tau_1    ~ normal(0, 1);

  // Subject-level parameters centered on group means
  lambda_0 ~ lognormal(log(mu_lambda_0) - 0.5 * square(sigma_lambda_0), sigma_lambda_0);
  lambda_1 ~ lognormal(log(mu_lambda_1) - 0.5 * square(sigma_lambda_1), sigma_lambda_1);

  // For alpha, use normal on inverse-logit scale or beta approx:
  // Here we do a Beta approximation using Normal with truncation:
  alpha_0 ~ beta(2, 1);  // Alternatively: you can model with transformed parameters, but for simplicity:
  alpha_1 ~ beta(2, 1);

  tau_0 ~ gamma(1.5, 0.5);  // could consider hierarchical gamma or lognormal if desired
  tau_1 ~ gamma(1.5, 0.5);

  // Likelihood
  for (t in 1:T) {
    int s = subject_id[t];
    real lambda = condition[t] == 0 ? lambda_0[s] : lambda_1[s];
    real alpha  = condition[t] == 0 ? alpha_0[s]  : alpha_1[s];
    real tau    = condition[t] == 0 ? tau_0[s]    : tau_1[s];

    real u_a = pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real u_b = pt_utility(to_vector(outcome_b[t]), alpha, lambda);

    choice[t] ~ bernoulli_logit(tau * (u_b - u_a));
  }
}
