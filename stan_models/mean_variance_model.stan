functions {
  real get_mean_variance_utility(
    vector outcome,
    real beta_variance
  ) {
    real ev = 0.5 * (outcome[1] + outcome[2]);
    real variance = 0.25 * square(outcome[1] - outcome[2]);

    return ev - beta_variance * variance;
  }

  real partial_log_lik(
    array[] int choice_slice,
    int start,
    int end,
    matrix outcomes_b,
    array[] int subject_id,
    vector beta_variance,
    vector tau
  ) {
    real acc = 0;

    for (i in start:end) {
      int s = subject_id[i];
      real utility_b = get_mean_variance_utility(
        outcomes_b[i]', beta_variance[s]
      );
      real logit_p = tau[s] * utility_b;

      acc += bernoulli_logit_lpmf(
        choice_slice[i - start + 1] | logit_p
      );
    }

    return acc;
  }
}

data {
  int<lower=1> T;
  int<lower=1> N;
  array[T] int<lower=1, upper=N> subject_id;
  matrix[T, 2] outcomes_b;
  array[T] int<lower=0, upper=1> choice;
}

parameters {
  real mu_beta_variance;
  real mu_tau;
  vector[N] z_beta_variance;
  vector[N] z_tau;
  real<lower=0> sigma_beta_variance;
  real<lower=0> sigma_tau;
}

transformed parameters {
  vector[N] beta_variance;
  vector[N] tau;

  for (s in 1:N) {
    beta_variance[s] = log1p_exp(
      mu_beta_variance + sigma_beta_variance * z_beta_variance[s]
    );
    tau[s] = log1p_exp(mu_tau + sigma_tau * z_tau[s]);
  }
}

model {
  mu_beta_variance ~ normal(-4, 1.5);
  mu_tau ~ normal(0.5, 2);
  sigma_beta_variance ~ normal(0, 1);
  sigma_tau ~ normal(0, 1);
  z_beta_variance ~ std_normal();
  z_tau ~ std_normal();

  target += reduce_sum(
    partial_log_lik,
    choice,
    1,
    outcomes_b,
    subject_id,
    beta_variance,
    tau
  );
}

generated quantities {
  real beta_variance_group_location = log1p_exp(mu_beta_variance);
  real tau_group_location = log1p_exp(mu_tau);

  array[T] real log_lik;
  array[T] int y_rep;

  for (i in 1:T) {
    int s = subject_id[i];
    real utility_b = get_mean_variance_utility(
      outcomes_b[i]', beta_variance[s]
    );
    real logit_p = tau[s] * utility_b;

    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
