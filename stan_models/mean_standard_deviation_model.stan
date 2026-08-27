functions {
  real get_mean_sd_utility(vector outcome, real kappa) {
    real ev = 0.5 * (outcome[1] + outcome[2]);
    real sd = 0.5 * abs(outcome[1] - outcome[2]);

    return ev - kappa * sd;
  }

  real partial_log_lik(
    array[] int choice_slice,
    int start,
    int end,
    matrix outcomes_b,
    array[] int subject_id,
    vector kappa,
    vector tau
  ) {
    real acc = 0;

    for (i in start:end) {
      int s = subject_id[i];
      real utility_b = get_mean_sd_utility(outcomes_b[i]', kappa[s]);
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
  real mu_kappa;
  real mu_tau;
  vector[N] z_kappa;
  vector[N] z_tau;
  real<lower=0> sigma_kappa;
  real<lower=0> sigma_tau;
}

transformed parameters {
  vector<lower=-1, upper=1>[N] kappa;
  vector<lower=0>[N] tau;

  for (s in 1:N) {
    // Map to (-1, 1), which corresponds exactly to lambda in (0, infinity).
    // Positive values indicate dispersion aversion; negative values indicate
    // dispersion seeking.
    kappa[s] = 2 * inv_logit(
      mu_kappa + sigma_kappa * z_kappa[s]
    ) - 1;
    tau[s] = log1p_exp(mu_tau + sigma_tau * z_tau[s]);
  }
}

model {
  // 2 * logit^{-1}(0.7) - 1 is approximately 1/3 (lambda approximately 2).
  mu_kappa ~ normal(0.7, 1.5);
  mu_tau ~ normal(0.5, 2);
  sigma_kappa ~ normal(0, 1);
  sigma_tau ~ normal(0, 1);
  z_kappa ~ std_normal();
  z_tau ~ std_normal();

  target += reduce_sum(
    partial_log_lik,
    choice,
    1,
    outcomes_b,
    subject_id,
    kappa,
    tau
  );
}

generated quantities {
  real kappa_group_location = 2 * inv_logit(mu_kappa) - 1;
  real tau_group_location = log1p_exp(mu_tau);
  real lambda_implied_group_location =
    (1 + kappa_group_location) / (1 - kappa_group_location);
  real tau_lambda_implied_group_location =
    tau_group_location * (1 - kappa_group_location);

  array[T] real log_lik;
  array[T] int y_rep;

  for (i in 1:T) {
    int s = subject_id[i];
    real utility_b = get_mean_sd_utility(outcomes_b[i]', kappa[s]);
    real logit_p = tau[s] * utility_b;

    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
