functions {
  real prelec_w(real p, real gamma) {
    return exp(-pow(-log(p), gamma));
  }

  real get_cpt_utility(vector x, real alpha, real lambda, real gamma) {
    vector[3] v;
    array[3] int idx;

    for (k in 1:3) {
      if (x[k] >= 0) {
        v[k] = pow(x[k], alpha);
      } else {
        v[k] = -lambda * pow(abs(x[k]), alpha);
      }
    }

    idx = sort_indices_asc(v);
    v = v[idx];

    {
      real w1 = prelec_w(1.0 / 3.0, gamma);
      real w2 = prelec_w(2.0 / 3.0, gamma);
      return w1 * v[1] + (w2 - w1) * v[2] + (1.0 - w2) * v[3];
    }
  }

  real inverse_utility(real u, real alpha, real lambda) {
    if (u >= 0) {
      return pow(u, 1.0 / alpha);
    }
    return -pow(abs(u) / lambda, 1.0 / alpha);
  }

  real get_outcome_variance(vector x) {
    int K = num_elements(x);
    real mu = mean(x);
    real variance_sum = 0;

    for (k in 1:K) {
      variance_sum += square(x[k] - mu);
    }

    // Population variance under the equiprobable outcome distribution.
    return variance_sum / K;
  }

  real get_cpt_variance_utility(
    vector x,
    real alpha,
    real lambda,
    real gamma,
    real beta_variance
  ) {
    real cpt_raw = get_cpt_utility(x, alpha, lambda, gamma);
    real cpt_ce = inverse_utility(cpt_raw, alpha, lambda);

    // Variance is calculated from the original, unscaled outcomes.
    return cpt_ce - beta_variance * get_outcome_variance(x);
  }

  real partial_log_lik(
    array[] int choice_slice,
    int start,
    int end,
    matrix outcomes_a,
    matrix outcomes_b,
    array[] int subject_id,
    array[] int gamble_type,
    matrix lambda,
    matrix alpha,
    matrix tau,
    matrix gamma,
    matrix beta_variance
  ) {
    real acc = 0;

    for (i in start:end) {
      int s = subject_id[i];
      int c = gamble_type[i];
      real utility_a = get_cpt_variance_utility(
        to_vector(outcomes_a[i]), alpha[c, s], lambda[c, s],
        gamma[c, s], beta_variance[c, s]
      );
      real utility_b = get_cpt_variance_utility(
        to_vector(outcomes_b[i]), alpha[c, s], lambda[c, s],
        gamma[c, s], beta_variance[c, s]
      );
      real logit_p = tau[c, s] * (utility_b - utility_a);

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
  array[T] int<lower=1, upper=2> gamble_type;
  matrix[T, 3] outcomes_a;
  matrix[T, 3] outcomes_b;
  array[T] int<lower=0, upper=1> choice;
}

parameters {
  real intercept_lambda;
  real b_lambda;
  real intercept_alpha;
  real b_alpha;
  real intercept_tau;
  real b_tau;
  real intercept_gamma;
  real b_gamma;
  real intercept_beta_variance;
  real b_beta_variance;

  matrix[N, 2] z_lambda;
  matrix[N, 2] z_alpha;
  matrix[N, 2] z_tau;
  matrix[N, 2] z_gamma;
  matrix[N, 2] z_beta_variance;

  vector[2] sigma_lambda_raw;
  vector[2] sigma_alpha_raw;
  vector[2] sigma_tau_raw;
  vector[2] sigma_gamma_raw;
  vector[2] sigma_beta_variance_raw;

  corr_matrix[2] Omega_lambda;
  corr_matrix[2] Omega_alpha;
  corr_matrix[2] Omega_tau;
  corr_matrix[2] Omega_gamma;
  corr_matrix[2] Omega_beta_variance;
}

transformed parameters {
  vector[2] effect_coding;
  vector[2] sigma_lambda = log1p_exp(sigma_lambda_raw);
  vector[2] sigma_alpha = log1p_exp(sigma_alpha_raw);
  vector[2] sigma_tau = log1p_exp(sigma_tau_raw);
  vector[2] sigma_gamma = log1p_exp(sigma_gamma_raw);
  vector[2] sigma_beta_variance = log1p_exp(sigma_beta_variance_raw);

  matrix[N, 2] lambda_raw;
  matrix[N, 2] alpha_raw;
  matrix[N, 2] tau_raw;
  matrix[N, 2] gamma_raw;
  matrix[N, 2] beta_variance_raw;

  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;
  matrix[2, N] gamma;
  matrix[2, N] beta_variance;

  effect_coding[1] = -0.5;
  effect_coding[2] = 0.5;

  lambda_raw = (
    diag_pre_multiply(sigma_lambda, cholesky_decompose(Omega_lambda))
    * transpose(z_lambda)
  )';
  alpha_raw = (
    diag_pre_multiply(sigma_alpha, cholesky_decompose(Omega_alpha))
    * transpose(z_alpha)
  )';
  tau_raw = (
    diag_pre_multiply(sigma_tau, cholesky_decompose(Omega_tau))
    * transpose(z_tau)
  )';
  gamma_raw = (
    diag_pre_multiply(sigma_gamma, cholesky_decompose(Omega_gamma))
    * transpose(z_gamma)
  )';
  beta_variance_raw = (
    diag_pre_multiply(
      sigma_beta_variance,
      cholesky_decompose(Omega_beta_variance)
    ) * transpose(z_beta_variance)
  )';

  for (s in 1:N) {
    for (c in 1:2) {
      real condition = effect_coding[c];

      lambda[c, s] = log1p_exp(
        intercept_lambda + b_lambda * condition
        + lambda_raw[s, 1] + lambda_raw[s, 2] * condition
      );
      alpha[c, s] = log1p_exp(
        intercept_alpha + b_alpha * condition
        + alpha_raw[s, 1] + alpha_raw[s, 2] * condition
      );
      tau[c, s] = log1p_exp(
        intercept_tau + b_tau * condition
        + tau_raw[s, 1] + tau_raw[s, 2] * condition
      );
      gamma[c, s] = log1p_exp(
        intercept_gamma + b_gamma * condition
        + gamma_raw[s, 1] + gamma_raw[s, 2] * condition
      );
      beta_variance[c, s] = log1p_exp(
        intercept_beta_variance + b_beta_variance * condition
        + beta_variance_raw[s, 1]
        + beta_variance_raw[s, 2] * condition
      );
    }
  }
}

model {
  intercept_lambda ~ normal(2, 1);
  b_lambda ~ normal(0, 0.5);
  intercept_alpha ~ normal(0, 1);
  b_alpha ~ normal(0, 0.5);
  intercept_tau ~ normal(0.5, 2);
  b_tau ~ normal(0, 0.5);
  intercept_gamma ~ normal(0, 0.5);
  b_gamma ~ normal(0, 0.5);

  // With outcomes on their original scale, beta is expected to be small.
  // softplus(-4.5) is approximately 0.011.
  intercept_beta_variance ~ normal(-4.5, 1.5);
  b_beta_variance ~ normal(0, 0.5);

  sigma_lambda_raw ~ normal(0, 1);
  sigma_alpha_raw ~ normal(0, 1);
  sigma_tau_raw ~ normal(0, 1);
  sigma_gamma_raw ~ normal(0, 1);
  sigma_beta_variance_raw ~ normal(-1, 1);

  Omega_lambda ~ lkj_corr(2);
  Omega_alpha ~ lkj_corr(2);
  Omega_tau ~ lkj_corr(2);
  Omega_gamma ~ lkj_corr(2);
  Omega_beta_variance ~ lkj_corr(2);

  to_vector(z_lambda) ~ std_normal();
  to_vector(z_alpha) ~ std_normal();
  to_vector(z_tau) ~ std_normal();
  to_vector(z_gamma) ~ std_normal();
  to_vector(z_beta_variance) ~ std_normal();

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, subject_id, gamble_type,
    lambda, alpha, tau, gamma, beta_variance
  );
}

generated quantities {
  vector[2] lambda_out;
  vector[2] alpha_out;
  vector[2] tau_out;
  vector[2] gamma_out;
  vector[2] beta_variance_out;

  array[T] real log_lik;
  array[T] int y_rep;

  lambda_out = log1p_exp(
    intercept_lambda + b_lambda * effect_coding
  );
  alpha_out = log1p_exp(
    intercept_alpha + b_alpha * effect_coding
  );
  tau_out = log1p_exp(
    intercept_tau + b_tau * effect_coding
  );
  gamma_out = log1p_exp(
    intercept_gamma + b_gamma * effect_coding
  );
  beta_variance_out = log1p_exp(
    intercept_beta_variance + b_beta_variance * effect_coding
  );

  for (i in 1:T) {
    int s = subject_id[i];
    int c = gamble_type[i];
    real utility_a = get_cpt_variance_utility(
      to_vector(outcomes_a[i]), alpha[c, s], lambda[c, s],
      gamma[c, s], beta_variance[c, s]
    );
    real utility_b = get_cpt_variance_utility(
      to_vector(outcomes_b[i]), alpha[c, s], lambda[c, s],
      gamma[c, s], beta_variance[c, s]
    );
    real logit_p = tau[c, s] * (utility_b - utility_a);

    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
