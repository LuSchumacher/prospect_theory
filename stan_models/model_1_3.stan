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
  
  real partial_log_lik(
    array[] int choice_slice,
    int start, int end,
    matrix outcomes_a,
    matrix outcomes_b,
    array[] int subject_id,
    array[] int gamble_type,
    matrix lambda,
    matrix alpha,
    matrix tau
  ) {
    real acc = 0;
    for (i in start:end) {
      real utility_a_raw = get_pt_utility(
        to_vector(outcomes_a[i]),
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]]
      );
      real utility_b_raw = get_pt_utility(
        to_vector(outcomes_b[i]),
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]]
      );
      real utility_a = inverse_utility(
        utility_a_raw, 
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]]
      );
      real utility_b = inverse_utility(
        utility_b_raw, 
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]]
      );
      real logit_p = tau[gamble_type[i], subject_id[i]] * (utility_b - utility_a);
      acc += bernoulli_logit_lpmf(choice_slice[i - start + 1] | logit_p);
    }
    return acc;
  }
}

data {
  int<lower=1>                   T;           // Number of trials
  int<lower=1>                   N;           // Number of subjects
  array[T] int<lower=1>          subject_id;  // Subject index per trial
  array[T] int<lower=1, upper=2> gamble_type; // 1 = confounded, 2 = unconfounded
  matrix[T, 3]                   outcomes_a;  // A option outcomes
  matrix[T, 3]                   outcomes_b;  // B option outcomes
  array[T] int<lower=0, upper=1> choice;      // 0 = chose A, 1 = chose B
}

parameters {
  real intercept_lambda;
  real b_lambda;
  real intercept_alpha;
  real b_alpha;
  real intercept_tau;
  real b_tau;
  matrix[N, 2] z_lambda;
  matrix[N, 2] z_alpha;
  matrix[N, 2] z_tau;
  vector[2] sigma_lambda;
  vector[2] sigma_alpha;
  vector[2] sigma_tau;
  corr_matrix[2] Omega_lambda;
  corr_matrix[2] Omega_alpha;
  corr_matrix[2] Omega_tau;
}

transformed parameters {
  vector[2] s_lambda = log1p_exp(sigma_lambda);
  vector[2] s_alpha  = log1p_exp(sigma_alpha);
  vector[2] s_tau    = log1p_exp(sigma_tau);
  vector[2] effect_coding = [-0.5, 0.5]';
  matrix[N, 2] lambda_raw =
    (diag_pre_multiply(sigma_lambda, cholesky_decompose(Omega_lambda)) * transpose(z_lambda))';
  matrix[N, 2] alpha_raw =
    (diag_pre_multiply(sigma_alpha, cholesky_decompose(Omega_alpha)) * transpose(z_alpha))';
  matrix[N, 2] tau_raw =
    (diag_pre_multiply(sigma_tau, cholesky_decompose(Omega_tau)) * transpose(z_tau))';

  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;

  for (s in 1:N) {
    for (c in 1:2) {
      real mu_lambda_c = intercept_lambda + b_lambda * effect_coding[c];
      real subj_lambda = lambda_raw[s,1] + lambda_raw[s,2] * effect_coding[c];
      lambda[c, s] = log1p_exp(mu_lambda_c + subj_lambda);

      real mu_alpha_c = intercept_alpha + b_alpha * effect_coding[c];
      real subj_alpha = alpha_raw[s,1] + alpha_raw[s,2] * effect_coding[c];
      alpha[c, s] = log1p_exp(mu_alpha_c + subj_alpha);

      real mu_tau_c = intercept_tau + b_tau * effect_coding[c];
      real subj_tau = tau_raw[s,1] + tau_raw[s,2] * effect_coding[c];
      tau[c, s] = log1p_exp(mu_tau_c + subj_tau);
    }
  }
}

model {
  // Priors
  intercept_lambda ~ normal(2, 1);
  intercept_alpha ~ normal(0, 1);
  intercept_tau ~ normal(0.5, 2);
  b_lambda ~ normal(0, 0.5);
  b_alpha ~ normal(0, 0.5);
  b_tau ~ normal(0, 0.5);
  sigma_lambda ~ normal(0, 1);
  sigma_alpha ~ normal(0, 1);
  sigma_tau ~ normal(0, 1);
  Omega_lambda ~ lkj_corr(2);
  Omega_alpha ~ lkj_corr(2);
  Omega_tau ~ lkj_corr(2);

  to_vector(z_lambda) ~ normal(0, 1);
  to_vector(z_alpha) ~ normal(0, 1);
  to_vector(z_tau) ~ normal(0, 1);

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b,
    subject_id, gamble_type,
    lambda, alpha, tau
  );
}

generated quantities {
  vector[2] lambda_out = log1p_exp(intercept_lambda + b_lambda * effect_coding);
  vector[2] alpha_out = log1p_exp(intercept_alpha + b_alpha * effect_coding);
  vector[2] tau_out = log1p_exp(intercept_tau + b_tau * effect_coding);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    real utility_a_raw = get_pt_utility(
      to_vector(outcomes_a[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_b_raw = get_pt_utility(
      to_vector(outcomes_b[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_a = inverse_utility(
      utility_a_raw, 
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_b = inverse_utility(
      utility_b_raw, 
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real logit_p = tau[gamble_type[t], subject_id[t]] * (utility_b - utility_a);
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | logit_p);
    y_rep[t] = bernoulli_logit_rng(logit_p);
  }
}
