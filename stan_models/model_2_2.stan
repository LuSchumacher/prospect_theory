functions {
  real prelec_w(real p, real gamma) {
    return exp(-pow(-log(p), gamma));
  }
  // real get_cpt_utility(
  //   vector x,
  //   real alpha,
  //   real lambda,
  //   real gamma
  // ) {
  //   vector[3] v;
  //   array[3] int idx;
  // 
  //   // Value function
  //   for (k in 1:3) {
  //     if (x[k] >= 0)
  //       v[k] = pow(x[k], alpha);
  //     else
  //       v[k] = -lambda * pow(abs(x[k]), alpha);
  //   }
  // 
  //   // Sort by value
  //   idx = sort_indices_asc(v);
  //   v = v[idx];
  // 
  //   // Fixed cumulative probabilities
  //   real w1 = prelec_w(1.0 / 3.0, gamma);
  //   real w2 = prelec_w(2.0 / 3.0, gamma);
  //   real w3 = 1.0;
  // 
  //   return
  //     (w1)       * v[1] +
  //     (w2 - w1)  * v[2] +
  //     (w3 - w2)  * v[3];
  // }
  real get_cpt_utility(
    vector x,
    real alpha,
    real lambda,
    real gamma
  ) {
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
  
    // CPT weighting is applied separately within the loss and gain domains.
    // Each option contains one loss and two gains with probability 1/3 each.
    real w_1_3 = prelec_w(1.0 / 3.0, gamma);
    real w_2_3 = prelec_w(2.0 / 3.0, gamma);
  
    return
      w_1_3               * v[1] +
      (w_2_3 - w_1_3)     * v[2] +
      w_1_3               * v[3];
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
    matrix tau,
    matrix gamma
  ) {
    real acc = 0;
    for (i in start:end) {
      int s = subject_id[i];
      int c = gamble_type[i];
      real utility_a = get_cpt_utility(
        to_vector(outcomes_a[i]),
        alpha[c, s],
        lambda[c, s],
        gamma[c, s]
      );
      real utility_b = get_cpt_utility(
        to_vector(outcomes_b[i]),
        alpha[c, s],
        lambda[c, s],
        gamma[c, s]
      );
      real logit_p = tau[c, s] * (utility_b - utility_a);
      acc += bernoulli_logit_lpmf(choice_slice[i - start + 1] | logit_p);
    }
    return acc;
  }
}

data {
  int<lower=1> T;                             // Number of trials
  int<lower=1> N;                             // Number of subjects
  array[T] int<lower=1> subject_id;           // Subject index per trial
  array[T] int<lower=1, upper=2> gamble_type; // 1=aligned, 2=opposed
  matrix[T, 3] outcomes_a;                    // A option outcomes
  matrix[T, 3] outcomes_b;                    // B option outcomes
  array[T] int<lower=0, upper=1> choice;      // 0=choose A, 1=choose B
}

parameters {
  // Group-level intercepts and slopes
  real intercept_lambda;
  real b_lambda;
  real intercept_alpha;
  real b_alpha;
  real intercept_tau;
  real b_tau;
  real intercept_gamma;
  real b_gamma;
  matrix[N, 2] z_lambda;
  matrix[N, 2] z_alpha;
  matrix[N, 2] z_tau;
  matrix[N, 2] z_gamma;
  vector[2] sigma_lambda;
  vector[2] sigma_alpha;
  vector[2] sigma_tau;
  vector[2] sigma_gamma;
  corr_matrix[2] Omega_lambda;
  corr_matrix[2] Omega_alpha;
  corr_matrix[2] Omega_tau;
  corr_matrix[2] Omega_gamma;
}

transformed parameters {
  vector[2] s_lambda = log1p_exp(sigma_lambda);
  vector[2] s_alpha  = log1p_exp(sigma_alpha);
  vector[2] s_tau    = log1p_exp(sigma_tau);
  vector[2] s_gamma  = log1p_exp(sigma_gamma);
  vector[2] effect_coding = [-0.5, 0.5]';
  matrix[N, 2] lambda_raw = 
    (diag_pre_multiply(s_lambda, cholesky_decompose(Omega_lambda)) * 
     transpose(z_lambda))';
  matrix[N, 2] alpha_raw =
    (diag_pre_multiply(s_alpha, cholesky_decompose(Omega_alpha)) *
     transpose(z_alpha))';
  matrix[N, 2] tau_raw =
    (diag_pre_multiply(s_tau, cholesky_decompose(Omega_tau)) *
     transpose(z_tau))';
  matrix[N, 2] gamma_raw =
    (diag_pre_multiply(s_gamma, cholesky_decompose(Omega_gamma)) *
     transpose(z_gamma))';

  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;
  matrix[2, N] gamma;

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

      real mu_gamma_c = intercept_gamma + b_gamma * effect_coding[c];
      real subj_gamma = gamma_raw[s,1] + gamma_raw[s,2] * effect_coding[c];
      gamma[c, s] = log1p_exp(mu_gamma_c + subj_gamma);
    }
  }
}

model {
  // Priors for fixed effects
  intercept_lambda ~ normal(2, 1);
  intercept_alpha  ~ normal(0, 1);
  intercept_tau    ~ normal(0.5, 2);
  intercept_gamma  ~ normal(0, 0.5);
  b_lambda ~ normal(0, 0.5);
  b_alpha ~ normal(0, 0.5);
  b_tau ~ normal(0, 0.5);
  b_gamma ~ normal(0, 0.5);

  // Priors for SDs
  sigma_lambda ~ normal(0, 1);
  sigma_alpha  ~ normal(0, 1);
  sigma_tau    ~ normal(0, 1);
  sigma_gamma  ~ normal(0, 1);

  // Priors for correlations
  Omega_lambda ~ lkj_corr(2);
  Omega_alpha ~ lkj_corr(2);
  Omega_tau ~ lkj_corr(2);
  Omega_gamma ~ lkj_corr(2);

  // Random effects
  to_vector(z_lambda) ~ normal(0, 1);
  to_vector(z_alpha) ~ normal(0, 1);
  to_vector(z_tau) ~ normal(0, 1);
  to_vector(z_gamma) ~ normal(0, 1);

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, 
    subject_id, gamble_type,
    lambda, alpha, tau, gamma
  );
}

generated quantities {
  vector[2] lambda_out = log1p_exp(intercept_lambda + b_lambda * effect_coding);
  vector[2] alpha_out  = log1p_exp(intercept_alpha + b_alpha * effect_coding);
  vector[2] tau_out    = log1p_exp(intercept_tau + b_tau * effect_coding);
  vector[2] gamma_out  = log1p_exp(intercept_gamma + b_gamma * effect_coding);

  array[T] real log_lik;
  array[T] int y_rep;

  for (i in 1:T) {
    int s = subject_id[i];
    int c = gamble_type[i];
    real utility_a = get_cpt_utility(
      to_vector(outcomes_a[i]),
      alpha[c, s],
      lambda[c, s],
      gamma[c, s]
    );
    real utility_b = get_cpt_utility(
      to_vector(outcomes_b[i]),
      alpha[c, s],
      lambda[c, s],
      gamma[c, s]
    );
    real logit_p = tau[c, s] * (utility_b - utility_a);
    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
