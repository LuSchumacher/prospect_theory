functions {
  real prelec_w(real p, real gamma) {
    return exp(-pow(-log(p), gamma));
  }
  real get_cpt_utility(
    vector x,
    real alpha,
    real lambda,
    real gamma
  ) {
    vector[3] v;
    array[3] int idx;
  
    // Value function
    for (k in 1:3) {
      if (x[k] >= 0)
        v[k] = pow(x[k], alpha);
      else
        v[k] = -lambda * pow(abs(x[k]), alpha);
    }
  
    // Sort by value
    idx = sort_indices_asc(v);
    v = v[idx];
  
    // Fixed cumulative probabilities
    real w1 = prelec_w(1.0 / 3.0, gamma);
    real w2 = prelec_w(2.0 / 3.0, gamma);
    real w3 = 1.0;
  
    return
      (w1)       * v[1] +
      (w2 - w1)  * v[2] +
      (w3 - w2)  * v[3];
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
    matrix tau,
    matrix gamma
  ) {
    real acc = 0;
    for (i in start:end) {
      real utility_a_raw = get_cpt_utility(
        to_vector(outcomes_a[i]),
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]],
        gamma[gamble_type[i], subject_id[i]]
      );
      real utility_b_raw = get_cpt_utility(
        to_vector(outcomes_b[i]),
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]],
        gamma[gamble_type[i], subject_id[i]]
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
  matrix[T, 3]                   outcomes_a;   // A option outcomes
  matrix[T, 3]                   outcomes_b;   // B option outcomes
  array[T] int<lower=0, upper=1> choice;      // 0 = chose A, 1 = chose B
}

parameters {
  // group-level parameters
  real intercept_lambda;
  real intercept_alpha;
  real intercept_tau;
  real intercept_gamma;
  real b_lambda;
  real b_alpha;
  real b_tau;
  real b_gamma;
  real sigma_lambda;
  real sigma_alpha;
  real sigma_tau;
  real sigma_gamma;
  // individual deviations
  vector[N] z_lambda;
  vector[N] z_alpha;
  vector[N] z_tau;
  vector[N] z_gamma;
}

transformed parameters {
  // Condition specific parameters
  vector[2] effect_coding = [-0.5, 0.5]';
  vector[2] mu_lambda = intercept_lambda  + b_lambda * effect_coding;
  vector[2] mu_alpha  = intercept_alpha   + b_alpha  * effect_coding;
  vector[2] mu_tau    = intercept_tau     + b_tau    * effect_coding;
  vector[2] mu_gamma  = intercept_gamma   + b_gamma  * effect_coding;
  // Subject specific parameters
  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;
  matrix[2, N] gamma;
  real<lower=0> s_lambda = log1p_exp(sigma_lambda);
  real<lower=0> s_alpha  = log1p_exp(sigma_alpha);
  real<lower=0> s_tau    = log1p_exp(sigma_tau);
  real<lower=0> s_gamma  = log1p_exp(sigma_gamma);
  for (c in 1:2) {
    lambda[c] = log1p_exp(mu_lambda[c] + s_lambda * z_lambda)';
    alpha[c]  = log1p_exp(mu_alpha[c]  + s_alpha  * z_alpha)';
    tau[c]    = log1p_exp(mu_tau[c]    + s_tau    * z_tau)';
    gamma[c]  = log1p_exp(mu_gamma[c]  + s_gamma  * z_gamma)';
  }
}

model {
  intercept_lambda ~ normal(2, 1);
  intercept_alpha  ~ normal(0, 1);
  intercept_tau    ~ normal(0.5, 2);
  intercept_gamma  ~ normal(0, 0.5);
  b_lambda         ~ normal(0, 0.5);
  b_alpha          ~ normal(0, 0.5);
  b_tau            ~ normal(0, 0.5);
  b_gamma          ~ normal(0, 0.5);
  sigma_lambda     ~ normal(0, 1);
  sigma_alpha      ~ normal(0, 1);
  sigma_tau        ~ normal(0, 1);
  sigma_gamma      ~ normal(0, 1);
  z_lambda         ~ std_normal();
  z_alpha          ~ std_normal();
  z_tau            ~ std_normal();
  z_gamma          ~ std_normal();

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, 
    subject_id, gamble_type,
    lambda, alpha, tau, gamma
  );
}

generated quantities {
  vector[2] lambda_out = log1p_exp(mu_lambda);
  vector[2] alpha_out  = log1p_exp(mu_alpha);
  vector[2] tau_out    = log1p_exp(mu_tau);
  vector[2] gamma_out  = log1p_exp(mu_gamma);

  array[T] real log_lik;
  array[T] int y_rep;

  for (i in 1:T) {
    real utility_a_raw = get_cpt_utility(
      to_vector(outcomes_a[i]),
      alpha[gamble_type[i], subject_id[i]],
      lambda[gamble_type[i], subject_id[i]],
      gamma[gamble_type[i], subject_id[i]]
    );
    real utility_b_raw = get_cpt_utility(
      to_vector(outcomes_b[i]),
      alpha[gamble_type[i], subject_id[i]],
      lambda[gamble_type[i], subject_id[i]],
      gamma[gamble_type[i], subject_id[i]]
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
    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
