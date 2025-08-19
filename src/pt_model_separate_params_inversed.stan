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
  int<lower=1> T;                           // Number of trials
  int<lower=1> N;                           // Number of subjects
  array[T] int<lower=1> subject_id;         // Subject index per trial
  array[T] int<lower=0, upper=1> gamble_type; // 0 = confounded, 1 = unconfounded
  matrix[T, 3] outcome_a;                   // A option outcomes
  matrix[T, 3] outcome_b;                   // B option outcomes
  array[T] int<lower=0, upper=1> choice;    // 0 = chose A, 1 = chose B
}

parameters {
  real mu_lambda_0;
  real mu_lambda_1;
  real mu_alpha_0;
  real mu_alpha_1;
  real mu_tau_0;
  real mu_tau_1;
  
  real sigma_lambda_0;
  real sigma_lambda_1;
  real sigma_alpha_0;
  real sigma_alpha_1;
  real sigma_tau_0;
  real sigma_tau_1;
  
  vector[N] z_lambda_0;
  vector[N] z_lambda_1;
  vector[N] z_alpha_0;
  vector[N] z_alpha_1;
  vector[N] z_tau_0;
  vector[N] z_tau_1;
}

transformed parameters {
  real<lower=0>      s_lambda_0 = log1p_exp(sigma_lambda_0);
  real<lower=0>      s_lambda_1 = log1p_exp(sigma_lambda_1);
  real<lower=0>      s_alpha_0  = log1p_exp(sigma_alpha_0);
  real<lower=0>      s_alpha_1  = log1p_exp(sigma_alpha_1);
  real<lower=0>      s_tau_0    = log1p_exp(sigma_tau_0);
  real<lower=0>      s_tau_1    = log1p_exp(sigma_tau_1);
  vector<lower=0>[N] lambda_0   = log1p_exp(mu_lambda_0 + s_lambda_0 * z_lambda_0);
  vector<lower=0>[N] lambda_1   = log1p_exp(mu_lambda_1 + s_lambda_1 * z_lambda_1);
  vector<lower=0>[N] alpha_0    = log1p_exp(mu_alpha_0 + s_alpha_0 * z_alpha_0);
  vector<lower=0>[N] alpha_1    = log1p_exp(mu_alpha_1 + s_alpha_1 * z_alpha_1);
  vector<lower=0>[N] tau_0      = log1p_exp(mu_tau_0 + s_tau_0 * z_tau_0);
  vector<lower=0>[N] tau_1      = log1p_exp(mu_tau_1 + s_tau_1 * z_tau_1);
  
}

model {
  mu_lambda_0 ~ normal(2, 1);
  mu_lambda_1 ~ normal(2, 1);
  mu_alpha_0  ~ normal(0, 1); 
  mu_alpha_1  ~ normal(0, 1);
  mu_tau_0    ~ normal(0.5, 2);
  mu_tau_1    ~ normal(0.5, 2);
  
  sigma_lambda_0 ~ normal(0, 1.5);
  sigma_lambda_1 ~ normal(0, 1.5);
  sigma_alpha_0  ~ normal(0, 1.5);
  sigma_alpha_1  ~ normal(0, 1.5);
  sigma_tau_0    ~ normal(0, 1.5);
  sigma_tau_1    ~ normal(0, 1.5);
  
  z_lambda_0 ~ std_normal();
  z_lambda_1 ~ std_normal();
  z_alpha_0  ~ std_normal();
  z_alpha_1  ~ std_normal();
  z_tau_0    ~ std_normal();
  z_tau_1    ~ std_normal();

  for (t in 1:T) {
    real lambda = gamble_type[t] == 0 ? lambda_0[subject_id[t]] : lambda_1[subject_id[t]];
    real alpha  = gamble_type[t] == 0 ? alpha_0[subject_id[t]]  : alpha_1[subject_id[t]];
    real tau    = gamble_type[t] == 0 ? tau_0[subject_id[t]]    : tau_1[subject_id[t]];
    real utility_a_raw = get_pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real utility_b_raw = get_pt_utility(to_vector(outcome_b[t]), alpha, lambda);
    real utility_a = inverse_utility(utility_a_raw, alpha, lambda);
    real utility_b = inverse_utility(utility_b_raw, alpha, lambda);
    real logit_p = tau * (utility_b - utility_a);
    choice[t] ~ bernoulli_logit(logit_p);
  }
}

generated quantities {
  real lambda_0_out = log1p_exp(mu_lambda_0);
  real lambda_1_out = log1p_exp(mu_lambda_1);
  real alpha_0_out  = log1p_exp(mu_alpha_0);
  real alpha_1_out  = log1p_exp(mu_alpha_1);
  real tau_0_out    = log1p_exp(mu_tau_0);
  real tau_1_out    = log1p_exp(mu_tau_1);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    real lambda = gamble_type[t] == 0 ? lambda_0[subject_id[t]] : lambda_1[subject_id[t]];
    real alpha  = gamble_type[t] == 0 ? alpha_0[subject_id[t]]  : alpha_1[subject_id[t]];
    real tau    = gamble_type[t] == 0 ? tau_0[subject_id[t]]    : tau_1[subject_id[t]];
    real utility_a_raw = get_pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real utility_b_raw = get_pt_utility(to_vector(outcome_b[t]), alpha, lambda);
    real utility_a = inverse_utility(utility_a_raw, alpha, lambda);
    real utility_b = inverse_utility(utility_b_raw, alpha, lambda);
    real logit_p = tau * (utility_b - utility_a);
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | logit_p);
    y_rep[t] = bernoulli_logit_rng(logit_p);
  }
}
