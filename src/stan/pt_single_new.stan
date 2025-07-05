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
  array[T] int<lower=0, upper=1> condition; // 0 = old, 1 = new
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
  real<lower=0> s_lambda_0 = log1p_exp(sigma_lambda_0);
  real<lower=0> s_lambda_1 = log1p_exp(sigma_lambda_1);
  real<lower=0> s_alpha_0  = log1p_exp(sigma_alpha_0);
  real<lower=0> s_alpha_1  = log1p_exp(sigma_alpha_1);
  real<lower=0> s_tau_0    = log1p_exp(sigma_tau_0);
  real<lower=0> s_tau_1    = log1p_exp(sigma_tau_1);
  
  vector<lower=0>[N]          lambda_0 = exp(mu_lambda_0 + s_lambda_0 * z_lambda_0);
  vector<lower=0>[N]          lambda_1 = exp(mu_lambda_1 + s_lambda_1 * z_lambda_1);
  vector<lower=0, upper=1>[N] alpha_0  = inv_logit(mu_alpha_0 + s_alpha_0 * z_alpha_0);
  vector<lower=0, upper=1>[N] alpha_1  = inv_logit(mu_alpha_1 + s_alpha_1 * z_alpha_1);
  vector<lower=0>[N]          tau_0    = exp(mu_tau_0 + s_tau_0 * z_tau_0);
  vector<lower=0>[N]          tau_1    = exp(mu_tau_1 + s_tau_1 * z_tau_1);
  
}

model {
  mu_lambda_0 ~ normal(log(2), 0.5);
  mu_lambda_1 ~ normal(log(2), 0.5);
  mu_alpha_0  ~ normal(1.0, 1.75); 
  mu_alpha_1  ~ normal(1.0, 1.75);
  mu_tau_0    ~ normal(0.5, 0.5);
  mu_tau_1    ~ normal(0.5, 0.5);
  
  sigma_lambda_0 ~ normal(0, 1);
  sigma_lambda_1 ~ normal(0, 1);
  sigma_alpha_0  ~ normal(0, 1);
  sigma_alpha_1  ~ normal(0, 1);
  sigma_tau_0    ~ normal(0, 1);
  sigma_tau_1    ~ normal(0, 1);
  
  z_lambda_0 ~ std_normal();
  z_lambda_1 ~ std_normal();
  z_alpha_0  ~ std_normal();
  z_alpha_1  ~ std_normal();
  z_tau_0    ~ std_normal();
  z_tau_1    ~ std_normal();

  for (t in 1:T) {
    real lambda = condition[t] == 0 ? lambda_0[subject_id[t]] : lambda_1[subject_id[t]];
    real alpha  = condition[t] == 0 ? alpha_0[subject_id[t]]  : alpha_1[subject_id[t]];
    real tau    = condition[t] == 0 ? tau_0[subject_id[t]]    : tau_1[subject_id[t]];

    real u_a_raw = get_pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real u_b_raw = get_pt_utility(to_vector(outcome_b[t]), alpha, lambda);
    real u_a = inverse_utility(u_a_raw, alpha, lambda);
    real u_b = inverse_utility(u_b_raw, alpha, lambda);

    // choice[t] ~ bernoulli_logit(tau * (u_b_raw - u_a_raw));
    choice[t] ~ bernoulli_logit(tau * (u_b - u_a));
  }
}

generated quantities {
  real lambda_0_out = exp(mu_lambda_0);
  real lambda_1_out = exp(mu_lambda_1);
  real alpha_0_out = inv_logit(mu_alpha_0);
  real alpha_1_out = inv_logit(mu_alpha_1);
  real tau_0_out = exp(mu_tau_0);
  real tau_1_out = exp(mu_tau_1);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    real lambda = condition[t] == 0 ? lambda_0[subject_id[t]] : lambda_1[subject_id[t]];
    real alpha  = condition[t] == 0 ? alpha_0[subject_id[t]]  : alpha_1[subject_id[t]];
    real tau    = condition[t] == 0 ? tau_0[subject_id[t]]    : tau_1[subject_id[t]];

    real u_a_raw = get_pt_utility(to_vector(outcome_a[t]), alpha, lambda);
    real u_b_raw = get_pt_utility(to_vector(outcome_b[t]), alpha, lambda);
    real logit_p = tau * (u_b_raw - u_a_raw);
    
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | logit_p);
    y_rep[t] = bernoulli_logit_rng(logit_p);
  }
}

