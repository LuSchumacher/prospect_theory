functions {
  real get_pt_utility(vector outcome, real alpha, real lambda) {
    vector[2] v;
    for (k in 1:2) {
      if (outcome[k] >= 0)
        v[k] = pow(outcome[k], alpha);
      else
        v[k] = -lambda * pow(abs(outcome[k]), alpha);
    }
    return mean(v);
  }
  
  real partial_log_lik(
    array[] int choice_slice,
    int start,
    int end,
    matrix outcomes_b,
    array[] int subject_id,
    vector alpha,
    vector tau
  ) {
    real acc = 0;
  
    for (i in start:end) {
      int s = subject_id[i];
  
      real utility_a = pow(20.0, alpha[s]);
      real utility_b = get_pt_utility(outcomes_b[i]', alpha[s], 1);
  
      real logit_p = tau[s] * (utility_b - utility_a);
  
      acc += bernoulli_logit_lpmf(choice_slice[i - start + 1] | logit_p);
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
  real mu_alpha;
  real mu_tau;
  vector[N] z_alpha;
  vector[N] z_tau;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_tau;
}

transformed parameters {
  vector[N] alpha;
  vector[N] tau;
  for (s in 1:N) {
    alpha[s] = log1p_exp(mu_alpha + sigma_alpha * z_alpha[s]);
    tau[s] = log1p_exp(mu_tau + sigma_tau * z_tau[s]);
  }
}

model {
  mu_alpha ~ normal(0, 1);
  mu_tau ~ normal(0.5, 2);
  sigma_alpha ~ normal(0, 1);
  sigma_tau ~ normal(0, 1);
  z_alpha ~ std_normal();
  z_tau ~ std_normal();
  
  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_b, subject_id, alpha, tau
  );
}

generated quantities {
  real alpha_group_location = log1p_exp(mu_alpha);
  real tau_group_location = log1p_exp(mu_tau);
  
  array[T] real log_lik;
  array[T] int y_rep;
  
  for (i in 1:T) {
    int s = subject_id[i];
    
    real utility_a = pow(20.0, alpha[s]);
    real utility_b = get_pt_utility(outcomes_b[i]', alpha[s], 1);
    
    real logit_p = tau[s] * (utility_b - utility_a);
    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
