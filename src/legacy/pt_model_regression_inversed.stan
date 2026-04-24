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
  int<lower=1>                   T;           // Number of trials
  int<lower=1>                   N;           // Number of subjects
  array[T] int<lower=1>          subject_id;  // Subject index per trial
  array[T] int<lower=1, upper=2> gamble_type; // 1 = confounded, 2 = unconfounded
  matrix[T, 3]                   outcome_a;   // A option outcomes
  matrix[T, 3]                   outcome_b;   // B option outcomes
  array[T] int<lower=0, upper=1> choice;      // 0 = chose A, 1 = chose B
}

parameters {
  real intercept_lambda;
  real intercept_alpha;
  real intercept_tau;

  real beta_lambda;
  real beta_alpha;
  real beta_tau;
  
  real sigma_lambda;
  real sigma_alpha;
  real sigma_tau;
  
  vector[N] z_lambda;
  vector[N] z_alpha;
  vector[N] z_tau;
}

transformed parameters {
  real<lower=0> s_lambda = log1p_exp(sigma_lambda);
  real<lower=0> s_alpha  = log1p_exp(sigma_alpha);
  real<lower=0> s_tau    = log1p_exp(sigma_tau);
  // Condition specific parameters
  vector[2] mu_lambda;
  vector[2] mu_alpha;
  vector[2] mu_tau;
  // Subject specific parameters
  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;
  vector[2] dummy = to_vector({0, 1});
  
  for (i in 1:2) {
    // Condition specific parameters
    mu_lambda[i] = intercept_lambda + beta_lambda * dummy[i];
    mu_alpha[i]  = intercept_alpha  + beta_alpha  * dummy[i];
    mu_tau[i]    = intercept_tau    + beta_tau    * dummy[i];
    // Subject specific parameters
    lambda[i]    = (log1p_exp(mu_lambda[i] + s_lambda * z_lambda))';
    alpha[i]     = (log1p_exp(mu_alpha[i]  + s_alpha  * z_alpha))';
    tau[i]       = (log1p_exp(mu_tau[i]    + s_tau    * z_tau))';
  }
}

model {
  intercept_lambda ~ normal(2, 1);
  intercept_alpha  ~ normal(0, 1);
  intercept_tau    ~ normal(0.5, 2);
  beta_lambda      ~ normal(0, 0.5);
  beta_alpha       ~ normal(0, 0.5);
  beta_tau         ~ normal(0, 0.5);
  sigma_lambda     ~ normal(0, 1.5);
  sigma_alpha      ~ normal(0, 1.5);
  sigma_tau        ~ normal(0, 1.5);
  z_lambda         ~ std_normal();
  z_alpha          ~ std_normal();
  z_tau            ~ std_normal();

  for (t in 1:T) {
    real utility_a_raw = get_pt_utility(
      to_vector(outcome_a[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_b_raw = get_pt_utility(
      to_vector(outcome_b[t]),
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
    choice[t] ~ bernoulli_logit(logit_p);
  }
}

generated quantities {
  vector[2] lambda_out = log1p_exp(mu_lambda);
  vector[2] alpha_out  = log1p_exp(mu_alpha);
  vector[2] tau_out    = log1p_exp(mu_tau);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    real utility_a_raw = get_pt_utility(
      to_vector(outcome_a[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_b_raw = get_pt_utility(
      to_vector(outcome_b[t]),
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

