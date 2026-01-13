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
      real utility_a = get_pt_utility(
        to_vector(outcomes_a[i]),
        alpha[gamble_type[i], subject_id[i]],
        lambda[gamble_type[i], subject_id[i]]
      );
      real utility_b = get_pt_utility(
        to_vector(outcomes_b[i]),
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
  real b_lambda;
  real b_alpha;
  real b_tau;
  real sigma_lambda;
  real sigma_alpha;
  real sigma_tau;
  // individual deviations
  vector[N] z_lambda;
  vector[N] z_alpha;
  vector[N] z_tau;
}

transformed parameters {
  // Condition specific parameters
  vector[2] effect_coding = [-0.5, 0.5]';
  vector[2] mu_lambda = intercept_lambda  + b_lambda * effect_coding;
  vector[2] mu_alpha  = intercept_alpha   + b_alpha  * effect_coding;
  vector[2] mu_tau    = intercept_tau     + b_tau    * effect_coding;
  // Subject specific parameters
  matrix[2, N] lambda;
  matrix[2, N] alpha;
  matrix[2, N] tau;
  real<lower=0> s_lambda = log1p_exp(sigma_lambda);
  real<lower=0> s_alpha  = log1p_exp(sigma_alpha);
  real<lower=0> s_tau    = log1p_exp(sigma_tau);
  for (c in 1:2) {
    lambda[c] = log1p_exp(mu_lambda[c] + s_lambda * z_lambda)';
    alpha[c]  = log1p_exp(mu_alpha[c]  + s_alpha  * z_alpha)';
    tau[c]    = log1p_exp(mu_tau[c]    + s_tau    * z_tau)';
  }
}

model {
  intercept_lambda ~ normal(2, 1);
  intercept_alpha  ~ normal(0, 1);
  intercept_tau    ~ normal(0.5, 2);
  b_lambda         ~ normal(0, 0.5);
  b_alpha          ~ normal(0, 0.5);
  b_tau            ~ normal(0, 0.5);
  sigma_lambda     ~ normal(0, 1);
  sigma_alpha      ~ normal(0, 1);
  sigma_tau        ~ normal(0, 1);
  z_lambda         ~ std_normal();
  z_alpha          ~ std_normal();
  z_tau            ~ std_normal();

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, 
    subject_id, gamble_type,
    lambda, alpha, tau
  );
}

generated quantities {
  vector[2] lambda_out = log1p_exp(mu_lambda);
  vector[2] alpha_out  = log1p_exp(mu_alpha);
  vector[2] tau_out    = log1p_exp(mu_tau);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    real utility_a = get_pt_utility(
      to_vector(outcomes_a[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real utility_b = get_pt_utility(
      to_vector(outcomes_b[t]),
      alpha[gamble_type[t], subject_id[t]],
      lambda[gamble_type[t], subject_id[t]]
    );
    real logit_p = tau[gamble_type[t], subject_id[t]] * (utility_b - utility_a);
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | logit_p);
    y_rep[t] = bernoulli_logit_rng(logit_p);
  }
}
