functions {
  real partial_sum(
    array[] int choice_slice, int start, int end,
    vector lambda_intercept, vector lambda_beta,
    vector alpha_intercept, vector alpha_beta,
    vector tau_intercept, vector tau_beta,
    array[] int subject_id, array[] int choice, array[] int condition,
    matrix outcome_a, matrix outcome_b
  ) {
    real ans = 0;
    for (t in start:end) {
      int s = subject_id[t];

      // Use individual-level parameters directly:
      real lambda = lambda_intercept[s] + lambda_beta[s] * condition[t];
      real alpha  = alpha_intercept[s]  + alpha_beta[s]  * condition[t];
      real tau    = tau_intercept[s]    + tau_beta[s]    * condition[t];

      real utility_a = pt_utility(to_vector(outcome_a[t]), alpha, lambda);
      real utility_b = pt_utility(to_vector(outcome_b[t]), alpha, lambda);

      ans += bernoulli_logit_lpmf(choice[t] | tau * (utility_a - utility_b));
    }
    return ans;
  }
  
  real pt_utility(vector outcomes, real alpha, real lambda) {
    real utility = 0;
    int K = num_elements(outcomes);
    for (i in 1:K) {
      real x = outcomes[i];
      utility += (x >= 0 ? pow(x, alpha) : -lambda * pow(-x, alpha)) / K;
    }
    return utility;
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
  // Group-level parameters
  real mu_lambda_intercept;
  real mu_lambda_beta;
  real sigma_lambda_intercept;
  real sigma_lambda_beta;
  
  real mu_alpha_intercept;
  real mu_alpha_beta;
  real sigma_alpha_intercept;
  real sigma_alpha_beta;
  
  real mu_tau_intercept;
  real mu_tau_beta;
  real sigma_tau_intercept;
  real sigma_tau_beta;
  
  // Non-centered parameters
  vector[N] z_lambda_intercept;
  vector[N] z_lambda_beta;
  vector[N] z_alpha_intercept;
  vector[N] z_alpha_beta;
  vector[N] z_tau_intercept;
  vector[N] z_tau_beta;
}

transformed parameters {
  // Individual-level parameters
  vector[N] lambda_intercept;
  vector[N] lambda_beta;
  vector[N] alpha_intercept;
  vector[N] alpha_beta;
  vector[N] tau_intercept;
  vector[N] tau_beta;
  
  // Transformed group-level standard deviation
  real<lower=0> s_lambda_intercept;
  real<lower=0> s_lambda_beta;
  real<lower=0> s_alpha_intercept;
  real<lower=0> s_alpha_beta;
  real<lower=0> s_tau_intercept;
  real<lower=0> s_tau_beta;

  s_lambda_intercept = log1p_exp(sigma_lambda_intercept);
  s_lambda_beta      = log1p_exp(sigma_lambda_beta);
  s_alpha_intercept  = log1p_exp(sigma_alpha_intercept);
  s_alpha_beta       = log1p_exp(sigma_alpha_beta);
  s_tau_intercept    = log1p_exp(sigma_tau_intercept);
  s_tau_beta         = log1p_exp(sigma_tau_beta);

  lambda_intercept = log1p_exp(mu_lambda_intercept + s_lambda_intercept * z_lambda_intercept);
  lambda_beta      = mu_lambda_beta + s_lambda_beta * z_lambda_beta;
  alpha_intercept  = inv_logit(mu_alpha_intercept + s_alpha_intercept * z_alpha_intercept + 1);
  alpha_beta       = mu_alpha_beta + s_alpha_beta * z_alpha_beta;
  tau_intercept    = log1p_exp(mu_tau_intercept + s_tau_intercept * z_tau_intercept);
  tau_beta         = mu_tau_beta + s_tau_beta * z_tau_beta;;
}

model {
  // Priors
  mu_lambda_intercept    ~ normal(2, 1.5);
  sigma_lambda_intercept ~ normal(0, 1);
  mu_lambda_beta         ~ normal(0, 1);
  sigma_lambda_beta      ~ normal(0, 1);
  
  mu_alpha_intercept     ~ normal(0, 1.5);
  sigma_alpha_intercept  ~ normal(0, 1);
  mu_alpha_beta          ~ normal(0, 1);
  sigma_alpha_beta       ~ normal(0, 1);
  
  mu_tau_intercept       ~ normal(1, 2);
  sigma_tau_intercept    ~ normal(0, 1);
  mu_tau_beta            ~ normal(0, 1);
  sigma_tau_beta         ~ normal(0, 1);

  z_lambda_intercept     ~ std_normal();
  z_lambda_beta          ~ std_normal();
  z_alpha_intercept      ~ std_normal();
  z_alpha_beta           ~ std_normal();
  z_tau_intercept        ~ std_normal();
  z_tau_beta             ~ std_normal();
  
  target += reduce_sum(
    partial_sum, choice, 1,
    lambda_intercept, lambda_beta,
    alpha_intercept, alpha_beta,
    tau_intercept, tau_beta,
    subject_id, choice, condition,
    outcome_a, outcome_b
  );
}

generated quantities {
  // Group-level transformed means under condition = 0
  real lambda_mean_0 = log1p_exp(mu_lambda_intercept);
  real alpha_mean_0  = inv_logit(mu_alpha_intercept + 1);
  real tau_mean_0    = log1p_exp(mu_tau_intercept);

  // Group-level transformed means under condition = 1
  real lambda_mean_1 = log1p_exp(mu_lambda_intercept + mu_lambda_beta);
  real alpha_mean_1  = inv_logit(mu_alpha_intercept + mu_alpha_beta + 1);
  real tau_mean_1    = log1p_exp(mu_tau_intercept + mu_tau_beta);

  // Condition effects (delta = condition 1 - condition 0)
  real delta_lambda = lambda_mean_1 - lambda_mean_0;
  real delta_alpha  = alpha_mean_1 - alpha_mean_0;
  real delta_tau    = tau_mean_1 - tau_mean_0;

  // Pointwise log-likelihood for each trial (for LOO/WAIC)
  vector[T] log_lik;
  for (t in 1:T) {
    int s = subject_id[t];

    real lambda = lambda_intercept[s] + lambda_beta[s] * condition[t];
    real alpha  = alpha_intercept[s]  + alpha_beta[s]  * condition[t];
    real tau    = tau_intercept[s]    + tau_beta[s]    * condition[t];

    vector[3] outcomes_a_t = to_vector(row(outcome_a, t));
    vector[3] outcomes_b_t = to_vector(row(outcome_b, t));

    real utility_a = pt_utility(outcomes_a_t, alpha, lambda);
    real utility_b = pt_utility(outcomes_b_t, alpha, lambda);

    log_lik[t] = bernoulli_logit_lpmf(choice[t] | tau * (utility_a - utility_b));
  }
}
