functions {
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
  int<lower=1>          T;             // Number of trials
  int<lower=1>          N;             // Number of subjects
  int<lower=1>          id[T];         // Subject index per trial
  int<lower=0, upper=1> condition[T];  // Condition per trial
  matrix[T, 3]          outcome_a;     // A option outcomes
  matrix[T, 3]          outcome_b;     // B option outcomes
  int<lower=0, upper=1> choice[T];     // 0 = chose A, 1 = chose B
}

parameters {
  // Non-centered parameters for hierarchical effects
  vector[N] z_log_lambda_intercept;
  vector[N] z_log_lambda_beta;
  
  vector[N] z_log_alpha_intercept;
  vector[N] z_log_alpha_beta;
  
  vector[N] z_log_tau_intercept;
  vector[N] z_log_tau_beta;

  // Group-level regression coefficients
  real mu_log_lambda_intercept;
  real mu_log_lambda_beta;
  real<lower=0> sigma_log_lambda_intercept;
  real<lower=0> sigma_log_lambda_beta;

  real mu_log_alpha_intercept;
  real mu_log_alpha_beta;
  real<lower=0> sigma_log_alpha_intercept;
  real<lower=0> sigma_log_alpha_beta;

  real mu_log_tau_intercept;
  real mu_log_tau_beta;
  real<lower=0> sigma_log_tau_intercept;
  real<lower=0> sigma_log_tau_beta;
}


model {
  y ~ normal(mu, sigma);
}

