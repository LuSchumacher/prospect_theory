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
}

model {
  mu_lambda_intercept ~ normal(2.5, 1);
  mu_alpha_intercept ~ beta(1, 1);
  mu_tau_intercept ~ normal(2.5, 1);
  
  
  mu_lamda_beta ~ normal(0, 1);
  
}