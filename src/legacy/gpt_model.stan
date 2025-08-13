data {
  int<lower=1> T;                 // number of trials
  int<lower=1> N;                 // number of subjects
  int<lower=1, upper=N> subject_id[T];
  int<lower=0, upper=1> resp[T];  // binary choice (0/1)
  int<lower=1, upper=2> condition[T]; // condition indicator (1 or 2)

  // Each lottery has 3 outcomes, outcomes bounded [-100, 100]
  // For each trial, you have two options with 3 outcomes each
  matrix[3, T] lottery1_outcomes; // outcomes for option 1
  matrix[3, T] lottery2_outcomes; // outcomes for option 2
}

parameters {
  // population-level (fixed effects)
  real mu_log_lambda_intercept;
  real mu_log_lambda_beta;
  real mu_log_alpha_intercept;
  real mu_log_alpha_beta;
  real mu_log_tau_intercept;
  real mu_log_tau_beta;

  // raw group-level SDs (unconstrained)
  real raw_sigma_log_lambda_intercept;
  real raw_sigma_log_lambda_beta;
  real raw_sigma_log_alpha_intercept;
  real raw_sigma_log_alpha_beta;
  real raw_sigma_log_tau_intercept;
  real raw_sigma_log_tau_beta;

  // subject-level z-scored effects
  vector[N] z_log_lambda_intercept;
  vector[N] z_log_lambda_beta;
  vector[N] z_log_alpha_intercept;
  vector[N] z_log_alpha_beta;
  vector[N] z_log_tau_intercept;
  vector[N] z_log_tau_beta;
}

transformed parameters {
  // apply softplus to get strictly positive SDs
  real sigma_log_lambda_intercept = log1p_exp(raw_sigma_log_lambda_intercept);
  real sigma_log_lambda_beta     = log1p_exp(raw_sigma_log_lambda_beta);
  real sigma_log_alpha_intercept = log1p_exp(raw_sigma_log_alpha_intercept);
  real sigma_log_alpha_beta      = log1p_exp(raw_sigma_log_alpha_beta);
  real sigma_log_tau_intercept   = log1p_exp(raw_sigma_log_tau_intercept);
  real sigma_log_tau_beta        = log1p_exp(raw_sigma_log_tau_beta);
}

model {
  // population-level priors
  mu_log_lambda_intercept ~ normal(0, 1);
  mu_log_lambda_beta     ~ normal(0, 1);
  mu_log_alpha_intercept ~ normal(0, 1);
  mu_log_alpha_beta      ~ normal(0, 1);
  mu_log_tau_intercept   ~ normal(0, 1);
  mu_log_tau_beta        ~ normal(0, 1);

  // raw SD priors
  raw_sigma_log_lambda_intercept ~ normal(0, 1);
  raw_sigma_log_lambda_beta     ~ normal(0, 1);
  raw_sigma_log_alpha_intercept ~ normal(0, 1);
  raw_sigma_log_alpha_beta      ~ normal(0, 1);
  raw_sigma_log_tau_intercept   ~ normal(0, 1);
  raw_sigma_log_tau_beta        ~ normal(0, 1);

  // z priors
  z_log_lambda_intercept ~ std_normal();
  z_log_lambda_beta      ~ std_normal();
  z_log_alpha_intercept  ~ std_normal();
  z_log_alpha_beta       ~ std_normal();
  z_log_tau_intercept    ~ std_normal();
  z_log_tau_beta         ~ std_normal();

  for (t in 1:T) {
    int s = subject_id[t];
    real log_lambda = mu_log_lambda_intercept + mu_log_lambda_beta * cond_code[t]
                    + sigma_log_lambda_intercept * z_log_lambda_intercept[s]
                    + sigma_log_lambda_beta * z_log_lambda_beta[s] * cond_code[t];

    real log_alpha = mu_log_alpha_intercept + mu_log_alpha_beta * cond_code[t]
                   + sigma_log_alpha_intercept * z_log_alpha_intercept[s]
                   + sigma_log_alpha_beta * z_log_alpha_beta[s] * cond_code[t];

    real log_tau = mu_log_tau_intercept + mu_log_tau_beta * cond_code[t]
                 + sigma_log_tau_intercept * z_log_tau_intercept[s]
                 + sigma_log_tau_beta * z_log_tau_beta[s] * cond_code[t];

    real lambda = log1p_exp(log_lambda);  // softplus
    real alpha = log1p_exp(log_alpha);
    real tau = log1p_exp(log_tau);

    real util1 = 0;
    real util2 = 0;
    for (i in 1:3) {
      real x1 = outcomes_lot1[t][i];
      real x2 = outcomes_lot2[t][i];

      util1 += (x1 >= 0 ? pow(x1, alpha) : -lambda * pow(-x1, alpha)) / 3;
      util2 += (x2 >= 0 ? pow(x2, alpha) : -lambda * pow(-x2, alpha)) / 3;
    }

    target += bernoulli_logit_lpmf(resp[t] | tau * (util1 - util2));
  }
}