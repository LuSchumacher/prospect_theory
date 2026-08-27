functions {
  real prelec_w(real p, real gamma) {
    return exp(-pow(-log(p), gamma));
  }

  real get_cpt_utility(vector x, real alpha, real lambda, real gamma) {
    vector[3] v;
    array[3] int idx;

    for (k in 1:3) {
      if (x[k] >= 0) {
        v[k] = pow(x[k], alpha);
      } else {
        v[k] = -lambda * pow(abs(x[k]), alpha);
      }
    }

    idx = sort_indices_asc(v);
    v = v[idx];

    {
      real w1 = prelec_w(1.0 / 3.0, gamma);
      real w2 = prelec_w(2.0 / 3.0, gamma);
      return w1 * v[1] + (w2 - w1) * v[2] + (1.0 - w2) * v[3];
    }
  }

  real partial_log_lik(
    array[] int choice_slice,
    int start,
    int end,
    matrix outcomes_a,
    matrix outcomes_b,
    array[] int gamble_type,
    vector lambda,
    vector alpha,
    vector tau,
    vector gamma
  ) {
    real acc = 0;

    for (i in start:end) {
      int c = gamble_type[i];
      real utility_a = get_cpt_utility(
        to_vector(outcomes_a[i]), alpha[c], lambda[c], gamma[c]
      );
      real utility_b = get_cpt_utility(
        to_vector(outcomes_b[i]), alpha[c], lambda[c], gamma[c]
      );
      real logit_p = tau[c] * (utility_b - utility_a);

      acc += bernoulli_logit_lpmf(
        choice_slice[i - start + 1] | logit_p
      );
    }

    return acc;
  }
}

data {
  int<lower=1> T;
  array[T] int<lower=1, upper=2> gamble_type;
  matrix[T, 3] outcomes_a;
  matrix[T, 3] outcomes_b;
  array[T] int<lower=0, upper=1> choice;
}

parameters {
  real intercept_lambda;
  real b_lambda;
  real intercept_alpha;
  real b_alpha;
  real intercept_tau;
  real b_tau;
  real intercept_gamma;
  real b_gamma;
}

transformed parameters {
  vector[2] effect_coding;
  vector[2] lambda;
  vector[2] alpha;
  vector[2] tau;
  vector[2] gamma;

  effect_coding[1] = -0.5;
  effect_coding[2] = 0.5;

  lambda = log1p_exp(intercept_lambda + b_lambda * effect_coding);
  alpha = log1p_exp(intercept_alpha + b_alpha * effect_coding);
  tau = log1p_exp(intercept_tau + b_tau * effect_coding);
  gamma = log1p_exp(intercept_gamma + b_gamma * effect_coding);
}

model {
  // Priors are placed on the unconstrained scale. The lambda intercept is
  // centered on lambda = 1 to avoid prior-induced apparent loss aversion.
  intercept_lambda ~ normal(0.54, 1.00);
  b_lambda ~ normal(0, 0.50);
  intercept_alpha ~ normal(0.54, 0.75);
  b_alpha ~ normal(0, 0.50);
  intercept_tau ~ normal(0.01, 1.50);
  b_tau ~ normal(0, 0.50);
  intercept_gamma ~ normal(0.54, 0.75);
  b_gamma ~ normal(0, 0.50);

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, gamble_type,
    lambda, alpha, tau, gamma
  );
}

generated quantities {
  real delta_lambda = lambda[2] - lambda[1];
  real delta_alpha = alpha[2] - alpha[1];
  real delta_tau = tau[2] - tau[1];
  real delta_gamma = gamma[2] - gamma[1];
}
