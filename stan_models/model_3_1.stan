functions {
  real get_mvl_utility(vector x, real b_var, real b_loss) {
    real ev = mean(x);
    real var_x = dot_self(x - rep_vector(ev, num_elements(x))) / num_elements(x);
    return ev - b_var * var_x - b_loss * abs(min(x));
  }

  real partial_log_lik(
    array[] int choice_slice,
    int start, int end,
    matrix outcomes_a,
    matrix outcomes_b,
    array[] int subject_id,
    array[] int gamble_type,
    matrix b_var,
    matrix b_loss,
    matrix tau
  ) {
    real acc = 0;

    for (i in start:end) {
      int s = subject_id[i];
      int c = gamble_type[i];

      real utility_a = get_mvl_utility(
        to_vector(outcomes_a[i]), b_var[c, s], b_loss[c, s]
      );

      real utility_b = get_mvl_utility(
        to_vector(outcomes_b[i]), b_var[c, s], b_loss[c, s]
      );

      real logit_p = tau[c, s] * (utility_b - utility_a);

      acc += bernoulli_logit_lpmf(
        choice_slice[i - start + 1] | logit_p
      );
    }

    return acc;
  }
}

data {
  int<lower=1>                   T;           // Number of trials
  int<lower=1>                   N;           // Number of subjects
  array[T] int<lower=1>          subject_id;  // Subject index per trial
  array[T] int<lower=1, upper=2> gamble_type; // 1 = confounded, 2 = unconfounded
  matrix[T, 3]                   outcomes_a;  // A option outcomes
  matrix[T, 3]                   outcomes_b;  // B option outcomes
  array[T] int<lower=0, upper=1> choice;      // 0 = chose A, 1 = chose B
}

parameters {
  real intercept_b_loss;
  real intercept_b_var;
  real intercept_tau;

  real beta_b_loss;
  real beta_b_var;
  real beta_tau;
  
  vector[2] sigma_b_loss;
  vector[2] sigma_b_var;
  vector[2] sigma_tau;
  
  matrix[N, 2] z_b_loss;
  matrix[N, 2] z_b_var;
  matrix[N, 2] z_tau;
  
  corr_matrix[2] Omega_b_loss;
  corr_matrix[2] Omega_b_var;
  corr_matrix[2] Omega_tau;
}

transformed parameters {
  vector[2] s_b_loss = log1p_exp(sigma_b_loss);
  vector[2] s_b_var  = log1p_exp(sigma_b_var);
  vector[2] s_tau    = log1p_exp(sigma_tau);
  vector[2] effect_coding = [-0.5, 0.5]';
  matrix[N, 2] b_loss_raw =
    (diag_pre_multiply(s_b_loss, cholesky_decompose(Omega_b_loss))
     * transpose(z_b_loss))';
  matrix[N, 2] b_var_raw =
    (diag_pre_multiply(s_b_var, cholesky_decompose(Omega_b_var))
     * transpose(z_b_var))';
  matrix[N, 2] tau_raw =
    (diag_pre_multiply(s_tau, cholesky_decompose(Omega_tau))
     * transpose(z_tau))';

  matrix[2, N] b_loss;
  matrix[2, N] b_var;
  matrix[2, N] tau;

  for (s in 1:N) {
    for (c in 1:2) {
      real mu_b_loss_c = intercept_b_loss + beta_b_loss * effect_coding[c];
      real subj_b_loss = b_loss_raw[s,1] + b_loss_raw[s,2] * effect_coding[c];
      b_loss[c, s] = mu_b_loss_c + subj_b_loss;
      real mu_b_var_c = intercept_b_var + beta_b_var * effect_coding[c];
      real subj_b_var = b_var_raw[s,1] + b_var_raw[s,2] * effect_coding[c];
      b_var[c, s] = mu_b_var_c + subj_b_var;
      real mu_tau_c = intercept_tau + beta_tau * effect_coding[c];
      real subj_tau = tau_raw[s,1] + tau_raw[s,2] * effect_coding[c];
      tau[c, s] = log1p_exp(mu_tau_c + subj_tau);
    }
  }
}


model {
  intercept_b_loss ~ normal(0, 2);
  intercept_tau    ~ normal(0.5, 2);
  intercept_b_var  ~ normal(0, 0.01);
  beta_b_loss      ~ normal(0, 0.5);
  beta_b_var       ~ normal(0, 0.005);
  beta_tau         ~ normal(0, 0.5);
  sigma_b_loss     ~ normal(0, 1);
  sigma_b_var      ~ normal(-5, 1);
  sigma_tau        ~ normal(0, 1);
  to_vector(z_b_loss) ~ normal(0, 1);
  to_vector(z_b_var) ~ normal(0, 1);
  to_vector(z_tau) ~ normal(0, 1);
  Omega_b_loss ~ lkj_corr(2);
  Omega_b_var  ~ lkj_corr(2);
  Omega_tau    ~ lkj_corr(2);

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, 
    subject_id, gamble_type,
    b_var, b_loss, tau
  );
}

generated quantities {
  vector[2] b_loss_out;
  vector[2] b_var_out;
  vector[2] tau_out;
  for (c in 1:2) {
    b_loss_out[c] = intercept_b_loss + beta_b_loss * (c == 1 ? -0.5 : 0.5);
    b_var_out[c]  = intercept_b_var  + beta_b_var  * (c == 1 ? -0.5 : 0.5);
    tau_out[c]    = log1p_exp(intercept_tau + beta_tau * (c == 1 ? -0.5 : 0.5));
  }

  array[T] real log_lik;
  array[T] int y_rep;

  for (i in 1:T) {
    int s = subject_id[i];
    int c = gamble_type[i];
    real utility_a = get_mvl_utility(
      to_vector(outcomes_a[i]),
      b_var[c, s],
      b_loss[c, s]
    );
    real utility_b = get_mvl_utility(
      to_vector(outcomes_b[i]),
      b_var[c, s],
      b_loss[c, s]
    );
    real logit_p = tau[gamble_type[i], subject_id[i]] * (utility_b - utility_a);
    log_lik[i] = bernoulli_logit_lpmf(choice[i] | logit_p);
    y_rep[i] = bernoulli_logit_rng(logit_p);
  }
}
