functions {
  real get_mvl_utility(vector x, real b_var, real b_loss) {
    return mean(x) - b_var * sd(x) - b_loss * abs(min(x));
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
        to_vector(outcomes_a[i]),
        b_var[c, s],
        b_loss[c, s]
      );
      real utility_b = get_mvl_utility(
        to_vector(outcomes_b[i]),
        b_var[c, s],
        b_loss[c, s]
      );
      real logit_p = tau[c, s] * (utility_b - utility_a);
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
  
  real sigma_b_loss;
  real sigma_b_var;
  real sigma_tau;
  
  vector[N] z_b_loss;
  vector[N] z_b_var;
  vector[N] z_tau;
}

transformed parameters {
  // Condition specific parameters
  vector[2] effect_coding = [-0.5, 0.5]';
  vector[2] mu_b_var  = intercept_b_var   + beta_b_var  * effect_coding;
  vector[2] mu_b_loss = intercept_b_loss  + beta_b_loss * effect_coding;
  vector[2] mu_tau    = intercept_tau     + beta_tau    * effect_coding;
  // Subject specific parameters
  matrix[2, N] b_var;
  matrix[2, N] b_loss;
  matrix[2, N] tau;
  real<lower=0> s_b_loss = log1p_exp(sigma_b_loss);
  real<lower=0> s_b_var  = log1p_exp(sigma_b_var);
  real<lower=0> s_tau    = log1p_exp(sigma_tau);
  for (i in 1:2) {
    b_loss[i]    = (mu_b_loss[i] + s_b_loss * z_b_loss)';
    b_var[i]     = (mu_b_var[i]  + s_b_var  * z_b_var)';
    tau[i]       = (log1p_exp(mu_tau[i] + s_tau * z_tau))';
  }
}

model {
  intercept_b_loss ~ normal(0, 2);
  intercept_b_var  ~ normal(0, 2);
  intercept_tau    ~ normal(0.5, 2);
  beta_b_loss      ~ normal(0, 0.5);
  beta_b_var       ~ normal(0, 0.5);
  beta_tau         ~ normal(0, 0.5);
  sigma_b_loss     ~ normal(0, 1);
  sigma_b_var      ~ normal(0, 1);
  sigma_tau        ~ normal(0, 1);
  z_b_loss         ~ std_normal();
  z_b_var          ~ std_normal();
  z_tau            ~ std_normal();

  target += reduce_sum(
    partial_log_lik, choice, 1,
    outcomes_a, outcomes_b, 
    subject_id, gamble_type,
    b_var, b_loss, tau
  );
}

generated quantities {
  vector[2] b_loss_out = mu_b_loss;
  vector[2] b_var_out  = mu_b_var;
  vector[2] tau_out    = log1p_exp(mu_tau);

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
