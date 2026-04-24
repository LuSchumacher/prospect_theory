functions {
  vector get_mvl_utility(vector outcomes, real b_var, real b_loss) {
    int K = num_elements(outcomes);
    int J = K / 2;
    vector[J] x_a = outcomes[1:J];
    vector[J] x_b = outcomes[(J+1):K];
    real mean_a = mean(x_a);
    real mean_b = mean(x_b);
    real sd_a   = sd(x_a);
    real sd_b   = sd(x_b);
    real min_a  = min(x_a);
    real min_b  = min(x_b);
    vector[2] util;
    util[1] = mean_a - b_var * sd_a - b_loss * abs(min_a);
    util[2] = mean_b - b_var * sd_b - b_loss * abs(min_b);
    return util;
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
  real<lower=0> s_b_loss = log1p_exp(sigma_b_loss);
  real<lower=0> s_b_var  = log1p_exp(sigma_b_var);
  real<lower=0> s_tau    = log1p_exp(sigma_tau);
  // Condition specific parameters
  vector[2] mu_b_loss;
  vector[2] mu_b_var;
  vector[2] mu_tau;
  // Subject specific parameters
  matrix[2, N] b_loss;
  matrix[2, N] b_var;
  matrix[2, N] tau;
  vector[2] dummy = to_vector({0, 1});
  
  for (i in 1:2) {
    // Condition specific parameters
    mu_b_loss[i] = intercept_b_loss + beta_b_loss * dummy[i];
    mu_b_var[i]  = intercept_b_var  + beta_b_var  * dummy[i];
    mu_tau[i]    = intercept_tau    + beta_tau    * dummy[i];
    // Subject specific parameters
    b_loss[i]    = (mu_b_loss[i] + s_b_loss * z_b_loss)';
    b_var[i]     = (mu_b_var[i]  + s_b_var  * z_b_var)';
    tau[i]       = (log1p_exp(mu_tau[i]    + s_tau    * z_tau))';
  }
}

model {
  intercept_b_loss ~ normal(0, 2);
  intercept_b_var  ~ normal(0, 2);
  intercept_tau    ~ normal(0.5, 2);
  beta_b_loss      ~ normal(0, 0.5);
  beta_b_var       ~ normal(0, 0.5);
  beta_tau         ~ normal(0, 0.5);
  sigma_b_loss     ~ normal(0, 1.5);
  sigma_b_var      ~ normal(0, 1.5);
  sigma_tau        ~ normal(0, 1.5);
  z_b_loss         ~ std_normal();
  z_b_var          ~ std_normal();
  z_tau            ~ std_normal();

  for (t in 1:T) {
    vector[2] util = get_mvl_utility(
        append_row(outcome_a[t]', outcome_b[t]'),
        b_var[gamble_type[t], subject_id[t]],
        b_loss[gamble_type[t], subject_id[t]]
    );
    real utility_a = util[1];
    real utility_b = util[2];
    real logit_p = tau[gamble_type[t], subject_id[t]] * (utility_b - utility_a);
    choice[t] ~ bernoulli_logit(logit_p);
  }
}

generated quantities {
  vector[2] b_loss_out = mu_b_loss;
  vector[2] b_var_out  = mu_b_var;
  vector[2] tau_out    = log1p_exp(mu_tau);

  array[T] real log_lik;
  array[T] int y_rep;

  for (t in 1:T) {
    vector[2] util = get_mvl_utility(
        append_row(outcome_a[t]', outcome_b[t]'),
        b_var[gamble_type[t], subject_id[t]],
        b_loss[gamble_type[t], subject_id[t]]
    );
    real utility_a = util[1];
    real utility_b = util[2];
    real logit_p = tau[gamble_type[t], subject_id[t]] * (utility_b - utility_a);
    log_lik[t] = bernoulli_logit_lpmf(choice[t] | logit_p);
    y_rep[t] = bernoulli_logit_rng(logit_p);
  }
}
