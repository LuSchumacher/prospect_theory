functions {
  
real prelec(real p, real rho) {
  real w;
  w = exp(-(-log(p))^rho);
  return w; 
}

}


data {
	int<lower=1> N;									         // number of data points 
	int<lower=1> L;									         // number of Participants
	int<lower=1> n_trials;									 // number of Lotteries (lot x L)
	int<lower=1> trials[N];	                 // Lottery number  

	real pa1[n_trials];				 
	real pa2[n_trials];
	real pa3[n_trials];
	real pa4[n_trials];
	real pa5[n_trials];
	
	real pb1[n_trials];
	real pb2[n_trials];
	real pb3[n_trials];
	real pb4[n_trials];
	real pb5[n_trials];
	
	real a1[n_trials];
	real a2[n_trials];
	real a3[n_trials];
	real a4[n_trials];
	real a5[n_trials];
	
	real b1[n_trials];
	real b2[n_trials];
	real b3[n_trials];
	real b4[n_trials];
	real b5[n_trials];
	
	int<lower=1, upper=L> participant[N];		 // participant per datapoint
	int<lower=0,upper=1> response[N];				 // response (0 or 1)
}

parameters {
  vector[3] mu;                          // mu_alpha and mu_sensitivity
  cholesky_factor_corr[3] L_p;           // Cholesky factor for z_p correlation
  matrix[3, L] z_p;                      // Uncorrelated matrix for participants
  vector<lower=0>[3] sigma_p;            // Scale vector for z_p
}


transformed parameters {
  matrix[3, L] z;
  real<lower=0> sensitivity_sbj[L];
  real<lower=0> alpha_sbj[L];
  real<lower=0> rho_sbj[L];
  real transf_mu_sensitivity = log(1 + exp(mu[1]));
  real transf_mu_alpha = log(1 + exp(mu[2]))*1.5;
  real transf_mu_rho = log(1 + exp(mu[3]))*1.5;
  real U_a[N];
	real U_b[N]; 
  
  // multiply uncorrelated z_p-matrix with correlations and scale according to sd
  z = diag_pre_multiply(sigma_p, L_p) * z_p;
  
  for (l in 1:L) {
    sensitivity_sbj[l] = log(1 + exp(mu[1] + z[1, l]));
    alpha_sbj[l] = log(1 + exp(mu[2] + z[2, l]))*1.5;
    rho_sbj[l] = log(1 + exp(mu[3] + z[3, l]))*1.5;
  }
  
 

  
  for(n in 1:N) {

    U_a[n] = a1[trials[n]]^alpha_sbj[participant[n]] * prelec(pa1[trials[n]], rho_sbj[participant[n]]) + 
          a2[trials[n]]^alpha_sbj[participant[n]] * (prelec(pa2[trials[n]], rho_sbj[participant[n]]) - prelec(pa1[trials[n]], rho_sbj[participant[n]])) + 
          a3[trials[n]]^alpha_sbj[participant[n]] * (prelec(pa3[trials[n]], rho_sbj[participant[n]]) - prelec(pa2[trials[n]], rho_sbj[participant[n]])) + 
          a4[trials[n]]^alpha_sbj[participant[n]] * (prelec(pa4[trials[n]], rho_sbj[participant[n]]) - prelec(pa3[trials[n]], rho_sbj[participant[n]])) + 
          a5[trials[n]]^alpha_sbj[participant[n]] * (prelec(pa5[trials[n]], rho_sbj[participant[n]]) - prelec(pa4[trials[n]], rho_sbj[participant[n]]));
    
    U_b[n] = b1[trials[n]]^alpha_sbj[participant[n]] * prelec(pb1[trials[n]], rho_sbj[participant[n]]) +
          b2[trials[n]]^alpha_sbj[participant[n]] * (prelec(pb2[trials[n]], rho_sbj[participant[n]]) - prelec(pb1[trials[n]], rho_sbj[participant[n]])) + 
          b3[trials[n]]^alpha_sbj[participant[n]] * (prelec(pb3[trials[n]], rho_sbj[participant[n]]) - prelec(pb2[trials[n]], rho_sbj[participant[n]])) + 
          b4[trials[n]]^alpha_sbj[participant[n]] * (prelec(pb4[trials[n]], rho_sbj[participant[n]]) - prelec(pb3[trials[n]], rho_sbj[participant[n]])) + 
          b5[trials[n]]^alpha_sbj[participant[n]] * (prelec(pb5[trials[n]], rho_sbj[participant[n]]) - prelec(pb4[trials[n]], rho_sbj[participant[n]]));
    
    U_a[n] = U_a[n]^(1/alpha_sbj[participant[n]]);
    U_b[n] = U_b[n]^(1/alpha_sbj[participant[n]]);
  }
  

}

model {
  mu ~ normal(0, 1);
  sigma_p ~ exponential(3);              // Prior for sigma
  L_p ~ lkj_corr_cholesky(2);            // Prior for Cholesky factor
  to_vector(z_p) ~ normal(0, 1);         // Prior for z_p

	// Likelihood function
  for (n in 1:N) {
    vector[2] p;
    real prob;
    p[1] = U_a[n];
    p[2] = U_b[n];
 
    p = p - max(p);
    prob = softmax(p * sensitivity_sbj[participant[n]])[1];
    
    response[n] ~ bernoulli( prob );
  }
}

generated quantities {
  matrix[3, 3] Omega;
  vector[N] log_lik;
  vector[N] pp;
  real prob;
  Omega = multiply_lower_tri_self_transpose(L_p);

	// Likelihood function
  for (n in 1:N) {
    vector[2] p;
    p[1] = U_a[n];
    p[2] = U_b[n];

    p = p - max(p);
    pp[n] = softmax(p * sensitivity_sbj[participant[n]])[1];

    log_lik[n] = bernoulli_lpmf(response[n] | pp[n]);
  }
}

