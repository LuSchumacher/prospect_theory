import numpy as np
import bayesflow as bf
from numba import njit

#--- Helper Functions ---#
@njit
def skewnorm(alpha, loc=0, scale=1, size=1):
    U = np.random.randn(size)
    V = np.random.randn(size)
    X = (alpha * np.abs(U) + V) / np.sqrt(1 + alpha**2)
    samples = loc + scale * X
    return samples

@njit
def truncated_normal(mean, sigma, lower, upper, size=1):
    samples = np.empty(size)
    for i in range(size):
        while True:
            sample = np.random.normal(mean, sigma)
            if lower <= sample <= upper:
                samples[i] = sample
                break
    return samples

#--- PT Prior ---#
pt_param_names = (r'$\lambda$', r'$\alpha$', r'$\tau$')

# inv_util_2
@njit
def sample_pt_prior(batch_size=32):
    # lamda = truncated_normal(2, 0.75, 1, 4, batch_size)
    lamda = np.random.uniform(0.8, 3.0, batch_size)
    # alpha = np.random.beta(2.5, 5, batch_size) * 2
    alpha = np.random.uniform(0.2, 1.2, batch_size)
    # tau = np.random.gamma(1, 10, batch_size)
    # tau = truncated_normal(0, 8, 0, 20, batch_size)
    tau = truncated_normal(0, 3, 0, 10, batch_size)
    return np.vstack((lamda, alpha, tau)).T

# @njit
# def sample_pt_prior(batch_size=32):
#     lamda = truncated_normal(2, 0.75, 1, 4, batch_size)
#     # lamda = np.random.uniform(0.8, 3.0, batch_size)
#     alpha = np.random.beta(2.5, 5, batch_size) * 2
#     # alpha = np.random.uniform(0.2, 1.2, batch_size)
#     tau = np.random.gamma(1, 10, batch_size)
#     # tau = truncated_normal(0, 8, 0, 20, batch_size)
#     # tau = truncated_normal(0, 3, 0, 10, batch_size)
#     return np.vstack((lamda, alpha, tau)).T

pt_prior = bf.simulation.Prior(
    batch_prior_fun=sample_pt_prior,
    param_names=pt_param_names
)

#--- MVL Prior ---#
mvl_param_names = (r'$b_{\text{var}}$', r'$b_{\text{loss}}$', r'$\tau$')

@njit
def sample_mvl_prior(batch_size=32):
    b_var = skewnorm(3, 0, 0.4, batch_size)
    b_loss = skewnorm(3, 0, 0.4, batch_size)
    tau = np.random.uniform(0.0, 2.0, batch_size)
    return np.vstack((b_var, b_loss, tau)).T

mvl_prior = bf.simulation.Prior(
    batch_prior_fun=sample_mvl_prior,
    param_names=mvl_param_names
)
