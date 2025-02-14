import numpy as np
from numba import njit

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

@njit
def sample_pt_prior(batch_size=32):
    lamda = truncated_normal(2, 0.8, 1, 3, batch_size)
    alpha = truncated_normal(0.8, 0.3, 0, 1.2, batch_size)
    tau = np.random.gamma(1, 1, batch_size)
    return np.vstack((lamda, alpha, tau)).T

@njit
def sample_mvl_prior(batch_size=32):
    b_var = skewnorm(3, 0, 0.4, batch_size)
    b_loss = skewnorm(3, 0, 0.4, batch_size)
    tau = np.random.uniform(0.0, 1.5, batch_size)
    return np.vstack((b_var, b_loss, tau)).T