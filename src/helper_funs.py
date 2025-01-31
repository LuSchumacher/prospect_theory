import numpy as np
from numba import njit

@njit
def numba_nansum(arr, axis=0):
    """Numba-compatible version of np.nansum."""
    return np.where(np.isnan(arr), 0, arr).sum(axis=axis)

@njit
def get_choices(utilities, tau):
    num_choices = utilities.shape[0]
    choices = np.zeros(num_choices, dtype=np.int64)
    # Compute softmax probabilities
    e_x = np.exp(utilities * tau)
    probs = e_x / np.sum(e_x, axis=1)[:, None]
    rand_vals = np.random.random(num_choices)
    # Determine choices using searchsorted
    for i in range(num_choices):
        cum_prob = np.cumsum(probs[i])
        choices[i] = np.searchsorted(cum_prob, rand_vals[i], side="right")
    return choices