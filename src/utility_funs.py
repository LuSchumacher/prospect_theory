import numpy as np
from numba import njit



@njit
def get_pt_utility(lamda, alpha, outcomes):
    num_outcomes = int(outcomes.shape[0] / 2)
    outcome_prob = 1 / num_outcomes
    value = np.zeros(outcomes.shape[0])
    for i, x in enumerate(outcomes):
        if x >= 0:
            value[i] = x**alpha
        else:
            value[i] = -lamda * np.abs(x)**alpha
    utility_a = outcome_prob * np.nansum(value[:num_outcomes])
    utility_b = outcome_prob * np.nansum(value[num_outcomes:])
    return np.array([utility_a, utility_b])