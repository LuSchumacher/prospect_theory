import numpy as np
from numba import njit

@njit
def get_choice(tau, utilities):
    e_x = np.exp(utilities * tau)
    probs = e_x / e_x.sum()
    alt = np.array([0, 1])
    return alt[np.searchsorted(np.cumsum(probs), np.random.random(), side="right")]

# --- PT Model ---

@njit
def get_inv_pt_utility(alpha, lamda, utilities):
    for i in range(utilities.shape[0]):
        if utilities[i] >= 0:
            utilities[i] = utilities[i] ** (1 / alpha)
        else:
            utilities[i] = -(np.abs(utilities[i]) / lamda) ** (1 / alpha)
    return utilities

@njit
def get_pt_utility(alpha, lamda, outcomes):
    num_outcomes = len(outcomes) // 2
    utility_a = 0.0
    utility_b = 0.0
    for j in range(num_outcomes):
        x_a = outcomes[j]
        x_b = outcomes[j + num_outcomes]
        if x_a >= 0:
            utility_a += x_a ** alpha
        else:
            utility_a += -lamda * ((-x_a) ** alpha)
        if x_b >= 0:
            utility_b += x_b ** alpha
        else:
            utility_b += -lamda * ((-x_b) ** alpha)
    utility_a /= num_outcomes
    utility_b /= num_outcomes
    utilities = np.array([utility_a, utility_b])
    return get_inv_pt_utility(alpha, lamda, utilities)

@njit
def sample_pt_model(alpha, lamda, tau, context):
    num_outcomes = context.shape[1] // 2
    choices = np.zeros(context.shape[0])
    for i in range(context.shape[0]):
        utilities = get_pt_utility(alpha, lamda, context[i])
        choices[i] = get_choice(tau, utilities)
    return choices

# --- MVL Model ---