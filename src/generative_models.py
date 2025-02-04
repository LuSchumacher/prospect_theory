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
    return np.where(
        utilities >= 0,
        utilities ** (1 / alpha),
        -(np.abs(utilities) / lamda) ** (1 / alpha)
    )

@njit
def get_pt_utility(alpha, lamda, outcomes):
    num_outcomes = len(outcomes) // 2
    utilities = np.where(
        outcomes >= 0, outcomes ** alpha, -lamda * np.abs(outcomes) ** alpha
    )
    utility_a = np.mean(utilities[:num_outcomes])
    utility_b = np.mean(utilities[num_outcomes:])
    return get_inv_pt_utility(alpha, lamda, np.array([utility_a, utility_b]))

@njit
def sample_pt_model(theta, context):
    alpha, lamda, tau = theta
    choices = np.zeros(context.shape[0])
    for i in range(context.shape[0]):
        utilities = get_pt_utility(alpha, lamda, context[i])
        choices[i] = get_choice(tau, utilities)
    return choices

# --- MVL Model ---
@njit
def get_mvl_utility(b_var, b_loss, outcomes):
    num_outcomes = len(outcomes) // 2
    x_a = outcomes[:num_outcomes]
    x_b = outcomes[num_outcomes:]
    utility_a = np.mean(x_a) - b_var * np.std(x_a) - b_loss * np.abs(np.min(x_a))
    utility_b = np.mean(x_b) - b_var * np.std(x_b) - b_loss * np.abs(np.min(x_b))
    return np.array([utility_a, utility_b])

@njit
def sample_mvl_model(theta, context):
    b_var, b_loss, tau = theta
    choices = np.zeros(context.shape[0])
    for i in range(context.shape[0]):
        utilities = get_mvl_utility(b_var, b_loss, context[i])
        choices[i] = get_choice(tau, utilities)
    return choices
