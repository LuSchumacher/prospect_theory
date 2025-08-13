import numpy as np
from numba import njit
import bayesflow as bf
from src.priors import pt_prior, mvl_prior, cpt_prior
from src.context import context_gen

#--- Helper Functions ---#
@njit
def get_choice(tau, utilities):
    z = utilities - max(utilities)
    numerator = np.exp(z * tau)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    alt = np.array([0, 1])
    return alt[np.searchsorted(np.cumsum(softmax), np.random.random(), side="right")]

#--- PT Model ---#
PT_PRIOR_MEAN  = np.array([1.9, 0.7, 1.9])
PT_PRIOR_STD  = np.array([0.6, 0.3, 1.4])

@njit
def get_inv_pt_utility(lamda, alpha, utilities):
    return np.where(
        utilities >= 0,
        utilities ** (1 / alpha),
        -(np.abs(utilities) / lamda) ** (1 / alpha)
    )

@njit
def get_pt_utility(lamda, alpha, outcomes):
    num_outcomes = len(outcomes) // 2
    utilities = np.where(
        outcomes >= 0, outcomes ** alpha, -lamda * np.abs(outcomes) ** alpha
    )
    utility_a = np.mean(utilities[:num_outcomes])
    utility_b = np.mean(utilities[num_outcomes:])
    return np.array([utility_a, utility_b])

@njit
def sample_pt_model(theta, context):
    lamda, alpha, tau = theta
    choices = np.zeros(context.shape[0])
    for i in range(context.shape[0]):
        utilities = get_pt_utility(lamda, alpha, context[i])
        inv_utilities = get_inv_pt_utility(lamda, alpha, utilities)
        choices[i] = get_choice(tau, inv_utilities)
    return choices

pt_simulator = bf.simulation.Simulator(
    simulator_fun=sample_pt_model,
    context_generator=context_gen
)

pt_model = bf.simulation.GenerativeModel(
    prior=pt_prior,
    simulator=pt_simulator,
    name="pt_model",
    skip_test=True
)

def pt_configurator(forward_dict):
    out_dict = {}
    data = forward_dict["sim_data"][:, :, None]
    context = np.array(forward_dict["sim_batchable_context"]) / 50
    out_dict["summary_conditions"] = np.c_[data, context].astype(np.float32)
    params = forward_dict["prior_draws"].astype(np.float32)
    out_dict["parameters"] = (params - PT_PRIOR_MEAN) / PT_PRIOR_STD
    return out_dict

# --- CPT Model ---
CPT_PRIOR_MEAN  = np.array([1.9, 0.7, 0.6, 1.9])
CPT_PRIOR_STD  = np.array([0.6, 0.3, 0.24, 1.4])

@njit
def get_decision_weights(gamma):
    cum_probs = np.array([1/3, 2/3, 3/3])
    prelec_weight = np.exp(-(-np.log(cum_probs)) ** gamma)
    decision_weights = np.empty(3)
    decision_weights[0] = prelec_weight[0]
    decision_weights[1] = prelec_weight[1] - prelec_weight[0]
    decision_weights[2] = 1.0 - prelec_weight[1]
    return decision_weights

@njit
def get_cpt_utility(lamda, alpha, gamma, outcomes):
    weights = get_decision_weights(gamma)
    num_outcomes = len(outcomes) // 2
    outcomes_a = np.sort(outcomes[:num_outcomes])
    outcomes_b = np.sort(outcomes[num_outcomes:])
    utilities_a = np.where(
        outcomes_a >= 0, outcomes_a ** alpha, -lamda * np.abs(outcomes_a) ** alpha
    )
    utilities_b = np.where(
        outcomes_b >= 0, outcomes_b ** alpha, -lamda * np.abs(outcomes_b) ** alpha
    )
    utility_a = utilities_a @ weights
    utility_b = utilities_b @ weights
    return np.array([utility_a, utility_b])

@njit
def sample_cpt_model(theta, context):
    lamda, alpha, gamma, tau = theta
    choices = np.zeros(context.shape[0])
    for i in range(context.shape[0]):
        utilities = get_cpt_utility(lamda, alpha, gamma, context[i])
        inv_utilities = get_inv_pt_utility(lamda, alpha, utilities)
        choices[i] = get_choice(tau, inv_utilities)
    return choices

cpt_simulator = bf.simulation.Simulator(
    simulator_fun=sample_cpt_model,
    context_generator=context_gen
)

cpt_model = bf.simulation.GenerativeModel(
    prior=cpt_prior,
    simulator=cpt_simulator,
    name="cpt_model",
    skip_test=True
)

def cpt_configurator(forward_dict):
    out_dict = {}
    data = forward_dict["sim_data"][:, :, None]
    context = np.array(forward_dict["sim_batchable_context"]) / 50
    out_dict["summary_conditions"] = np.c_[data, context].astype(np.float32)
    params = forward_dict["prior_draws"].astype(np.float32)
    out_dict["parameters"] = (params - CPT_PRIOR_MEAN) / CPT_PRIOR_STD
    return out_dict

# --- MVL Model ---
MVL_PRIOR_MEAN = np.array([0.3, 0.3, 1])
MVL_PRIOR_STD = np.array([0.26, 0.26, 0.57])

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

mvl_simulator = bf.simulation.Simulator(
    simulator_fun=sample_mvl_model,
    context_generator=context_gen
)

mvl_model = bf.simulation.GenerativeModel(
    prior=mvl_prior,
    simulator=mvl_simulator,
    name="mvl_model",
    skip_test=True
)

def mvl_configurator(forward_dict):
    out_dict = {}
    data = forward_dict["sim_data"][:, :, None]
    context = np.array(forward_dict["sim_batchable_context"]) / 50
    out_dict["summary_conditions"] = np.c_[data, context].astype(np.float32)
    params = forward_dict["prior_draws"]
    out_dict["parameters"] = ((params - MVL_PRIOR_MEAN) / MVL_PRIOR_STD).astype(np.float32)
    return out_dict