import numpy as np
import pandas as pd
import bayesflow as bf
from numba import njit

context_1 = pd.read_csv('../data/lotteries_gain_new.csv')
context_2 = pd.read_csv('../data/lotteries_gain_old.csv')
context = [context_1, context_2]

CONTEXT = np.zeros((2, 140, 6))
for i in range(2):
    gamble_df = context[i]
    CONTEXT[i] = gamble_df[[
        'outcome_a1', 'outcome_a2', 'outcome_a3',
        'outcome_b1', 'outcome_b2', 'outcome_b3'
    ]].to_numpy()

@njit
def get_context():
    rand_idx = np.random.randint(0, 1 + 1)
    return CONTEXT[rand_idx]

context_gen = bf.simulation.ContextGenerator(
    batchable_context_fun=get_context
)