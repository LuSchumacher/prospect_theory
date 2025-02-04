import numpy as np
import pandas as pd
from numba import njit

CONTEXT = np.zeros((5, 160, 6))
for i in range(5):
    gamble_df = pd.read_csv(f'data/three_outcome_lotteries_{i}.csv')
    CONTEXT[i] = gamble_df[[
        'outcome_a1', 'outcome_a2', 'outcome_a3',
        'outcome_b1', 'outcome_b2', 'outcome_b3'
    ]].to_numpy()

@njit
def get_context():
    rand_idx = np.random.randint(0, 5)
    return CONTEXT[rand_idx]