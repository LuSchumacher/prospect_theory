import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew

RNG = np.random.default_rng(2025)

columns = [
    "outcome_a1", "outcome_a2", "outcome_a3",
    "outcome_b1", "outcome_b2", "outcome_b3",
    "ev_a", "ev_b", "ev_diff",
    "var_a", "var_b", "var_diff",
    "loss_a", "loss_b", "loss_diff",
    "skew_a", "skew_b", "skew_diff",
]
gamble_df = pd.DataFrame(columns=columns)

num_gambles_per_ev_diff = 100
ev_diff = np.array([-20, -10, 0, 10, 20])
num_trials = np.array([40, 40, 160, 40, 40])
for i in tqdm(range(len(ev_diff))):
    counter = 0
    while True:
        # sample random gamble
        loss = RNG.uniform(-1000, 5, 2).astype(np.int32).round(-1)
        middle = RNG.uniform(5, 1000, 2).astype(np.int32).round(-1)
        gain = RNG.uniform(5, 1000, 2).astype(np.int32).round(-1)
        gamble_a = np.sort(np.array([loss[0], middle[0], gain[0]]))[::-1]
        gamble_b = np.sort(np.array([loss[1], middle[1], gain[1]]))[::-1]
        # get ev
        ev_a = (1/3) * np.sum(gamble_a)
        ev_b = (1/3) * np.sum(gamble_b)
        # get loss
        loss_a = np.min(gamble_a)
        loss_b = np.min(gamble_b)
        # get var
        var_a = np.var(gamble_a)
        var_b = np.var(gamble_b)
        # get skewness|
        skew_a = np.round(skew(gamble_a), 2)
        skew_b = np.round(skew(gamble_b), 2)
        # check if two outcomes are the same
        if (len(gamble_a) == len(np.unique(gamble_a))) and (len(gamble_b) == len(np.unique(gamble_b))):
            # check if all outcomes are non-zero
            if np.all(np.concatenate([gamble_a, gamble_b]) != 0):
                # check for equal ev
                if ev_a - ev_b == ev_diff[i]:
                    # check for loss and var confound
                    if loss_a < loss_b and np.var(gamble_a) < np.var(gamble_b):
                        gamble = [
                            {                            
                                "outcome_a1": gamble_a[0], "outcome_a2": gamble_a[1],
                                "outcome_a3": gamble_a[2], "outcome_b1": gamble_b[0],
                                "outcome_b2": gamble_b[1], "outcome_b3": gamble_b[2],
                                "ev_a": ev_a, "ev_b": ev_b, "ev_diff": ev_a - ev_b,
                                "var_a": var_a, "var_b": var_b, "var_diff": var_a - var_b,
                                "loss_a": loss_a, "loss_b": loss_b, "loss_diff": loss_a - loss_b,
                                "skew_a": skew_a, "skew_b": skew_b, "skew_diff": skew_a - skew_b
                            }
                        ]
                        gamble_df = pd.concat([gamble_df, pd.DataFrame(gamble)], ignore_index=True)
                        counter += 1
                        if counter == num_gambles_per_ev_diff:
                            break

gamble_df.to_csv("data/three_outcome_lotteries.csv", index=False)