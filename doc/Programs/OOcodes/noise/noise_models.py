import numpy as np
import random

def bit_flip(state, p):
    noisy_state = state.copy()
    for i in range(len(state)):
        if random.random() < p:
            flipped = i ^ 1  # flips qubit 0
            noisy_state[flipped] += state[i]
            noisy_state[i] = 0
    return noisy_state / np.linalg.norm(noisy_state)

def depolarizing(state, p):
    d = len(state)
    noisy_state = (1-p) * state + p/d * np.ones(d)
    return noisy_state / np.linalg.norm(noisy_state)

