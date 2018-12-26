import numpy as np
import scipy.linalg as sla


A = np.zeros((5, 5))

time_step = 1e-2

for t in np.arange(0, 10, time_step):
    alp = -1
    bet = 0.2

    v = np.random.rand(5)
    next_A = A + time_step * (alp * A + bet * np.outer(v, v))

    A = next_A
