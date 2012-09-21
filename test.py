import numpy as np

import hmm_mp

t = np.array([[0.5, 0.3, 0.2], [0.3, 0.3, 0.4], [0.2, 0.6, 0.2]])

e = np.array([[0.1, 0.2, 0.3], [0.5, 0.2, 0.1], [0.3, 0.1, 0.4], [0.1, 0.5, 0.2]])

init = np.array([0.3, 0.5, 0.2])

def gen_data(num, length, maximum):
    """Generate dataset as a list of integers."""
    return np.array([[np.random.randint(0, maximum)
                     for i in xrange(length)]
                  for n in xrange(num)])

data = gen_data(100, 50, 4)

h = hmm_mp.MultiProcessHMM(t, e, init, 2)
h.baum_welch(data, threshold=1e-6, pseudocounts=[1e-5, 1e-5, 1e-5])
