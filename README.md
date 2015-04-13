hmm
=====================
An implementation of hidden Markov model in Python.  
The Baum-welch and Viterbi algorithms for discrete emissions
are implemented so far.
It is a generic implementation, so one may need to write
some wrapper to apply some real data.


Usage
-----
Configure three parameters, which are transition, emission and initial
probabilities. It is assumed that these are instances of numpy.array 
class. All emissions should be represented as integer, which are the indices
of emission probability matrix.

    import numpy as np
    import hmm

    t = np.array([[0.55, 0.45], [0.4, 0.6]])
    e = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
    i = np.array([0.4, 0.6])
    h = hmm.HMM(t, e, i)

    observations = np.array([[2, 2, 2, 2, 1, 0, 0, 1, 1, 1, 0, 2],
            [2, 2, 2, 1, 2, 1, 0, 0, 0, 1, 1, 2, 1, 2],
            [1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 2, 0]])
    h.baum_welch(observations, iter_limit=1000, 
            threshold=1e-5, pseudocounts=[0, 1e-4, 0])

    path, l = h.viterbi([2, 2, 1, 0, 0, 2, 1, 1, 1, 0, 0])

Using multiprocessing module, you can speed up the calculation.

    import numpy as np
    import hmm_mp
    
    ... # Configure parameters t, e, i

    h = hmm_mp.MultiProcessHMM(t, e, i, worker_num=4)

    h.baum_welch(observations)

This module also offers importing an XML file of GHMM.

    import hmm

    (t, e, i) = hmm.load_ghmmxml('filename.xml')
    h = hmm.MultiProcessHMM(t, e, i, 4)

Methods
-------
+ `baum_welch` :
A basic Baum-Welch algorithm. It requires a list of observations, where 
each observation is represented as a list of integers.
