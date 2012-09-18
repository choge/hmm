#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
#import ghmm   # not used now
import multiprocessing
import hmm

class MultiProcessHMM(hmm.HMM):
    """Implementation of HMM with multiprocessing.

    Using multiprocessing module to fasten calculation in estimation
    step."""
    def baum_welch(self,
                   observations,
                   iter_limit=1000,
                   threshold=1e-5,
                   pseudocounts=[0, 0, 0],
                   worker_num=2):
        """Perform Baum-Welch algorithm.

        Require a list of observations."""
        x_digits = [ np.array(
            [[1 if x[n] == i else 0 for i in xrange(self._M)]
                for n in xrange(len(x))] ).T
            for x in observations]
        l_prev = 0
        p = [multiprocessing.Process(
            observations[n:len(observations):worker_num])
            for n in xrange(worker_num)]
        for n in xrange(iter_limit):
            gammas, xisums, cs = None, None, None
            ### do something
            l = self.maximize(gammas, xisums, cs, x_digits)
            if hmm.has_positive(pseudocounts):
                self.add_pseudocounts(pseudocounts)
            dif = l - l_prev
            print n, l, dif
            l_prev = l
            if n > 0 and dif < threshold:
                break


class Estimator(multiprocessing.Process):
    """Tasks for estimation step."""
    def __init__(self, xs):
        """Requires a list of observations."""
        multiprocessing.Process.__init__(self)
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()
        self.xs = xs

def estimate(x, t, e, i):
    """Calculate alpha

    @param x  is an observation, which should be a list of integers.
    @param t  is a transition matrix, which is a numpy.array class.
    @param e  is an emission matrix, DxK dimension.
              (where D is number of symbols and K is number of classes.
    @param i  is an initial probabilities in a numpy.array object.
    """
    N = len(x)
    K = len(t[0])
    # \hat{alpha}: p(z_n | x_1, ..., x_n)
    alpha = np.zeros([N, K], float)
    alpha[0] = i * e[x[0]]
    alpha[0] /= alpha[0].sum()
    beta  = np.zeros([N, K], float)
    beta[-1] = 1.0
    c = np.zeros([N], float)
    c[0] = alpha[0].sum()
    # Calculate Alpha
    for n in xrange(1, N):
        a = e[x[n]] * np.dot(alpha[n -1], t)
        c[n] = a.sum()
        alpha[n] = a / c[n]
    # Calculate Beta
    for n in xrange(N - 2, -1, -1):
        beta[n] = np.dot(beta[n + 1] * e[x[n + 1]], t.T) / c[n + 1]
    gamma = alpha * beta
    xisum = sum(
        np.outer(alpha[n-1], e[x[n]] * beta[n]) / c[n] for n in xrange(1, N)
        ) * t
    return gamma, xisum, c

