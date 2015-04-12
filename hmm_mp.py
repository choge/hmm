#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
#import ghmm   # not used now
import multiprocessing
from . import hmm
import logging

class MultiProcessHMM(hmm.HMM):
    """Implementation of HMM with multiprocessing.

    Using multiprocessing module to fasten calculation of estimation
    step."""
    def __init__(self, t, e, i, worker_num=2):
        """Constructer."""
        hmm.HMM.__init__(self, t, e, i)
        self.worker_num = worker_num

    def baum_welch(self,
                   observations,
                   iter_limit=1000,
                   threshold=1e-5,
                   pseudocounts=[0, 0, 0],
                   worker_num=None):
        """Perform Baum-Welch algorithm.

        Require a list of observations."""
        worker_num = worker_num if worker_num is not None else self.worker_num
        x_digits = [ np.array(
            [[x[n] == i for i in range(self._M)]
                for n in range(len(x))] ).T
            for x in observations]
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(i, tasks, results) for i in range(worker_num)]
        for w in workers:
            w.start()
            logging.info("Starting process %d...", w.id_num)
        l_prev = 0
        for n in range(iter_limit):
            for i in range(len(observations)):
                tasks.put(Estimator(observations[i], self._t, self._e, self._i, i))
            tasks.join()
            estimations = {}
            for i in range(len(observations)):
                estimations.update(results.get())
            gammas, xisums, cs = np.array(
                [estimations[i] for i in range(len(observations))]
            ).T
            ### do something
            l = self.maximize(gammas, xisums, cs, x_digits)
            if hmm.has_positive(pseudocounts):
                self.add_pseudocounts(pseudocounts)
            dif = l - l_prev
            logging.info("iter: %d, likelihood=%f, delta=%f", n, l, dif)
            l_prev = l
            if n > 0 and dif < threshold:
                break
        for i in range(worker_num):
            tasks.put(None)


class Worker(multiprocessing.Process):
    """Tasks for estimation step."""
    def __init__(self, number, tasks, results):
        """Requires a list of observations."""
        multiprocessing.Process.__init__(self)
        self.id_num = number
        self.tasks = tasks
        self.results = results

    def run(self):
        """Run calculation."""
        while True:
            next_task = self.tasks.get()
            if next_task is None:
                self.tasks.task_done()
                break
            estimation = next_task()
            self.tasks.task_done()
            self.results.put(estimation)
        return

class Estimator(object):
    def __init__(self, x, t, e, i, seq_number):
        self.x = x
        self.t = t
        self.e = e
        self.i = i
        self.seq_number = seq_number

    def __call__(self):
        """Calculate alpha
        """
        x = self.x
        t = self.t
        e = self.e
        i = self.i
        N = len(x)
        K = len(t[0])
        # \hat{alpha}: p(z_n | x_1, ..., x_n)
        alpha = np.zeros([N, K], float)
        alpha[0] = i * e[x[0]]
        alpha[0] /= alpha[0].sum()
        beta  = np.zeros([N, K], float)
        beta[-1] = 1.0
        c = np.zeros([N], float)
        c[0] = 1.0
        # Calculate Alpha
        for n in range(1, N):
            a = e[x[n]] * np.dot(alpha[n -1], t)
            c[n] = a.sum()
            alpha[n] = a / c[n]
        # Calculate Beta
        for n in range(N - 2, -1, -1):
            beta[n] = np.dot(beta[n + 1] * e[x[n + 1]], t.T) / c[n + 1]
        gamma = alpha * beta
        xisum = sum(
            np.outer(alpha[n-1], e[x[n]] * beta[n]) / c[n] for n in range(1, N)
            ) * t
        return {self.seq_number: [gamma, xisum, c]}
        #alpha = np.zeros((N, K))
        #c = np.ones(N)   # c[0] = 1
        #a = i * e[x[0]]
        #alpha[0] = a / a.sum()
        #for n in xrange(1, N):
        #    a = e[x[n]] * np.dot(alpha[n-1], t)
        #    c[n] = z = a.sum()
        #    alpha[n] = a / z

        #beta = np.zeros((N, K))
        #beta[N-1] = 1
        #for n in xrange(N-1, 0, -1):
        #    beta[n-1] = np.dot(t, beta[n] * e[x[n]]) / c[n]

        #gamma = alpha * beta

        #xi_sum = np.outer(alpha[0], e[x[1]] * beta[1]) / c[1]
        #for n in range(2, N):
        #    xi_sum += np.outer(alpha[n-1], e[x[n]] * beta[n]) / c[n]
        #xi_sum *= t
        #return {self.seq_number: [gamma, xi_sum, c]}

def split_data(x, num):
    """Split the data set x into num subsets."""
    data_num = len(x)
    num_per_subset = data_num / num
    frac = data_num % num
    subsets = {}
    splitted = 0
    for i in range(num):
        start = splitted
        end   = start + num_per_subset
        if frac > 0:
            end += 1
            frac -= 1
        splitted = end
        subsets[i] = x[start:end+1]
    return subsets

def merge_results(xs, num):
    """Merge estimation into one numpy.array object."""
    return np.array([xs[i] for i in range(num)]).T
