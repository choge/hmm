#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
#import ghmm   # not used now
import multiprocessing
import hmm

class MultiProcessHMM(hmm.HMM):
    """Implementation of HMM with multiprocessing.

    Using multiprocessing module to fasten calculation of estimation
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
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(i, tasks, results) for i in xrange(worker_num)]
        for w in workers:
            w.start()
        l_prev = 0
        for n in xrange(iter_limit):
            for i in xrange(len(observations)):
                tasks.put(Estimator(observations[i], self._t, self._e, self._i))
            tasks.join()
            gammas, xisums, cs = np.array(
                [results.get() for i in xrange(len(observations))]
            ).T
            ### do something
            l = self.maximize(gammas, xisums, cs, x_digits)
            if hmm.has_positive(pseudocounts):
                self.add_pseudocounts(pseudocounts)
            dif = l - l_prev
            print n, l, dif
            l_prev = l
            if n > 0 and dif < threshold:
                for i in xrange(worker_num):
                    tasks.put(None)
                break


class Worker(multiprocessing.Process):
    """Tasks for estimation step."""
    def __init__(self, number, tasks, results):
        """Requires a list of observations."""
        multiprocessing.Process.__init__(self)
        self.id_num = number
        self.tasks = tasks
        self.results = results
        print "Process (%d) has been made." % self.id_num

    def run(self):
        """Run calculation."""
        while True:
            next_task = self.tasks.get()
            if next_task is None:
                print "%d exiting..." % self.id_num
                self.tasks.task_done()
                break
            print "Proccess (%d) calculating estimations..." % self.id_num
            estimation = next_task()
            self.tasks.task_done()
            self.results.put(estimation)
        return

class Estimator(object):
    def __init__(self, x, t, e, i):
        self.x = x
        self.t = t
        self.e = e
        self.i = i

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

def split_data(x, num):
    """Split the data set x into num subsets."""
    data_num = len(x)
    num_per_subset = data_num / num
    frac = data_num % num
    subsets = {}
    splitted = 0
    for i in xrange(num):
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
    return np.array([xs[i] for i in xrange(num)]).T
