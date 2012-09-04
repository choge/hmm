#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
# Log Likelihood が下がってしまう！！！直す！！！！
# 直した！！！
# 原因はbetaの更新式のself._e[x[n]]だった(正解はself._e[x[n+1]])
#

import numpy as np
import ghmm

class HMM(object):
    """Implementation of hidden Markov model.

    An instance of this class has transition probabilities, emission
    probabilities and initial probabilities. There are some methods that
    calculates likelihood with given observations."""

    def __init__(self, transition, emission, initial):
        """Constructer method, which requires a priori probabilities.

        @params
        transition: numpy matrix class object
        emission:   numpy matrix class object
        initial:

        self._t: Transition probabilities. KxK array.
        self._e: Emission probabilities. KxM array.
        """
        self._t = transition
        self._e = emission
        self._i = initial
        self._K = len(initial)      # Number of classes
        self._M = len(emission)  # Number of symbols


    def baum_welch(self,
            observations,
            iter_limit=100,
            threshold=1e-5,
            pseudocounts=[0, 0, 0]):
        """Perform Baum-Welch algorithm.

        Requires a list of observations."""
        # Make 1-of-K representation
        # This is used to update emission probs in maximization step
        x_digits = [np.array(
                [[1 if x[n]==i else 0 for i in xrange(self._M)]
                    for n in xrange(len(x))]
                ).T
                for x in observations]
        l_prev = 0
        for n in xrange(iter_limit):
            gammas, xisums, cs = np.array([self.estimate(x) for x in observations]).T
            l = self.maximize(gammas, xisums, cs, x_digits)
            #if pseudocounts != [0, 0, 0]:  # At least one pseudocount is set
            if has_positive(pseudocounts):
                self.add_pseudocounts(pseudocounts)
            dif = l - l_prev
            print n, l, dif
            l_prev = l
            if n > 0 and dif < threshold:
                break

    def bw_onestep(self, observations, x_digits):
        """One step of baum welch algorithm."""
        gammas, xisums, cs = np.array([self.estimate(x) for x in observations]).T
        l = self.maximize(gammas, xisums, cs, x_digits)
        return gammas, xisums, cs, l

    def bw_single(self, x):
        """Perform Baum-Welch algorithm with a single observation"""
        # Make 1-of-K representation
        x_digits = np.array([[1 if x[n] == i else 0 for i in xrange(self._M)]
            for n in xrange(len(x))]).T
        l_prev = 0
        for n in xrange(100):
            g, xi, c = self.estimate(x)
            l = np.log(c).sum()
            self.maximize_one(g, xi, c, x_digits)
            dif = l - l_prev
            print n, l, dif
            l_prev = l

    def estimate(self, observation):
        """Calculate alpha"""
        x = observation
        N = len(x)
        # \hat{alpha}: p(z_n | x_1, ..., x_n)
        alpha = np.zeros([N, self._K], float)
        alpha[0] = self._i * self._e[x[0]]
        alpha[0] /= alpha[0].sum()
        beta  = np.zeros([N, self._K], float)
        beta[-1] = 1.0
        c = np.zeros([N], float)
        c[0] = alpha[0].sum()
        # Calculate Alpha
        for n in xrange(1, N):
            a = self._e[x[n]] * np.dot(alpha[n -1], self._t)
            c[n] = a.sum()
            alpha[n] = a / c[n]
        # Calculate Beta
        for n in xrange(N - 2, -1, -1):
            beta[n] = np.dot(beta[n + 1] * self._e[x[n + 1]], self._t.T) / c[n + 1]
        gamma = alpha * beta
        xisum = sum(
            np.outer(alpha[n-1], self._e[x[n]] * beta[n]) / c[n] for n in xrange(1, N)
            ) * self._t
        return gamma, xisum, c


    def maximize(self, gammas, xisums, cs, x_digits):
        """Maximization step of EM algorithm.

        @param x_digits A matrix of 1-of-K. DxN dimension"""
        log_likelihood = sum(np.log(c).sum() for c in cs)
        R = len(gammas)
        sumxisums = sum(xisums)

        gammas_init = [gammas[r][0] for r in xrange(R)]
        self._i = sum(gammas_init) / sum(gammas_init[r].sum() for r in xrange(R))
        self._t = (sumxisums.T / sumxisums.sum(1)).T
        self._e = sum(np.dot(x_digits[i], gammas[i]) for i in xrange(R))
        self._e /= sum(gammas[i].sum(0) for i in xrange(R))
        return log_likelihood

    def maximize_one(self, gamma, xisum, c, x_digits):
        """Maximization with a single observation."""
        log_likelihood = np.log(c).sum()
        self._i = gamma[0] / gamma[0].sum()
        self._t = (xisum.T / xisum.sum(1)).T
        self._e = np.dot(x_digits, gamma) / gamma.sum(0)
        return log_likelihood

    def viterbi(self, observations):
        """Decode observations."""
        pass

    def normalize_transition(self):
        """Normalize transition probabilities to 1."""
        self._t /= self._t.sum(1)[:, np.newaxis]

    def normalize_emission(self):
        """Normalize emission probabilities to 1."""
        self._e /= self._e.sum(0)

    def normalize_initial(self):
        """Normalize initial probabilities to 1."""
        self._i /= self._i.sum()

    def add_pseudocounts(self, pseudocounts):
        """Add pseudocounts to avoid zero probabilities.

        @param pseudocounts  a list of pseudocounts, which consists of
                             pseudocounts in float or double for transition
                             probs, emission probs and initial probs.
        """
        if pseudocounts[0] > 0:
            self._t += pseudocounts[0]
            self.normalize_transition()
        if pseudocounts[1] > 0:
            self._e += pseudocounts[1]
            self.normalize_emission()
        if pseudocounts[2] > 0:
            self._i += pseudocounts[2]
            self.normalize_initial()

def convert_ghmm(g):
    """Currently only discrete hmms can be converted.."""
    t = np.array([[g.getTransition(i, j) for j in xrange(g.N)] for i in xrange(g.N)])
    e = np.array([g.getEmission(i) for i in xrange(g.N)]).T
    i = np.array([g.getInitial(j) for j in xrange(g.N)])
    return HMM(t, e, i)

def has_positive(l):
    """Return if there are positive number in a list."""
    return any([p > 0 for p in l])

def convert2ghmm(h):
    """Convert an HMM object to a GHMM object."""
    alphabets = ghmm.Alphabet("ACDEFGHIKLMNPQRSTVWY")
    g = ghmm.HMMFromMatrices(alphabets, ghmm.DiscreteDistribution(alphabets),
            h._t, h._e.T, h._i)
    return g
