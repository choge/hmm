#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#
#

import numpy as np
import ghmm
import logging
import pickle

class HMM(object):
    """A simple implementation of hidden Markov model.

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
        self._e: Emission probabilities. MxK array.
        """
        self._t = transition
        self._e = emission
        self._i = initial
        self._K = len(initial)      # Number of classes
        self._M = len(emission)  # Number of symbols
        logging.basicConfig(format='[%(asctime)s] %(message)s')


    def baum_welch(self,
            observations,
            iter_limit=100,
            threshold=1e-5,
            pseudocounts=[0, 0, 0],
            do_logging=True,
            **args):
        """Perform Baum-Welch algorithm.

        Requires a list of observations."""
        # Make 1-of-K representation
        # This is used to update emission probs in maximization step
        if do_logging:
            logging.info("Baum Welch Algorithm started.")
        x_digits = [np.array(
                [[1 if x[n]==i else 0 for i in xrange(self._M)]
                    for n in xrange(len(x))]
                ).T
                for x in observations]
        if do_logging is True:
            logging.info("1 of K representation has been made.")
        l_prev = 0
        for n in xrange(iter_limit):
            if do_logging:
                logging.info("Estimation step began.")
            ## gammas: array of R elements. Each element is also an array
            ##        (matrix) of N x K (N is length)
            gammas, xisums, cs = np.array([self.estimate(x) for x in observations]).T
            if do_logging:
                logging.info("Estimation step ended.")
            l = self.maximize(gammas, xisums, cs, x_digits)
            if np.isnan(l):
                logging.error("Log Likelihood is nan. Here are some information:")
                logging.error("Parameters are pickled into params.pickle")
                pickle.dump({'t': self._t, 'e': self._e, 'i': self._i, 'xisums': xisums},
                            open('params.pickle', 'wb'))
                raise ValueError("nan detected. The scaling factors are: %s" % cs)
            #if pseudocounts != [0, 0, 0]:  # At least one pseudocount is set
            if has_positive(pseudocounts):
                self.add_pseudocounts(pseudocounts)
            dif = l - l_prev
            if do_logging:
                logging.info("iter: " + str(n))
                logging.info("Likelihood: " + str(l))
                logging.info("Delta: " + str(dif))
            l_prev = l
            pickle.dump({'t': self._t, 'e': self._e, 'i': self._i},
                        open('params.pickle', 'wb'))
            if n > 0 and dif < threshold:
                break


    def estimate(self, x, want_alpha=False, **args):
        """Calculate alpha

        @param x  is an observation, which should be a list of integers.
        @param want_alpha  is a boolean,"""
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
        if want_alpha:
            return alpha, c
        # Calculate Beta
        for n in xrange(N - 2, -1, -1):
            beta[n] = np.dot(beta[n + 1] * self._e[x[n + 1]], self._t.T) / c[n + 1]
        gamma = alpha * beta
        xisum = sum(
            np.outer(alpha[n-1], self._e[x[n]] * beta[n]) / c[n] for n in xrange(1, N)
            ) * self._t
        return gamma, xisum, c


    def maximize(self, gammas, xisums, cs, x_digits, do_logging=True,
                 del_state=0):
        """Maximization step of EM algorithm.

        @param x_digits A matrix of 1-of-K representation. DxN dimension
        @del_state  a threshold for deletion of invalid states"""
        if do_logging:
            logging.info("Maximization step began.")
        log_likelihood = sum(np.log(c).sum() for c in cs)
        # R: number of sequences.
        # r: identifier of sequences (integer, 0 .. R-1)
        R = len(gammas)
        sumxisums = sum(xisums)
        sumxisums, gammas = self.delete_invalid_states(sumxisums, gammas, del_state)

        #gammas_init = [gammas[r][0] for r in xrange(R)]
        #self._i = sum(gammas_init) / sum(gammas_init[r].sum() for r in xrange(R))
        self._i = gammas[:, 0].sum(0) / gammas[:, 0].sum()
        self._t = (sumxisums.T / sumxisums.sum(1)).T
        self._e = sum(np.dot(x_digits[i], gammas[i]) for i in xrange(R))
        self._e /= sum(gammas[i].sum(0) for i in xrange(R))
        if do_logging:
            logging.info("Maximization step ended.")
        return log_likelihood

    def delete_invalid_states(self, sumxisums, gammas, threshold=0):
        """Check if sumxisums contain all zero rows.

        @param sumxisums  A matrix of \simga \simga xi.
        @param threshold  float """
        # check if there are all zero columns
        # (a state with such column cannot be reached from any states)
        valid = sumxisums.sum(0) > threshold
        if np.all(valid):
            return sumxisums, gammas
        valid_cross = np.outer(valid, valid)
        new_K = valid.sum()

        # slice transition probabilities
        self._t = np.reshape(self._t[valid_cross], (new_K, new_K))
        # slice emission probabilities
        self._e = self._e[:, valid]
        # slice initial probabilities
        self._i = self._i[valid]
        # set the new number of states
        self._K = new_K

        return sumxisums[valid_cross], gammas[:, :, valid]

    def maximize_one(self, gamma, xisum, c, x_digits):
        """Maximization with a single observation."""
        log_likelihood = np.log(c).sum()
        self._i = gamma[0] / gamma[0].sum()
        self._t = (xisum.T / xisum.sum(1)).T
        self._e = np.dot(x_digits, gamma) / gamma.sum(0)
        return log_likelihood

    def viterbi(self, x, do_logging=True, **args):
        """Decode observations."""
        if do_logging:
            logging.info("Started calculating Viterbi path.")
        N = len(x)
        alpha, c = self.estimate(x, want_alpha=True)
        alpha, c = np.log(alpha), np.log(c)
        alpha = np.array([alpha[n] + c[:n+1].sum() for n in xrange(N)])
        # ^ log \alpha (Not \hat{\alpha})
        logt, loge = np.log(self._t), np.log(self._e)
        omega = np.log(self._i) + loge[x[0]]
        # ^ omega: probability at current position (at position 0 here)
        path = np.array([[i for i in xrange(self._K)] for n in xrange(N)])
        # calculate the most probable path at each position of the observation
        for n in xrange(1, N):
            prob = loge[x[n]] + omega + logt.T
            # NxN matrix  row: transition from, col: transition to
            omega = np.max(prob, axis=1)
            path[n] = np.argmax(prob, axis=1)
        # Seek the most likely route (From N-1 to 0)
        route = [np.argmax(omega)]
        for n in xrange(N - 2, -1, -1):
            route.append(path[n][route[-1]])
        if do_logging:
            logging.info("Finished calculating Viterbi path.")
        return route[::-1], omega.max()

    def sample(self, length):
        """Sample the sequence for specified length"""
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
        """Add pseudocounts (Laplace correction).

        @param pseudocounts  a list of pseudocounts, which consists of
                             pseudocounts in float or double for transition
                             probs, emission probs and initial probs.
        """
        logging.info("Adding pseudocounts...")
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
