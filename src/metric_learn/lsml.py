"""
Liu et al.
"Metric Learning from Relative Comparisons by Minimizing Squared Residual".
ICDM 2012.

Adapted from https://gist.github.com/kcarnold/5439917
Paper: http://www.cs.ucla.edu/~weiwang/paper/ICDM12.pdf
"""

from __future__ import print_function, absolute_import

import numpy as np
import scipy.linalg
from six.moves import xrange
from sklearn.utils.validation import check_array, check_X_y

from .base_metric import BaseMetricLearner
from .constraints import Constraints


class LSML(BaseMetricLearner):
    def __init__(self, tol=1e-3, max_iter=1000, prior=None, verbose=False):
        """Initialize LSML.

        Parameters
        ----------
        tol : float, optional
        max_iter : int, optional
        prior : (d x d) matrix, optional
                guess at a metric [default: inv(covariance(X))]
        verbose : bool, optional
                if True, prints information while learning
        """
        self.prior = prior
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _prepare_inputs(self, X, constraints, weights):
        self.X_ = X = check_array(X)
        a, b, c, d = constraints
        self.vab_ = X[a] - X[b]
        self.vcd_ = X[c] - X[d]
        if self.vab_.shape != self.vcd_.shape:
            raise ValueError('Constraints must have same length')
        if weights is None:
            self.w_ = np.ones(self.vab_.shape[0])
        else:
            self.w_ = weights
        self.w_ /= self.w_.sum()  # weights must sum to 1
        if self.prior is None:
            self.M_ = np.cov(X, rowvar=False)
            if self.M_.ndim == 0:
                self._M = 1. / self.M_
            else:
                self.M_ = np.linalg.inv(self.M_)
        else:
            self.M_ = self.prior

    def metric(self):
        return self.M_

    def fit(self, X, constraints, weights=None):
        """Learn the LSML model.

        Parameters
        ----------
        X : (n x d) data matrix
                each row corresponds to a single instance
        constraints : 4-tuple of arrays
                (a,b,c,d) indices into X, such that d(X[a],X[b]) < d(X[c],X[d])
        weights : (m,) array of floats, optional
                scale factor for each constraint
        """
        self._prepare_inputs(X, constraints, weights)
        prior_inv = scipy.linalg.inv(self.M_)
        s_best = self._total_loss(self.M_, prior_inv)
        step_sizes = np.logspace(-10, 0, 10)
        if self.verbose:
            print('initial loss', s_best)
        for it in xrange(1, self.max_iter + 1):
            grad = self._gradient(self.M_, prior_inv)
            grad_norm = scipy.linalg.norm(grad)
            if grad_norm < self.tol:
                break
            if self.verbose:
                print('gradient norm', grad_norm)
            M_best = None
            for step_size in step_sizes:
                step_size /= grad_norm
                new_metric = self.M_ - step_size * grad
                w, v = scipy.linalg.eigh(new_metric)
                new_metric = v.dot((np.maximum(w, 1e-8) * v).T)
                cur_s = self._total_loss(new_metric, prior_inv)
                if cur_s < s_best:
                    l_best = step_size
                    s_best = cur_s
                    M_best = new_metric
            if self.verbose:
                print('iter', it, 'cost', s_best, 'best step', l_best * grad_norm)
            if M_best is None:
                break
            self.M_ = M_best
        else:
            if self.verbose:
                print("Didn't converge after", it, "iterations. Final loss:", s_best)
        self.n_iter_ = it
        return self

    def _comparison_loss(self, metric):
        dab = np.sum(self.vab_.dot(metric) * self.vab_, axis=1)
        dcd = np.sum(self.vcd_.dot(metric) * self.vcd_, axis=1)
        violations = dab > dcd
        return self.w_[violations].dot((np.sqrt(dab[violations]) - np.sqrt(dcd[violations])) ** 2)

    def _total_loss(self, metric, prior_inv):
        closs = self._comparison_loss(metric)
        rloss = _regularization_loss(metric, prior_inv)
        regweight = 1.0 / (metric.shape[0] * metric.shape[1])
        return (closs + regweight * rloss)

    def _gradient(self, metric, prior_inv):
        regweight = 1.0 / (metric.shape[0] * metric.shape[1])
        dMetric = regweight * (prior_inv - scipy.linalg.inv(metric))
        dabs = np.sum(self.vab.dot(metric) * self.vab, axis=1)
        dcds = np.sum(self.vcd.dot(metric) * self.vcd, axis=1)
        violations = dabs > dcds
        # TODO: vectorize
        for vab, dab, vcd, dcd in zip(self.vab[violations], dabs[violations],
                                      self.vcd[violations], dcds[violations]):
            dMetric += ((1 - np.sqrt(dcd / dab)) * np.outer(vab, vab) +
                        (1 - np.sqrt(dab / dcd)) * np.outer(vcd, vcd))
        return dMetric


def _regularization_loss(metric, prior_inv):
    trace = np.trace(metric * prior_inv)
    sign, logdet = np.linalg.slogdet(metric)
    logdet = sign * logdet
    return trace + logdet


class LSML_Supervised(LSML):
    def __init__(self, tol=1e-3, max_iter=1000, prior=None, num_labeled=np.inf,
                 num_constraints=None, weights=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        tol : float, optional
        max_iter : int, optional
        prior : (d x d) matrix, optional
                guess at a metric [default: covariance(X)]
        num_labeled : int, optional
                number of labels to preserve for training
        num_constraints: int, optional
                number of constraints to generate
        weights : (m,) array of floats, optional
                scale factor for each constraint
        verbose : bool, optional
                if True, prints information while learning
        """
        LSML.__init__(self, tol=tol, max_iter=max_iter, prior=prior,
                      verbose=verbose)
        self.num_labeled = num_labeled
        self.num_constraints = num_constraints
        self.weights = weights

    def fit(self, X, y, random_state=np.random):
        """Create constraints from labels and learn the LSML model.

        Parameters
        ----------
        X : (n x d) matrix
                Input data, where each row corresponds to a single instance.

        y : (n) array-like
                Data labels.

        random_state : numpy.random.RandomState, optional
                If provided, controls random number generation.
        """
        X, y = check_X_y(X, y)
        num_constraints = self.num_constraints
        if num_constraints is None:
            num_classes = len(np.unique(y))
            num_constraints = 20 * num_classes ** 2

        c = Constraints.random_subset(y, self.num_labeled,
                                      random_state=random_state)
        pairs = c.positive_negative_pairs(num_constraints, same_length=True,
                                          random_state=random_state)
        return LSML.fit(self, X, pairs, weights=self.weights)
