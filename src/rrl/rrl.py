from datetime import datetime

import numpy as np
import pyspark
import utils
from six.moves import xrange
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class RRL():
    def __init__(self, tol=1e-10, epochs=100, regularization_alpha=1, verbose=False):
        """Initialize RRL.
        Parameters
        ----------
        tol : float, optional
        epochs : int, optional
        verbose : bool, optional
                if True, prints information while learning
        """
        self.alpha = regularization_alpha
        self.tol = tol
        self.epochs = epochs
        self.verbose = verbose

    def _prepare_inputs(self, vectors, relscores):
        X, constraints, wordpairs = utils.get_train_parameters(relscores, vectors)
        self.vectors = vectors
        self.prep_eval_dfs = {
            evaldfname: utils.prepare_dataset(utils.load_eval_df(evaldfname), self.vectors) for evaldfname in
            ["ws353", "ws353rel", "ws353sim", "men", "mturk", "mturk771", "simlex999"]
        }
        self.X_ = X
        self.constraints = constraints
        self.wordpairs = wordpairs
        self.M_ = np.identity(self.X_.shape[1])

    def metric(self):
        return self.M_

    def transform(self):
        return {entry[0]: np.linalg.cholesky(self.M_).T.dot(entry[1]) for entry in self.vectors.items()}

    def fit(self, vectors, relscores, learning_rate=0.05, batchsize=50,
            eval_steps=False, save_steps=False, output_dir=None):
        """Learn the LSML model.
        Parameters
        ----------
        X : (n x d) word embedding matrix
                each row corresponds to a single word embedding
        vectors : matrix (2 x c)
                (wp1, wp2) indices into wordpairs, such that rel(wp1) > rel(wp2)
        relscores : matrix (2 x p)
                (w1, w2) with two word indices into X
        learning_rate : 1-dimensional numpy array, default None
                step sizes for the gradient descent step. If None, np.logspace(-2, 1, 10) is the default value.
        eval_steps : evaluate metric after each epoch on a set of word similarity datasets
        save_steps : not yet implemented. should save the model (aka the transformation matrix) after each step.
        """
        self._prepare_inputs(vectors, relscores)
        oldloss = self._loss(self.M_)
        if self.verbose:
            print('initial loss', oldloss)
        # iterations
        for epoch in xrange(1, self.epochs + 1):
            # shuffle constraints
            violations, cosines, _ = self._violations(self.M_)
            current_constraints = np.hstack([self.constraints[violations], cosines[violations]])
            np.random.shuffle(current_constraints)
            for constraint_batch in batch(current_constraints, n=batchsize):
                transformed = np.dot(self.X_, np.linalg.cholesky(self.M_))
                grad = self.batchgradient(self.M_, constraint_batch, transformed)
                grad_norm = np.linalg.norm(grad)
                self.M_ = self.M_ - learning_rate * grad / grad_norm
                self.M_ = _make_psd(self.M_)
            if eval_steps:
                print("epoch", epoch, "loss", self._loss(self.M_),
                      [(dfname, utils.evaluate(self.prep_eval_dfs[dfname], metric=self.M_)) for dfname in
                       self.prep_eval_dfs])
            if save_steps and output_dir:
                if output_dir[-1] != "/":
                    output_dir += "/"
                import pickle
                pickle.dump(self.M_, open(output_dir + "M_" + str(epoch) + ".pkl", "wb"))
                pickle.dump(("epoch", epoch, "loss", self._loss(self.M_),
                             [(dfname, utils.evaluate(self.prep_eval_dfs[dfname], metric=self.M_)) for dfname in
                              self.prep_eval_dfs]), open(output_dir + "state_" + str(epoch) + ".pkl", "wb"))
            if abs(oldloss - self._loss(self.M_)) < self.tol:
                break
            oldloss = self._loss(self.M_)
        else:
            print("Didn't converge after", epoch, "iterations. Final loss:", self._loss(self.M_))
        print("Finished after ", epoch, "iterations. Final loss:", self._loss(self.M_))
        # turn off any running spark context
        pyspark.SparkContext.getOrCreate().stop()
        return self

    def _violations(self, metric):
        transformed = normalize(self.X_.dot(np.linalg.cholesky(metric)))
        cosines = np.diag(cosine_similarity(transformed[self.wordpairs[:, 0]],
                                            transformed[self.wordpairs[:, 1]]))[
            self.constraints]
        # this then looks for cosab - coscd < 0
        vio = cosines[:, 0] - cosines[:, 1] < 0
        return vio, cosines, transformed

    def _loss(self, metric):
        vio, cosines, _ = self._violations(metric)
        closs = np.sum(((cosines[vio, 0] - cosines[vio, 1]) ** 2))  # / len(cosines)
        rloss = _regularization_loss(metric)
        if self.verbose:
            print(str(datetime.now()) + "\tviolations: " + str(len(cosines[vio])) + "\tCLoss: " + str(
                closs) + "\tRLoss: " + str(rloss))
        return closs + rloss

    def batchgradient(self, metric, constraint_batch, transformed):
        # unused
        dMetric = (np.identity(metric.shape[0]) - np.linalg.inv(metric))  # / (metric.shape[0] ** 2)
        if self.verbose:
            print(str(datetime.now()) + "\tCalculating gradient for " + str(len(constraint_batch)) + " violations")
        sc = pyspark.SparkContext.getOrCreate()
        entries = sc.parallelize(zip(range(len(constraint_batch)), constraint_batch))
        Xbc = sc.broadcast(self.X_)
        wordpairsbc = sc.broadcast(self.wordpairs)
        transformedbc = sc.broadcast(transformed)
        clossgradient = entries.mapValues(
            lambda entry: get_loss_gradient(entry, Xbc, wordpairsbc, transformedbc)).values().sum()  # / len(cosines)
        dMetric += clossgradient
        return dMetric


def _regularization_loss(metric):
    trace = np.trace(metric)
    sign, logdet = np.linalg.slogdet(metric)
    logdet = sign * logdet
    return (trace + logdet)  # / (metric.shape[0] ** 2)


def _make_psd(metric, tol=1e-8):
    w, v = np.linalg.eigh(metric)
    return v.dot((np.maximum(w, tol) * v).T)


def makeouter(X, col1, col2, transformed):
    return np.outer(X[col1], X[col2])\
           / (np.linalg.norm(transformed[col1]) * np.linalg.norm(transformed[col2]))


def get_loss_gradient(entry, X, wordpairs, transformed):
    wp1, wp2, cosab, coscd = entry
    wp1 = int(wp1)
    wp2 = int(wp2)
    vavbT = makeouter(X.value, wordpairs.value[wp1][0], wordpairs.value[wp1][1], transformed.value)
    vavaTvbvbT = makeouter(X.value, wordpairs.value[wp1][0], wordpairs.value[wp1][0], transformed.value) \
                 + makeouter(X.value, wordpairs.value[wp1][1], wordpairs.value[wp1][1], transformed.value)
    vcvdT = makeouter(X.value, wordpairs.value[wp2][0], wordpairs.value[wp2][1], transformed.value)
    vcvcTvdvdT = makeouter(X.value, wordpairs.value[wp2][0], wordpairs.value[wp2][0], transformed.value) \
                 + makeouter(X.value, wordpairs.value[wp2][1], wordpairs.value[wp2][1], transformed.value)
    return (coscd - cosab) * (vcvdT - vcvcTvdvdT * coscd - (vavbT - vavaTvbvbT * cosab))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
