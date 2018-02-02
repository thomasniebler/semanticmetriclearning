from datetime import datetime

import numpy as np
from six.moves import xrange
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import rrl.utils


class RRL():
    def __init__(self, tol=1e-3, max_iter=1000, verbose=False):
        """Initialize RRL.
        Parameters
        ----------
        tol : float, optional
        max_iter : int, optional
        verbose : bool, optional
                if True, prints information while learning
        """
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _prepare_inputs(self, vectors, relscores):
        X, constraints, wordpairs = rrl.utils.get_train_parameters(relscores, vectors)
        self.vectors = vectors
        self.prep_eval_dfs = {
            evaldfname: rrl.utils.prepare_dataset(rrl.utils.load_eval_df(evaldfname), self.vectors) for evaldfname in
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

    def fit(self, vectors, relscores, step_sizes=None,
            eval_steps=False, save_steps=False, outputpath=None):
        """Learn the LSML model.
        Parameters
        ----------
        X : (n x d) word embedding matrix
                each row corresponds to a single word embedding
        constraints : matrix (2 x c)
                (wp1, wp2) indices into wordpairs, such that rel(wp1) > rel(wp2)
        wordpairs : matrix (2 x p)
                (w1, w2) with two word indices into X
        step_sizes : 1-dimensional numpy array, default None
                step sizes for the gradient descent step. If None, np.logspace(-2, 1, 10) is the default value.
        """
        if step_sizes is None:
            step_sizes = np.logspace(-2, 1, 10)
        self._prepare_inputs(vectors, relscores)
        # this is only for plotting and debugging purposes
        self._Ms = [self.M_]
        s_best = self._loss(self.M_)
        if self.verbose:
            print('initial loss', s_best)
        # iterations
        for it in xrange(1, self.max_iter + 1):
            grad = self._gradient(self.M_)
            grad_norm = np.linalg.norm(grad)
            if self.verbose:
                print(str(datetime.now()) + "\tGradient done. Norm: " + str(grad_norm))
            if grad_norm < self.tol:
                if self.verbose:
                    print(str(datetime.now()) + "\tgradient norm is lower than tolerance ")
                break
            M_best = None
            l_best = 100
            # TODO: realize that by applying a "min" step, as we have to calculate
            # each step anyway. And if we are convex, we can even break after finding a minimum.
            for step_size in step_sizes:
                if self.verbose:
                    print(str(datetime.now()) + "\tStep size: " + str(step_size))
                step_size /= grad_norm
                new_metric = self.M_ - step_size * grad
                new_metric = _make_psd(new_metric)
                step_loss = self._loss(new_metric)
                if step_loss < s_best:
                    l_best = step_size
                    s_best = step_loss
                    M_best = new_metric
            print('iter', it, 'cost', s_best, 'best step', l_best, 'gradient norm', grad_norm)
            if eval_steps and M_best is not None:
                print([(dfname, rrl.utils.evaluate(self.prep_eval_dfs[dfname], metric=M_best)) for dfname in
                       self.prep_eval_dfs])
            if M_best is None:
                # this is due to the convexity of RRL: IS IT CONVEX???
                # If we reached a minimum, it is the global minimum.
                break
            self._Ms.append(M_best)
            self.M_ = M_best
        else:
            if self.verbose:
                print("Didn't converge after", it, "iterations. Final loss:", s_best)
        self.n_iter_ = it
        print("Finished after ", it, "iterations.")
        return self

    def _violations(self, metric):
        if self.verbose:
            print(str(datetime.now()) + "\tCalculating violations")
        # TODO: Is the transpose correct here?
        transformed = normalize(self.X_.dot(np.linalg.cholesky(metric).T))
        if self.verbose:
            print(str(datetime.now()) + "\t\tcosaMbs")
        cosines = np.diag(cosine_similarity(transformed[self.wordpairs[:, 0]], transformed[self.wordpairs[:, 1]]))[
            self.constraints]
        # this then looks for cosab < coscd
        vio = cosines[:, 0] - cosines[:, 1] < 0
        if self.verbose:
            print(str(len(cosines[vio])) + " violations found")
        return vio, cosines, transformed

    def _comparison_loss(self, metric):
        """
        loss function. very easy to compute. furthermore it is convex.
        :param metric: a PSD symmetric metric.
        :return:
        """
        vio, cosines, _ = self._violations(metric)
        closs = np.sum(((cosines[:, 0] - cosines[:, 1]) ** 2)[vio] / len(cosines))
        if self.verbose:
            print("comparison loss: " + str(closs))
        return closs

    def _loss(self, metric):
        if self.verbose:
            print(str(datetime.now()) + "\tCalculating loss")
        closs = self._comparison_loss(metric)
        rloss = _regularization_loss(metric)
        if self.verbose:
            print(str(datetime.now()) + "\t:" + str(closs) + "\t" + str(rloss))
        return closs + rloss

    def _gradient(self, metric):
        dMetric = (np.identity(metric.shape[0]) - np.linalg.inv(metric)) / (metric.shape[0] ** 2)
        violations, cosines, transformed = self._violations(metric)
        if self.verbose:
            print(str(datetime.now()) + "\tCalculating gradient for " + str(len(cosines[violations])) + " violations")
        vio_count = 0
        import pyspark
        sc = pyspark.SparkContext.getOrCreate()
        entries = sc.parallelize(np.hstack([self.constraints[violations], cosines[violations]]))
        Xbc = sc.broadcast(self.X_)
        wordpairsbc = sc.broadcast(self.wordpairs)
        transformedbc = sc.broadcast(transformed)
        dMetric += entries.map(lambda entry: get_loss_gradient(entry, Xbc, wordpairsbc, transformedbc)).sum()
        if self.verbose:
            print(str(datetime.now()) + "\tGradient done")
        sc.stop()
        return dMetric


def _regularization_loss(metric):
    trace = np.trace(metric)
    sign, logdet = np.linalg.slogdet(metric)
    logdet = sign * logdet
    return (trace + logdet) / (metric.shape[0] ** 3)


def _make_psd(metric, tol=1e-8):
    w, v = np.linalg.eigh(metric)
    return v.dot((np.maximum(w, tol) * v).T)


def makeouter(X, col1, col2, transformed):
    return np.outer(X[col1], X[col2]) / (np.linalg.norm(transformed[col1]) * np.linalg.norm(transformed[col2]))


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
