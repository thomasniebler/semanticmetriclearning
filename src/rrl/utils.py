try:
    import urllib2
except Exception:
    import urllib.request as urllib2

import numpy as np
import pandas
from scipy.sparse import vstack, spmatrix
from scipy.special import erf


def t(x, column=True):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if column:
        if x.shape[0] < x.shape[1]:
            return x.T
        else:
            return x
    else:
        if x.shape[0] > x.shape[1]:
            return x.T
        else:
            return x


def dotA(x, y, A=None):
    if A is None:
        return t(x, column=False).dot(t(y))[0, 0]
    result = t(x, column=False).dot(A).dot(t(y))
    if type(result) == int:
        return result
    else:
        return result[0, 0]


def normA(x, A=None):
    return np.sqrt(dotA(x, x, A))


def cosA(x, y, A=None):
    return dotA(x, y, A) / (normA(x, A) * normA(y, A))


def prepare_dataset(evaldf, vecs):
    result = pandas.DataFrame()
    for row in evaldf[["termA", "termB", "relatedness"]].values:
        if row[0].lower() in vecs and row[1].lower() in vecs:
            result = result.append([tuple(row) + (
                vecs[row[0].lower()], vecs[row[1].lower()], cosA(vecs[row[0].lower()], vecs[row[1].lower()]))])
    result.columns = ["termA", "termB", "relatedness", "vecA", "vecB", "cos"]
    return result.reset_index(drop=True)


def load_vecs(path, vocabulary=None, encoding="iso-8859-1"):
    file = open(path)
    if vocabulary is not None:
        vecs = {parts[0].lower(): np.array([float(k) for k in parts[1:]])
                for parts in [line.strip().split(" ") for line in file]
                if len(parts) > 2 and parts[0].lower() in vocabulary}
    else:
        vecs = {parts[0].lower(): np.array([float(k) for k in parts[1:]])
                for parts in [line.strip().split(" ") for line in file] if len(parts) > 2}
    vecs = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    return vecs


def load_eval_df(name="ws353"):
    return pandas.read_csv(urllib2.urlopen("https://www.thomas-niebler.de/evaldf/" + name + ".csv"), sep="\t",
                           header=0)


def get_train_parameters(evaldf, vecs):
    examples = prepare_dataset(evaldf, vecs)
    a = examples[["termA", "vecA"]]
    b = examples[["termB", "vecB"]]
    a.columns = ["term", "vec"]
    b.columns = ["term", "vec"]
    term_to_vector = a.append(b).drop_duplicates(subset=["term"]).reset_index()
    term_to_vector["index"] = range(len(term_to_vector))
    examples["idA"] = examples.termA.map(
        lambda termA: int(term_to_vector[term_to_vector.term == termA]["index"]))
    examples["idB"] = examples.termB.map(
        lambda termB: int(term_to_vector[term_to_vector.term == termB]["index"]))
    # generate data matrix
    X = np.array(list(term_to_vector["vec"]))
    if issubclass(type(X[0]), spmatrix):
        X = vstack(X).tocsc()
    # create constraints
    wordpairs = np.array(examples[["idA", "idB"]])
    relscores = np.array(examples[["relatedness"]])
    constraints = []
    for i in range(len(relscores)):
        for j in range(i, len(relscores)):
            if relscores[i] > relscores[j]:
                constraints.append([i, j])
    constraints = np.array(constraints)
    return X, constraints, wordpairs


def evaluate(test_eval_df, metric=None):
    results = [round(cosA(row[0], row[1]), 3) for row in test_eval_df[["vecA", "vecB"]].values]
    test_eval_df["cos"] = results
    results_metric = [round(cosA(row[0], row[1], metric), 3) for row in test_eval_df[["vecA", "vecB"]].values]
    test_eval_df["met"] = results_metric
    return [test_eval_df["cos"].corr(test_eval_df["relatedness"], method="spearman"),
            test_eval_df["met"].corr(test_eval_df["relatedness"], method="spearman")]


def z(r):
    return 0.5 * np.log((1 + r) / (1 - r))


def p(r1, r2, n):
    return 1 - erf(np.abs(z(r1) - z(r2)) / (np.sqrt(1.06 / (n - 3))))
