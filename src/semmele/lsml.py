import numpy as np

from metric_learn import LSML


def get_constraints(values):
    n = values.shape[0]
    constraints = np.empty((int(n * (n - 1) / 2), 4), dtype=np.int)
    i = 1
    j = 0
    # make sure that the dataframe is sorted in descending order, such that sim(a_i, b_i) >= sim(a_j, b_j) for i < j
    values = values.sort_values(by="relatedness", ascending=False)
    for vec_a1, vec_b1, rel_ab1 in values.values:
        for vec_a2, vec_b2, rel_ab2 in values.values[i:, ]:
            # constraints are interpreted as d(a1, b1) < d(a2, b2)
            # sim(a2, b2) < sim(a1, b1) <=> d(a1, b1) < d(a2, b2)
            constraints[j,] = [vec_a1, vec_b1, vec_a2, vec_b2]
            j += 1
        i += 1
    return constraints


def get_weights(values):
    n = values.shape[0]
    weights = np.empty((int(n * (n - 1) / 2)), dtype=np.float32)
    i = 1
    j = 0
    # make sure that the dataframe is sorted in descending order, such that sim(a_i, b_i) >= sim(a_j, b_j) for i < j
    values = values.sort_values(by="relatedness", ascending=False)
    for vec_a1, vec_b1, rel_ab1 in values.values:
        for vec_a2, vec_b2, rel_ab2 in values.values[i:, ]:
            # constraints are interpreted as d(a1, b1) < d(a2, b2)
            # sim(a2, b2) < sim(a1, b1) <=> d(a1, b1) < d(a2, b2)
            weights[j] = rel_ab1 - rel_ab2
            j += 1
        i += 1
    return np.array(weights)


def get_train_test(eval_df, samples):
    train_data = eval_df.sample(n=samples).sort_values(by="relatedness")
    test_data = eval_df[~eval_df.isin(train_data)].dropna()
    return train_data, test_data


def get_train_parameters(examples):
    a = examples[["termA", "vecA"]]
    b = examples[["termB", "vecB"]]
    a.columns = ["term", "vec"]
    b.columns = ["term", "vec"]
    term_to_vector = a.append(b).drop_duplicates(subset=["term"]).reset_index()
    term_to_vector["index"] = range(len(term_to_vector))
    examples["idA"] = examples.termA.map(lambda termA: int(term_to_vector[term_to_vector.term == termA]["index"]))
    examples["idB"] = examples.termB.map(lambda termB: int(term_to_vector[term_to_vector.term == termB]["index"]))
    # generate data matrix
    X = np.array(list(term_to_vector["vec"]))
    # create constraints
    constraints = np.array(get_constraints(examples[["idA", "idB", "relatedness"]])).T
    weights = np.array(get_weights(examples[["idA", "idB", "relatedness"]])).T
    weights /= np.linalg.norm(weights)
    # use identity matrix as prior
    prior = np.identity(X.shape[1])
    return X, constraints, weights, prior


def train(examples):
    # train on these examples
    X, constraints, weights, prior = get_train_parameters(examples)
    return LSML().fit(X, constraints, weights=weights)


def train_until_completed(data, n, evaldata=None, attempts=10):
    for i in range(attempts):
        try:
            if str(type(data)) == "<class 'pyspark.broadcast.Broadcast'>":
                (train_data, test_data) = get_train_test(data.value, n)
            else:
                (train_data, test_data) = get_train_test(data, n)
            metric = train(train_data).metric()
            if evaldata is not None:
                if str(type(evaldata)) == "<class 'pyspark.broadcast.Broadcast'>":
                    test_data = evaldata.value
                else:
                    test_data = evaldata
            return eval(test_data, metric) + eval(train_data, metric), metric
        except NameError as e:
            return [-2, -2, -2, -2], e
    return [-2, -2, -2, -2], None


def train_spark(sc, data, ns=[10, 20, 30, 40, 50, 60, 100], runs=25, kaputtmaching=False):
    params = [int(0.8 * k) for k in ns] * runs
    blob = sc.parallelize(params)
    partitions_count = len(params)
    smpl = int(0.8 * len(data))
    train_data, test_data = get_train_test(data, samples=smpl)
    if kaputtmaching:
        rel = list(train_data["relatedness"])
        np.random.shuffle(rel)
        train_data["relatedness"] = rel
    train_data_bc = sc.broadcast(train_data)
    test_data_bc = sc.broadcast(test_data)
    results = blob.zipWithIndex() \
        .map(lambda x: (x[1], x[0])) \
        .partitionBy(partitions_count, lambda k: k) \
        .glom() \
        .map(
        lambda n: (n[0][1],) + train_until_completed(train_data_bc, n[0][1],
                                                     evaldata=test_data_bc)) \
        .map(lambda n_with_res: (n_with_res[0], np.array(n_with_res[1]), n_with_res[2])) \
        .collect()
    i = 0
    while i < len(results) and 2 < len(results[i]) and (results[i][2] is None or type(results[i][2]) is ValueError):
        if type(results[i][2]) is ValueError:
            print(results[i][2])
        i += 1
    if i >= len(results):
        print("no results found.")
        return [], results
    desired_shape = results[i][2].shape
    metrics = {k: np.zeros(desired_shape) for k in params[:len(ns)]}
    scores = {}
    errorCount = 0
    for element in results:
        if element[1][0] != -2 and element[2] is not None and type(element[2]) is not ValueError:
            try:
                if element[0] in scores:
                    scores[element[0]] = np.vstack([scores[element[0]], element[1]])
                else:
                    scores[element[0]] = element[1]
                metrics[element[0]] += element[2]  # wtf am I doing here
            except TypeError:
                pass
        else:
            errorCount += 1
    print("error count: " + str(errorCount))
    result_scores = {}
    for key in scores.keys():
        if len(scores[key].shape) == 1:
            scores[key] = scores[key].reshape(1, -1)
        result_scores[key] = [np.mean(scores[key][:, 0]), np.std(scores[key][:, 0]),
                              np.mean(scores[key][:, 1]), np.std(scores[key][:, 1]),
                              np.mean(scores[key][:, 2]), np.std(scores[key][:, 2]),
                              np.mean(scores[key][:, 3]), np.std(scores[key][:, 3]),
                              ]
        metrics[key] /= runs
    return metrics, result_scores
