import itertools


def can_be_printed(term):
    try:
        str(term)
        return True
    except UnicodeEncodeError:
        return False


def filter_by_count(tas_rdd, entity="tag", minfrequency=1, top=0):
    tas_with_counts = tas_rdd \
        .filter(lambda tas: can_be_printed(tas[entity])) \
        .map(lambda tas: (tas[entity], 1))
    if top > 0:
        toptags = [tag_with_count[1] for tag_with_count in tas_with_counts
            .reduceByKey(lambda a, b: a + b)
            .map(lambda (a, b): (b, a))
            .sortByKey(ascending=False)
            .take(top)]
        return dict(zip(toptags, range(top)))
    else:
        return dict(tas_with_counts
                    .reduceByKey(lambda a, b: a + b)
                    .filter(lambda (ent, count): count >= minfrequency)
                    .map(lambda (ent, _): ent)
                    .zipWithIndex()
                    .collect())


def get_transformed_tas_rdd(tas_rdd, vocabulary_map_bc, user_map_bc, resource_map_bc):
    return tas_rdd \
        .filter(lambda tas: tas["tag"] in vocabulary_map_bc.value and
                            tas["user"] in user_map_bc.value and
                            tas["res"] in resource_map_bc.value) \
        .map(lambda tas: {"tag": vocabulary_map_bc.value[tas["tag"]],
                          "user": user_map_bc.value[tas["user"]],
                          "res": resource_map_bc.value[tas["res"]]}) \
        .cache()


def get_cooccs_with_counts(transformed_tas_rdd, grouping_function=lambda tas: (tas["user"],)):
    """
    :param transformed_tas_rdd:
    :param grouping_function:
    :param minfrequency:
    :return:
    """
    return transformed_tas_rdd \
        .map(lambda tas: (grouping_function(tas), tas["tag"])) \
        .groupByKey() \
        .map(lambda (key, tag_list): (key, list(tag_list))) \
        .flatMap(lambda (key, tag_list): list(itertools.product(tag_list, tag_list))) \
        .filter(lambda kv: kv[0] != kv[1]) \
        .map(lambda kv: (kv, 1)) \
        .reduceByKey(lambda a, b: a + b)


def tas_to_edgelist(sc, tas_rdd, min_tag=1, min_user=1, min_resource=1, top_tag=0, top_user=0, top_resource=0):
    vocabulary_map_bc = sc.broadcast(filter_by_count(tas_rdd, entity="tag", minfrequency=min_tag, top=top_tag))
    reverse_voc_map_bc = sc.broadcast({v: k for k, v in vocabulary_map_bc.value.items()})
    user_map_bc = sc.broadcast(filter_by_count(tas_rdd, entity="user", minfrequency=min_user, top=top_user))
    resource_map_bc = sc.broadcast(filter_by_count(tas_rdd, entity="res", minfrequency=min_resource, top=top_resource))
    transformed_tas_rdd = get_transformed_tas_rdd(tas_rdd, vocabulary_map_bc, user_map_bc, resource_map_bc)
    cooccs_with_counts = get_cooccs_with_counts(transformed_tas_rdd,
                                                grouping_function=lambda tas: (tas["user"], tas["res"]))
    return cooccs_with_counts \
        .map(lambda (kv, count): kv + (count,)) \
        .map(lambda (k, v, count): (reverse_voc_map_bc.value[k], reverse_voc_map_bc.value[v], count))
