from embeddings.preprocessing import edgelist_to_vec
from .preprocessing import tas_to_edgelist

'''
the tas file should be in the format
tag\tuser\tresource\n
'''
delicious_tas = sc.textFile("path_to_delicious_tas_in_hdfs") \
    .map(lambda entry: entry.strip().split("\t")) \
    .map(lambda entry_parts: dict(zip(["tag", "user", "res"], entry_parts))) \
    .cache()

delicious_edgelist, delicious_vocab_map_bc = tas_to_edgelist(sc, delicious_tas, top_tag=10000, min_user=5,
                                                             min_resource=10)
delicious_edgelist = delicious_edgelist \
    .map(lambda entry: "\t".join([str(x) for x in entry])) \
    .cache()

delicious_vectors = edgelist_to_vec(delicious_edgelist, delicious_vocab_map_bc)

with open("cooccs_delicious", "w") as outfile:
    for line in delicious_edgelist.collect():
        outfile.write(line + "\n")
    outfile.close()
