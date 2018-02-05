import argparse
from datetime import datetime

import pandas

import rrl
from rrl import utils

parser = argparse.ArgumentParser(description="Retrain Word Embeddings using Relative Relatedness Learning")
parser.add_argument('embeddings', help="word embedding text file. file format is \"name val1 val2 val3 ...\"")
parser.add_argument('relscores', help="a tab separated csv file with word pairs and numeric relatedness scores, columns"
                                      " headers must be in the first line")
parser.add_argument('-l', '--learningrate', type=float, help="learning rate")
parser.add_argument('-v', '--verbose', action="store_true", help="more elaborate output")
parser.add_argument('--outputfile', help="path where the transformed vectors should be saved to")
parser.add_argument('--evalsteps', action="store_true",
                    help="show evaluation scores on a range of datasets after every iteration")
args = parser.parse_args()

print(str(datetime.now()) + "\tLoading data...")
vectors = utils.load_vecs(args.embeddings)
relscores = pandas.read_csv(args.relscores, header=0, sep="\t")
relscores.columns = ["termA", "termB", "relatedness"]
relscores["relatedness"] = relscores["relatedness"].apply(float)

alg = rrl.RRL(verbose=args.verbose)
print(str(datetime.now()) + "\tTraining...")
model = alg.fit(vectors, relscores, eval_steps=args.evalsteps, step_sizes=[args.learningrate])

for met in model._Ms:
    print(
        "ws353", rrl.utils.evaluate(model.prep_eval_dfs["ws353"], metric=met),
        "ws353sim", rrl.utils.evaluate(model.prep_eval_dfs["ws353sim"], metric=met),
        "ws353rel", rrl.utils.evaluate(model.prep_eval_dfs["ws353rel"], metric=met),
        "men", rrl.utils.evaluate(model.prep_eval_dfs["men"], metric=met),
        "simlex999", rrl.utils.evaluate(model.prep_eval_dfs["simlex999"], metric=met),
        "mturk", rrl.utils.evaluate(model.prep_eval_dfs["mturk"], metric=met),
        "mturk771", rrl.utils.evaluate(model.prep_eval_dfs["mturk771"], metric=met),
    )

print(str(datetime.now()) + "\tTransforming vectors...")
transformedvecs = model.transform()
outputfile = open(args.outputfile, "w")
for namevec in transformedvecs.items():
    outputfile.write(namevec[0] + " " + " ".join(map(str, map(lambda x: round(x, 3), namevec[1]))) + "\n")

outputfile.close()
print(str(datetime.now()) + "\tDone with training.")
