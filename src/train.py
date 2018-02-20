import argparse
from datetime import datetime

import pandas

import rrl
from rrl import utils

parser = argparse.ArgumentParser(description="Retrain Word Embeddings using Relative Relatedness Learning")
parser.add_argument('-i', '--inputfile', help="word embedding text file. file format is \"name val1 val2 val3 ...\"")
parser.add_argument('-r', '--relscores',
                    help="a tab separated csv file with word pairs and numeric relatedness scores, columns"
                                      " headers must be in the first line")
parser.add_argument('-l', '--learningrate', type=float, help="initial learning rate", default=0.01)
parser.add_argument('-a', '--learningrateadaption', type=int,
                    help="learning rate adaption mode 0/1/2: no adaption/adaption only if loss increases/adaption every 10 steps",
                    default=1)
parser.add_argument('-b', '--batchsize', type=int, default=100, help="batchsize")
parser.add_argument('-v', '--verbose', action="store_true", help="more elaborate output")
parser.add_argument('-o', '--outputdir', help="path where the transformed vectors should be saved to")
parser.add_argument('-c', '--maxsparkcores', type=int, default=30,
                    help="maximum number of cores that gradient computation can use")
parser.add_argument('-e', '--evalsteps', action="store_true",
                    help="show evaluation scores on a range of datasets after every iteration")
parser.add_argument('-p', '--epochs', type=int, default=100, help="Number of training epochs")
args = parser.parse_args()

print(str(datetime.now()) + "\tLoading data...")
vectors = utils.load_vecs(args.inputfile)
relscores = pandas.read_csv(args.relscores, header=0, sep="\t")
relscores.columns = ["termA", "termB", "relatedness"]
relscores["relatedness"] = relscores["relatedness"].apply(float)

alg = rrl.RRL(verbose=args.verbose, epochs=args.epochs)
print(str(datetime.now()) + "\tTraining...")
model = alg.fit(vectors, relscores, eval_steps=args.evalsteps, learning_rate=args.learningrate,
                batchsize=args.batchsize, output_dir=args.outputdir, max_spark_cores=args.maxsparkcores,
                learning_rate_adaption=args.learningrateadaption)

print(str(datetime.now()) + "\tTransforming vectors...")
transformedvecs = model.transform()
if args.outputdir[-1] != "/":
    args.outputdir += "/"
outputfile = open(
    args.inputfile + "_rrl_" + str(args.learningrate) + "_" + str(args.batchsize), "w")
for namevec in transformedvecs.items():
    outputfile.write(namevec[0] + " " + " ".join(map(str, map(lambda x: round(x, 3), namevec[1]))) + "\n")

outputfile.close()
print(str(datetime.now()) + "\tDone with training.")
