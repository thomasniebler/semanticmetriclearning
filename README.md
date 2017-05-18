# semantics-metriclearning

This repository contains the used code and a description of the used materials
for the ISWC'17 submission "Learning Semantic Relatedness from Human Feedback using Metric Learning",
which are necessary to recreate the experiments in the paper.

For more information, see [the GitHub page](https://thomasniebler.github.io/semantics-metriclearning/)

Python (2.7, didn't test it on 3.x) dependencies are:
* NumPy
* SciPy
* Scikit-learn
* pandas

If you want to speed up the training process, we also implemented a rough distributed training procedure
which uses [Spark](http://spark.apache.org/).