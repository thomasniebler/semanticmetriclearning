# Learning Semantic Relatedness from Human Feedback Using Metric Learning

This page contains all the necessary information to reproduce the results given in the ISWC'17 submission
["Learning Semantic Relatedness from Human Feedback Using Metric Learning"](https://arxiv.org/abs/1705.07425)
by
[Thomas Niebler](http://www.dmir.uni-wuerzburg.de/staff/niebler),
[Martin Becker](http://www.dmir.uni-wuerzburg.de/staff/martinbecker),
[Christian Pölitz](http://www.dmir.uni-wuerzburg.de/staff/christian_poelitz) and
[Andreas Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho).

## Overview
In our work, we learned a semantic relatedness measure from human feedback, using a metric learning approach.
Human Intuition Datasets contain direct human judgments about the relatedness of words, i.e. human feedback.
We exploit these datasets to then learn a parameterization of the cosine measure, while resorting to
a metric learning approach, which is based on relative distance comparisons. We validate our approach on four
different embedding datasets, which we make public or provide a download a link here.

Furthermore and to the best of our knowledge, we were the first to explore the possibility of learning
word embeddings from tagging data.

## Reference Implementations
### Tag Co-Occurrence Graph
To calculate the tag cooccurrence graph as input for the GloVe algorithm, we applied
the method presented in ["Semantic Grounding of Tag Relatedness in Social Bookmarking Systems"](https://www.bibsonomy.org/bibtex/23d13a333db2d59968b6afda906006286/thoni)
by Cattuto et al.. More specifically, we used the co-occurrence based on posts as described in
Equation (1) in the linked paper:
<a href="https://www.codecogs.com/eqnedit.php?latex=coocc(t_1,&space;t_2)&space;:=&space;card\left((u,&space;r)&space;\in&space;U&space;\times&space;R&space;|&space;t_1,&space;t_2&space;\in&space;T_{ur}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?coocc(t_1,&space;t_2)&space;:=&space;card\left((u,&space;r)&space;\in&space;U&space;\times&space;R&space;|&space;t_1,&space;t_2&space;\in&space;T_{ur}&space;\right)" title="coocc(t_1, t_2) := card\left((u, r) \in U \times R | t_1, t_2 \in T_{ur} \right)" /></a>
Here, <a href="https://www.codecogs.com/eqnedit.php?latex=T_{ur}&space;:=&space;\left\{&space;t&space;\in&space;T&space;|&space;(u,t,r)&space;\in&space;Y&space;\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{ur}&space;:=&space;\left\{&space;t&space;\in&space;T&space;|&space;(u,t,r)&space;\in&space;Y&space;\right\}" title="T_{ur} := \left\{ t \in T | (u,t,r) \in Y \right\}" /></a>, i.e. all tags t, which hav been assigned
to resource r by user u.

In [src/embeddings/example_call.py](https://github.com/thomasniebler/semantics-metriclearning/blob/master/src/embeddings/example_call.py), we provided an example
on how to call the corresponding methods to construct the co-occurrence graph. It then needs to be saved to a file, before
the GloVe algorithm can be called on that file.

### LSML
For LSML, we used a modified implementation from the one in the [metric_learn](https://github.com/all-umass/metric-learn) package.
It can be found under [src/metric_learn/lsml.py](https://github.com/thomasniebler/semantics-metriclearning/blob/master/src/metric_learn/lsml.py)
in our repository.

We added a diminishing factor to the matrix regularization term, as Euclidean distances on a unit sphere tend to become
rather small in comparison to the trace of a 100x100 matrix. The initial matrix M_0 was chosen as the Identity matrix,
since we want to modify the cosine measure.

### GloVe
We used the [published code of GloVe](https://nlp.stanford.edu/projects/glove/) to create the tag embeddings of
dimension 100. We used the predefined values of alpha=0.75 and x_max=100.

## Vector Embeddings
These are the datasets that we used for our experiments.

### Delicious
The Delicious tagging dataset is [publicly available](http://www.zubiaga.org/datasets/socialbm0311).
The generated word embeddings are published in this repository. 

### BibSonomy
The BibSonomy tagging data can be retrieved from [the BibSonomy homepage](https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/).
We also provide the generated word embeddings as a public download in this repository. 

### WikiGlove
Pennington et al. made some of their vector collections [publicly available](https://nlp.stanford.edu/projects/glove/).
Specifically, we used to GloVe6B corpus, which is generated from a Wikipedia dump from 2014 and the Gigaword5 corpus.

### WikiNav
The WikiNav vectors are publicly available at [Wikimedia Research](https://meta.wikimedia.org/wiki/Research:Wikipedia_Navigation_Vectors).
Specifically, we used the 100-dimensional vectors from [FigShare](https://figshare.com/articles/Wikipedia_Vectors/3146878), created
with data ranging from 01-01-2017 till 31-01-2017.

## Human Intuition Datasets
The Human Intuition Datasets (HIDs) can be retrieved as preprocessed pandas-friendly csv files 
[here](http://www.thomas-niebler.de/dataset-collection-for-evaluating-semantic-relatedness/)
or from the corresponding original locations.
* [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.html)
* [MEN collection](https://staff.fnwi.uva.nl/e.bruni/MEN)
* [Bib100](http://dmir.org/datasets/bib100/)


## Appendix for ISWC reviews
As an additional experiment, we trained a metric using MEN on the [ConceptNet Numberbatch Vectors](https://github.com/commonsense/conceptnet-numberbatch/tree/16.09)
as presented in [ConceptNet 5.5: An Open Multilingual Graph of General Knowledge](https://arxiv.org/abs/1612.03975) (AAAI 2017).
The published embeddings currently pose the state-of-the-art on correlation with human intuition on the MEN dataset (0.868 Spearman Correlation).

We generated 6 training and holdout splits of the MEN word pairs with a ratio of 80:20 each. From the 80% training pairs, we randomly sampled
90% and trained a metric on these pairs. We repeated this process 10 times. The correlation of the embeddings with the test pairs using the 
standard cosine measure was taken as baseline.

In the following table, we report the mean correlations for both the baseline and the metric for each split.
Then, we show that our improvement is in fact significant with p < 0.05.

|         | split 1|split 2|split 3|split 4|split 5| mean  |
| ------- | -----:| -----:| -----:| -----:| -----:| ---:  |
| cos     | 0.859 | 0.857 | 0.864 | 0.862 | 0.865 | 0.861 |
| metric  | 0.875 | 0.880 | 0.873 | 0.880 | 0.878 | 0.877 |
| p       | **0.030** | **0.002** | 0.219 | **0.012** | 0.069 | **0.028**|

First off, we slightly outperform the ConceptNet Embeddings in each split. Testing the significance of each 
difference gives us a p-value < 0.05 in most cases. Furthermore, the difference between the mean scores across
all splits is also significant.

Unfortunately, we haven't yet performed all experiments described in the paper on this embeddings dataset, since the figures
presented in the paper encourage increased performance when training a metric on the whole training dataset. However, we plan on
doing this in the near future.