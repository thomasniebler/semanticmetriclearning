# Learning Semantic Relatedness from Human Feedback Using Relative Relatedness Learning

This page contains all the necessary information to reproduce the results given in the ISWC'17 poster
["Learning Semantic Relatedness from Human Feedback Using Relative Relatedness Learning"](https://www.thomas-niebler.de/pub/niebler2017rrl.pdf)
by
[Thomas Niebler](http://www.dmir.uni-wuerzburg.de/staff/niebler),
[Martin Becker](http://www.dmir.uni-wuerzburg.de/staff/martinbecker),
[Christian Pölitz](http://www.dmir.uni-wuerzburg.de/staff/christian_poelitz) and
[Andreas Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho).

## Overview
In our work, we learned a semantic relatedness measure from human feedback,
using a metric learning approach.
Human Intuition Datasets contain direct human judgments about the
relatedness of words, i.e. human feedback.
We exploit these datasets to then learn a parameterization of the cosine
measure, while resorting to a metric learning approach, which is based
on relative distance comparisons.
We validate our approach on several different embedding datasets, which
we either make public or provide a download a link here.

Furthermore and to the best of our knowledge, we were the first to
explore the possibility of learning word embeddings from tagging data.
We further elaborated on this in a [different paper](https://www.thomas-niebler.de/pub/niebler2017embeddings.pdf).

## Reference Implementations
### From Tag Co-Occurrences to Tag Embeddings
To calculate the tag cooccurrence graph as input for the GloVe algorithm,
we applied the method presented in ["Semantic Grounding of Tag Relatedness
in Social Bookmarking Systems"](https://www.bibsonomy.org/bibtex/23d13a333db2d59968b6afda906006286/thoni)
by Cattuto et al.

More specifically, we used the co-occurrence based on posts as described
in Equation (1) in the linked paper:
<a href="https://www.codecogs.com/eqnedit.php?latex=coocc(t_1,&space;t_2)&space;:=&space;card\left((u,&space;r)&space;\in&space;U&space;\times&space;R&space;|&space;t_1,&space;t_2&space;\in&space;T_{ur}&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?coocc(t_1,&space;t_2)&space;:=&space;card\left((u,&space;r)&space;\in&space;U&space;\times&space;R&space;|&space;t_1,&space;t_2&space;\in&space;T_{ur}&space;\right)" title="coocc(t_1, t_2) := card\left((u, r) \in U \times R | t_1, t_2 \in T_{ur} \right)" /></a>
Here, <a href="https://www.codecogs.com/eqnedit.php?latex=T_{ur}&space;:=&space;\left\{&space;t&space;\in&space;T&space;|&space;(u,t,r)&space;\in&space;Y&space;\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T_{ur}&space;:=&space;\left\{&space;t&space;\in&space;T&space;|&space;(u,t,r)&space;\in&space;Y&space;\right\}" title="T_{ur} := \left\{ t \in T | (u,t,r) \in Y \right\}" /></a>,
i.e. all tags t, which hav been assigned to resource r by user u.

In [src/embeddings/example_call.py](https://github.com/thomasniebler/semantics-metriclearning/blob/master/src/embeddings/example_call.py),
we provided an example on how to call the corresponding methods to
construct the co-occurrence graph. It then needs to be saved to a file,
before the GloVe algorithm can be called on that file.

### LSML
RRL is inspired by the LSML metric learning algorithm.
We built on the LSML implementation contained in the
[metric_learn](https://github.com/all-umass/metric-learn) python package.

### GloVe
We used the [published code of GloVe](https://nlp.stanford.edu/projects/glove/)
to create the tag embeddings of dimension 100.
We used the predefined parameter values of alpha=0.75 and x_max=100.

## Word Embedding Datasets
These are the datasets that we used for our experiments.

* **Delicious**
The Delicious tagging dataset is [publicly available](http://www.zubiaga.org/datasets/socialbm0311).
The generated word embeddings are published in this repository. 

* **BibSonomy**
The BibSonomy tagging data can be retrieved from [the BibSonomy homepage](https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/).
We also provide the generated word embeddings as a public download in this repository. 

* **WikiGlove**
Pennington et al. made some of their vector collections [publicly available](https://nlp.stanford.edu/projects/glove/).
Specifically, we used to GloVe6B corpus, which is generated from a Wikipedia dump from 2014 and the Gigaword5 corpus.

* **WikiNav**
The WikiNav vectors are publicly available at [Wikimedia Research](https://meta.wikimedia.org/wiki/Research:Wikipedia_Navigation_Vectors).
Specifically, we used the 100-dimensional vectors from [FigShare](https://figshare.com/articles/Wikipedia_Vectors/3146878), created
with data ranging from 01-01-2017 till 31-01-2017.

* **ConceptNet Numberbatch** Finally, we applied our algorithm on the
[ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch/tree/16.09) vectors, which currently yield state-of-the-art
performance in [a series of competitions](https://blog.conceptnet.io/2017/03/02/how-luminoso-made-conceptnet-into-the-best-word-vectors-and-won-at-semeval/).


## Human Intuition Datasets
The Human Intuition Datasets (HIDs) can be retrieved as preprocessed pandas-friendly csv files 
[here](http://www.thomas-niebler.de/dataset-collection-for-evaluating-semantic-relatedness/)
or from the corresponding original locations.
* [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.html)
* [MEN collection](https://staff.fnwi.uva.nl/e.bruni/MEN)
* [Bib100](http://dmir.org/datasets/bib100/)

