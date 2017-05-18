# Learning Semantic Relatedness from Human Feedback Using Metric Learning

This page contains all the necessary information to reproduce the results given in the ISWC'17 submission
"Learning Semantic Relatedness from Human Feedback Using Metric Learning" (link to arxiv coming soon)
by
[Thomas Niebler](http://www.dmir.uni-wuerzburg.de/staff/niebler),
[Martin Becker](http://www.dmir.uni-wuerzburg.de/staff/martinbecker),
[Christian Pölitz](http://www.dmir.uni-wuerzburg.de/staff/christian_poelitz) and
[Andreas Hotho](http://www.dmir.uni-wuerzburg.de/staff/hotho).

## Reference Implementations
### LSML
For LSML, we used a modified implementation from the one in the [metric_learn](https://github.com/all-umass/metric-learn) package.
It can be found in src/metric_learn/lsml.py

We added a diminishing factor to the matrix regularization term, as Euclidean distances on a unit sphere tend to become
rather small in comparison to the trace of a 100x100 matrix. The initial matrix M_0 was chosen as the Identity matrix,
since we want to modify the cosine measure.

### GloVe
We used the [published code of GloVe](https://nlp.stanford.edu/projects/glove/) to create the tag embeddings.
We used the predefined values of alpha=0.75 and x_max=100.

## Vector Embeddings
These are the datasets that we used for our experiments.

### Delicious
The Delicious tagging dataset is [publicly available](http://www.zubiaga.org/datasets/socialbm0311).
The generated word embeddings can be retrieved from [PUT THAT IN] 

### BibSonomy
The BibSonomy tagging data can be retrieved from [the BibSonomy homepage](https://www.kde.cs.uni-kassel.de/bibsonomy/dumps/).
We also provide the generated word embeddings as a public download at [PUT THAT IN] 

### WikiGlove
Pennington et al. made some of their vector collections [publicly available](https://nlp.stanford.edu/projects/glove/).
Specifically, we used to GloVe6B corpus, which is generated from a Wikipedia dump from 2014 and the Gigaword5 corpus.

### WikiNav
The WikiNav vectors are publicly available at [https://meta.wikimedia.org/wiki/Research:Wikipedia_Navigation_Vectors].
Specifically, we used the 100-dimensional vectors from [https://figshare.com/articles/Wikipedia_Vectors/3146878], created
with data ranging from 01-01-2017 till 31-01-2017.

## Human Intuition Datasets
The Human Intuition Datasets (HIDs) can be retrieved as preprocessed pandas-friendly csv files 
from [http://www.thomas-niebler.de/dataset-collection-for-evaluating-semantic-relatedness/]
or from the corresponding original locations.
* [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.html)
* [MEN collection](https://staff.fnwi.uva.nl/e.bruni/MEN)
* [Bib100](http://dmir.org/datasets/) (TODO: WE MUST ADD A DESCRIPTION HERE)