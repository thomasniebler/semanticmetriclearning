#!/bin/bash

mkdir test
cd test
echo "Download GloVe vectors..."
wget http://nlp.stanford.edu/data/glove.6B.zip
echo "Unzipping..."
unzip glove.6B.zip
rm glove.6B.zip
wget https://www.thomas-niebler.de/evaldf/men.csv
cd ..
echo "Training..."
python train.py test/glove.6B.200d.txt men --output test/glove_men_200.txt
