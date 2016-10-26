#!/bin/bash
if [[ ! -e $2 ]]; then
	mkdir $2
fi
echo "[glove] preprocessing"
python2.7 preprocessing_glove.py --corpus $1
echo "[glove] training"
python2.7 glove.py
echo "[glove] filtering"
python2.7 filterVocab.py fullVocab.txt < ./g_vector.txt > $2/filter_glove.txt
echo "[word2vec] preprocessing"
python2.7 preprocessing_word2vec.py --corpus $1
echo "[word2vec] training"
python2.7 word2vec.py
echo "[word2vec] filtering"
python2.7 filterVocab.py fullVocab.txt < ./w2v_vector.txt > $2/filter_word2vec.txt