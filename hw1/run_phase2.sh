#!/bin/bash
if [[ ! -e $2 ]]; then
	mkdir $2
fi
echo "[ptt_glove] preprocessing"
python2.7 preprocessing_glove.py --corpus $1
echo "[ptt_glove] training"
python2.7 glove.py
echo "[ptt_glove] filtering"
python2.7 filterVocab.py fullVocab_phase2.txt < ./g_vector.txt > $2/filter_vec.txt

