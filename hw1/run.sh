echo "[glove] preprocessing"
python2.7 preprocessing_glove.py --corpus $1/text8
echo "[glove] training"
python2.7 glove.py
echo "[glove] filtering"
python2.7 filterVocab.py fullVocab.txt < ./g_vector.txt > $2/filter_glove.txt
echo "[word2vec] preprocessing"
python2.7 preprocessing_word2vec.py --corpus $1/text8
echo "[word2vec] training"
python2.7 word2vec.py
echo "[word2vec] filtering"
python2.7 filterVocab.py fullVocab.txt < ./w2v_vector.txt > $2/filter_word2vec.txt
echo "[ptt_glove] preprocessing"
python2.7 preprocessing_glove.py --corpus $1/ptt_corpus.txt
echo "[ptt_glove] training"
python2.7 glove.py
echo "[ptt_glove] filtering"
python2.7 filterVocab.py fullVocab_phase2.txt < ./g_vector.txt > $2/filter_vec.txt


