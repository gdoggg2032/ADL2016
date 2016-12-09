test_data=$1
answer_file=$2

# combine the split model
cat ./nlg/model_nlg/model_nlg.* > ./nlg/model_nlg/model_nlg

# mapping the content of entities to entity-like names
python2.7 ./nlg/nlg_preprocessing.py \
			--test True \
			--test_data $test_data \
			--test_output ./test.in

# predict
python2.7 ./nlg/nlg.py \
			--decode True \
			--test_data ./test.in \
			--predict_file ./nlg.predict \
			--data_dir ./nlg/data_nlg \
			--train_dir ./nlg/model_nlg

# mapping the entity-like name to content of entities
python2.7 ./nlg/nlg_postprocessing.py \
			--test_data $test_data \
			--predict_data ./nlg.predict \
			--output $answer_file
