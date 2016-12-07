test_data=$1
answer_file=$2

cat ./nlg/model_nlg/model_nlg.* > ./nlg/model_nlg/model_nlg


python2.7 ./nlg/nlg_preprocessing.py \
			--test True \
			--test_data $test_data \
			--test_output ./test.in

python2.7 ./nlg/nlg.py \
			--decode True \
			--test_data ./test.in \
			--predict_file ./nlg.predict \
			--data_dir ./nlg/data_nlg \
			--train_dir ./nlg/model_nlg

python2.7 ./nlg/nlg_postprocessing.py \
			--test_data $test_data \
			--predict_data ./nlg.predict \
			--output $answer_file
