test_data=$1
answer_file=$2

cat ./translate/model_translate/model_translate.* > ./translate/model_translate/model_translate

python2.7 ./translate/translate.py \
			--decode True \
			--test_data $test_data \
			--predict_file $answer_file \
			--data_dir ./translate/data_translate \
			--train_dir ./translate/model_translate
