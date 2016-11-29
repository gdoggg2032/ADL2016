python pre_format.py --test $1 --mode test

data_dir=data/hw3
train_dir=model_tagging
max_sequence_length=50  # max length for train/valid/test sequence
task=tagging  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False
mode=test
answer=$2
word_embedding_size=300

CUDA_VISIBLE_DEVICES="" python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $train_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn \
      --mode $mode \
      --answer $answer \
	  --word_embedding_size $word_embedding_size

python pred_format.py --input $answer --output $answer
