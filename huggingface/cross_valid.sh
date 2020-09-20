#export CUDA_VISIBLE_DEVICES=4,5,6,7
#
#python cross_validation.py \
#  --model_type bert \
#  --model_name_or_path ../models/bert-large-uncased-wwm-squad2/ \
#  --model_arch bert-large-uncased-wwm-squad2-last2h \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 8 \
#  --per_gpu_eval_batch_size 8 \
#  --learning_rate 7e-5 \
#  --num_train_epochs 2.0 \
#  --splits 5 \
#  --max_seq_length 70 \
#  --use_jaccard_soft \
#  --fp16

# 注意 use_jaccard_soft 的特殊设置
export CUDA_VISIBLE_DEVICES=0,1,2,3

python cross_validation.py \
  --model_type roberta \
  --model_name_or_path ../models/roberta-base-squad2/ \
  --model_arch roberta-base-squad2-last3h-conv-jaccard-soft \
  --do_lower_case \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --per_gpu_train_batch_size 48 \
  --per_gpu_eval_batch_size 48 \
  --learning_rate 5e-5 \
  --num_train_epochs 5.0 \
  --splits 5 \
  --max_seq_length 128