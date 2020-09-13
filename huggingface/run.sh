export CUDA_VISIBLE_DEVICES=4,5,6,7

#python main.py \
#  --model_type roberta \
#  --model_name_or_path ../models/roberta-base/ \
#  --model_arch roberta-base-last2h-jaccard-soft \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file debug_train.json \
#  --predict_file debug_valid.json \
#  --per_gpu_train_batch_size 64 \
#  --per_gpu_eval_batch_size 16 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --doc_stride 128 \
#  --use_jaccard_soft


#python main.py \
#  --model_type bert \
#  --model_name_or_path ../models/bert-base-cased/ \
#  --model_arch bert-base-cased-last2h-jaccard-soft \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file train.json \
#  --predict_file valid.json \
#  --per_gpu_train_batch_size 32 \
#  --per_gpu_eval_batch_size 16 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --doc_stride 128

  python main1.py \
  --model_type bert \
  --model_name_or_path ../models/bert-large-uncased-wwm-squad/ \
  --model_arch bert-large-uncased-wwm-squad-last2h \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 7e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --overwrite_cache \
  --eval_all_checkpoints \
  --use_jaccard_soft \
  --fp16