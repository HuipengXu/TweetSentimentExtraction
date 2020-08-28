#export SQUAD_DIR=/data/xhp/TweetSentimentExtr/data
export CUDA_VISIBLE_DEVICES=4,5,6,7

python run_squad.py \
  --model_type bert \
  --model_name_or_path ../models/bert/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 45 \
  --logging_steps 300 \
  --learning_rate 3e-5 \
  --num_train_epochs 6.0 \
  --max_seq_length 180 \
  --doc_stride 128 \
  --overwrite_output_dir