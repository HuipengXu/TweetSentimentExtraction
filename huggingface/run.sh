export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py \
  --model_type bert \
  --model_name_or_path ../models/bertweet-base/ \
  --model_arch bertweet-base \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 200 \
  --per_gpu_eval_batch_size 200 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 70 \
  --overwrite_cache \
  --eval_all_checkpoints \
  --weight_decay 0.001 \
  --doc_stride 70