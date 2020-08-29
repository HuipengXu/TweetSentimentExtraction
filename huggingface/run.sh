export CUDA_VISIBLE_DEVICES=4,5,6

python run_squad.py \
  --model_type bert \
  --model_name_or_path ../models/bert/ \
  --do_train true \
  --do_eval true \
  --do_lower_case true \
  --overwrite_cache true \
  --train_file debug_train.json \
  --predict_file debug_valid.json \
  --per_gpu_train_batch_size 36 \
  --per_gpu_eval_batch_size 36 \
  --logging_steps 300 \
  --learning_rate 3e-5 \
  --num_train_epochs 6.0 \
  --max_seq_length 180 \
  --doc_stride 128