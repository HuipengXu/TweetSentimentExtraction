export CUDA_VISIBLE_DEVICES=4,5,6,7
python main.py \
  --model_type roberta \
  --model_name_or_path ../models/roberta-base/ \
  --model_arch roberta-base \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 60 \
  --per_gpu_eval_batch_size 60 \
  --logging_steps 300 \
  --learning_rate 3e-5 \
  --num_train_epochs 6.0 \
  --max_seq_length 180 \
  --overwrite_cache \
  --doc_stride 128
