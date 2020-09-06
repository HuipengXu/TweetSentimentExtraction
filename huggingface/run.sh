export CUDA_VISIBLE_DEVICES=4,5,6,7

python main.py \
  --model_type roberta \
  --model_name_or_path ../models/roberta-base-squad2/ \
  --model_arch roberta-base-squad2-last2h \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 70 \
  --overwrite_cache \
  --eval_all_checkpoints \
  --doc_stride 70