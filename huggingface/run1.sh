export CUDA_VISIBLE_DEVICES=0,1,2,3

python main1.py \
  --model_type roberta \
  --model_name_or_path ../models/roberta-base/ \
  --model_arch roberta-base-last2h \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 7e-5 \
  --num_train_epochs 6.0 \
  --max_seq_length 128 \
  --overwrite_cache \
  --eval_all_checkpoints