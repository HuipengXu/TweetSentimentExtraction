export CUDA_VISIBLE_DEVICES=4,5,6,7

python main.py \
  --model_type albert \
  --model_name_or_path ../models/albert-large-v2/ \
  --model_arch albert-large-v2-last2h-dropout \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 128 \
  --overwrite_cache \
  --eval_all_checkpoints \
  --doc_stride 128 \
  --fp16