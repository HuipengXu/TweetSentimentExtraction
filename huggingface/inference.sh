export CUDA_VISIBLE_DEVICES=4,5,6

python main.py \
  --model_type bert \
  --do_test \
  --do_lower_case \
  --predict_file test.json \
  --per_gpu_eval_batch_size 36 \
  --max_seq_length 180 \
  --overwrite_cache \
  --doc_stride 128
