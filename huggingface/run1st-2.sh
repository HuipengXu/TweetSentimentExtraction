export CUDA_VISIBLE_DEVICES=0,1,2



python main1st.py \
  --model_type roberta \
  --model_name_or_path ../models/distilroberta-base-squad2/ \
  --model_arch distilroberta-base-squad2-last3h-conv \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 128 \
  --eval_all_checkpoints \
  --use_jaccard_soft