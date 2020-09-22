export CUDA_VISIBLE_DEVICES=7

#python main2nd.py \
#  --model_type roberta-base-last2h-conv-jaccard-soft-1 \
#  --model_arch LSTM \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --max_seq_length 150 \
#  --do_train \
#  --do_eval \
#  --char_embed_dim 8 \
#  --n_models 1 \
#  --lstm_hidden_size 16 \
#  --sentiment_dim 16 \
#  --encode_size 64 \
#  --per_gpu_train_batch_size 128 \
#  --per_gpu_eval_batch_size 512 \
#  --learning_rate 5e-3 \
#  --num_train_epochs 10 \
#  --eval_all_checkpoints \
#  --swa_first_epoch 5 \
#  --seed 42


python main2nd.py \
  --model_type roberta-base-squad2-last3h-conv-jaccard-soft-1/ \
  --model_arch CNN \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --max_seq_length 180 \
  --do_train \
  --do_eval \
  --char_embed_dim 16 \
  --n_models 1 \
  --cnn_dim 16 \
  --sentiment_dim 16 \
  --encode_size 32 \
  --kernel_size 3 \
  --per_gpu_train_batch_size 128 \
  --per_gpu_eval_batch_size 512 \
  --learning_rate 4e-3 \
  --num_train_epochs 5 \
  --eval_all_checkpoints \
  --swa_first_epoch 1 \
  --seed 42
#  --use_swa
#  --use_beam_search