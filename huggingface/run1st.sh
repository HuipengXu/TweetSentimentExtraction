export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2
CUDA_LAUNCH_BLOCKING=1

# ('distilbert', 'albert', 'camembert', 'bart', 'longformer', 'xlm-roberta', '
# roberta', 'bert', 'xlnet', 'flaubert', 'mobilebert', 'xlm', 'electra', 'reformer')

#python main1st.py \
#  --model_type roberta \
#  --model_name_or_path ../models/distilroberta-base/ \
#  --model_arch distilroberta-base-last3h \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 64 \
#  --per_gpu_eval_batch_size 64 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 3.0 \
#  --max_seq_length 128 \
#  --eval_all_checkpoints \
#  --use_jaccard_soft


#python main1st.py \
#  --model_type distilbert \
#  --model_name_or_path ../models/distilbert-base-uncased-squad2/ \
#  --model_arch distilbert-base-uncased-squad2-last2h \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 64 \
#  --per_gpu_eval_batch_size 16 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 8.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints


#python main1st.py \
#  --model_type bert \
#  --model_name_or_path ../models/bert-base-cased-squad2/ \
#  --model_arch bert-base-cased-squad2-last2h \
#  --do_train \
#  --do_eval \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 16 \
#  --per_gpu_eval_batch_size 16 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints

#
#python main1st.py \
#  --model_type bert \
#  --model_name_or_path ../models/bert-base-uncased-squad2/ \
#  --model_arch bert-base-uncased-squad2-last2h \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 64 \
#  --per_gpu_eval_batch_size 16 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --use_jaccard_soft


#python main1st.py \
#  --model_type roberta \
#  --model_name_or_path ../models/roberta-base-squad2/ \
#  --model_arch roberta-base-squad2-last3h-conv \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 48 \
#  --per_gpu_eval_batch_size 48 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --use_jaccard_soft


#python main1st.py \
#  --model_type roberta \
#  --model_name_or_path ../models/roberta-large/ \
#  --model_arch roberta-large-last2h-conv \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 8 \
#  --per_gpu_eval_batch_size 4 \
#  --learning_rate 5e-5 \
#  --num_train_epochs 5.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --use_jaccard_soft \
#  --fp16

python main1st.py \
  --model_type roberta \
  --model_name_or_path ../models/roberta-large/ \
  --model_arch roberta-large-last3h-conv \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file clean_train.csv \
  --predict_file clean_valid.csv \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --learning_rate 7e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --save_steps 100 \
  --eval_all_checkpoints \
  --use_jaccard_soft \
  --fp16


#python main1st.py \
#  --model_type bert \
#  --model_name_or_path ../models/bert-large-uncased-wwm-squad \
#  --model_arch bert-large-uncased-wwm-squad-last2h \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 24 \
#  --per_gpu_eval_batch_size 24 \
#  --learning_rate 7e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --use_jaccard_soft \
#  --fp16


#python main1st.py \
#  --model_type albert \
#  --model_name_or_path ../models/albert-xxlarge-v2-squad2 \
#  --eval_model_dir ./results1/albert-xxlarge-v2-squad2-last2h-result-2 \
#  --model_arch albert-xxlarge-v2-squad2-last2h \
#  --do_eval \
#  --do_lower_case \
#  --train_file clean_train.csv \
#  --predict_file clean_valid.csv \
#  --per_gpu_train_batch_size 4 \
#  --per_gpu_eval_batch_size 4 \
#  --learning_rate 2e-5 \
#  --num_train_epochs 4.0 \
#  --max_seq_length 128 \
#  --overwrite_cache \
#  --eval_all_checkpoints \
#  --fp16