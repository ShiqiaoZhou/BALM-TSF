export CUDA_VISIBLE_DEVICES=0,1
model_name=BALM
train_epochs=100
learning_rate=0.01
llama_layers=6 # 6 for GPT2-small, 32 for llama
llm_dim=768
llm_model=GPT2

master_port=29501
num_process=2 # 1 for single GPU, 2 for two GPUs
batch_size=24
d_model=32
d_ff=128

comment='BALM-ETTm1'
wd_project=BALM-fewshot
version_num=BALM-fewshot

percent=10

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --llm_model 'GPT2'\
  --llm_dim 768 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --llm_model 'GPT2'\
  --llm_dim 768 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 32 \
  --lradj 'TST'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs 10 \
  --patience 20 \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --llm_model 'GPT2'\
  --llm_dim 768 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 16 \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 20 \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --llm_model 'GPT2'\
  --llm_dim 768 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 16 \
  --learning_rate 0.002 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --patience 20 \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \