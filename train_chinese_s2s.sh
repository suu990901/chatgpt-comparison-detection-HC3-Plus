device=0
lr=1e-4
epochs=4
wp=0
name=chinese_t5
CUDA_VISIBLE_DEVICES=$device nohup /share/miniconda3/envs/llama/bin/python run_s2s.py \
--train_path data/zh/train.jsonl \
--hc3_val_path data/zh/val_hc3_QA.jsonl \
--hc3_si_val_path data/zh/val_hc3_si.jsonl \
--model chinese_tk_base \
--max_length 512 \
--batch_size 32 \
--save_path model/$name \
--tensorboard_dir tflog/$name \
--num_test_times 10 \
--lang zh \
--epochs $epochs \
--lr $lr \
--seed 42 \
--warm_up_ratio $wp \
--weight_decay 0.0 \
--accumulation_steps 1 \
>log/${name}.log 2>&1 &
