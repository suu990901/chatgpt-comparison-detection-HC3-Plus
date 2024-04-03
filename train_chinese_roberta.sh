name=chinese_roberta
CUDA_VISIBLE_DEVICES=7 nohup /share/miniconda3/envs/llama/bin/python run_roberta.py \
--train_path data/zh/train.jsonl \
--hc3_val_path data/zh/val_hc3_QA.jsonl \
--hc3_si_val_path data/zh/val_hc3_si.jsonl \
--model hfl/chinese-roberta-wwm-ext \
--max_length 256 \
--batch_size 32 \
--save_path model/$name \
--tensorboard_dir tflog/$name \
--num_test_times 10 \
--epochs 2 \
--lr 5e-5 \
--seed 42 \
>log/$name.log 2>&1 &
