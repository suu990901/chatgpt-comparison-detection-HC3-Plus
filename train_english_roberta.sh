name=english_roberta
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
CUDA_VISIBLE_DEVICES=6 nohup /share/miniconda3/envs/llama/bin/python run_roberta.py \
--train_path data/en/train.jsonl \
--hc3_val_path data/en/val_hc3_QA.jsonl \
--hc3_si_val_path data/en/val_hc3_si.jsonl \
--model roberta-base \
--max_length 256 \
--batch_size 32 \
--save_path model/$name \
--tensorboard_dir tflog/$name \
--num_test_times 10 \
--epochs 2 \
--lr 5e-5 \
--seed 42 \
>log/$name.log 2>&1 &