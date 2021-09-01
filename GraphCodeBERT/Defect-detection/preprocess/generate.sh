CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_0_2000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 0 2000 &



CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
        --store_path ./train_subs_2000_4000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 2000 4000 &



CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_4000_6000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 4000 6000 &



CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
        --store_path ./train_subs_6000_8000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 6000 8000 &



CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_8000_10000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 8000 10000 &



CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
        --store_path ./train_subs_10000_12000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 10000 12000 &



CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_12000_14000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 12000 14000 &



CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
        --store_path ./train_subs_14000_16000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl  \
        --block_size 512 \
        --index 14000 16000



CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_16000_18000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 16000 18000 &



CUDA_VISIBLE_DEVICES=4 python get_substitutes.py \
        --store_path ./train_subs_18000_20000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 18000 20000 &



CUDA_VISIBLE_DEVICES=0 python get_substitutes.py \
        --store_path ./train_subs_20000_22000.jsonl \
        --base_model=microsoft/graphcodebert-base \
        --eval_data_file=./dataset/train.jsonl \
        --block_size 512 \
        --index 20000 22000 &



