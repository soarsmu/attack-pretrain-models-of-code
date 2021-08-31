for index in range(11):
    a = index * 2000
    b = (index + 1) * 2000
    if (index + 1) % 8 == 0:
        print("""CUDA_VISIBLE_DEVICES={2} python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_no_gi_train_subs_{0}_{1}.csv \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=../preprocess/train_subs_{0}_{1}.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 16 \
    --seed 123456  2>&1 | tee attack_no_gi_train_subs_{0}_{1}.log""".format(a, b, (index % 2) * 4))
        print('\n\n')
    else:
        print("""CUDA_VISIBLE_DEVICES={2} python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_no_gi_train_subs_{0}_{1}.csv \
    --base_model=microsoft/graphcodebert-base \
    --eval_data_file=../preprocess/train_subs_{0}_{1}.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 16 \
    --seed 123456  2>&1 | tee attack_no_gi_train_subs_{0}_{1}.log &""".format(a, b, (index % 2) * 4))
        print('\n\n')


# CUDA_VISIBLE_DEVICES={2} python gi_attack.py \
#     --output_dir=./saved_models \
#     --model_type=roberta \
#     --tokenizer_name=microsoft/graphcodebert-base \
#     --model_name_or_path=microsoft/graphcodebert-base \
#     --csv_store_path ./attack_no_gi_train_subs_{0}_{1}.csv \
#     --base_model=microsoft/graphcodebert-base \
#     --eval_data_file=../preprocess/train_subs_{0}_{1}.jsonl \
#     --code_length 512 \
#     --data_flow_length 128 \
#     --eval_batch_size 16 \
#     --seed 123456  2>&1 | tee attack_no_gi_train_subs_{0}_{1}.log

