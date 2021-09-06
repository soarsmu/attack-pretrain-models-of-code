## How to run

Please use the downloaded datasets.

### Fine-tune

```
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --epoch 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log
```

### Attack

### ALERT

```
cd code
python gi_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_genetic_2400_2800.csv \
    --base_model=microsoft/graphcodebert-base \
    --use_ga \
    --eval_data_file=../preprocess/dataset/test_subs_2400_2800.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 8 \
    --seed 123456  2>&1 | tee attack_gi_2400_2800.log
```

#### MHM-NS

```
cd code
python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhm_0_400.csv \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 16 \
    --seed 123456  2>&1 | tee attack_mhm_0_400.log
```
