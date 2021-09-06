## How to run

Please use the downloaded datasets.

### Fine-tune

```
cd code
python run.py \
    --output_dir=./adv_saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../preprocess/dataset/adv_train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 24 \
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
    --tokenizer_name=microsoft/codebert-base-mlm \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --csv_store_path ./attack_genetic.csv \
    --base_model=microsoft/codebert-base-mlm \
    --use_ga \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_gi.log
```

#### MHM-NS

```
cd code
python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm_ls.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/train_subs.jsonl \
    --eval_data_file=../preprocess/dataset/test_subs_0_400.jsonl \
    --test_data_file=../preprocess/dataset/test_subs.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm_ls.log
```
