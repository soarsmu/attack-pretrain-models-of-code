## How to run

Please use the downloaded datasets.

### Fine-tune

```
cd code
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/valid_sampled.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --epoch 2 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log
```

### Attack

### ALERT

```
cd code
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --csv_store_path ./attack_base_result_GA.csv \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --use_ga \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 32 \
    --seed 123456 2>&1| tee attack_GA.log
```

#### MHM-NS

```
cd code
python mhm_attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/train_sampled.txt \
    --eval_data_file=../dataset/test_sampled_0_500.txt \
    --test_data_file=../dataset/test_sampled.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm.log
```
