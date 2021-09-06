## How to run

Please use the downloaded datasets.

### Fine-tune

```
cd code
python run.py \
    --output_dir=./saved_models/gcjpy \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --language_type python \
    --number_labels 66 \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 30 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train_gcjpy.log
```

### Attack

### ALERT

```
cd code
python gi_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_gi.csv \
    --number_labels 66 \
    --use_ga \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_gi.log
```

#### MHM-NS

```
cd code
python mhm_attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --csv_store_path ./attack_mhm.csv \
    --number_labels 66 \
    --base_model=microsoft/graphcodebert-base \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --code_length 512 \
    --data_flow_length 128 \
    --eval_batch_size 32 \
    --seed 123456  2>&1 | tee attack_mhm.log
```
