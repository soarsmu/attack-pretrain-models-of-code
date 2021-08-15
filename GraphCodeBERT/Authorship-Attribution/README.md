# Code Authorship Attribution Task

## Data Preparation

下载[数据集](https://drive.google.com/drive/u/1/folders/1UGFFC5KYMRA-9F_VTsG_VcsZjAv7SG4i)到`dataset/data_folder`下，然后运行

```
python process.py
```

来准备数据集.

## Fine-tuning

### On OJ dataset
```
python run.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 70 \
    --do_train \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 20 \
    --code_length 350 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_gcjpy.log
```
{acc: 0.9214}

### On Java dataset
```
CUDA_VISIBLE_DEVICES=1 python run.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 41 \
    --do_train \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 10 \
    --code_length 350 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_java40.log
```
{acc: 0.9904}

## Attack

### On Python dataset

```
python attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 70 \
    --do_eval \
    --language_type python \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 20 \
    --code_length 350 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_gcjpy.log
```

### On Java dataset

```
CUDA_VISIBLE_DEVICES=1 python attack.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --config_name=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --number_labels 41 \
    --do_eval \
    --language_type java \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 10 \
    --code_length 350 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_java40.log
```