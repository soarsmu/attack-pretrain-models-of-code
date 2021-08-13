# Code Authorship Attribution Task

## Data Preparation

First, you need to download datasets to [dataset](https://drive.google.com/drive/u/1/folders/1UGFFC5KYMRA-9F_VTsG_VcsZjAv7SG4i) the `dataset/data_folder`. Then, you need to decompress the three `tar.xz` files. For example:

```
xz -d gcjpy.tar.xz
tar -xvf gcjpy.tar
```

Then, you can run the following command to preprocess the datasets:

```
python process.py
```

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
    --train_batch_size 10 \
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
    --epoch 5 \
    --code_length 350 \
    --data_flow_length 128 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_java40.log
```
<<<<<<< HEAD
{acc: 0.9371}


#### ATTACK
=======
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
>>>>>>> 075cf6e8ec7bfd1d18c6751602d887f0f7fb7170
