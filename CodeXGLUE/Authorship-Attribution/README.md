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

### On Python dataset

16 mins on 4*P100-16G

```
CUDA_VISIBLE_DEVICES=2,4,5,7 python run.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 70 \
    --do_train \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 20 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_gcjpy.log
```


#### Evaluate
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 70 \
    --do_eval \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test_gcjpy.log
```

#### Attack
```shell
cd code
python attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 70 \
    --do_eval \
    --language_type python \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/test.txt \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_gcjpy.log
```


### On Java dataset
#### Train

20 mins in 8*P100-16G

```
python run.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 41 \
    --do_train \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_java40.log
```
#### Evaluate
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 41 \
    --do_eval \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_java40.log
```

#### Attack
```shell
cd code
python attack.py \
    --output_dir=./saved_models/java40 \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --language_type java \
    --number_labels 41 \
    --train_data_file=../dataset/data_folder/processed_java40/train.txt \
    --eval_data_file=../dataset/data_folder/processed_java40/valid.txt \
    --test_data_file=../dataset/data_folder/processed_java40/test.txt \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee attack_java40.log
```
## results 

on python 
eval_f1 = 0.8767
eval_precision = 0.9105
eval_recall = 0.8857

on java

eval_f1 = 0.974
eval_precision = 0.982
eval_recall = 0.9713