# Attack CodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download the dataset from [link](https://drive.google.com/file/d/1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe/view?usp=sharing). Then, you need to decompress the `.zip` file to the `dataset/data_folder`. For example:

```
pip install gdown
gdown https://drive.google.com/uc?id=1t0lmgVHAVpB1GxVqMXpXdU8ArJEQQfqe
unzip gcjpy.zip
cd dataset
mkdir data_folder
cd data_folder
mv ../../gcjpy ./
```

Then, you can run the following command to preprocess the datasets:

```
python process.py
```

❕**Notes:** The labels of preprocessed dataset rely on the directory list of your machine, so it's possible that the data generated on your side is quite different from ours. You may need to fine-tune your model again.

## Fine-tune CodeBERT

### Dependency

Users can try with the following docker image.

```
docker pull zhouyang996/codebert-attack:v1
```

Then, create a container using this docker image. An example is:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace zhouyang996/codebert-attack:v1
```

All the following scripts should run inside the docker container. 

❕**Notes:** This docker works fine with RTX 2080Ti GPUs and Tesla P100 GPUs. But if on RTX 30XX GPUs, it may take very long time to load the models to cuda. We think it's related to the CUDA version. Users can use the following command for a lower version:

```
docker run --name=codebert-attack --gpus all -it --mount type=bind,src=<codebase_path>,dst=/workspace pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
```

### On Python dataset

We use full train data for fine-tuning. The training cost is 10 mins on 4*P100-16G. We use full valid data to evaluate during training.

```shell
cd code
CUDA_VISIBLE_DEVICES=4,6 python run.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 66 \
    --do_train \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --epoch 30 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_gcjpy.log
```

## Attack

### On Python dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/gcjpy/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/gcjpy/checkpoint-best-f1
gdown https://drive.google.com/uc?id=14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w
mv model.bin code/saved_models/gcjpy/checkpoint-best-f1/
```

```
cd preprocess
CUDA_VISIBLE_DEVICES=1 python get_substitutes.py \
    --store_path ./data_folder/processed_gcjpy/valid_subs.jsonl \
    --base_model=microsoft/codebert-base-mlm \
    --eval_data_file=./data_folder/processed_gcjpy/valid.txt \
    --block_size 512
```

#### GA-Attack

```shell
cd code
python attack.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 66 \
    --do_eval \
    --use_ga \
    --csv_store_path ./attack_gi.csv \
    --language_type python \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --block_size 512 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee attack_gcjpy.log
```


#### MHM-LS
```shell
cd code
CUDA_VISIBLE_DEVICES=6 python mhm.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --number_labels 66 \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm_LS.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm_LS.log
```


#### MHM-Original
```shell
cd code
CUDA_VISIBLE_DEVICES=1 python mhm.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --number_labels 66 \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --csv_store_path ./attack_mhm_original.csv \
    --base_model=microsoft/codebert-base-mlm \
    --is_original_mhm \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee attack_mhm_original.log
```


## results 

| Dataset  |    ACC    |  ACC (attacked)    | F1| F1(attacked) |Recall| Recall(attacked)|
| -------- | :-------: |   :-------: | :-------: | :-------: | :-------: | :-------: |
| Python(66 labels) | **0.8806** |  |**0.824**| |**0.8258**| |
