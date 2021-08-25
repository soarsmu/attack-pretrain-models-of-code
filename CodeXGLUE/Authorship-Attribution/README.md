# Attack CodeBERT on Code Authorship Attribution Task

## Dataset

First, you need to download 3 datasets from [dataset](https://drive.google.com/drive/u/1/folders/1UGFFC5KYMRA-9F_VTsG_VcsZjAv7SG4i). Then, you need to decompress the 3 `tar.xz` files to the `dataset/data_folder`. For example:

```
pip install gdown
gdown https://drive.google.com/uc?id=1qMpwdaPASYFbX0gPEMSUlRtf_ErRkI-r
gdown https://drive.google.com/uc?id=1TXaLKEIVvkWZRwPQhUYeNAL4e11FgzDj
gdown https://drive.google.com/uc?id=1bBx04zqrpxNC0H5F6QObKByPDZ6QGZO2
xz -d gcjpy.tar.xz
tar -xvf gcjpy.tar
xz -d gcj.tar.xz
tar -xvf gcj.tar
xz -d java40.tar.xz
tar -xvf java40.tar
mkdir dataset/data_folder
mv gcjpy dataset/data_folder/
mv gcj dataset/data_folder/
mv java40 dataset/data_folder/
```

Then, you can run the following command to preprocess the datasets:

```
cd dataset
python process.py
```

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
CUDA_VISIBLE_DEVICES=0,2,4,5 python run.py \
    --output_dir=./saved_models/gcjpy \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --number_labels 66 \
    --do_eval \
    --train_data_file=../dataset/data_folder/processed_gcjpy/train.txt \
    --eval_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --test_data_file=../dataset/data_folder/processed_gcjpy/valid.txt \
    --epoch 40 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_gcjpy.log
```

### On Java dataset

We use full train data for fine-tuning. The training cost is 20 mins on 8*P100-16G. We use full valid data to evaluate during training.

```shell
cd code
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

## Attack

### On Python dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/gcjpy/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/gcjpy/checkpoint-best-f1
gdown https://drive.google.com/uc?id=14dOsW-_C0D1IINP2J4l2VqB-IAlGB15w
mv model.bin code/saved_models/gcjpy/checkpoint-best-f1/
```

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
    --csv_store_path ./attack_no_gi.csv \
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

### On Java dataset

If you don't want to be bothered by fine-tuning models, you can download the victim model into `code/saved_models/java40/checkpoint-best-f1` by [this link](https://drive.google.com/file/d/14_XxUo9Pcx4QnK50BUazBK6VbnP93pIC/view?usp=sharing).

```shell
pip install gdown
mkdir code/saved_models/java40/checkpoint-best-f1
gdown https://drive.google.com/uc?id=14_XxUo9Pcx4QnK50BUazBK6VbnP93pIC
mv model.bin code/saved_models/java40/checkpoint-best-f1/
```

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

| Dataset  |    ACC    |  ACC (attacked)    | F1| F1(attacked) |Recall| Recall(attacked)|
| -------- | :-------: |   :-------: | :-------: | :-------: | :-------: | :-------: |
| Python(70 labels) | **0.9129** |  |**0.8777**| |**0.8857**| |
| Java(41 labels) | **0.982** |  |**0.974**| |**0.9713**| |
